"""Hybrid retrieval pipeline: semantic + FTS + trigram with RRF fusion."""

import asyncio
import logging
import re
from dataclasses import dataclass
from db.connection import get_pool
from memory.manager import embed, touch_memories
from persistence import runtime_toggles
import config

log = logging.getLogger(__name__)

# Gate for query resolution. A turn pays the resolver call
# (llm.client.resolve_query on :8093) ONLY when its utterance is a SHORT,
# query-like elliptical follow-up carrying a deictic reference -- e.g.
# "what does he do?", "remind me about her". Three conjoined conditions:
#   1. deixis present (a pronoun/reference needing a conversational antecedent),
#   2. within config.RESOLVE_MAX_WORDS, and
#   3. query-like: ends with '?' or opens with a wh-/aux-/recall-verb.
# The resolver is decode-bound (regenerates ~the utterance), so a long
# declarative/banter turn that merely *contains* a pronoun ("That's why I made
# you, because he has so much independence") costs 500-800ms while gaining
# nothing over the embedding blend, which already carries its lexical signal.
# Skipped turns fall back to the blend -- same fail-safe contract as a resolver
# miss, so tightening this gate can never poison retrieval. Word-boundary,
# case-insensitive. Measured 2026-06-18 (ops/elliptical_*.py); latency-gated
# 2026-06-22 (supervisor_issues.md, long-declarative spikes).
_DEIXIS_RE = re.compile(
    r"\b(it|its|it's|that|there|them|they|their|theirs|he|him|his|she|her|hers|those)\b",
    re.IGNORECASE,
)
# Query-like opener: wh-words, auxiliaries/modals, or recall imperatives. Paired
# with the trailing '?' check this keeps the resolver on follow-up *questions*
# and off declarative statements that happen to contain a pronoun.
_QUERY_LEAD_RE = re.compile(
    r"^(what|whats|who|whom|whose|which|where|when|why|how|"
    r"do|does|did|is|are|am|was|were|can|could|would|should|will|"
    r"has|have|had|may|might|"
    r"tell|remind|remember|name|list|show|give|find)\b",
    re.IGNORECASE,
)


def _needs_resolution(query: str) -> bool:
    """True only for short, query-like utterances with a deictic reference.

    See the module comment above _DEIXIS_RE: the resolver is decode-bound, so
    restricting it to elliptical follow-up *questions* (deixis + word cap +
    query-like opener/'?') keeps the latency win without poisoning retrieval --
    everything skipped falls back to the embedding blend."""
    q = (query or "").strip()
    if not _DEIXIS_RE.search(q):
        return False
    if len(q.split()) > config.RESOLVE_MAX_WORDS:
        return False
    return q.endswith("?") or bool(_QUERY_LEAD_RE.match(q))


@dataclass
class RetrievedMemory:
    id: int
    type: str
    content: str
    score: float
    created_at: object


# Cosine-distance floor for the semantic channel. pgvector's <=> returns
# cosine distance in [0, 2] (0=identical, 1=orthogonal, 2=opposite).
# Tuned 2026-05-06 against nomic-embed-text + the live LT memory corpus:
#   genuinely relevant queries land at distance 0.24-0.40
#   tangentially related ~0.45-0.55
#   nonsense / completely unrelated ~0.55-0.67 (note: nomic-embed-text never
#   produces large distances between English sentences — there's no "1.0+"
#   noise floor like with raw averaged word vectors)
# 0.50 is the cliff that keeps real hits and rejects everything below.
SEMANTIC_DISTANCE_MAX = 0.50


async def _semantic_search(pool, query_embedding, limit: int) -> list[tuple[int, int, float]]:
    """Returns list of (memory_id, rank, cosine_distance), filtered by the
    cosine-distance floor. The distance is surfaced (not just used for the
    floor) so _fuse can fold it back in as a tiebreaker for the semantic
    channel -- see RRF_COSINE_BONUS in config."""
    rows = await pool.fetch(
        """SELECT id, embedding <=> $1 AS distance FROM memories
           WHERE embedding <=> $1 < $3
           ORDER BY embedding <=> $1
           LIMIT $2""",
        query_embedding,
        limit,
        SEMANTIC_DISTANCE_MAX,
    )
    return [(r["id"], i, float(r["distance"])) for i, r in enumerate(rows)]


async def _fts_search(pool, query: str, limit: int) -> list[tuple[int, int]]:
    rows = await pool.fetch(
        """SELECT id FROM memories
           WHERE content_tsv @@ plainto_tsquery('english', $1)
           ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $1)) DESC
           LIMIT $2""",
        query,
        limit,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


async def _trigram_search(pool, query: str, limit: int) -> list[tuple[int, int]]:
    rows = await pool.fetch(
        """SELECT id FROM memories
           WHERE content % $1
           ORDER BY similarity(content, $1) DESC
           LIMIT $2""",
        query,
        limit,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


def _fuse(
    semantic: list[tuple[int, int, float]],
    fts: list[tuple[int, int]],
    trigram: list[tuple[int, int]],
    *,
    k: int | None = None,
    w_semantic: float | None = None,
    w_fts: float | None = None,
    w_trigram: float | None = None,
    cosine_bonus: float | None = None,
) -> dict[int, float]:
    """Weighted reciprocal-rank fusion. Returns {memory_id: fused_score}.

    Each channel contributes weight * 1/(k + rank + 1) -- still scale-free and
    rank-based, so RRF's robustness holds; the weights just rebalance how
    loudly each channel votes (semantic > fts > trigram). The semantic channel
    additionally gets an additive cosine bonus: distances are normalized within
    the kept band [0, SEMANTIC_DISTANCE_MAX) to a (0,1] similarity so genuinely
    close hits outrank barely-admitted ones instead of tying on rank alone.

    Setting all weights to 1.0 and cosine_bonus to 0.0 reproduces the original
    equal-weight, rank-only RRF exactly (the A/B control).
    """
    if k is None:
        k = config.RRF_K
    if w_semantic is None:
        w_semantic = config.RRF_W_SEMANTIC
    if w_fts is None:
        w_fts = config.RRF_W_FTS
    if w_trigram is None:
        w_trigram = config.RRF_W_TRIGRAM
    if cosine_bonus is None:
        cosine_bonus = config.RRF_COSINE_BONUS

    scores: dict[int, float] = {}

    for mem_id, rank, distance in semantic:
        scores[mem_id] = scores.get(mem_id, 0.0) + w_semantic * (1.0 / (k + rank + 1))
        if cosine_bonus and SEMANTIC_DISTANCE_MAX > 0:
            # Normalize the kept band to (0,1]: dist=0 -> 1.0, dist=floor -> ~0.
            sim = 1.0 - (distance / SEMANTIC_DISTANCE_MAX)
            if sim > 0:
                scores[mem_id] += w_semantic * cosine_bonus * sim

    for mem_id, rank in fts:
        scores[mem_id] = scores.get(mem_id, 0.0) + w_fts * (1.0 / (k + rank + 1))

    for mem_id, rank in trigram:
        scores[mem_id] = scores.get(mem_id, 0.0) + w_trigram * (1.0 / (k + rank + 1))

    return scores


def _build_semantic_query(query: str, context_turns: list | None) -> str:
    """Build the embedding-query string for the semantic channel.

    When coreference is enabled and prior turns are supplied, prepend the last
    few turns (role-tagged, char-capped) with the current utterance last and
    full so it stays the dominant signal in the averaged embedding. This lets
    elliptical/pronoun follow-ups ("what about her?") embed near the antecedent
    from earlier in the conversation. Returns the bare query unchanged when
    coreference is off or there's no context.
    """
    if not config.COREFERENCE_ENABLED or not context_turns:
        return query
    cap = config.CONTEXT_TURN_CHAR_CAP
    # The caller passes the WIDER resolver window (RESOLVE_CONTEXT_TURNS); the
    # embedding blend stays at CONTEXT_TURNS so the extra history can't dilute the
    # averaged vector. Take the most-recent CONTEXT_TURNS only.
    context_turns = context_turns[-config.CONTEXT_TURNS:]
    parts = []
    for t in context_turns:
        content = (getattr(t, "content", "") or "").strip()
        if not content:
            continue
        role = getattr(t, "role", "user")
        parts.append(f"{role}: {content[:cap]}")
    if not parts:
        return query
    parts.append(query)  # current utterance last, uncapped -> dominant
    return "\n".join(parts)


async def retrieve(
    query: str,
    top_k: int | None = None,
    context_turns: list | None = None,
) -> list[RetrievedMemory]:
    """Run hybrid retrieval: semantic + FTS + trigram, merge with weighted RRF.

    `query` is the bare current utterance; it drives the FTS and trigram
    channels and is the trailing segment of the semantic query. `context_turns`
    (prior conversation Turns, oldest-first, current excluded) are blended into
    the semantic channel's embedding query only -- see _build_semantic_query.
    """
    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K
    candidates = config.RETRIEVAL_CANDIDATES

    pool = await get_pool()

    # Semantic-query construction. Default: the role-tagged context blend
    # (_build_semantic_query). When query_resolution_enabled AND the utterance is
    # a short query-like deictic follow-up (_needs_resolution) AND we have
    # context, rewrite it into a standalone query via :8093
    # FIRST and embed THAT instead -- measured to beat the blend on elliptical
    # follow-ups (MRR 0.71->0.85). Resolver failure/empty -> fall back to the
    # blend (graceful). FTS/trigram below always use the bare `query`, unaffected.
    semantic_query = None
    resolved = None
    if (runtime_toggles.get("query_resolution_enabled")
            and context_turns and _needs_resolution(query)):
        from llm import client  # local import: avoid any import-time cycle
        # Resolver sees the full (wider) window. Char-cap each turn so a long
        # monologue across RESOLVE_CONTEXT_TURNS can't overflow :8093's -c 2048.
        rcap = config.CONTEXT_TURN_CHAR_CAP
        context_text = "\n".join(
            f"{getattr(t, 'role', 'user')}: {(getattr(t, 'content', '') or '').strip()[:rcap]}"
            for t in context_turns if (getattr(t, "content", "") or "").strip()
        )
        import time
        t0 = time.perf_counter()
        resolved = await client.resolve_query(query, context_text)
        resolution_ms = int((time.perf_counter() - t0) * 1000)
        # Publish the per-turn resolve cost to the HUDs (LT-OS WS + Booth poll),
        # mirroring the classifier-latency pip. Non-fatal if web.app isn't up.
        try:
            from web.app import broadcast_event, update_metrics, record_stage
            update_metrics(last_resolution_ms=resolution_ms)
            # Only logged when the deixis gate fires -> this series' sample count
            # is the resolution-fired turn count (the "deictic turn" condition).
            record_stage("stage:resolution", resolution_ms)
            await broadcast_event("resolution_metric",
                                  {"ms": resolution_ms, "resolved": bool(resolved)})
        except Exception:
            log.debug("[RESOLVE] latency publish failed (non-fatal)", exc_info=True)
        if resolved:
            semantic_query = resolved
            log.info("[RESOLVE] %r -> %r (%dms)", query, resolved, resolution_ms)
    if semantic_query is None:
        semantic_query = _build_semantic_query(query, context_turns)
    query_emb = await embed(semantic_query)

    # Run all three searches in parallel. Semantic uses the (possibly
    # context-augmented) embedding; FTS/trigram use the bare utterance.
    semantic, fts, trigram = await asyncio.gather(
        _semantic_search(pool, query_emb, candidates),
        _fts_search(pool, query, candidates),
        _trigram_search(pool, query, candidates),
    )

    fused = _fuse(semantic, fts, trigram)
    merged_ids = sorted(fused, key=lambda mid: fused[mid], reverse=True)[:top_k]
    if not merged_ids:
        return []

    # Fetch full rows for top results
    placeholders = ", ".join(f"${i+1}" for i in range(len(merged_ids)))
    rows = await pool.fetch(
        f"""SELECT id, type, content, created_at FROM memories
            WHERE id IN ({placeholders})""",
        *merged_ids,
    )
    row_map = {r["id"]: r for r in rows}

    results = []
    for mid in merged_ids:
        if mid in row_map:
            r = row_map[mid]
            results.append(RetrievedMemory(
                id=r["id"],
                type=r["type"],
                content=r["content"],
                score=fused[mid],  # real weighted-RRF score (was 1/(rank+1))
                created_at=r["created_at"],
            ))

    # Batch access-stat update in background (one query for all hits)
    if results:
        asyncio.create_task(touch_memories([r.id for r in results]))

    # best_dist = closest cosine distance among admitted semantic hits (the
    # A/B observability hook; None when the semantic channel returned nothing).
    best_dist = min((d for _, _, d in semantic), default=None)
    top_score = results[0].score if results else 0.0
    log.info(
        "Retrieved %d memories (semantic=%d, fts=%d, trigram=%d; "
        "top_score=%.4f, best_dist=%s%s)",
        len(results), len(semantic), len(fts), len(trigram),
        top_score,
        f"{best_dist:.3f}" if best_dist is not None else "n/a",
        ", coref" if (config.COREFERENCE_ENABLED and context_turns) else "",
    )
    return results
