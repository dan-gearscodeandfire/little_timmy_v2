"""Hybrid retrieval pipeline: semantic + FTS + trigram with RRF fusion."""

import asyncio
import logging
from dataclasses import dataclass
from db.connection import get_pool
from memory.manager import embed, touch_memories
import config

log = logging.getLogger(__name__)


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
