"""Proposition tier: atomic single-claim rows derived from episode summaries.

An episode is a ~16-minute, multi-speaker, multi-topic summary in ONE 768-d
vector. That average is mildly near everything and close to nothing, so
specific questions collapse the candidate pool and whatever wins is a grab-bag
paragraph whose relevant clause is a fraction of the injected text. Splitting
each episode into standalone claims gives every vector exactly one meaning.

Three parts, deliberately separable so each is testable alone:
  - generate_propositions(text)  -> LLM split, pure text in / list[str] out
  - store_propositions(...)      -> write + embed (idempotent per episode)
  - search_propositions(...)     -> the same 3-channel hybrid + decay as
                                    memory.episodic_search, over propositions

The LLM call goes through llm.client.generate_memory, which already blocks on
BOTH the conversation-idle and vision-idle gates -- so backfilling a large
corpus cannot steal the GPU from a live turn.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone

import config
from db.connection import get_pool
from memory.manager import embed
from memory.decay import decay_multiplier
from memory.retrieval import _fuse, SEMANTIC_DISTANCE_MAX

log = logging.getLogger(__name__)


PROPOSITION_PROMPT = """Break the following conversation summary into atomic \
statements — the individual things worth remembering later.

Rules:
- ONE claim per line. No numbering, no bullets, no preamble, no commentary.
- Each line must stand completely on its own. Replace every pronoun and \
reference with the actual name ("Dan's cats are named Dexter and Preston", \
never "his cats are named that").
- Keep the specific, recallable details: names, numbers, preferences, \
decisions, events, opinions, things someone said or did.
- Drop pure pleasantries, greetings, and filler. If someone merely said hello \
and left, that is not worth a line.
- Write between 1 and {max_n} lines. Fewer is fine if the summary is thin.
- Do not invent anything. Only state what the summary actually says.

SUMMARY:
{text}

STATEMENTS:"""


_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s*")
# A model that ignores "no preamble" typically opens with one of these.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here (?:are|is)\b|statements?\s*:|the following\b|sure[,!]|okay[,!])",
    re.IGNORECASE,
)


def _clean_line(raw: str) -> str | None:
    """Normalize one model output line to a proposition, or None to drop it."""
    s = _BULLET_RE.sub("", (raw or "").strip())
    s = s.strip().strip("`").strip()
    if not s or _PREAMBLE_RE.match(s):
        return None
    if len(s) < config.PROPOSITION_MIN_CHARS or len(s) > config.PROPOSITION_MAX_CHARS:
        return None
    # A line with no letters is punctuation noise / a separator rule.
    if not any(c.isalpha() for c in s):
        return None
    return s


def parse_propositions(raw: str) -> list[str]:
    """Newline-delimited model output -> deduped, cleaned proposition list.

    Deliberately NOT structured output: llama.cpp drops strict `oneOf` and
    Qwen3.6 omits optional schema fields, so a JSON contract here buys
    fragility rather than safety (see the structured-output gotchas in
    CLAUDE.md). Line-per-claim is trivially parseable and degrades to "fewer
    lines" instead of "parse failure -> lost episode"."""
    out: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        s = _clean_line(line)
        if s is None:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= config.PROPOSITION_MAX_PER_EPISODE:
            break
    return out


async def generate_propositions(episode_text: str) -> list[str]:
    """Split one episode summary into atomic claims via the memory-tier LLM.

    Returns [] on any failure -- a missing proposition set degrades retrieval
    to the episode tier, which still works. Never raises into the caller's
    write path."""
    if not episode_text or not episode_text.strip():
        return []
    from llm.client import generate_memory

    prompt = PROPOSITION_PROMPT.format(
        text=episode_text.strip(),
        max_n=config.PROPOSITION_MAX_PER_EPISODE,
    )
    try:
        # thinking=False: this is a mechanical decomposition, not reasoning --
        # and the thinking trace would blow past the line-per-claim contract.
        raw = await generate_memory(prompt, thinking=False)
    except Exception:
        log.warning("[PROP] generation failed; episode keeps episode-tier only",
                    exc_info=True)
        return []
    props = parse_propositions(raw)
    if not props:
        log.warning("[PROP] model returned no usable lines (%d chars raw)", len(raw or ""))
    return props


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()


async def store_propositions(episode_id: int, span_end, texts: list[str]) -> int:
    """Embed and insert propositions for one episode. Returns rows written.

    Idempotent per (episode_id, content_hash) -- re-running a backfill over an
    already-split episode inserts nothing. `span_end` is denormalized from the
    parent so decay ranking needs no join; accepts a datetime or epoch float."""
    if not texts:
        return 0
    if isinstance(span_end, (int, float)):
        span_end = datetime.fromtimestamp(span_end, tz=timezone.utc)

    pool = await get_pool()
    written = 0
    for t in texts:
        try:
            emb = await embed(t)
        except Exception:
            log.warning("[PROP] embed failed for ep%s, skipping line", episode_id,
                        exc_info=True)
            continue
        row = await pool.fetchrow(
            """INSERT INTO propositions (episode_id, text, embedding, span_end, content_hash)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (episode_id, content_hash) WHERE content_hash IS NOT NULL
               DO NOTHING
               RETURNING id""",
            episode_id, t, emb, span_end, _hash(t),
        )
        if row is not None:
            written += 1
    return written


async def split_and_store(episode_id: int, span_end, episode_text: str) -> int:
    """generate + store in one call. Used by the write path and the backfill."""
    props = await generate_propositions(episode_text)
    if not props:
        return 0
    n = await store_propositions(episode_id, span_end, props)
    log.info("[PROP] ep%s -> %d proposition(s) stored", episode_id, n)
    return n


# --------------------------------------------------------------------------
# Retrieval -- mirrors memory.episodic_search channel-for-channel so the two
# tiers cannot drift (the divergence that silently orphaned the coref resolver
# for two weeks in 2026-07).
# --------------------------------------------------------------------------

async def _semantic(pool, query_embedding, limit):
    rows = await pool.fetch(
        """SELECT id, embedding <=> $1 AS distance FROM propositions
           WHERE embedding IS NOT NULL AND embedding <=> $1 < $3
           ORDER BY embedding <=> $1
           LIMIT $2""",
        query_embedding, limit, SEMANTIC_DISTANCE_MAX,
    )
    return [(r["id"], i, float(r["distance"])) for i, r in enumerate(rows)]


# IDF-weighted lexical ranking (2026-08-15, default OFF).
#
# `ts_rank` over an OR-of-lexemes weights every query term equally, so a
# throwaway word outranks the one term that carries the question. Measured
# 2026-08-14: "we've spoken about Radiohead before, correct?" ranked five
# propositions about Timmy CORRECTING people (matching "correct") above the two
# that actually mention Radiohead, and Timmy answered "I don't remember that
# conversation" -- an honest reply built on a broken lookup, which is worse than
# a fabrication because nothing about it looks wrong. Same night, "do you
# remember anything from OpenSauce?" retrieved none of the 40 stored OpenSauce
# items.
#
# Score = sum over matched query terms of ln(N/df), divided by (1 + ln(doclen)).
#   - IDF is what ts_rank lacks: it makes "radiohead" (df 2) worth far more
#     than "correct" (df ~40).
#   - Length normalisation is load-bearing. Without it, long propositions win
#     by matching more terms; with MAX(idf) instead of SUM, a hapax like
#     "N-O-T-S-A-M" (df 1, idf 7.16) beats "opensauce" (df 31, idf 3.73), so
#     rare JUNK wins. Sum-then-normalise was the only variant of the three that
#     improved both probe queries.
#
# Measured on the live corpus (1289 propositions): Radiohead 0/5 -> 3/5
# relevant in the top five, including the correct "Dan's favorite Radiohead
# song is Paranoid Android". OpenSauce improved but remains noisy -- the query
# carries rare-but-meaningless filler ("hopefully", "curious", "seriously")
# that still earns IDF. Default OFF pending a real eval set; flip
# `prop_idf_ranking` live to A/B it.
_IDF_FTS_SQL = """
    WITH terms AS (
      SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('english', $1))) AS w),
    total AS (SELECT GREATEST(count(*),1)::float n FROM propositions),
    df AS (
      SELECT t.w,
             GREATEST((SELECT count(*) FROM propositions p
                        WHERE p.content_tsv @@ plainto_tsquery('english', t.w)),1)::float d
      FROM terms t),
    tq AS (SELECT string_agg(quote_literal(w), ' | ')::tsquery q FROM terms),
    cand AS (
      SELECT p.id, p.content_tsv, GREATEST(length(p.content_tsv),1)::float dl
      FROM propositions p, tq
      WHERE tq.q IS NOT NULL AND p.content_tsv @@ tq.q)
    SELECT c.id
    FROM cand c, df, total
    GROUP BY c.id, c.dl
    ORDER BY SUM(CASE WHEN c.content_tsv @@ plainto_tsquery('english', df.w)
                      THEN ln(total.n / df.d) ELSE 0 END) / (1 + ln(c.dl)) DESC,
             c.id
    LIMIT $2"""


async def _fts(pool, query, limit):
    """OR-of-lexemes. See memory.episodic_search._fts -- plainto_tsquery ANDs
    every term and returned nothing on half of real questions.

    With `prop_idf_ranking` on, ranking switches to IDF-weighted scoring (see
    _IDF_FTS_SQL above); the candidate set is identical either way, only the
    ORDER BY changes."""
    try:
        from persistence import runtime_toggles as _rt
        _idf = bool(_rt.get("prop_idf_ranking"))
    except Exception:
        _idf = False
    if _idf:
        rows = await pool.fetch(_IDF_FTS_SQL, query, limit)
        return [(r["id"], i) for i, r in enumerate(rows)]
    rows = await pool.fetch(
        """WITH tq AS (
             SELECT string_agg(quote_literal(w), ' | ')::tsquery AS q
             FROM unnest(tsvector_to_array(to_tsvector('english', $1))) AS w)
           SELECT p.id FROM propositions p, tq
           WHERE tq.q IS NOT NULL AND p.content_tsv @@ tq.q
           ORDER BY ts_rank(p.content_tsv, tq.q) DESC, p.id
           LIMIT $2""",
        query, limit,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


async def _trigram(pool, query, limit):
    """word_similarity against the best-matching extent. Propositions are SHORT
    (one claim), so unlike the episode tier a whole-document `%` would actually
    have been viable here -- but keeping the two tiers identical matters more
    than saving one function call, and word_similarity is still the correct
    comparison when a long question meets a short claim."""
    rows = await pool.fetch(
        """SELECT id FROM propositions
           WHERE word_similarity($1, text) >= $3
           ORDER BY word_similarity($1, text) DESC, id
           LIMIT $2""",
        query, limit, config.TRIGRAM_WORD_SIM_FLOOR,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


async def touch_propositions(ids: list[int]) -> None:
    if not ids:
        return
    pool = await get_pool()
    await pool.execute(
        """UPDATE propositions SET access_count = COALESCE(access_count,0) + 1,
           accessed_at = NOW() WHERE id = ANY($1::int[])""", ids)


async def search_propositions(query_text: str, now: datetime, *,
                              top_k: int | None = None,
                              embed_query: str | None = None) -> list[dict]:
    """Top-k propositions for `query_text`, recency-decayed.

    Same shape as memory.episodic_search.search_episodes: each dict carries id,
    text, episode_id, span_end, access_count, score (post-decay) and base_score.
    `embed_query` lets the caller embed a coref-resolved / context-blended
    string while FTS+trigram keep the bare utterance."""
    if top_k is None:
        top_k = config.PROPOSITION_TOP_K
    candidates = config.RETRIEVAL_CANDIDATES
    pool = await get_pool()

    query_emb = await embed(embed_query or query_text)
    semantic = await _semantic(pool, query_emb, candidates)
    fts = await _fts(pool, query_text, candidates)
    trigram = await _trigram(pool, query_text, candidates)

    # Tier-specific weights -- short claims want a quieter trigram channel than
    # long episode summaries do. See config.PROPOSITION_RRF_W_* for the measured
    # rationale (and for the intuitive-but-wrong fix not to retry).
    fused = _fuse(semantic, fts, trigram,
                  w_semantic=config.PROPOSITION_RRF_W_SEMANTIC,
                  w_fts=config.PROPOSITION_RRF_W_FTS,
                  w_trigram=config.PROPOSITION_RRF_W_TRIGRAM)
    if not fused:
        log.info("[prop_search] no candidates for %r", query_text[:60])
        return []

    rows = await pool.fetch(
        """SELECT id, text, episode_id, span_end, COALESCE(access_count, 0) AS access_count
           FROM propositions WHERE id = ANY($1::int[])""",
        list(fused.keys()),
    )
    scored = []
    for r in rows:
        base = fused[r["id"]]
        mult = decay_multiplier(r["span_end"], now, r["access_count"])
        scored.append({
            "id": r["id"], "text": r["text"], "episode_id": r["episode_id"],
            "span_end": r["span_end"], "access_count": r["access_count"],
            "base_score": base, "score": base * mult,
        })
    scored.sort(key=lambda e: e["score"], reverse=True)

    if config.PROPOSITION_DEDUPE_BY_EPISODE:
        # Keep the highest-scoring claim per parent episode. A chatty episode
        # otherwise spends every slot on its own claims, narrowing what the
        # prompt sees to a single conversation.
        seen_eps: set[int] = set()
        deduped = []
        for e in scored:
            if e["episode_id"] in seen_eps:
                continue
            seen_eps.add(e["episode_id"])
            deduped.append(e)
            if len(deduped) >= top_k:
                break
        scored = deduped

    top = scored[:top_k]
    if top:
        await touch_propositions([e["id"] for e in top])
    log.info("[prop_search] %r -> %d candidate(s), top score=%.5f",
             query_text[:60], len(scored), top[0]["score"] if top else 0.0)
    return top
