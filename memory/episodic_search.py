"""Semantic search over `episodes`, recency-decayed (plan Session 5).

Three layers, kept DISTINCT from the other two recall paths:
  - memory.retrieval  -> the `memories` vector tier (frozen),
  - memory.temporal + manager.query_episodes_by_range -> deterministic DATE-RANGE
    recall over episodes (recall_temporal),
  - THIS module -> SIMILARITY recall over episodes (recall_semantic): "find the
    time I mentioned something like X", with no date in the query.

Pipeline: embed query -> vector + FTS + trigram channels over `episodes` ->
weighted RRF fusion (reuses memory.retrieval._fuse, same channel weights) ->
multiply each episode's fused score by its RECENCY×USAGE decay (memory.decay) so
a fresh episode outranks a stale one of equal similarity. Reads only rows with a
non-NULL embedding (the partial HNSW index), so it returns nothing until
EMBED_EPISODES has been writing/backfilled — which is exactly the corpus gate.

Pure retrieval: no flag checks here (the router gates on RECALL_SEMANTIC_ENABLED),
so this is unit-testable in isolation against a seeded local DB.
"""

from __future__ import annotations

import logging
from datetime import datetime

import config
from db.connection import get_pool
from memory.manager import embed, touch_episodes
from memory.decay import decay_multiplier
from memory.retrieval import _fuse, SEMANTIC_DISTANCE_MAX

log = logging.getLogger(__name__)


async def _semantic(pool, query_embedding, limit):
    rows = await pool.fetch(
        """SELECT id, embedding <=> $1 AS distance FROM episodes
           WHERE embedding IS NOT NULL AND embedding <=> $1 < $3
           ORDER BY embedding <=> $1
           LIMIT $2""",
        query_embedding, limit, SEMANTIC_DISTANCE_MAX,
    )
    return [(r["id"], i, float(r["distance"])) for i, r in enumerate(rows)]


async def _fts(pool, query, limit):
    rows = await pool.fetch(
        """SELECT id FROM episodes
           WHERE content_tsv @@ plainto_tsquery('english', $1)
           ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $1)) DESC
           LIMIT $2""",
        query, limit,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


async def _trigram(pool, query, limit):
    rows = await pool.fetch(
        """SELECT id FROM episodes
           WHERE text % $1
           ORDER BY similarity(text, $1) DESC
           LIMIT $2""",
        query, limit,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


async def search_episodes(query_text: str, now: datetime, *,
                          top_k: int | None = None,
                          embed_query: str | None = None) -> list[dict]:
    """Return up to `top_k` episodes most relevant to `query_text`, recency-
    decayed. Each dict: id, text, span_start, span_end, access_count, score
    (post-decay), base_score (pre-decay fusion). `now` is the tz-aware query
    instant used for decay. Empty list when nothing clears the distance floor /
    no embedded episodes exist yet.

    `embed_query` (optional): the string to embed for the SEMANTIC channel when
    it should differ from the lexical `query_text` used by FTS/trigram. Mirrors
    memory.retrieval.retrieve(): the caller can pass a coref-blended context
    string so elliptical follow-ups embed near their antecedent, while FTS/
    trigram keep the bare utterance. Defaults to `query_text`."""
    if top_k is None:
        top_k = config.EPISODE_SEMANTIC_TOP_K
    candidates = config.RETRIEVAL_CANDIDATES
    pool = await get_pool()

    query_emb = await embed(embed_query or query_text)
    semantic = await _semantic(pool, query_emb, candidates)
    fts = await _fts(pool, query_text, candidates)
    trigram = await _trigram(pool, query_text, candidates)

    fused = _fuse(semantic, fts, trigram)  # {episode_id: base_score}
    if not fused:
        log.info("[recall_semantic] no episode candidates for %r", query_text[:60])
        return []

    rows = await pool.fetch(
        """SELECT id, text, span_start, span_end, COALESCE(access_count, 0) AS access_count
           FROM episodes WHERE id = ANY($1::int[])""",
        list(fused.keys()),
    )
    scored = []
    for r in rows:
        base = fused[r["id"]]
        mult = decay_multiplier(r["span_end"], now, r["access_count"])
        scored.append({
            "id": r["id"], "text": r["text"],
            "span_start": r["span_start"], "span_end": r["span_end"],
            "access_count": r["access_count"],
            "base_score": base, "score": base * mult,
        })
    scored.sort(key=lambda e: e["score"], reverse=True)
    top = scored[:top_k]

    if top:
        await touch_episodes([e["id"] for e in top])
    log.info("[recall_semantic] %r -> %d candidate(s), top score=%.5f",
             query_text[:60], len(scored), top[0]["score"] if top else 0.0)
    return top
