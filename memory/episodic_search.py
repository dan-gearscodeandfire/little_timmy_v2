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
    """OR-of-lexemes, ranked by ts_rank (2026-08-13).

    Was plainto_tsquery, which ANDs every content term -- so any question with
    three content words matched nothing. Measured on the Open Sauce corpus:
    5 of 10 real questions returned ZERO rows, including "what's your favorite
    Radiohead album?" while the word 'radiohead' sat in two episodes. The AND
    made this channel dead exactly when it was most needed (a proper noun the
    embedder doesn't know is the case FTS exists to rescue).

    Now: lex the query with to_tsvector, OR the lexemes, and let ts_rank order
    by how many/how rare the matched terms are. quote_literal keeps a lexeme
    with punctuation from breaking tsquery parsing. A query that lexes to
    nothing (pure stopwords) yields NULL -> no rows, same as before."""
    rows = await pool.fetch(
        """WITH tq AS (
             SELECT string_agg(quote_literal(w), ' | ')::tsquery AS q
             FROM unnest(tsvector_to_array(to_tsvector('english', $1))) AS w)
           SELECT e.id FROM episodes e, tq
           WHERE tq.q IS NOT NULL AND e.content_tsv @@ tq.q
           ORDER BY ts_rank(e.content_tsv, tq.q) DESC, e.id
           LIMIT $2""",
        query, limit,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


async def _trigram(pool, query, limit):
    """word_similarity against the best-matching extent of the episode.

    Was `text % $1` (whole-document similarity, 0.3 default threshold), which
    has NEVER returned a row: the best similarity any real question achieves
    against any episode is 0.157, because a ~30-char query cannot be 30%
    trigram-similar to a ~490-char summary. The channel was geometrically
    incapable of firing, so its RRF weight described a contribution that never
    existed.

    word_similarity(query, text) scores the query against the best-matching
    WINDOW of the document instead of the whole thing, which is the right
    comparison for short-query/long-document. Floor is configurable; 0.35
    measured best on the Open Sauce eval set.

    SCALING NOTE: this is a seq scan (a function call in the predicate is not
    indexable) -- 9.8ms over 201 episodes, and it grows linearly. The
    index-friendly form is the `<%` operator, which CAN use idx_episodes_text_trgm
    but takes its threshold from the pg_trgm.word_similarity_threshold session
    GUC -- fragile to set correctly through a connection pool. At today's corpus
    the planner picks a seq scan for `<%` anyway (measured: 8.8ms, same), so
    there is nothing to buy yet. Revisit past ~2k episodes."""
    rows = await pool.fetch(
        """SELECT id FROM episodes
           WHERE word_similarity($1, text) >= $3
           ORDER BY word_similarity($1, text) DESC, id
           LIMIT $2""",
        query, limit, config.TRIGRAM_WORD_SIM_FLOOR,
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
