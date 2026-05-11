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


async def _semantic_search(pool, query_embedding, limit: int) -> list[tuple[int, int]]:
    """Returns list of (memory_id, rank), filtered by cosine-distance floor."""
    rows = await pool.fetch(
        """SELECT id FROM memories
           WHERE embedding <=> $1 < $3
           ORDER BY embedding <=> $1
           LIMIT $2""",
        query_embedding,
        limit,
        SEMANTIC_DISTANCE_MAX,
    )
    return [(r["id"], i) for i, r in enumerate(rows)]


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


def _reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, int]]], k: int = 60) -> list[int]:
    """Merge multiple ranked lists using RRF. Returns sorted memory IDs."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for mem_id, rank in ranked:
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda mid: scores[mid], reverse=True)


async def retrieve(query: str, top_k: int | None = None) -> list[RetrievedMemory]:
    """Run hybrid retrieval: semantic + FTS + trigram, merge with RRF."""
    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K
    candidates = config.RETRIEVAL_CANDIDATES

    pool = await get_pool()
    query_emb = await embed(query)

    # Run all three searches in parallel
    semantic, fts, trigram = await asyncio.gather(
        _semantic_search(pool, query_emb, candidates),
        _fts_search(pool, query, candidates),
        _trigram_search(pool, query, candidates),
    )

    merged_ids = _reciprocal_rank_fusion([semantic, fts, trigram])[:top_k]
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
    for rank, mid in enumerate(merged_ids):
        if mid in row_map:
            r = row_map[mid]
            results.append(RetrievedMemory(
                id=r["id"],
                type=r["type"],
                content=r["content"],
                score=1.0 / (rank + 1),
                created_at=r["created_at"],
            ))

    # Batch access-stat update in background (one query for all hits)
    if results:
        asyncio.create_task(touch_memories([r.id for r in results]))

    log.info("Retrieved %d memories (semantic=%d, fts=%d, trigram=%d)",
             len(results), len(semantic), len(fts), len(trigram))
    return results
