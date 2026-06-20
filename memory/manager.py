"""Memory CRUD and embedding generation."""

import json
import logging
import httpx
import numpy as np
from db.connection import get_pool
import config

log = logging.getLogger(__name__)

_embed_client: httpx.AsyncClient | None = None


async def _get_embed_client() -> httpx.AsyncClient:
    global _embed_client
    if _embed_client is None:
        _embed_client = httpx.AsyncClient(timeout=30.0)
    return _embed_client


async def embed(text: str) -> np.ndarray:
    """Generate 768-dim embedding via Ollama nomic-embed-text."""
    client = await _get_embed_client()
    resp = await client.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBEDDING_MODEL, "prompt": text},
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


async def store_memory(
    mem_type: str,
    content: str,
    speaker_id: int | None = None,
    metadata: dict | None = None,
) -> int:
    """Store a memory with its embedding. Returns the memory id."""
    emb = await embed(content)
    pool = await get_pool()
    row = await pool.fetchrow(
        """INSERT INTO memories (type, content, speaker_id, embedding, metadata)
           VALUES ($1, $2, $3, $4, $5::jsonb) RETURNING id""",
        mem_type,
        content,
        speaker_id,
        emb,
        json.dumps(metadata or {}),
    )
    log.info("Stored %s memory id=%d (%d chars)", mem_type, row["id"], len(content))
    return row["id"]


async def store_episode(
    span_start: float,
    span_end: float,
    text: str,
    token_count: int | None = None,
    source: dict | None = None,
) -> int:
    """Store an episodic memory (a rollup summary with a real time span).

    Writes to the `episodes` table — distinct from `store_memory`'s vector
    `memories` tier. `span_start`/`span_end` are epoch seconds (turn
    `time.time()` timestamps), converted to TIMESTAMPTZ via `to_timestamp`.
    The `embedding` column is left NULL: episodic recall is date-range only
    until Session 5 restores (fixed) vector search. Returns the episode id.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        """INSERT INTO episodes (span_start, span_end, text, token_count, source)
           VALUES (to_timestamp($1), to_timestamp($2), $3, $4, $5::jsonb)
           RETURNING id""",
        span_start,
        span_end,
        text,
        token_count,
        json.dumps(source or {}),
    )
    log.info(
        "Stored episode id=%d span=%.0f..%.0f (%d chars)",
        row["id"], span_start, span_end, len(text),
    )
    return row["id"]


async def touch_memory(memory_id: int):
    """Update access timestamp and count for a single memory."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE memories SET accessed_at = NOW(), access_count = access_count + 1 WHERE id = $1",
        memory_id,
    )


async def touch_memories(memory_ids: list[int]):
    """Batched access-stat update for multiple memories in one query."""
    if not memory_ids:
        return
    pool = await get_pool()
    await pool.execute(
        "UPDATE memories SET accessed_at = NOW(), access_count = access_count + 1 "
        "WHERE id = ANY($1::int[])",
        memory_ids,
    )
