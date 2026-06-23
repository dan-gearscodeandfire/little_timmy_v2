"""Memory CRUD and embedding generation."""

import hashlib
import json
import logging
import re
import httpx
import numpy as np
from db.connection import get_pool
import config

log = logging.getLogger(__name__)


def _episode_content_hash(text: str) -> str:
    """SHA-256 of normalized episode text (lowercased, whitespace-collapsed) —
    the dedup-at-write floor. Normalization absorbs trivial formatting churn so
    a verbatim re-summary collides even if spacing differs."""
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

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
    Dedup-at-write (Session 5): an exact content-hash UNIQUE floor (ALWAYS on)
    skips a verbatim re-summary instead of double-writing it — guarding the
    rollup double-encode re-rot. With config.EMBED_EPISODES on, the embedding is
    also computed and stored (else left NULL = pre-S5 behavior), and an optional
    near-dupe similarity layer (config.EPISODE_DEDUP_SIM_ENABLED) can skip
    almost-identical re-summaries. Returns the episode id — the NEW row's id, or
    the EXISTING row's id when the write was deduped.
    """
    # Redaction: scrub any blocked term (e.g. Dan's last name) from episode text
    # before hashing/embedding/storing -- it must not persist in any memory.
    _terms = getattr(config, "REDACT_TERMS", ())
    if _terms and text:
        import re as _re
        for _t in _terms:
            text = _re.sub(rf"\b{_re.escape(_t)}\b", "[redacted]", text, flags=_re.IGNORECASE)

    pool = await get_pool()
    content_hash = _episode_content_hash(text)
    embedding = await embed(text) if config.EMBED_EPISODES else None

    # Near-dupe layer (opt-in, on top of the hash floor): if the closest already
    # embedded episode is within the threshold, treat this as the same episode.
    if embedding is not None and config.EPISODE_DEDUP_SIM_ENABLED:
        near = await pool.fetchrow(
            """SELECT id, embedding <=> $1 AS distance FROM episodes
               WHERE embedding IS NOT NULL
               ORDER BY embedding <=> $1 LIMIT 1""",
            embedding,
        )
        if near is not None and float(near["distance"]) <= config.EPISODE_DEDUP_SIM_MAX_DIST:
            log.info("Episode dedup (similarity %.4f <= %.4f): reusing id=%d",
                     float(near["distance"]), config.EPISODE_DEDUP_SIM_MAX_DIST, near["id"])
            return near["id"]

    row = await pool.fetchrow(
        """INSERT INTO episodes (span_start, span_end, text, token_count, source,
                                 content_hash, embedding)
           VALUES (to_timestamp($1), to_timestamp($2), $3, $4, $5::jsonb, $6, $7)
           ON CONFLICT (content_hash) WHERE content_hash IS NOT NULL DO NOTHING
           RETURNING id""",
        span_start,
        span_end,
        text,
        token_count,
        json.dumps(source or {}),
        content_hash,
        embedding,
    )
    if row is None:
        # Hash collision -> a verbatim duplicate already exists; return its id.
        existing = await pool.fetchrow(
            "SELECT id FROM episodes WHERE content_hash = $1", content_hash)
        log.info("Episode dedup (content-hash): reusing id=%s for verbatim re-summary",
                 existing["id"] if existing else "?")
        return existing["id"] if existing else None
    log.info(
        "Stored episode id=%d span=%.0f..%.0f (%d chars%s)",
        row["id"], span_start, span_end, len(text),
        ", embedded" if embedding is not None else "",
    )
    return row["id"]


async def query_episodes_by_range(start, end, limit: int = 20) -> list[dict]:
    """Return episodes whose [span_start, span_end] overlaps [start, end).

    Overlap test: `span_start < end AND span_end >= start` (the query window is
    half-open, episode spans are inclusive). Ordered oldest-first by span_start
    and capped at `limit`. Text is returned UNTRUNCATED — episodic recall does
    not go through the 200-char retrieval guillotine. `start`/`end` are tz-aware
    datetimes (see memory.temporal.resolve_date_range). Read-only.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT id, span_start, span_end, created_at, text, token_count, source
           FROM episodes
           WHERE span_start < $2 AND span_end >= $1
           ORDER BY span_start
           LIMIT $3""",
        start, end, limit,
    )
    return [dict(r) for r in rows]


async def list_episodes(start=None, end=None, limit: int = 200) -> list[dict]:
    """Read-only episode listing for the Memory Inspector. With no bounds,
    returns the whole timeline newest-first. With `start`/`end` (tz-aware
    datetimes), filters to episodes whose [span_start, span_end] overlaps the
    window — same overlap test as query_episodes_by_range — and orders
    oldest-first to read as a timeline. Text returned untruncated. Read-only.
    """
    pool = await get_pool()
    if start is not None and end is not None:
        rows = await pool.fetch(
            """SELECT id, span_start, span_end, created_at, text, token_count,
                      source, access_count, accessed_at
               FROM episodes
               WHERE span_start < $2 AND span_end >= $1
               ORDER BY span_start
               LIMIT $3""",
            start, end, limit,
        )
    else:
        rows = await pool.fetch(
            """SELECT id, span_start, span_end, created_at, text, token_count,
                      source, access_count, accessed_at
               FROM episodes
               ORDER BY span_start DESC
               LIMIT $1""",
            limit,
        )
    return [dict(r) for r in rows]


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


async def touch_episodes(episode_ids: list[int]):
    """Batched access-stat update for episodes (recall_semantic re-rank signal).
    Feeds memory.decay.access_boost — the formerly-unused access_count, now read."""
    if not episode_ids:
        return
    pool = await get_pool()
    await pool.execute(
        "UPDATE episodes SET accessed_at = NOW(), access_count = COALESCE(access_count, 0) + 1 "
        "WHERE id = ANY($1::int[])",
        episode_ids,
    )
