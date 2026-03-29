"""Asyncpg connection pool singleton."""

import asyncpg
from pgvector.asyncpg import register_vector
import config

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(config.DB_DSN, min_size=2, max_size=10,
                                           init=_init_connection)
    return _pool


async def _init_connection(conn: asyncpg.Connection):
    """Register pgvector type on each new connection."""
    await register_vector(conn)


async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
