"""Tests for Session 5 episode dedup-at-write + semantic search + decay re-rank.

Hits LOCAL Postgres only (NOT :8083, NOT Ollama — embeddings are faked with
deterministic one-hot vectors so cosine geometry is controllable). Every test
seeds marker-tagged episodes and DELETEs them in a finally, so the live
`episodes` table is left as found.

Run: .venv/bin/pytest tests/test_episodic_search.py -v
"""
import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import config
import memory.manager as manager
import memory.episodic_search as es
from db.connection import get_pool

MARK = "__EPISODIC_SEARCH_TEST__"


def _run(coro):
    """Run a coroutine in a FRESH event loop with a FRESH asyncpg pool. The pool
    is a process-global bound to its creating loop; multiple asyncio.run() calls
    each make a new loop, so we reset + close the pool per test to avoid reusing
    one bound to a closed loop ('Event loop is closed')."""
    async def wrapped():
        import db.connection as dbc
        dbc._pool = None
        try:
            return await coro
        finally:
            await dbc.close_pool()
    return asyncio.run(wrapped())


def _vec(text: str) -> np.ndarray:
    """Deterministic one-hot embedding keyed on a topic word in the text.
    Same topic -> identical vector (cosine distance 0); different topic ->
    orthogonal (distance 1.0, excluded by SEMANTIC_DISTANCE_MAX=0.5)."""
    v = np.zeros(768, dtype=np.float32)
    t = text.lower()
    v[0 if "apple" in t else 1 if "banana" in t else 2] = 1.0
    return v


async def _fake_embed(text: str) -> np.ndarray:
    return _vec(text)


def _patch_embed(monkeypatch):
    monkeypatch.setattr(manager, "embed", _fake_embed)
    monkeypatch.setattr(es, "embed", _fake_embed)


async def _cleanup(ids):
    pool = await get_pool()
    await pool.execute("DELETE FROM episodes WHERE id = ANY($1::int[])",
                       [i for i in ids if i is not None])


# --------------------------------------------------------------------------
# dedup-at-write: exact content-hash floor (ALWAYS on, no embedding needed)
# --------------------------------------------------------------------------

def test_verbatim_resummary_is_deduped(monkeypatch):
    monkeypatch.setattr(config, "EMBED_EPISODES", False)
    base = time.time() - 200 * 86400

    async def go():
        ids = []
        try:
            text = f"{MARK} we planted the apple tree by the fence"
            id1 = await manager.store_episode(base, base + 60, text)
            id2 = await manager.store_episode(base + 3600, base + 3660, text)  # verbatim
            ids += [id1, id2]
            assert id1 == id2, "verbatim re-summary must reuse the same row"
            # Case/whitespace variant normalizes to the same hash.
            id3 = await manager.store_episode(base, base + 60,
                                              f"  {MARK}  WE PLANTED the APPLE tree  by the fence ")
            ids.append(id3)
            assert id3 == id1
            pool = await get_pool()
            # Exactly one physical row exists, and only id1 was inserted (id2/id3
            # hit the hash conflict and did NOT store their own text variant).
            n = await pool.fetchval(
                "SELECT count(*) FROM episodes WHERE text = $1", text)
            assert n == 1, "only one physical row for the deduped episode"
            assert set(ids) == {id1}, "all three calls collapsed to one id"
        finally:
            await _cleanup(ids)

    _run(go())


def test_distinct_text_makes_new_row(monkeypatch):
    monkeypatch.setattr(config, "EMBED_EPISODES", False)
    base = time.time() - 200 * 86400

    async def go():
        ids = []
        try:
            id1 = await manager.store_episode(base, base + 60, f"{MARK} apple harvest notes")
            id2 = await manager.store_episode(base, base + 60, f"{MARK} banana bread recipe")
            ids += [id1, id2]
            assert id1 != id2
        finally:
            await _cleanup(ids)

    _run(go())


# --------------------------------------------------------------------------
# dedup-at-write: optional near-dupe similarity layer
# --------------------------------------------------------------------------

def test_similarity_layer_dedupes_near_dupe(monkeypatch):
    _patch_embed(monkeypatch)
    monkeypatch.setattr(config, "EMBED_EPISODES", True)
    monkeypatch.setattr(config, "EPISODE_DEDUP_SIM_ENABLED", True)
    base = time.time() - 200 * 86400

    async def go():
        ids = []
        try:
            id1 = await manager.store_episode(base, base + 60, f"{MARK} apple orchard plan A")
            # Different text (distinct hash) but identical fake embedding -> distance 0.
            id2 = await manager.store_episode(base, base + 60, f"{MARK} apple orchard plan B")
            ids += [id1, id2]
            assert id2 == id1, "near-identical embedding should reuse the first row"
        finally:
            await _cleanup(ids)

    _run(go())


def test_similarity_layer_off_keeps_both(monkeypatch):
    _patch_embed(monkeypatch)
    monkeypatch.setattr(config, "EMBED_EPISODES", True)
    monkeypatch.setattr(config, "EPISODE_DEDUP_SIM_ENABLED", False)
    base = time.time() - 200 * 86400

    async def go():
        ids = []
        try:
            id1 = await manager.store_episode(base, base + 60, f"{MARK} apple orchard plan A")
            id2 = await manager.store_episode(base, base + 60, f"{MARK} apple orchard plan B")
            ids += [id1, id2]
            assert id1 != id2, "with the layer off, distinct-text rows both persist"
        finally:
            await _cleanup(ids)

    _run(go())


# --------------------------------------------------------------------------
# semantic search + recency decay
# --------------------------------------------------------------------------

def test_search_ranks_fresh_over_stale_same_topic(monkeypatch):
    _patch_embed(monkeypatch)
    monkeypatch.setattr(config, "EMBED_EPISODES", True)
    monkeypatch.setattr(config, "EPISODE_DEDUP_SIM_ENABLED", False)
    from datetime import datetime
    now = datetime.now().astimezone()
    now_s = now.timestamp()

    async def go():
        ids = []
        try:
            stale = await manager.store_episode(now_s - 120 * 86400, now_s - 120 * 86400 + 60,
                                                f"{MARK} apple pruning was the topic (stale)")
            fresh = await manager.store_episode(now_s - 1 * 86400, now_s - 1 * 86400 + 60,
                                                f"{MARK} apple pruning was the topic (fresh)")
            ids += [stale, fresh]
            results = await es.search_episodes("apple", now, top_k=5)
            ours = [r for r in results if MARK in r["text"]]
            assert len(ours) == 2, "both apple episodes match the topic"
            assert ours[0]["id"] == fresh, "recency decay must rank the fresh one first"
            assert ours[0]["score"] > ours[1]["score"]
        finally:
            await _cleanup(ids)

    _run(go())


def test_search_excludes_other_topic(monkeypatch):
    _patch_embed(monkeypatch)
    monkeypatch.setattr(config, "EMBED_EPISODES", True)
    monkeypatch.setattr(config, "EPISODE_DEDUP_SIM_ENABLED", False)
    from datetime import datetime
    now = datetime.now().astimezone()
    now_s = now.timestamp()

    async def go():
        ids = []
        try:
            ids.append(await manager.store_episode(now_s - 2 * 86400, now_s - 2 * 86400 + 60,
                                                   f"{MARK} banana bread baking session"))
            results = await es.search_episodes("apple", now, top_k=5)
            assert not [r for r in results if MARK in r["text"]], \
                "an orthogonal-topic episode must not surface for an apple query"
        finally:
            await _cleanup(ids)

    _run(go())
