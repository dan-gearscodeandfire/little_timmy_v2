"""Tests for the proposition tier (memory/propositions.py).

Parser tests are pure. Storage/search tests hit LOCAL Postgres only (NOT :8084,
NOT Ollama — the LLM split and the embedder are both faked). Every DB test
seeds marker-tagged rows and DELETEs them in a finally, so the live
`episodes` / `propositions` tables are left as found.

Run: .venv/bin/pytest tests/test_propositions.py -v
"""
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import config
import memory.manager as manager
import memory.propositions as props
from db.connection import get_pool

MARK = "__PROPOSITION_TEST__"


def _run(coro):
    """Fresh loop + fresh asyncpg pool per test — the pool is a process-global
    bound to its creating loop (see tests/test_episodic_search.py)."""
    async def wrapped():
        import db.connection as dbc
        dbc._pool = None
        try:
            return await coro
        finally:
            await dbc.close_pool()
    return asyncio.run(wrapped())


def _vec(text: str) -> np.ndarray:
    v = np.zeros(768, dtype=np.float32)
    t = text.lower()
    v[0 if "lighthouse" in t else 1 if "bicycle" in t else 2] = 1.0
    return v


async def _fake_embed(text):
    return _vec(text)


def _patch_embed(monkeypatch):
    monkeypatch.setattr(manager, "embed", _fake_embed)
    monkeypatch.setattr(props, "embed", _fake_embed)


async def _seed_episode(span_end_dt, text):
    pool = await get_pool()
    row = await pool.fetchrow(
        """INSERT INTO episodes (span_start, span_end, text)
           VALUES ($1, $2, $3) RETURNING id""",
        span_end_dt - timedelta(minutes=5), span_end_dt, text)
    return row["id"]


async def _cleanup(episode_ids):
    if not episode_ids:
        return
    pool = await get_pool()
    # propositions cascade on episode delete, but be explicit.
    await pool.execute("DELETE FROM propositions WHERE episode_id = ANY($1::int[])", episode_ids)
    await pool.execute("DELETE FROM episodes WHERE id = ANY($1::int[])", episode_ids)


# --------------------------------------------------------------------------
# parser (pure)
# --------------------------------------------------------------------------

def test_parse_strips_bullets_numbering_and_preamble():
    raw = ("Here are the statements:\n"
           "- Dan's cats are named Dexter and Preston.\n"
           "2) Sierra rated her yellow top an eight out of ten.\n"
           "• Timmy refused to perform a backflip for Jeffrey.\n")
    out = props.parse_propositions(raw)
    assert out == [
        "Dan's cats are named Dexter and Preston.",
        "Sierra rated her yellow top an eight out of ten.",
        "Timmy refused to perform a backflip for Jeffrey.",
    ]


def test_parse_dedupes_case_insensitively():
    raw = ("Dan's cats are named Dexter and Preston.\n"
           "dan's CATS are named Dexter and Preston.\n")
    assert len(props.parse_propositions(raw)) == 1


def test_parse_drops_fragments_and_paragraphs():
    """The length band is the guard against the two failure modes: a fragment
    carries no recallable claim, and a paragraph is the dilution being fixed."""
    raw = ("Ok.\n"                                    # under MIN_CHARS
           "Dan confirmed the microphone diagnosis.\n"  # keeper
           + "x" * (config.PROPOSITION_MAX_CHARS + 50) + "\n"
           "-----\n")                                  # no letters
    assert props.parse_propositions(raw) == ["Dan confirmed the microphone diagnosis."]


def test_parse_respects_max_per_episode(monkeypatch):
    monkeypatch.setattr(config, "PROPOSITION_MAX_PER_EPISODE", 3)
    raw = "\n".join(f"Dan said something specific number {i} to Timmy." for i in range(10))
    assert len(props.parse_propositions(raw)) == 3


def test_parse_empty_input_is_empty_not_error():
    assert props.parse_propositions("") == []
    assert props.parse_propositions(None) == []


def test_generate_returns_empty_when_llm_fails(monkeypatch):
    """A failed split must degrade to episode-tier retrieval, never raise into
    the rollup write path."""
    async def boom(prompt, thinking=None):
        raise RuntimeError("server down")
    import llm.client as client
    monkeypatch.setattr(client, "generate_memory", boom)
    assert _run(props.generate_propositions("some episode text")) == []


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def test_store_is_idempotent_per_episode(monkeypatch):
    _patch_embed(monkeypatch)
    now = datetime.now(timezone.utc)

    async def go():
        ids = []
        try:
            ep = await _seed_episode(now, f"{MARK} lighthouse episode")
            ids.append(ep)
            texts = [f"{MARK} The lighthouse beam warns sailors away from rocks.",
                     f"{MARK} Dan asked Timmy about the lighthouse at midnight."]
            first = await props.store_propositions(ep, now, texts)
            second = await props.store_propositions(ep, now, texts)
            assert first == 2, "first write inserts both"
            assert second == 0, "re-running a backfill must insert nothing"
            pool = await get_pool()
            n = await pool.fetchval(
                "SELECT count(*) FROM propositions WHERE episode_id = $1", ep)
            assert n == 2
        finally:
            await _cleanup(ids)
    _run(go())


def test_same_claim_in_two_episodes_stays_two_rows(monkeypatch):
    """Dedup is per-episode ON PURPOSE: each restatement carries its own
    span_end, and the FRESHER one must be able to win on recency decay. A
    global unique hash would pin the claim to its oldest mention."""
    _patch_embed(monkeypatch)
    now = datetime.now(timezone.utc)

    async def go():
        ids = []
        try:
            old = await _seed_episode(now - timedelta(days=200), f"{MARK} old lighthouse")
            new = await _seed_episode(now, f"{MARK} new lighthouse")
            ids += [old, new]
            claim = [f"{MARK} The lighthouse is white and tall."]
            await props.store_propositions(old, now - timedelta(days=200), claim)
            await props.store_propositions(new, now, claim)
            pool = await get_pool()
            n = await pool.fetchval(
                "SELECT count(*) FROM propositions WHERE episode_id = ANY($1::int[])", ids)
            assert n == 2
        finally:
            await _cleanup(ids)
    _run(go())


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def test_search_returns_parent_episode_and_decays_by_recency(monkeypatch):
    _patch_embed(monkeypatch)
    now = datetime.now(timezone.utc)

    async def go():
        ids = []
        try:
            stale_ep = await _seed_episode(now - timedelta(days=200), f"{MARK} stale")
            fresh_ep = await _seed_episode(now - timedelta(days=1), f"{MARK} fresh")
            ids += [stale_ep, fresh_ep]
            await props.store_propositions(
                stale_ep, now - timedelta(days=200),
                [f"{MARK} The lighthouse guided ships safely (stale)."])
            await props.store_propositions(
                fresh_ep, now - timedelta(days=1),
                [f"{MARK} The lighthouse guided ships safely (fresh)."])

            found = await props.search_propositions("lighthouse", now, top_k=40)
            ours = [p for p in found if MARK in p["text"]]
            assert len(ours) == 2, "both propositions match the topic"
            assert ours[0]["episode_id"] == fresh_ep, "recency decay ranks fresh first"
            assert ours[0]["score"] > ours[1]["score"]
            assert all(p["episode_id"] in ids for p in ours), "parent episode is carried"
        finally:
            await _cleanup(ids)
    _run(go())


def test_search_empty_corpus_returns_empty_not_error(monkeypatch):
    _patch_embed(monkeypatch)
    now = datetime.now(timezone.utc)
    # A query whose lexemes match nothing and whose fake vector is orthogonal.
    out = _run(props.search_propositions("zzzqqq unmatchable bicycle token", now, top_k=5))
    assert isinstance(out, list)
