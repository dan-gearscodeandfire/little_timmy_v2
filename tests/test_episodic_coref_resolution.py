"""Tests for coref resolution on the EPISODIC retrieval path (2026-07-08).

Regression guard: the episodic always-on migration (EPISODIC_ALWAYS_ON_RETRIEVAL,
~2026-06-24) replaced retrieve() with _retrieve_episodes_as_memories on the live
path — which built only the embedding blend and never called
resolve_for_retrieval, silently orphaning the :8093 resolver (healthy server,
zero traffic, found live 2026-07-08). These tests pin the re-wire: the episodic
path must mirror retrieve()'s resolver contract exactly — inline resolve when
not pre-resolved, trust a doorway pre-resolution, and fall back to the blend on
decline/miss.

Hermetic (no :8093, no DB): search_episodes and the resolver are monkeypatched.

2026-08-13: the proposition tier now runs AHEAD of search_episodes on this path,
so `fake_search` also forces PROPOSITION_RETRIEVAL_ENABLED off -- otherwise the
episode spy never fires and the real DB pool gets touched from the test loop.
The resolver contract must hold on BOTH tiers, so `fake_prop_search` mirrors the
spy for the proposition path and test_proposition_tier_gets_same_embed_query
pins it there too.

Run:
    .venv/bin/pytest tests/test_episodic_coref_resolution.py -v
"""

import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import config
import memory.retrieval as retrieval
import memory.episodic_search as episodic_search
import memory.propositions as propositions
from conversation.turn import _retrieve_episodes_as_memories


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _Turn:
    role: str
    content: str


_CTX = [_Turn("user", "I just adopted a dog."), _Turn("assistant", "Congratulations!")]
_REWRITE = "what is the dog's name"


@pytest.fixture
def fake_search(monkeypatch):
    """Capture the embed_query search_episodes receives; return no episodes."""
    seen = {"embed_query": None, "query": None}

    async def fake_search_episodes(query, now, top_k=5, embed_query=None):
        seen["query"] = query
        seen["embed_query"] = embed_query
        return []

    monkeypatch.setattr(episodic_search, "search_episodes", fake_search_episodes)
    # Exercise the EPISODE path: the proposition tier would otherwise short-
    # circuit ahead of it (and hit the real pool from this loop).
    monkeypatch.setattr(config, "PROPOSITION_RETRIEVAL_ENABLED", False)
    return seen


@pytest.fixture
def fake_prop_search(monkeypatch):
    """Same spy, for the proposition tier, with the tier forced ON."""
    seen = {"embed_query": None, "query": None}

    async def fake_search_propositions(query, now, top_k=5, embed_query=None):
        seen["query"] = query
        seen["embed_query"] = embed_query
        return []

    monkeypatch.setattr(propositions, "search_propositions", fake_search_propositions)
    monkeypatch.setattr(config, "PROPOSITION_RETRIEVAL_ENABLED", True)
    return seen


@pytest.fixture
def fake_resolver(monkeypatch):
    """Count resolve_for_retrieval calls; return the canned rewrite."""
    calls = {"n": 0}

    async def fake_resolve(query, context_turns):
        calls["n"] += 1
        return _REWRITE

    monkeypatch.setattr(retrieval, "resolve_for_retrieval", fake_resolve)
    return calls


def test_inline_resolve_feeds_embed_query(fake_search, fake_resolver):
    """Default (not pre-resolved): the episodic path calls the resolver and
    embeds the standalone rewrite, not the blend."""
    _run(_retrieve_episodes_as_memories("what's its name?", 5, _CTX))
    assert fake_resolver["n"] == 1
    assert fake_search["embed_query"] == _REWRITE
    # FTS/trigram side still sees the bare utterance.
    assert fake_search["query"] == "what's its name?"


def test_resolver_miss_falls_back_to_blend(fake_search, monkeypatch):
    """Resolver decline/miss (None) -> embedding blend, the pre-fix behaviour."""
    async def fake_resolve(query, context_turns):
        return None

    monkeypatch.setattr(retrieval, "resolve_for_retrieval", fake_resolve)
    _run(_retrieve_episodes_as_memories("what's its name?", 5, _CTX))
    expected_blend = retrieval._build_semantic_query("what's its name?", _CTX)
    assert fake_search["embed_query"] == expected_blend


def test_pre_resolved_trusts_doorway_no_second_call(fake_search, fake_resolver):
    """query_pre_resolved=True with a rewrite -> use it verbatim, do NOT call
    :8093 again (the doorway owns resolution)."""
    _run(_retrieve_episodes_as_memories(
        "what's its name?", 5, _CTX,
        resolved_query=_REWRITE, query_pre_resolved=True))
    assert fake_resolver["n"] == 0
    assert fake_search["embed_query"] == _REWRITE


def test_pre_resolved_none_falls_back_to_blend(fake_search, fake_resolver):
    """query_pre_resolved=True with None (doorway gate declined / resolver
    missed) -> blend, still no second :8093 call."""
    _run(_retrieve_episodes_as_memories(
        "what's its name?", 5, _CTX,
        resolved_query=None, query_pre_resolved=True))
    assert fake_resolver["n"] == 0
    expected_blend = retrieval._build_semantic_query("what's its name?", _CTX)
    assert fake_search["embed_query"] == expected_blend


def test_nominal_ellipsis_ones_passes_gate():
    """"ones" (2026-07-08): "what about the tall ones?" is anaphoric like
    "them" -- must pass _needs_resolution. Bare "one" stays excluded (noisy)."""
    assert retrieval._needs_resolution("what about the tall ones?") is True
    assert retrieval._needs_resolution("what about one of my friends?") is False


def test_proposition_tier_gets_same_embed_query(fake_prop_search, fake_resolver):
    """The resolver contract is tier-agnostic: whichever tier serves the turn
    must embed the coref-RESOLVED query, not the bare deictic utterance. Pins
    the 2026-08-13 proposition branch against re-orphaning the resolver the way
    the 2026-06 episodic migration did."""
    _run(_retrieve_episodes_as_memories("what is its name?", 5, _CTX))
    assert fake_resolver["n"] == 1, "proposition path must still call the resolver"
    assert fake_prop_search["embed_query"] == _REWRITE
    assert fake_prop_search["query"] == "what is its name?", "lexical channels keep the bare utterance"


def test_proposition_tier_falls_back_to_episodes_when_empty(monkeypatch, fake_search):
    """A partially-split corpus must degrade to episodes, not go blind."""
    async def empty_props(query, now, top_k=5, embed_query=None):
        return []
    monkeypatch.setattr(propositions, "search_propositions", empty_props)
    monkeypatch.setattr(config, "PROPOSITION_RETRIEVAL_ENABLED", True)
    _run(_retrieve_episodes_as_memories("anything at all", 5, None))
    assert fake_search["query"] == "anything at all", "episode tier must be consulted"
