"""Tests for the S4 read-path gate (needs_retrieval).

Two layers, both hermetic (no :8083, no :8092, no DB):
  1. the pure _needs_retrieval banter heuristic, over realistic phrasing;
  2. LiveMemory.gather() actually elides the vector retrieve() when the gate is
     on AND the turn is banter -- and only then. The retrieve() function and the
     facts lookups are monkeypatched, so this never touches Postgres or a server.

Live routing/latency is validated separately against the running service.

Run:
    .venv/bin/pytest tests/test_needs_retrieval.py -v
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import conversation.turn as turn
from conversation.turn import LiveMemory, _needs_retrieval


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# 1. the pure heuristic
# --------------------------------------------------------------------------

# Banter -> SKIP. Plain declaratives / acks with no question, recall verb, or
# possessive referent. Synthetic only (no real PII).
_BANTER = [
    "lol nice one",
    "that's hilarious",
    "good morning timmy",
    "I'm heading out for a bit",
    "okay cool",
    "sounds good to me",
    "haha yeah",
    "thanks buddy",
    "Marcus just walked in",
    "the weather is gross today",
]

# Recall / question / referent -> RETRIEVE.
_RECALL = [
    "what's my dog's name?",
    "do you remember the leak I told you about",
    "who came over last night",
    "did I tell you about the trip",
    "where did I leave my keys",
    "what do you know about Dana",
    "how does that thing work",
    "tell me what we discussed",
    "is there anything I forgot",
    "my appointment got moved",          # possessive referent
    "remind me what my address is",
]


@pytest.mark.parametrize("text", _BANTER)
def test_banter_is_skipped(text):
    assert _needs_retrieval(text) is False, f"banter should skip: {text!r}"


@pytest.mark.parametrize("text", _RECALL)
def test_recall_is_retrieved(text):
    assert _needs_retrieval(text) is True, f"recall should retrieve: {text!r}"


def test_empty_and_none_text_skip_safely():
    assert _needs_retrieval("") is False
    assert _needs_retrieval(None) is False


def test_question_mark_alone_retrieves():
    # No wh-word / verb, but the '?' marks intent.
    assert _needs_retrieval("really?") is True


# --------------------------------------------------------------------------
# 2. gather() actually elides retrieve() — gate x intent matrix
# --------------------------------------------------------------------------

class _Mem:
    """Minimal stand-in for a retrieved memory (gather only reads it into the
    Retrieved tuple; downstream .type/.content/.score aren't exercised here)."""
    type = "episodic"
    content = "sentinel memory"
    score = 0.9


def _install_fakes(monkeypatch, *, gate: bool, use_episodes: bool):
    """Patch BOTH memory-channel tiers + facts lookups + the gate toggle, and
    pin which tier LiveMemory.gather selects (EPISODIC_ALWAYS_ON_RETRIEVAL).
    Returns PER-TIER counters {'legacy','episodes'} so a test asserts the RIGHT
    tier ran and the other did not -- a shared counter would pass vacuously if a
    gate bug hit only the tier the ambient config didn't happen to select."""
    import config as config_mod
    import memory.retrieval as retrieval_mod
    import memory.facts as facts_mod
    from persistence import runtime_toggles

    calls = {"legacy": 0, "episodes": 0}

    async def fake_retrieve(user_text, top_k=5, context_turns=None,
                            resolved_query=None, query_pre_resolved=False):
        calls["legacy"] += 1
        return [_Mem()]

    async def fake_episodes(user_text, top_k, context_turns,
                            resolved_query=None, query_pre_resolved=False):
        calls["episodes"] += 1
        return [_Mem()]

    async def fake_all_facts(subjects, limit=5):
        return []

    async def fake_speaker_facts(name, db_id, limit=5):
        return []

    monkeypatch.setattr(config_mod, "EPISODIC_ALWAYS_ON_RETRIEVAL", use_episodes)
    monkeypatch.setattr(retrieval_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(turn, "_retrieve_episodes_as_memories", fake_episodes)
    monkeypatch.setattr(facts_mod, "get_all_facts_for_prompt", fake_all_facts)
    monkeypatch.setattr(facts_mod, "get_facts_about_speaker", fake_speaker_facts)
    monkeypatch.setattr(runtime_toggles, "get",
                        lambda key: gate if key == "needs_retrieval_gate" else None)
    return calls


def _gather(text):
    return LiveMemory().gather(
        user_text=text, speaker_name="marcus", speaker_db_id=2,
        subjects=[], context_turns=None,
    )


# Each gate x intent case is run against BOTH tiers so neither branch can go
# unexercised (the episodic path silently orphaned the resolver for ~2 weeks
# exactly because nothing pinned it here). `active`/`idle` name the tier that
# should / should not run for the given use_episodes.
@pytest.mark.parametrize("use_episodes,active,idle", [
    (True, "episodes", "legacy"),
    (False, "legacy", "episodes"),
])
def test_gate_on_banter_skips_retrieve(monkeypatch, use_episodes, active, idle):
    calls = _install_fakes(monkeypatch, gate=True, use_episodes=use_episodes)
    result = _run(_gather("lol nice one"))
    assert calls[active] == 0, "banter with gate ON must NOT retrieve on the active tier"
    assert calls[idle] == 0, "the inactive tier never runs"
    assert result.memories == [], "skipped turn injects no memories"


@pytest.mark.parametrize("use_episodes,active,idle", [
    (True, "episodes", "legacy"),
    (False, "legacy", "episodes"),
])
def test_gate_on_recall_still_retrieves(monkeypatch, use_episodes, active, idle):
    calls = _install_fakes(monkeypatch, gate=True, use_episodes=use_episodes)
    result = _run(_gather("what's my dog's name?"))
    assert calls[active] == 1, "recall with gate ON must retrieve on the active tier"
    assert calls[idle] == 0, "the inactive tier must not run"
    assert len(result.memories) == 1


@pytest.mark.parametrize("use_episodes,active,idle", [
    (True, "episodes", "legacy"),
    (False, "legacy", "episodes"),
])
def test_gate_off_banter_still_retrieves(monkeypatch, use_episodes, active, idle):
    # Default behaviour (gate OFF): retrieve on every turn, banter included.
    calls = _install_fakes(monkeypatch, gate=False, use_episodes=use_episodes)
    result = _run(_gather("lol nice one"))
    assert calls[active] == 1, "gate OFF must preserve retrieve-always on the active tier"
    assert calls[idle] == 0
    assert len(result.memories) == 1


@pytest.mark.parametrize("use_episodes,active,idle", [
    (True, "episodes", "legacy"),
    (False, "legacy", "episodes"),
])
def test_gate_off_recall_retrieves(monkeypatch, use_episodes, active, idle):
    calls = _install_fakes(monkeypatch, gate=False, use_episodes=use_episodes)
    result = _run(_gather("do you remember the leak"))
    assert calls[active] == 1
    assert calls[idle] == 0
    assert len(result.memories) == 1
