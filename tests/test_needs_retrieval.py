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


def _install_fakes(monkeypatch, *, gate: bool):
    """Patch BOTH memory-channel tiers (legacy retrieve() + the live episodic
    path — see EPISODIC_ALWAYS_ON_RETRIEVAL) + facts lookups + the toggle.
    Returns a dict whose 'retrieve' counter counts whichever tier ran, so the
    gate assertions hold regardless of the configured tier."""
    import memory.retrieval as retrieval_mod
    import memory.facts as facts_mod
    from persistence import runtime_toggles

    calls = {"retrieve": 0}

    async def fake_retrieve(user_text, top_k=5, context_turns=None,
                            resolved_query=None, query_pre_resolved=False):
        calls["retrieve"] += 1
        return [_Mem()]

    async def fake_episodes(user_text, top_k, context_turns,
                            resolved_query=None, query_pre_resolved=False):
        calls["retrieve"] += 1
        return [_Mem()]

    async def fake_all_facts(subjects, limit=5):
        return []

    async def fake_speaker_facts(name, db_id, limit=5):
        return []

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


def test_gate_on_banter_skips_retrieve(monkeypatch):
    calls = _install_fakes(monkeypatch, gate=True)
    result = _run(_gather("lol nice one"))
    assert calls["retrieve"] == 0, "banter with gate ON must NOT call retrieve()"
    assert result.memories == [], "skipped turn injects no memories"


def test_gate_on_recall_still_retrieves(monkeypatch):
    calls = _install_fakes(monkeypatch, gate=True)
    result = _run(_gather("what's my dog's name?"))
    assert calls["retrieve"] == 1, "recall with gate ON must retrieve"
    assert len(result.memories) == 1


def test_gate_off_banter_still_retrieves(monkeypatch):
    # Default behaviour (gate OFF): retrieve on every turn, banter included.
    calls = _install_fakes(monkeypatch, gate=False)
    result = _run(_gather("lol nice one"))
    assert calls["retrieve"] == 1, "gate OFF must preserve today's retrieve-always behaviour"
    assert len(result.memories) == 1


def test_gate_off_recall_retrieves(monkeypatch):
    calls = _install_fakes(monkeypatch, gate=False)
    result = _run(_gather("do you remember the leak"))
    assert calls["retrieve"] == 1
    assert len(result.memories) == 1
