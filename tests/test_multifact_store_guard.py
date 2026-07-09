"""Tests for the multifact store-corruption fix (2026-07-08).

Bug (found live 2026-06-23, project_lt_multifact_store_corruption): a
conjunction utterance ("my cat is named Mittens AND my dog is named Rex") made
the Tier-2 single-triple grammar bleed structural JSON chars into the first
value ("Mittens},{") and drop the second fact entirely. Two-layer fix:

  1. _ARGS_GRAMMAR / _RECALL_ARGS_GRAMMAR char class excludes {} -- the bleed
     is grammatically impossible (spoken values never contain braces).
  2. _multifact_utterance gates the store route: a conjunction of
     possessed-entity clauses DECLINES the fast path (before Tier-2), so the
     background extractor (array-native, loops store_fact) owns the turn.

Hermetic: classifier (:8092), store_fact, conversation and tts are stubbed.

Run:
    .venv/bin/pytest tests/test_multifact_store_guard.py -v
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import conversation.tool_router as tr
from persistence import runtime_toggles


def _run(coro):
    return asyncio.run(coro)


_CONJUNCTION = "Remember my cat is named Mittens and my dog is named Rex."


# --------------------------------------------------------------------------
# layer 1: grammar hardening
# --------------------------------------------------------------------------

def test_args_grammar_forbids_braces_in_strings():
    """The str char class must exclude {} so a second-object attempt has no
    legal continuation except closing the string -- the '"Mittens},{"' bleed
    becomes unrepresentable."""
    assert r'[^"\\{}]' in tr._ARGS_GRAMMAR
    assert r'[^"\\]' + "\n" not in tr._ARGS_GRAMMAR


def test_recall_grammar_forbids_braces_in_strings():
    assert r'[^"\\{}]' in tr._RECALL_ARGS_GRAMMAR


# --------------------------------------------------------------------------
# layer 2: conjunction detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("utterance", [
    _CONJUNCTION,
    "my cat is Mittens and my dog is Rex",
    "his truck is red and her car is blue",
    "remember our anniversary is June 3rd and their address is 12 Oak Lane",
    "My CAT is Mittens AND MY dog is Rex",  # case-insensitive
])
def test_multifact_detected(utterance):
    assert tr._multifact_utterance(utterance) is True


@pytest.mark.parametrize("utterance", [
    "my dog is named Rex",
    "my favorite band is Florence and the Machine",   # "and the" -> single fact
    "I like fish and chips",
    "my brother lives in Sandy Hook",                  # no conjunction
    "remember my anniversary is June 3rd",
    "",
    # Compound SUBJECT sharing one predicate -- one fact, no copula before the
    # "and", must keep the deterministic fast path (2026-07-09 tightening: the
    # original "and <possessive>" pattern wrongly declined these onto the
    # droppable background extractor).
    "my brother and my sister live in Ohio",
    "my mom and my dad are visiting next week",
    "remember my son and my daughter go to Lincoln Elementary",
])
def test_single_fact_not_detected(utterance):
    assert tr._multifact_utterance(utterance) is False


# --------------------------------------------------------------------------
# route-level behaviour
# --------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    async def add_assistant_turn(self, text):
        self.calls.append(("ack", text))

    async def speak(self, text):
        self.calls.append(("speak", text))


@pytest.fixture
def classifier_on(monkeypatch):
    real_get = runtime_toggles.get
    monkeypatch.setattr(
        tr.runtime_toggles, "get",
        lambda k: True if k == "classifier_enabled" else real_get(k),
    )


def test_conjunction_declines_before_tier2(monkeypatch, classifier_on):
    """store_fact route + conjunction -> fall through WITHOUT paying the Tier-2
    args call (and without storing anything)."""
    tier2 = {"n": 0}

    async def fake_route(user_text):
        return "store_fact"

    async def fake_args(user_text):
        tier2["n"] += 1
        return {"subject": "user's cat", "predicate": "name", "value": "Mittens"}

    monkeypatch.setattr(tr, "classify_intent", fake_route)
    monkeypatch.setattr(tr, "extract_store_fact_args", fake_args)
    out = _run(tr.maybe_handle_tool_call(
        _CONJUNCTION, "dan", 1, conversation=None, tts=None))
    assert out.handled is False
    assert tier2["n"] == 0


def test_single_fact_path_still_stores(monkeypatch, classifier_on):
    """Regression guard: a plain single-fact utterance still routes, stores the
    clean value, and ACKs -- the gate must not touch it."""
    stored = {}

    async def fake_route(user_text):
        return "store_fact"

    async def fake_args(user_text):
        return {"subject": "user's dog", "predicate": "name", "value": "Rex"}

    async def fake_store(subject, predicate, value, **kw):
        stored.update(subject=subject, predicate=predicate, value=value)

    monkeypatch.setattr(tr, "classify_intent", fake_route)
    monkeypatch.setattr(tr, "extract_store_fact_args", fake_args)
    monkeypatch.setattr(tr, "store_fact", fake_store)
    io = _Recorder()
    out = _run(tr.maybe_handle_tool_call(
        "my dog is named Rex", "dan", 1, conversation=io, tts=io))
    assert out.handled is True
    assert stored["value"] == "Rex"
    assert stored["subject"] == "dan's dog"
    kinds = [k for k, _ in io.calls]
    assert "ack" in kinds and "speak" in kinds
