"""Tests for the speculative-coref gating contract (2026-06-22).

Hermetic (no :8093, no DB): exercises memory.retrieval.resolve_for_retrieval,
the function the doorway launches in PARALLEL with the tool-call classifier and
hands into retrieve(query_pre_resolved=True). The crux invariant for the
speculative path's safety: resolve_for_retrieval must return None (and NOT call
the :8093 resolver) whenever its gate declines -- toggle off, no context, or a
non-deictic/long/non-query utterance -- so a declined turn cleanly falls back to
the embedding blend inside retrieve(). client.resolve_query is monkeypatched, so
this never touches a server.

Run:
    .venv/bin/pytest tests/test_speculative_coref.py -v
"""

import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import memory.retrieval as retrieval
from llm import client as llm_client
from persistence import runtime_toggles


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _Turn:
    role: str
    content: str


_CTX = [_Turn("user", "I just adopted a dog."), _Turn("assistant", "Congratulations!")]


@pytest.fixture
def fake_resolver(monkeypatch):
    """Record resolve_query calls and return a canned rewrite."""
    calls = {"n": 0, "last": None}

    async def fake_resolve_query(utterance, context_text):
        calls["n"] += 1
        calls["last"] = (utterance, context_text)
        return "what is the dog's name"

    monkeypatch.setattr(llm_client, "resolve_query", fake_resolve_query)
    return calls


def _force_toggle(monkeypatch, query_resolution: bool):
    real_get = runtime_toggles.get
    monkeypatch.setattr(
        runtime_toggles, "get",
        lambda k: query_resolution if k == "query_resolution_enabled" else real_get(k),
    )


def test_gate_on_deictic_query_resolves(monkeypatch, fake_resolver):
    """Toggle on + short, deictic, query-like utterance + context -> calls :8093
    and returns the rewrite."""
    _force_toggle(monkeypatch, True)
    out = _run(retrieval.resolve_for_retrieval("what's its name?", _CTX))
    assert out == "what is the dog's name"
    assert fake_resolver["n"] == 1


def test_toggle_off_never_calls_resolver(monkeypatch, fake_resolver):
    """query_resolution_enabled off -> None, no :8093 call (even on a deictic
    query). The speculative task is still launched by the doorway; it must cost
    nothing here."""
    _force_toggle(monkeypatch, False)
    out = _run(retrieval.resolve_for_retrieval("what's its name?", _CTX))
    assert out is None
    assert fake_resolver["n"] == 0


def test_non_deictic_query_skips_resolver(monkeypatch, fake_resolver):
    """No deixis -> gate declines -> None, no :8093 call."""
    _force_toggle(monkeypatch, True)
    out = _run(retrieval.resolve_for_retrieval("tell me about the weather", _CTX))
    assert out is None
    assert fake_resolver["n"] == 0


def test_empty_context_skips_resolver(monkeypatch, fake_resolver):
    """No context window -> nothing to resolve against -> None, no call."""
    _force_toggle(monkeypatch, True)
    assert _run(retrieval.resolve_for_retrieval("what's its name?", None)) is None
    assert _run(retrieval.resolve_for_retrieval("what's its name?", [])) is None
    assert fake_resolver["n"] == 0


def test_long_utterance_skips_resolver(monkeypatch, fake_resolver):
    """A long declarative that merely contains a pronoun is decode-bound and
    gains nothing over the blend -> gate declines."""
    _force_toggle(monkeypatch, True)
    long_q = "that is exactly why I built you " + "and ".join(["it works"] * 12)
    out = _run(retrieval.resolve_for_retrieval(long_q, _CTX))
    assert out is None
    assert fake_resolver["n"] == 0


def test_speculative_toggle_defaults_off():
    """Ships default OFF (2026-06-23): the speculative resolve fires a 4B :8093
    call concurrent with the brain, and cross-process GPU contention on the one
    Strix Halo Vulkan card means that is not free. Kept behind the toggle for
    future hardware; re-A/B before ever flipping ON. (Was default ON in the
    initial 2026-06-22 build; reverted same night.)"""
    assert runtime_toggles._DEFAULTS["speculative_coref_enabled"] is False
