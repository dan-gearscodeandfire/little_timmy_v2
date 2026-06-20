"""Tests for the recall_temporal router intent (plan Session 3).

Covers the deterministic block-building + resolve flow with the classifier
(:8092) and DB stubbed out, so it's hermetic. Live routing accuracy is
validated separately against the running service.

Run:
    .venv/bin/pytest tests/test_recall_temporal.py -v
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import conversation.tool_router as tr

UTC = timezone.utc


def _ep(span_start, span_end, text):
    return {"id": 1, "span_start": span_start, "span_end": span_end,
            "text": text, "token_count": 10, "source": {}}


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# block formatting
# --------------------------------------------------------------------------

def test_build_block_with_episodes_is_untruncated_and_grounding():
    s = datetime(2026, 6, 13, 18, 5, tzinfo=UTC)  # 2:05 PM EDT
    e = datetime(2026, 6, 13, 18, 13, tzinfo=UTC)
    long_text = "Topic: budget. " + "x" * 500  # well over the 200-char vector cap
    block = tr._build_recall_block("last Saturday", s, e, [_ep(s, e, long_text)])
    assert block.startswith("[WHAT WE TALKED ABOUT]")
    assert "last Saturday" in block
    assert long_text in block            # untruncated
    assert "using ONLY these" in block   # grounding instruction


def test_build_block_empty_is_honest_marker():
    s = datetime(2026, 6, 13, 0, 0, tzinfo=UTC)
    e = datetime(2026, 6, 14, 0, 0, tzinfo=UTC)
    block = tr._build_recall_block("last Saturday", s, e, [])
    assert "NO recorded conversation summaries" in block
    assert "do NOT invent" in block
    assert "last Saturday" in block


def test_window_label_single_day_vs_range():
    s = datetime(2026, 6, 13, 4, 0, tzinfo=UTC)   # local Sat Jun 13
    e_day = s + timedelta(days=1)
    e_week = s + timedelta(days=7)
    one = tr._fmt_window_label("", s, e_day)
    many = tr._fmt_window_label("", s, e_week)
    assert "June 13" in one and "–" not in one
    assert "–" in many  # multi-day range uses an en-dash span


# --------------------------------------------------------------------------
# _resolve_recall_block — stub classifier + DB
# --------------------------------------------------------------------------

def test_resolve_hit(monkeypatch):
    s = datetime.now(UTC) - timedelta(days=1)
    e = s + timedelta(minutes=8)

    async def fake_phrase(_text):
        return "yesterday"

    async def fake_query(start, end, limit):
        # The resolver should have produced a window that brackets yesterday.
        assert start < end
        return [_ep(s, e, "Topic: dinner. Dana picked the venue.")]

    monkeypatch.setattr(tr, "extract_recall_phrase", fake_phrase)
    monkeypatch.setattr("memory.manager.query_episodes_by_range", fake_query)

    block = _run(tr._resolve_recall_block("what did we talk about yesterday"))
    assert block is not None
    assert "Dana picked the venue" in block
    assert "yesterday" in block


def test_resolve_empty_range_returns_marker(monkeypatch):
    async def fake_phrase(_text):
        return "yesterday"

    async def fake_query(start, end, limit):
        return []  # nothing recorded

    monkeypatch.setattr(tr, "extract_recall_phrase", fake_phrase)
    monkeypatch.setattr("memory.manager.query_episodes_by_range", fake_query)

    block = _run(tr._resolve_recall_block("what did we discuss yesterday"))
    assert block is not None
    assert "NO recorded conversation summaries" in block


def test_resolve_unparseable_phrase_falls_through(monkeypatch):
    async def fake_phrase(_text):
        return ""  # extractor found no time phrase

    called = {"n": 0}

    async def fake_query(start, end, limit):
        called["n"] += 1
        return []

    monkeypatch.setattr(tr, "extract_recall_phrase", fake_phrase)
    monkeypatch.setattr("memory.manager.query_episodes_by_range", fake_query)

    # No resolvable date anywhere -> None (caller falls through, no DB hit).
    block = _run(tr._resolve_recall_block("tell me a joke about cats"))
    assert block is None
    assert called["n"] == 0


# --------------------------------------------------------------------------
# maybe_handle_tool_call — gate + ToolOutcome contract
# --------------------------------------------------------------------------

def test_recall_route_gated_off_falls_through(monkeypatch):
    monkeypatch.setattr(tr.runtime_toggles, "get",
                        lambda k, *a: True if k == "classifier_enabled" else None)

    async def fake_classify(_text):
        return "recall_temporal"

    monkeypatch.setattr(tr, "classify_intent", fake_classify)
    monkeypatch.setattr(tr.config, "RECALL_TEMPORAL_ENABLED", False)

    outcome = _run(tr.maybe_handle_tool_call("what did we talk about yesterday",
                                             "dan", 1, None, None))
    assert outcome.handled is False
    assert outcome.recall_block is None


def test_recall_route_on_returns_block(monkeypatch):
    monkeypatch.setattr(tr.runtime_toggles, "get",
                        lambda k, *a: True if k == "classifier_enabled" else None)

    async def fake_classify(_text):
        return "recall_temporal"

    async def fake_resolve(_text):
        return "[WHAT WE TALKED ABOUT] stub block"

    monkeypatch.setattr(tr, "classify_intent", fake_classify)
    monkeypatch.setattr(tr, "_resolve_recall_block", fake_resolve)
    monkeypatch.setattr(tr.config, "RECALL_TEMPORAL_ENABLED", True)

    outcome = _run(tr.maybe_handle_tool_call("what did we talk about yesterday",
                                             "dan", 1, None, None))
    assert outcome.handled is False
    assert outcome.recall_block == "[WHAT WE TALKED ABOUT] stub block"
