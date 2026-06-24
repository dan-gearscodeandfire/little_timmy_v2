"""Tests for write-through-at-mint episodic persistence + the SUBSTANTIVE/BANTER
usefulness gate (2026-06-24).

Fully hermetic: NO Postgres, NO Ollama, NO :8083/:8084. `generate_summary`,
`store_episode`, and `store_memory` are monkeypatched, so this exercises ONLY
the `maybe_rollup` control flow and the `_parse_summary` verdict parser.

Behaviour under test:
  - the summarizer leads with a SUBSTANTIVE/BANTER verdict; the verdict is
    stripped before the summary reaches the warm tier and never leaks into an
    episode;
  - SUBSTANTIVE rollups are written through to `episodes` AT MINT (not on
    eviction), so a summary that survives in the warm tier is already durable;
  - BANTER rollups are NOT persisted, but the rolling window STILL advances
    (hot drains + warm appends) — the buffer must never stall on a banter span;
  - eviction no longer writes an episode (write-through is the sole writer), so
    a warm overflow does not double-write.

Run: .venv/bin/pytest tests/test_rollup_writethrough.py -v
"""
import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import config
import memory.rollup as rollup
from conversation.models import WarmSummary


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class _Turn:
    def __init__(self, role, content, speaker=None, ts=None, tok=50):
        self.role = role
        self.content = content
        self.speaker = speaker
        self.timestamp = ts if ts is not None else time.time()
        self.token_count = tok


class _Conv:
    def __init__(self, hot):
        self.hot_turns = list(hot)
        self.warm_summaries = []


def _force_age_trigger(conv):
    """Backdate the oldest hot turn past ROLLUP_AGE_SECONDS so maybe_rollup fires
    regardless of token count."""
    old = time.time() - config.ROLLUP_AGE_SECONDS - 10
    for i, t in enumerate(conv.hot_turns):
        t.timestamp = old + i


def _patch(monkeypatch, summary_text):
    """Stub the three I/O calls maybe_rollup makes. Returns (episodes, cold)
    capture lists."""
    episodes = []
    cold = []

    async def fake_generate_summary(text):
        return summary_text

    async def fake_store_episode(s, e, text, token_count=None, source=None):
        episodes.append({"span": (s, e), "text": text,
                         "token_count": token_count, "source": source})
        return len(episodes)

    async def fake_store_memory(mem_type, content):
        cold.append((mem_type, content))
        return 1

    monkeypatch.setattr(rollup, "generate_summary", fake_generate_summary)
    monkeypatch.setattr(rollup, "store_episode", fake_store_episode)
    monkeypatch.setattr(rollup, "store_memory", fake_store_memory)
    return episodes, cold


def _conv4(prefix="m"):
    return _Conv([
        _Turn("user", f"{prefix}1", "Dan"),
        _Turn("assistant", f"{prefix}2"),
        _Turn("user", f"{prefix}3", "Dan"),
        _Turn("assistant", f"{prefix}4"),
    ])


# --------------------------------------------------------------------------
# _parse_summary — verdict extraction
# --------------------------------------------------------------------------
def test_parse_substantive_strips_verdict():
    sub, summ = rollup._parse_summary(
        "SUBSTANTIVE\nDan decided EMBED_EPISODES=True.")
    assert sub is True
    assert summ == "Dan decided EMBED_EPISODES=True."


def test_parse_banter_keeps_text_for_warm():
    sub, summ = rollup._parse_summary("BANTER\nDan and Timmy joked around.")
    assert sub is False
    assert summ == "Dan and Timmy joked around."


def test_parse_tolerates_markdown_and_inline_colon():
    sub, summ = rollup._parse_summary(
        "**SUBSTANTIVE**: Dan set the decay half-life to 30 days.")
    assert sub is True
    assert summ == "Dan set the decay half-life to 30 days."


def test_parse_missing_verdict_failsafe_substantive():
    # No verdict line: keep the (real) first line AND default to persisting.
    raw = "Dan committed to the trip in March."
    sub, summ = rollup._parse_summary(raw)
    assert sub is True
    assert summ == raw


def test_parse_refusal_returns_none():
    sub, summ = rollup._parse_summary("There is no conversation to summarize.")
    assert summ is None


def test_parse_verdict_only_no_body_returns_none():
    sub, summ = rollup._parse_summary("SUBSTANTIVE\n")
    assert summ is None


def test_verdict_word_never_leaks_into_summary():
    for raw in ("SUBSTANTIVE\nreal body", "[BANTER] - greetings only",
                "substantive: lowercase works"):
        _, summ = rollup._parse_summary(raw)
        assert summ and summ.upper().split()[0] not in ("SUBSTANTIVE", "BANTER")


# --------------------------------------------------------------------------
# maybe_rollup — write-through at mint
# --------------------------------------------------------------------------
def test_substantive_writes_episode_at_mint(monkeypatch):
    episodes, cold = _patch(
        monkeypatch, "SUBSTANTIVE\nDan asked Timmy to embed episodes.")
    conv = _conv4()
    _force_age_trigger(conv)
    n_before = len(conv.hot_turns)

    assert asyncio.run(rollup.maybe_rollup(conv)) is True
    assert len(episodes) == 1
    # verdict stripped from the durable text
    assert not episodes[0]["text"].upper().startswith("SUBSTANTIVE")
    assert episodes[0]["source"]["trigger"] == "warm_mint"
    # window advanced + warm got the (clean) summary
    assert len(conv.hot_turns) < n_before
    assert len(conv.warm_summaries) == 1
    assert not cold  # legacy memories tier untouched


def test_banter_not_persisted_but_window_still_advances(monkeypatch):
    episodes, cold = _patch(monkeypatch, "BANTER\nDan and Timmy joked around.")
    conv = _conv4("b")
    _force_age_trigger(conv)
    n_before = len(conv.hot_turns)

    asyncio.run(rollup.maybe_rollup(conv))
    assert episodes == []                       # gated out of durable memory
    assert len(conv.hot_turns) < n_before       # but hot STILL drains
    assert len(conv.warm_summaries) == 1        # and warm still gets context


def test_eviction_does_not_double_write_episode(monkeypatch):
    episodes, cold = _patch(
        monkeypatch, "SUBSTANTIVE\nDan reviewed the plan and approved phase one.")
    conv = _conv4("c")
    # pre-fill warm to the cap so this rollup overflows + evicts the oldest
    for i in range(config.WARM_MAX_SUMMARIES):
        conv.warm_summaries.append(WarmSummary(
            text=f"old{i}", timestamp=time.time(), turn_count=2,
            span_start=1.0, span_end=2.0))
    _force_age_trigger(conv)

    asyncio.run(rollup.maybe_rollup(conv))
    # exactly ONE episode — the mint write — and none from eviction
    assert len(episodes) == 1
    assert not cold
    assert len(conv.warm_summaries) == config.WARM_MAX_SUMMARIES  # capped


def test_episode_span_matches_summarized_turns(monkeypatch):
    episodes, _ = _patch(monkeypatch, "SUBSTANTIVE\nConcrete decision recorded.")
    conv = _conv4("s")
    _force_age_trigger(conv)
    # the summarized block is the oldest half of hot
    split = max(1, len(conv.hot_turns) // 2)
    summarized = conv.hot_turns[:split]
    exp_start = min(t.timestamp for t in summarized)
    exp_end = max(t.timestamp for t in summarized)

    asyncio.run(rollup.maybe_rollup(conv))
    assert episodes[0]["span"] == (exp_start, exp_end)
