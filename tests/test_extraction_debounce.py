"""Offline tests for the debounce + speaker-coalesce extraction pipeline
(memory/extraction.py), the 2026-06-06 cancel-churn structural fix.

No GPU / brain / DB: generate_memory, store_fact, store_memory are stubbed.
The point of these tests is the *scheduling* contract that keeps abandoned
extraction generations from stacking on the single-slot Strix Halo brain:

  - nothing extracts mid-burst (debounce keeps deferring),
  - the lull produces ONE coalesced pass per speaker,
  - an unbroken monologue still flushes (max-hold ceiling),
  - short STT-noise turns never even buffer.

Run: .venv/bin/pytest tests/test_extraction_debounce.py -v
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import memory.extraction as ex


def _run(coro):
    return asyncio.run(coro)


def _install_fakes():
    """Stub the LLM + stores; return the list extraction prompts land in.
    Classifier always says yes so every flush reaches the extraction pass."""
    calls: list[str] = []

    async def fake_generate_memory(prompt, thinking=None):
        if "one word" in prompt:          # classifier prompt
            return "yes"
        calls.append(prompt)              # extraction prompt
        return '{"facts": [], "memories": []}'

    async def fake_store_fact(*a, **k):
        pass

    async def fake_store_memory(*a, **k):
        pass

    ex.generate_memory = fake_generate_memory
    ex.store_fact = fake_store_fact
    ex.store_memory = fake_store_memory
    return calls


def _reset():
    ex._pending.clear()
    ex._queue.clear()
    ex._extraction_running = False
    if ex._flush_handle is not None:
        ex._flush_handle.cancel()
    ex._flush_handle = None
    ex._buffer_started = None


async def _drain():
    for _ in range(300):
        if not ex._queue and not ex._extraction_running and ex._flush_handle is None:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("extraction never drained")


def _fast_timings():
    config.EXTRACTION_DEBOUNCE_SECONDS = 0.2
    config.EXTRACTION_MAX_HOLD_SECONDS = 1.0


def test_burst_coalesces_into_one_pass():
    async def go():
        _fast_timings()
        _reset()
        calls = _install_fakes()
        for i in range(4):
            await ex.extract_and_store(
                f"I have a fact number {i} about me",
                f"reply {i}", speaker_id=1, speaker_name="dan",
            )
            await asyncio.sleep(0.05)  # gap < debounce: must NOT fire mid-burst
            assert not calls, f"extraction fired mid-burst after turn {i}"
        await asyncio.sleep(0.4)
        await _drain()
        assert len(calls) == 1, f"expected 1 coalesced pass, got {len(calls)}"
        for i in range(4):
            assert f"fact number {i}" in calls[0], f"turn {i} missing from block"
    _run(go())


def test_two_speakers_get_separate_passes():
    async def go():
        _fast_timings()
        _reset()
        calls = _install_fakes()
        await ex.extract_and_store("I am dan and I like coffee here",
                                   "ok", speaker_id=1, speaker_name="dan")
        await ex.extract_and_store("I am erin and I like tea here now",
                                   "ok", speaker_id=2, speaker_name="erin")
        await asyncio.sleep(0.4)
        await _drain()
        assert len(calls) == 2, f"expected per-speaker split, got {len(calls)}"
    _run(go())


def test_max_hold_flushes_an_unbroken_monologue():
    async def go():
        _fast_timings()
        _reset()
        calls = _install_fakes()
        fired = False
        for i in range(40):  # 40 * 0.05s = 2s > 1.0s max-hold, all gaps < debounce
            await ex.extract_and_store(
                f"continuous chatter line {i} with content",
                "r", speaker_id=1, speaker_name="dan",
            )
            await asyncio.sleep(0.05)
            if calls:
                fired = True
                break
        assert fired, "max-hold never forced a flush during unbroken chatter"
    _run(go())


def test_short_text_is_skipped_before_buffering():
    async def go():
        _fast_timings()
        _reset()
        _install_fakes()
        await ex.extract_and_store("hi", "yo", speaker_id=1, speaker_name="dan")
        assert len(ex._pending) == 0
    _run(go())
