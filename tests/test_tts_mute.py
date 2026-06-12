"""Tests for the mouth-mute (tts_muted) toggle.

2026-06-12: a runtime mute for Timmy's CONVERSATIONAL voice (replies + THINKING
fillers) so the mic stays fully open for clean two-voice attribution tests and
guest enrollment. speak()/speak_filler() skip the playback enqueue when muted
(so capture.suppressed never sets); the supervisor /api/announce channel passes
force=True to bypass it. Fails OPEN — a persistence glitch never silences Timmy.

Run:
    .venv/bin/pytest tests/test_tts_mute.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import tts.engine as engine_mod
from tts.engine import TTSEngine


def _fake_synth(text, model_path):
    return np.array([0.1, 0.2], dtype=np.float32), 16000


@pytest.fixture
def engine(monkeypatch):
    # Avoid loading Piper: stub synthesis so the unmuted/force path can enqueue.
    monkeypatch.setattr(engine_mod, "_synthesize_raw", _fake_synth)
    return TTSEngine("dummy.onnx")


@pytest.mark.asyncio
async def test_muted_speak_does_not_enqueue(engine, monkeypatch):
    monkeypatch.setattr(engine_mod, "_tts_muted", lambda: True)
    await engine.speak("hello there friend")
    assert engine._playback_queue.empty()


@pytest.mark.asyncio
async def test_muted_speak_filler_does_not_enqueue_or_fall_through(engine, monkeypatch):
    monkeypatch.setattr(engine_mod, "_tts_muted", lambda: True)
    # Even on a cache miss, muted filler must NOT fall through to speak().
    await engine.speak_filler("Huh.")
    assert engine._playback_queue.empty()


@pytest.mark.asyncio
async def test_force_bypasses_mute(engine, monkeypatch):
    """The announce channel (force=True) speaks even when muted."""
    monkeypatch.setattr(engine_mod, "_tts_muted", lambda: True)
    await engine.speak("This is Claude talking.", force=True)
    assert not engine._playback_queue.empty()


@pytest.mark.asyncio
async def test_unmuted_speak_enqueues(engine, monkeypatch):
    monkeypatch.setattr(engine_mod, "_tts_muted", lambda: False)
    await engine.speak("normal reply")
    assert not engine._playback_queue.empty()


def test_tts_muted_fails_open(monkeypatch):
    """If persistence raises, _tts_muted must return False (never silence)."""
    import persistence.runtime_toggles as rt
    monkeypatch.setattr(rt, "get", lambda k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert engine_mod._tts_muted() is False


def test_tts_muted_default_is_false():
    """Default toggle state is unmuted (Timmy speaks)."""
    from persistence.runtime_toggles import _DEFAULTS
    assert _DEFAULTS["tts_muted"] is False
