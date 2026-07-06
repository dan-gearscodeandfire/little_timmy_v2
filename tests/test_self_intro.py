"""Hermetic tests for passive self-intro detection (LED-mic anchor, 2026-07-06).

detect_self_intro fires ONLY on framed names from an unsolicited turn —
"my name is X", "call me X", "I go by X" — never on bare tokens or the weak
"I'm X" frame ("I'm tired" is the false-positive vector). Negated frames
("my name is not Walter") fail safe through _clean_name_tokens. The composed
flow: offer_confirm arms the introductions confirm; the visitor's "yes"
completes assign_name + (toggle-gated) the co-sampled face commit.

Run:
    .venv/bin/pytest tests/test_self_intro.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import runtime_toggles
from conversation.enroll_intent import detect_self_intro
from conversation.introductions import Introductions
from presence.cosample import CoSampleBuffer


# --- detector: positives -----------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("my name is Flynn", "flynn"),
    ("Hi there, my name is Flynn.", "flynn"),
    ("my name's Mary Jane", "mary_jane"),
    ("call me Flynn", "flynn"),
    ("I go by Flynn", "flynn"),
    ("well, my name is Buddy", "buddy"),          # soft name, explicit frame
    ("my name is Dan the Barbarian", "dan_the_barbarian"),
    ("my name is O'Brien", "obrien"),
])
def test_detector_positives(text, want):
    assert detect_self_intro(text) == want


# --- detector: negatives -----------------------------------------------------

@pytest.mark.parametrize("text", [
    "I'm tired",                       # weak frame excluded entirely
    "I'm Flynn",                       # weak frame excluded even with a name
    "this is Flynn",                   # weak frame excluded
    "Flynn",                           # bare token — no frame, no fire
    "my name is not Walter",           # negated -> 'not' kills the span
    "my name is hard to pronounce",    # predicate, not a name (7-06 class)
    "my name is on the whiteboard",
    "call me later",                   # non-name after the frame...
    "call me, buddy",                  # vocative filler stays rejected
    "not telling you my name",         # evasive
    "never mind, my name is secret",   # evasive wins
    "what's your name?",               # no frame at all
    "",
])
def test_detector_negatives(text):
    assert detect_self_intro(text) is None


def test_detector_second_match_wins_over_dead_first():
    # First frame cleans to None ('not'), the claim lives at the second.
    assert detect_self_intro(
        "my name is not Walter, my name is Flynn") == "flynn"


# --- composed flow: offer_confirm -> yes -> assign + face commit --------------

@pytest.fixture
def toggles(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_toggles, "STATE_PATH",
                        tmp_path / "lt_runtime_toggles.json")
    return runtime_toggles


class FakeTurn:
    def __init__(self):
        self.said: list[str] = []

    async def say(self, prompt_text: str):
        self.said.append(prompt_text)
        return SimpleNamespace(text=f"[said: {prompt_text[:24]}...]")


class FakeSpeakerID:
    def __init__(self):
        self._known_speakers = [SimpleNamespace(name="dan", speaker_id=1)]
        self._unknown_speakers = [
            SimpleNamespace(temp_id="unknown_1", name_asked=False)]
        self.assigned: list[tuple[str, str]] = []

    def assign_name(self, temp_id, name):
        self.assigned.append((temp_id, name))
        self._known_speakers.append(SimpleNamespace(name=name, speaker_id=99))
        return True


class FakeCommitter:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, name, **kwargs):
        self.calls.append({"name": name, **kwargs})
        return SimpleNamespace(face_committed=True, voice_committed=False,
                               status="committed", speaker_id=42)


@pytest.mark.asyncio
async def test_offer_confirm_then_yes_assigns_and_commits(toggles):
    toggles.set("intro_face_commit_enabled", True)
    spk, turn, committer = FakeSpeakerID(), FakeTurn(), FakeCommitter()
    cos = CoSampleBuffer()
    cos.add("unknown_1", [np.zeros((112, 112, 3), dtype=np.uint8)])
    intro = Introductions(speaker_id_module=spk, turn=turn,
                          cosample=cos, committer=committer)

    name = detect_self_intro("hey Timmy, my name is Flynn")
    assert name == "flynn"
    await intro.offer_confirm("unknown_1", name)
    assert intro.awaiting
    assert any("Flynn" in s for s in turn.said)   # spoke the confirm

    out = await intro.handle("yes that's right", "unknown_1")
    assert out.handled is False and out.speaker_name == "flynn"
    assert spk.assigned == [("unknown_1", "flynn")]
    assert len(committer.calls) == 1
    assert committer.calls[0]["name"] == "flynn"
    assert cos.crops_for("unknown_1") == []       # cleared post-commit


@pytest.mark.asyncio
async def test_offer_confirm_then_no_rejects_cleanly(toggles):
    spk, turn = FakeSpeakerID(), FakeTurn()
    intro = Introductions(speaker_id_module=spk, turn=turn)
    await intro.offer_confirm("unknown_1", "flynn")
    out = await intro.handle("no that's wrong", "unknown_1")
    assert out.handled is False
    assert spk.assigned == []
    # Rejection resets name_asked so the ordinary ask can still happen.
    assert spk._unknown_speakers[0].name_asked is False


@pytest.mark.asyncio
async def test_offer_confirm_supersedes_pending_capture(toggles):
    """A passive intro landing while a name-ask capture is pending replaces
    it — the volunteered name IS the answer."""
    spk, turn = FakeSpeakerID(), FakeTurn()
    intro = Introductions(speaker_id_module=spk, turn=turn)
    await intro.ask_name(SimpleNamespace(temp_id="unknown_1", last_text="hi"))
    await intro.offer_confirm("unknown_1", "flynn")
    out = await intro.handle("yes correct", "unknown_1")
    assert out.speaker_name == "flynn"
    assert not intro.awaiting
