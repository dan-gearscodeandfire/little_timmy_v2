"""Offline tests for the Introductions sub-dialog — the multi-turn name
exchange, driven with fakes. Run:

    .venv/bin/pytest tests/test_introductions.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation.introductions import Introductions, _extract_name


class FakeTurn:
    """Records the lines Timmy was asked to say; returns them as the result."""
    def __init__(self):
        self.said: list[str] = []

    async def say(self, prompt_text: str):
        self.said.append(prompt_text)
        return SimpleNamespace(text=f"[said: {prompt_text[:24]}...]")


class FakeSpeakerID:
    def __init__(self, known=("dan", "timmy"), unknown_temp_ids=()):
        self._known_speakers = [SimpleNamespace(name=n) for n in known]
        self._unknown_speakers = [
            SimpleNamespace(temp_id=t, name_asked=True) for t in unknown_temp_ids
        ]
        self.assigned: list[tuple[str, str]] = []

    def assign_name(self, temp_id, name):
        self.assigned.append((temp_id, name))
        return True


def _make(unknown_temp_ids=("unknown_1",)):
    spk = FakeSpeakerID(unknown_temp_ids=unknown_temp_ids)
    turn = FakeTurn()
    return Introductions(speaker_id_module=spk, turn=turn), spk, turn


# --- the full happy path -------------------------------------------------

@pytest.mark.asyncio
async def test_ask_then_capture_then_confirm_then_assign():
    intro, spk, turn = _make()
    assert not intro.awaiting

    # 1. ask the new speaker for their name
    await intro.ask_name(SimpleNamespace(temp_id="unknown_1", last_text="hey there"))
    assert intro.awaiting
    assert "Ask them for their name" in turn.said[0]

    # 2. they answer with a name -> Timmy confirms, turn is "handled"
    out = await intro.handle("I'm Bob", "unknown_1")
    assert out.handled is True
    assert "Did you say Bob?" in turn.said[1]

    # 3. they say yes -> name committed, fall through to a normal turn as Bob
    out = await intro.handle("yes", "unknown_1")
    assert out.handled is False
    assert out.speaker_name == "bob"
    assert spk.assigned == [("unknown_1", "bob")]
    assert not intro.awaiting


@pytest.mark.asyncio
async def test_rejection_reopens_for_reask():
    intro, spk, turn = _make()
    await intro.ask_name(SimpleNamespace(temp_id="unknown_1", last_text="hi"))
    await intro.handle("I'm Bob", "unknown_1")     # -> pending confirm

    out = await intro.handle("no", "unknown_1")
    assert out.handled is False
    assert spk.assigned == []                       # nothing committed
    assert not intro.awaiting                        # confirm cleared
    # name_asked reset so the speaker can be re-asked
    assert spk._unknown_speakers[0].name_asked is False


@pytest.mark.asyncio
async def test_no_pending_known_speaker_passes_through():
    intro, spk, turn = _make()
    out = await intro.handle("what's the weather", "dan")
    assert out.handled is False
    assert out.speaker_name == "dan"
    assert turn.said == []


@pytest.mark.asyncio
async def test_evasive_answer_is_not_captured_as_a_name():
    intro, spk, turn = _make()
    await intro.ask_name(SimpleNamespace(temp_id="unknown_1", last_text="hi"))
    out = await intro.handle("none of your business", "unknown_1")
    # no name extracted -> not handled, capture cleared, nothing said beyond the ask
    assert out.handled is False
    assert len(turn.said) == 1
    assert not intro.awaiting


# --- the extractor itself ------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I'm Erin", "erin"),
    ("My name is Preston", "preston"),
    ("Dexter", "dexter"),
    ("call me Max", "max"),
    ("not sure", None),
    ("fine", None),
    ("yes", None),
    # F9 (review 7-07): the extractor now delegates to enroll_intent's
    # canonical extract_reply_name, so fixes landed there apply here too —
    # these were parsed as names by the retired duplicate.
    ("call me later", None),
    ("call me tomorrow", None),
    ("I go by Dex", "dex"),
    ("My name is Mary Jane", "mary_jane"),
])
def test_extract_name(text, expected):
    assert _extract_name(text) == expected


# --- F2 (review 7-07): refused assign_name must not promote ------------------

class RefusingSpeakerID(FakeSpeakerID):
    def assign_name(self, temp_id, name):
        super().assign_name(temp_id, name)
        return False    # tombstoned / reserved / already-known


@pytest.mark.asyncio
async def test_refused_assign_keeps_unknown_speaker():
    spk = RefusingSpeakerID(unknown_temp_ids=("unknown_1",))
    turn = FakeTurn()
    intro = Introductions(speaker_id_module=spk, turn=turn)
    await intro.offer_confirm("unknown_1", "dan")   # claim an enrolled name
    out = await intro.handle("yes that is right", "unknown_1")
    assert out.handled is False
    # The refused name must NOT become the turn's speaker (facts would file
    # under the real person's name); the speaker stays the unknown.
    assert out.speaker_name == "unknown_1"
