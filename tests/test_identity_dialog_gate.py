"""Hermetic tests for the EXPO identity-dialog gate (Dan 2026-07-06).

ONE gate for the whole identity-dialog class — enroll latches, misID
correction, introductions name-ask, face auto-enroll consent: every
multi-turn identity FSM that seizes turns and can end in a store write.
Gated behavior is FULLY SILENT (Dan's ruling): detectors/triggers are
skipped so the utterance falls through to the LLM as ordinary speech, and
dialog state armed BEFORE the gate closed is dropped without a spoken line.

runtime_toggles is redirected onto a tmp file so these tests never read or
write the live ~/little_timmy/data/lt_runtime_toggles.json (and never see
whatever regime the running service is in).

Run:
    .venv/bin/pytest tests/test_identity_dialog_gate.py -v
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import runtime_toggles
from conversation.introductions import Introductions
from presence.face_enroller import EnrollerConfig, FaceEnroller, State


@pytest.fixture
def toggles(monkeypatch, tmp_path):
    """Isolated runtime_toggles: state lives in a per-test tmp file."""
    monkeypatch.setattr(runtime_toggles, "STATE_PATH",
                        tmp_path / "lt_runtime_toggles.json")
    return runtime_toggles


# --- the predicate itself --------------------------------------------------

def test_default_shop_regime_allows(toggles):
    # Fresh state: regime '' (Shop) -> dialogs allowed, override irrelevant.
    assert toggles.identity_dialogs_allowed() is True


def test_expo_regime_gates(toggles):
    toggles.set("situation_regime", "EXPO")
    assert toggles.identity_dialogs_allowed() is False


def test_expo_with_override_allows(toggles):
    # The supervised-enroll override forces dialogs back on mid-show.
    toggles.set("situation_regime", "EXPO")
    toggles.set("identity_dialogs_override", True)
    assert toggles.identity_dialogs_allowed() is True


def test_legacy_party_value_still_gates(toggles):
    # The /api/situation whitelist is binary ''/'EXPO' since 2026-07-05, but
    # a stale on-disk PARTY must still gate (mirrors identifier.py's
    # CONTINUITY_DISABLED_REGIMES tolerance).
    toggles.set("situation_regime", "PARTY")
    assert toggles.identity_dialogs_allowed() is False


def test_override_is_inert_in_shop(toggles):
    # Override left on after a show changes nothing back in the shop.
    toggles.set("identity_dialogs_override", True)
    assert toggles.get("situation_regime") == ""
    assert toggles.identity_dialogs_allowed() is True


def test_flip_applies_live(toggles):
    # The predicate re-reads per call — no caching across a webui flip.
    toggles.set("situation_regime", "EXPO")
    assert toggles.identity_dialogs_allowed() is False
    toggles.set("identity_dialogs_override", True)
    assert toggles.identity_dialogs_allowed() is True
    toggles.set("identity_dialogs_override", False)
    assert toggles.identity_dialogs_allowed() is False
    toggles.set("situation_regime", "")
    assert toggles.identity_dialogs_allowed() is True


# --- FaceEnroller.drop_gated ------------------------------------------------

def _enroller():
    spoken: list[str] = []

    async def _speak(text):
        spoken.append(text)

    async def _say(text):
        spoken.append(text)
        return SimpleNamespace(text=text)

    async def _verify():
        return []

    fe = FaceEnroller(
        say=_say,
        speak=_speak,
        enroll_stream=None,       # never reached by these tests
        verify_faces=_verify,
        turn_lock=asyncio.Lock(),
        cfg=EnrollerConfig(enabled=True),
    )
    return fe, spoken


@pytest.mark.parametrize("state", [State.OFFERING, State.ASK_NAME,
                                   State.CONFIRM_NAME])
def test_drop_gated_clears_awaiting_dialog_silently(state):
    fe, spoken = _enroller()
    fe.state = state
    fe._deadline = fe._now() + 45.0
    assert fe.awaiting

    fe.drop_gated()

    assert fe.state == State.IDLE
    assert not fe.awaiting
    assert spoken == []                       # SILENT — no "maybe another time"
    assert fe._cooldown_until > fe._now()     # no instant re-offer on re-open


def test_drop_gated_noop_when_idle():
    fe, spoken = _enroller()
    assert fe.state == State.IDLE

    fe.drop_gated()

    assert fe.state == State.IDLE
    assert spoken == []
    assert fe._cooldown_until == 0.0          # no cooldown minted from nothing


# --- Introductions.drop_pending ----------------------------------------------


class _FakeTurn:
    def __init__(self):
        self.said: list[str] = []

    async def say(self, prompt_text: str):
        self.said.append(prompt_text)
        return SimpleNamespace(text=prompt_text)


class _FakeSpeakerID:
    def __init__(self):
        self._known_speakers = [SimpleNamespace(name="dan")]
        self.assigned: list[tuple[str, str]] = []

    def assign_name(self, temp_id, name):
        self.assigned.append((temp_id, name))
        return True


@pytest.mark.asyncio
async def test_drop_pending_clears_name_ask_silently():
    spk, turn = _FakeSpeakerID(), _FakeTurn()
    intro = Introductions(speaker_id_module=spk, turn=turn)
    await intro.ask_name(SimpleNamespace(temp_id="unknown_1",
                                         last_text="hello there"))
    assert intro.awaiting
    said_before = list(turn.said)

    intro.drop_pending()

    assert not intro.awaiting
    assert turn.said == said_before           # SILENT drop

    # The visitor's next utterance is ordinary speech: no capture, no write.
    out = await intro.handle("I'm Bob", "unknown_1")
    assert out.handled is False
    assert spk.assigned == []


def test_drop_pending_noop_when_nothing_pending():
    intro = Introductions(speaker_id_module=_FakeSpeakerID(), turn=_FakeTurn())
    assert not intro.awaiting
    intro.drop_pending()                      # must not raise or log-spam
    assert not intro.awaiting
