"""Hermetic tests for the LED-mic anchor gate split (Dan 2026-07-06).

The doorway computes TWO predicates from the shipped gate:
    _consent_ok = runtime_toggles.identity_dialogs_allowed()   # face-consent FSM
    _dialogs_ok = _consent_ok or anchor.gate_disjunct()        # speech dialogs

A fresh anchor (lit mic in hand) un-darks the SPEECH identity dialogs at
EXPO but never the consent FSM — mic-in-hand is implied consent, the offer
would be noise. No anchor / feature off -> both predicates are byte-identical
to the shipped global-dark behavior.

runtime_toggles redirected to a tmp file; time monkeypatched for TTL cases.

Run:
    .venv/bin/pytest tests/test_anchor_gate.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import runtime_toggles
from presence import anchor


@pytest.fixture
def toggles(monkeypatch, tmp_path):
    """Isolated runtime_toggles: state lives in a per-test tmp file."""
    monkeypatch.setattr(runtime_toggles, "STATE_PATH",
                        tmp_path / "lt_runtime_toggles.json")
    return runtime_toggles


@pytest.fixture(autouse=True)
def clean_anchor():
    anchor.clear_anchor()
    yield
    anchor.clear_anchor()


@pytest.fixture
def clock(monkeypatch):
    state = {"now": 1000.0}
    monkeypatch.setattr(anchor.time, "time", lambda: state["now"])

    def advance(s):
        state["now"] += s
    return advance


def _predicates():
    """Mirror main.py's split exactly."""
    consent_ok = runtime_toggles.identity_dialogs_allowed()
    dialogs_ok = consent_ok or anchor.gate_disjunct()
    return consent_ok, dialogs_ok


# --- truth table: regime x override x anchor_enabled x fresh/stale ----------

@pytest.mark.parametrize(
    "regime, override, enabled, set_anchor, age, want_consent, want_dialogs",
    [
        # Shop: everything allowed regardless of anchor.
        ("",     False, False, False, 0.0,  True,  True),
        ("",     False, True,  True,  0.0,  True,  True),
        # EXPO dark: no override, no anchor -> both dark (shipped behavior).
        ("EXPO", False, False, False, 0.0,  False, False),
        # EXPO + fresh anchor but feature OFF -> still dark (master switch).
        ("EXPO", False, False, True,  0.0,  False, False),
        # EXPO + feature ON + fresh anchor -> speech dialogs back, consent dark.
        ("EXPO", False, True,  True,  0.0,  False, True),
        # EXPO + feature ON + STALE anchor -> dark again (TTL decay).
        ("EXPO", False, True,  True,  31.0, False, False),
        # EXPO + override -> BOTH allowed (supervised enroll, consent included).
        ("EXPO", True,  False, False, 0.0,  True,  True),
        # Legacy PARTY value still gates; anchor un-darks it the same way.
        ("PARTY", False, True, True,  0.0,  False, True),
    ],
)
def test_gate_truth_table(toggles, clock, regime, override, enabled,
                          set_anchor, age, want_consent, want_dialogs):
    toggles.set("situation_regime", regime)
    toggles.set("identity_dialogs_override", override)
    toggles.set("anchor_enabled", enabled)
    if set_anchor:
        anchor.set_anchor(source="stub")
        clock(age)
    consent_ok, dialogs_ok = _predicates()
    assert consent_ok is want_consent
    assert dialogs_ok is want_dialogs


def test_consent_predicate_unmoved_by_fresh_anchor(toggles, clock):
    """identity_dialogs_allowed() itself must never see the anchor — it also
    guards the face-consent FSM (main doorway, _generate_response, and the
    face_enroll_monitor tick all call it bare)."""
    toggles.set("situation_regime", "EXPO")
    toggles.set("anchor_enabled", True)
    anchor.set_anchor(source="stub")
    assert anchor.gate_disjunct() is True
    assert runtime_toggles.identity_dialogs_allowed() is False


def test_ttl_expiry_flips_combined_verdict_live(toggles, clock):
    toggles.set("situation_regime", "EXPO")
    toggles.set("anchor_enabled", True)
    anchor.set_anchor(source="stub")
    assert _predicates() == (False, True)
    clock(31.0)  # default anchor_ttl_s = 30.0
    assert _predicates() == (False, False)


def test_clear_flips_combined_verdict_immediately(toggles, clock):
    toggles.set("situation_regime", "EXPO")
    toggles.set("anchor_enabled", True)
    anchor.set_anchor(source="stub")
    assert _predicates() == (False, True)
    anchor.clear_anchor()  # mic racked
    assert _predicates() == (False, False)
