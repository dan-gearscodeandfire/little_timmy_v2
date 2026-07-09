"""Hermetic tests for the LED-mic anchor core (presence/anchor.py, 2026-07-06).

Covers the in-process anchor state (TTL freshness, clear, per-set override),
the gate disjunct (anchor_enabled AND fresh), and the pure nearest-face-above
geometry with its abstain-on-ambiguity contract (two candidates in tolerance
or LED-not-in-frame -> None, never a guess).

runtime_toggles is redirected onto a tmp file (never touches the live
service state); time is monkeypatched on the anchor module so TTL cases are
deterministic.

Run:
    .venv/bin/pytest tests/test_anchor.py -v
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
    """Every test starts and ends with no anchor (module-global state)."""
    anchor.clear_anchor()
    yield
    anchor.clear_anchor()


@pytest.fixture
def clock(monkeypatch):
    """Deterministic time for the anchor module."""
    state = {"now": 1000.0}
    monkeypatch.setattr(anchor.time, "time", lambda: state["now"])

    def advance(s):
        state["now"] += s
    return advance


# --- state / TTL ------------------------------------------------------------

def test_never_set_is_inactive(toggles):
    assert anchor.get_anchor() is None
    assert anchor.anchor_active() is False


def test_active_within_ttl(toggles, clock):
    anchor.set_anchor(source="stub")
    assert anchor.anchor_active() is True
    clock(29.0)  # default anchor_ttl_s = 30.0
    assert anchor.anchor_active() is True


def test_inactive_after_ttl(toggles, clock):
    anchor.set_anchor(source="stub")
    clock(31.0)
    assert anchor.anchor_active() is False


def test_refresh_restarts_ttl(toggles, clock):
    anchor.set_anchor(source="stub")
    clock(25.0)
    anchor.set_anchor(source="cv")  # CV refresh restamps
    clock(25.0)
    assert anchor.anchor_active() is True
    assert anchor.get_anchor().source == "cv"


def test_clear_is_immediate(toggles, clock):
    anchor.set_anchor(source="stub")
    anchor.clear_anchor()
    assert anchor.anchor_active() is False
    assert anchor.get_anchor() is None


def test_per_set_ttl_override_beats_toggle(toggles, clock):
    anchor.set_anchor(source="stub", ttl_s=300.0)  # long bench hold
    clock(200.0)
    assert anchor.anchor_active() is True
    clock(101.0)
    assert anchor.anchor_active() is False


def test_ttl_toggle_read_live(toggles, clock):
    anchor.set_anchor(source="stub")
    toggles.set("anchor_ttl_s", 5.0)
    clock(6.0)
    assert anchor.anchor_active() is False


# --- gate disjunct ----------------------------------------------------------

def test_disjunct_false_when_disabled_even_fresh(toggles, clock):
    anchor.set_anchor(source="stub")
    assert anchor.anchor_active() is True
    assert anchor.gate_disjunct() is False  # anchor_enabled defaults False


def test_disjunct_true_when_enabled_and_fresh(toggles, clock):
    toggles.set("anchor_enabled", True)
    anchor.set_anchor(source="stub")
    assert anchor.gate_disjunct() is True


def test_disjunct_false_when_enabled_but_stale(toggles, clock):
    toggles.set("anchor_enabled", True)
    anchor.set_anchor(source="stub")
    clock(31.0)
    assert anchor.gate_disjunct() is False


# --- geometry: nearest-face-above + abstain ----------------------------------

FRAME = (640, 480)  # x_tol at default 0.25 = 160 px


def test_single_face_above_led_is_picked(toggles):
    # Face centered at x=320, bottom y=200; LED at (320, 300) below it.
    bboxes = [(280, 100, 360, 200)]
    assert anchor.pick_anchored_face(bboxes, (320, 300), FRAME) == 0


def test_two_candidates_in_tolerance_abstain(toggles):
    # Both faces above the LED and within 160 px horizontally -> ambiguous.
    bboxes = [(280, 100, 360, 200), (400, 100, 480, 200)]
    assert anchor.pick_anchored_face(bboxes, (380, 300), FRAME) is None


def test_face_below_led_excluded(toggles):
    # bbox vertical center (y_ctr=330) below the LED y=300 -> not "above
    # the mic" (F8: rule is center-above, not bottom-above).
    bboxes = [(280, 280, 360, 380)]
    assert anchor.pick_anchored_face(bboxes, (320, 300), FRAME) is None


def test_mouth_height_led_is_picked(toggles):
    # F8 regression (live frame 2026-07-08 21:21:39): mic at the MOUTH puts
    # the LED at chin height, 2 px INSIDE the bbox (led_y=200 < y_max=202).
    # The old y_max-rule rejected the natural speaking hold; the vertical-
    # center rule (y_ctr=130 < 200) picks it.
    bboxes = [(299, 59, 404, 202)]
    assert anchor.pick_anchored_face(bboxes, (343, 200), FRAME) == 0


def test_led_at_face_center_excluded(toggles):
    # Boundary: LED exactly AT the bbox vertical center -> not below it
    # (strict <), abstain. Guards the eye-height/forehead absurdity.
    bboxes = [(280, 100, 360, 200)]  # y_ctr = 150
    assert anchor.pick_anchored_face(bboxes, (320, 150), FRAME) is None


def test_out_of_horizontal_tolerance_excluded(toggles):
    # Face center x=320, LED x=500 -> 180 px > 160 px tolerance.
    bboxes = [(280, 100, 360, 200)]
    assert anchor.pick_anchored_face(bboxes, (500, 300), FRAME) is None


def test_one_in_one_out_picks_the_one(toggles):
    # Second face is out of horizontal tolerance -> exactly one candidate.
    bboxes = [(280, 100, 360, 200), (10, 100, 90, 200)]  # centers 320, 50
    assert anchor.pick_anchored_face(bboxes, (320, 300), FRAME) == 0


def test_tolerance_boundary_inclusive(toggles):
    # |center - led_x| == exactly x_tol (160) -> included (<=).
    bboxes = [(280, 100, 360, 200)]  # center 320
    assert anchor.pick_anchored_face(bboxes, (480, 300), FRAME) == 0


def test_explicit_tol_param_overrides_toggle(toggles):
    bboxes = [(280, 100, 360, 200)]  # center 320
    # 50 px off the LED x; tight 0.05 tolerance (32 px) -> abstain.
    assert anchor.pick_anchored_face(bboxes, (370, 300), FRAME,
                                     x_tol_frac=0.05) is None


def test_empty_inputs_abstain(toggles):
    assert anchor.pick_anchored_face([], (320, 300), FRAME) is None
    assert anchor.pick_anchored_face([(0, 0, 10, 10)], None, FRAME) is None
    assert anchor.pick_anchored_face([(0, 0, 10, 10)], (5, 20), None) is None


# --- voice<->anchor binding (F1/F7, review 7-07) ------------------------------

def _enable(toggles):
    toggles.set("anchor_enabled", True)


def test_binding_stub_always_binds(toggles):
    _enable(toggles)
    anchor.set_anchor((320, 300), source="stub")
    assert anchor.binding_ok("dan") is True
    assert anchor.binding_ok("unknown_3") is True
    assert anchor.gate_disjunct("dan") is True


def test_binding_cv_unrecognized_binds_only_unknowns(toggles):
    _enable(toggles)
    anchor.set_anchor((320, 300), (280, 100, 360, 200), source="cv",
                      anchored_name=None)
    # Ordinary visitor: unknown voice + unrecognized anchored face -> bind.
    assert anchor.binding_ok("unknown_3") is True
    assert anchor.gate_disjunct("unknown_3") is True
    # Off-mic-collapse signature: ENROLLED voice + unrecognized face -> dark.
    assert anchor.binding_ok("dan") is False
    assert anchor.gate_disjunct("dan") is False


def test_binding_cv_recognized_binds_only_that_speaker(toggles):
    _enable(toggles)
    anchor.set_anchor((320, 300), (280, 100, 360, 200), source="cv",
                      anchored_name="dan")
    assert anchor.binding_ok("dan") is True
    assert anchor.binding_ok("devon") is False
    assert anchor.binding_ok("unknown_3") is False
    assert anchor.gate_disjunct("devon") is False


def test_gate_disjunct_without_speaker_is_plain_freshness(toggles):
    _enable(toggles)
    anchor.set_anchor((320, 300), source="cv", anchored_name=None)
    # Status surfaces (web/app.py) keep the unbound verdict.
    assert anchor.gate_disjunct() is True


def test_binding_no_anchor_is_false(toggles):
    _enable(toggles)
    assert anchor.binding_ok("unknown_3") is False


def test_speech_dialogs_allowed_shop_regime_ignores_binding(toggles):
    # Shop ('') regime: pure predicate True -> allowed for anyone, no anchor.
    assert anchor.speech_dialogs_allowed("dan") is True
    assert anchor.consent_allowed() is True


def test_speech_dialogs_allowed_expo_requires_bound_anchor(toggles):
    toggles.set("situation_regime", "EXPO")
    assert anchor.speech_dialogs_allowed("unknown_3") is False
    assert anchor.consent_allowed() is False
    _enable(toggles)
    anchor.set_anchor((320, 300), source="cv", anchored_name=None)
    assert anchor.speech_dialogs_allowed("unknown_3") is True
    assert anchor.speech_dialogs_allowed("dan") is False   # unbound
    assert anchor.consent_allowed() is False               # anchor never un-darks consent
