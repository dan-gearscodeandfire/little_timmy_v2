"""Hermetic tests for anchored-crop plumbing (LED-mic anchor, 2026-07-06).

_recognize_many threads an optional led_xy into recognize_frame, which pairs
it against the per-frame face bboxes via presence.anchor.pick_anchored_face
(abstain on ambiguity) and surfaces the picked crops as
FaceObservation.anchored_face_crops. Since the F6 fix (review 7-07) a grab's
picks must ALSO agree cross-frame (bbox x-centers within tolerance, no two
different recognized names) — a mic handoff mid-grab abstains instead of
mixing two people's crops. led_xy=None must be byte-identical to the
pre-anchor behavior.

recognize_frame is monkeypatched (no cv2/YuNet/EdgeFace); runtime_toggles
redirected to a tmp file for the geometry-tolerance read.

Run:
    .venv/bin/pytest tests/test_anchored_crops.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import runtime_toggles
from presence import face_recognize
from presence.types import FaceObservation


@pytest.fixture
def toggles(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_toggles, "STATE_PATH",
                        tmp_path / "lt_runtime_toggles.json")
    # The STATE_PATH swap must not serve a stale parse of the real file.
    monkeypatch.setattr(runtime_toggles, "_cache_stamp", None)
    monkeypatch.setattr(runtime_toggles, "_cache_state", None)
    return runtime_toggles


def _crop(v):
    return np.full((112, 112, 3), v, dtype=np.uint8)


def _fake_frames(frames_spec, names=None):
    """Build a recognize_frame stand-in from per-frame specs:
    each spec is a list of (crop, bbox) pairs; frames keyed by jpeg bytes.
    ``names`` optionally maps jpeg -> recognized name of that frame's pick."""
    def fake(jpeg, led_xy=None, detect_led=False):
        crops = frames_spec[jpeg]
        preds = []
        size = (640, 480)
        # Mirror the live contract: sole is (crop, frontal_ratio); anchored
        # carries a "frontal" key (frontality shadow, 2026-07-15).
        sole = (crops[0][0], 0.0) if len(crops) == 1 else None
        anchored = None
        if led_xy is not None and crops:
            from presence.anchor import pick_anchored_face
            idx = pick_anchored_face([b for _, b in crops], led_xy, size)
            if idx is not None:
                anchored = {"crop": crops[idx][0], "bbox": crops[idx][1],
                            "led_xy": led_xy, "cv_led": False,
                            "name": (names or {}).get(jpeg),
                            "frontal": 0.0}
        return preds, size, len(crops), sole, anchored
    return fake


# bboxes: A centered at x=320 above y=300; B at x=480; C below the LED.
BOX_A = (280, 100, 360, 200)
BOX_B = (440, 100, 520, 200)
BOX_C = (280, 320, 360, 420)
LED = (320, 300)


def test_anchored_crop_selected_in_crowd(toggles, monkeypatch):
    """Two faces in frame: sole-face rule starves, the anchor still picks."""
    spec = {b"f1": [(_crop(1), BOX_A), (_crop(2), BOX_B)]}
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    # LED at x=320: A in tolerance+above, B is 160px off center... use tight
    # tolerance via toggle so only A qualifies.
    toggles.set("anchor_x_tol_frac", 0.1)  # 64 px
    preds, size, detected, sole, anchored, name = face_recognize._recognize_many(
        [b"f1"], led_xy=LED)
    assert detected == 2
    assert sole == []            # sole-face rule abstains (2 faces)
    assert len(anchored) == 1
    assert anchored[0][0, 0, 0] == 1   # crop A
    assert name is None


def test_ambiguous_frame_contributes_nothing(toggles, monkeypatch):
    """Both faces above the LED and in tolerance -> per-frame abstain."""
    spec = {b"f1": [(_crop(1), BOX_A), (_crop(2), BOX_B)]}
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    toggles.set("anchor_x_tol_frac", 0.5)  # 320 px — both qualify
    *_, anchored, name = face_recognize._recognize_many([b"f1"], led_xy=(400, 300))
    assert anchored == []
    assert name is None


def test_face_below_led_not_anchored(toggles, monkeypatch):
    spec = {b"f1": [(_crop(3), BOX_C)]}
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    *_, anchored, name = face_recognize._recognize_many([b"f1"], led_xy=LED)
    assert anchored == []
    assert name is None


def test_led_none_is_todays_behavior(toggles, monkeypatch):
    """No anchor -> anchored empty, sole-face rule untouched."""
    spec = {b"f1": [(_crop(1), BOX_A)]}
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    preds, size, detected, sole, anchored, name = face_recognize._recognize_many(
        [b"f1"], led_xy=None)
    assert detected == 1
    assert len(sole) == 1        # sole-face rule fills as before
    assert anchored == []
    assert name is None


def test_multi_frame_anchored_crops_accumulate(toggles, monkeypatch):
    """Each unambiguous frame contributes a crop (pose diversity), even when
    another frame of the grab was ambiguous."""
    spec = {
        b"f1": [(_crop(1), BOX_A), (_crop(2), BOX_B)],  # A anchors (tight tol)
        b"f2": [(_crop(4), BOX_A)],                     # A anchors again
    }
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    toggles.set("anchor_x_tol_frac", 0.1)
    preds, size, detected, sole, anchored, name = face_recognize._recognize_many(
        [b"f1", b"f2"], led_xy=LED)
    assert detected == 2         # max across frames
    assert sole == []            # cross-frame !=1 rule drops sole crops
    assert [c[0, 0, 0] for c in anchored] == [1, 4]


# --- F6 cross-frame consistency (review 7-07) ---------------------------------

def test_cross_frame_position_disagreement_abstains(toggles, monkeypatch):
    """A mic handoff mid-grab: per-frame picks land on two far-apart faces ->
    the whole grab abstains (no mixed-person crops)."""
    spec = {
        b"f1": [(_crop(1), BOX_A)],   # x-center 320
        b"f2": [(_crop(2), BOX_B)],   # x-center 480 — a different person
    }
    toggles.set("anchor_x_tol_frac", 0.1)  # 64 px << the 160 px spread
    # The LED moved with the mic: give each frame its own led_xy so the
    # per-frame geometry picks BOTH faces, then the cross-frame check trips.
    fake = _fake_frames(spec)

    def fake2(jpeg, led_xy=None, detect_led=False):
        return fake(jpeg, led_xy=LED if jpeg == b"f1" else (480, 300))
    monkeypatch.setattr(face_recognize, "recognize_frame", fake2)
    *_, anchored, name = face_recognize._recognize_many([b"f1", b"f2"])
    assert anchored == []
    assert name is None


def test_cross_frame_identity_disagreement_abstains(toggles, monkeypatch):
    """Two frames pick faces recognized as DIFFERENT enrolled people ->
    abstain, even if spatially close."""
    spec = {b"f1": [(_crop(1), BOX_A)], b"f2": [(_crop(2), BOX_A)]}
    names = {b"f1": "dan", b"f2": "devon"}
    monkeypatch.setattr(face_recognize, "recognize_frame",
                        _fake_frames(spec, names))
    toggles.set("anchor_x_tol_frac", 0.25)
    *_, anchored, name = face_recognize._recognize_many([b"f1", b"f2"], led_xy=LED)
    assert anchored == []
    assert name is None


def test_intermittent_recognition_names_the_track(toggles, monkeypatch):
    """One frame recognizes the anchored face, the other doesn't: spatially
    consistent -> the track carries the recognized name."""
    spec = {b"f1": [(_crop(1), BOX_A)], b"f2": [(_crop(2), BOX_A)]}
    names = {b"f1": "dan"}          # f2 pick unrecognized
    monkeypatch.setattr(face_recognize, "recognize_frame",
                        _fake_frames(spec, names))
    toggles.set("anchor_x_tol_frac", 0.25)
    *_, anchored, name = face_recognize._recognize_many([b"f1", b"f2"], led_xy=LED)
    assert len(anchored) == 2
    assert name == "dan"


# --- dataclass compat ---------------------------------------------------------

def test_faceobservation_constructs_without_new_field():
    obs = FaceObservation(captured_at=0.0, predictions=(), behavior=None)
    assert obs.anchored_face_crops == ()
    assert obs.sole_face_crops == ()
    assert obs.anchored_face_name is None


def test_faceobservation_constructs_with_new_field():
    obs = FaceObservation(captured_at=0.0, predictions=(), behavior=None,
                          anchored_face_crops=(_crop(1),),
                          anchored_face_name="dan")
    assert len(obs.anchored_face_crops) == 1
    assert obs.anchored_face_name == "dan"
