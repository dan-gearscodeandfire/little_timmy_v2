"""Hermetic tests for anchored-crop plumbing (LED-mic anchor, 2026-07-06).

_recognize_many threads an optional led_xy into recognize_frame, which pairs
it against the per-frame face bboxes via presence.anchor.pick_anchored_face
(abstain on ambiguity) and surfaces the picked crops as
FaceObservation.anchored_face_crops. No cross-frame ==1-face rule for
anchored crops — every crop that survived was individually unambiguous under
the LED. led_xy=None must be byte-identical to the pre-anchor behavior.

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
    return runtime_toggles


def _crop(v):
    return np.full((112, 112, 3), v, dtype=np.uint8)


def _fake_frames(frames_spec):
    """Build a recognize_frame stand-in from per-frame specs:
    each spec is a list of (crop, bbox) pairs; frames keyed by jpeg bytes."""
    def fake(jpeg, led_xy=None, detect_led=False):
        crops = frames_spec[jpeg]
        preds = []
        size = (640, 480)
        sole = crops[0][0] if len(crops) == 1 else None
        anchored = None
        if led_xy is not None and crops:
            from presence.anchor import pick_anchored_face
            idx = pick_anchored_face([b for _, b in crops], led_xy, size)
            if idx is not None:
                anchored = crops[idx][0]
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
    preds, size, detected, sole, anchored = face_recognize._recognize_many(
        [b"f1"], led_xy=LED)
    assert detected == 2
    assert sole == []            # sole-face rule abstains (2 faces)
    assert len(anchored) == 1
    assert anchored[0][0, 0, 0] == 1   # crop A


def test_ambiguous_frame_contributes_nothing(toggles, monkeypatch):
    """Both faces above the LED and in tolerance -> per-frame abstain."""
    spec = {b"f1": [(_crop(1), BOX_A), (_crop(2), BOX_B)]}
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    toggles.set("anchor_x_tol_frac", 0.5)  # 320 px — both qualify
    *_, anchored = face_recognize._recognize_many([b"f1"], led_xy=(400, 300))
    assert anchored == []


def test_face_below_led_not_anchored(toggles, monkeypatch):
    spec = {b"f1": [(_crop(3), BOX_C)]}
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    *_, anchored = face_recognize._recognize_many([b"f1"], led_xy=LED)
    assert anchored == []


def test_led_none_is_todays_behavior(toggles, monkeypatch):
    """No anchor -> anchored empty, sole-face rule untouched."""
    spec = {b"f1": [(_crop(1), BOX_A)]}
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    preds, size, detected, sole, anchored = face_recognize._recognize_many(
        [b"f1"], led_xy=None)
    assert detected == 1
    assert len(sole) == 1        # sole-face rule fills as before
    assert anchored == []


def test_multi_frame_anchored_crops_accumulate(toggles, monkeypatch):
    """Each unambiguous frame contributes a crop (pose diversity), even when
    another frame of the grab was ambiguous."""
    spec = {
        b"f1": [(_crop(1), BOX_A), (_crop(2), BOX_B)],  # A anchors (tight tol)
        b"f2": [(_crop(4), BOX_A)],                     # A anchors again
    }
    monkeypatch.setattr(face_recognize, "recognize_frame", _fake_frames(spec))
    toggles.set("anchor_x_tol_frac", 0.1)
    preds, size, detected, sole, anchored = face_recognize._recognize_many(
        [b"f1", b"f2"], led_xy=LED)
    assert detected == 2         # max across frames
    assert sole == []            # cross-frame !=1 rule drops sole crops
    assert [c[0, 0, 0] for c in anchored] == [1, 4]


# --- dataclass compat ---------------------------------------------------------

def test_faceobservation_constructs_without_new_field():
    obs = FaceObservation(captured_at=0.0, predictions=(), behavior=None)
    assert obs.anchored_face_crops == ()
    assert obs.sole_face_crops == ()


def test_faceobservation_constructs_with_new_field():
    obs = FaceObservation(captured_at=0.0, predictions=(), behavior=None,
                          anchored_face_crops=(_crop(1),))
    assert len(obs.anchored_face_crops) == 1
