"""Smoke test for okDemerzel-side YuNet detection + alignment glue."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence import face_detect as fd  # noqa: E402
from presence.face_align import INPUT_SIZE  # noqa: E402

SCENE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "scene.jpg"

pytestmark = pytest.mark.skipif(
    not (SCENE.exists() and fd.YUNET_PATH.exists()),
    reason="scene fixture or YuNet model missing")


def test_detects_face_in_scene():
    frame = cv2.imread(str(SCENE))
    faces = fd.detect_faces(frame)
    assert len(faces) >= 1
    box, lm = faces[0]
    assert box.shape == (4,) and lm.shape == (5, 2)


def test_blank_frame_no_faces():
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    assert fd.detect_faces(blank) == []
    assert fd.aligned_crops(blank) == []


def test_aligned_crops_shape():
    frame = cv2.imread(str(SCENE))
    crops = fd.aligned_crops(frame)
    assert len(crops) >= 1
    aligned, bbox = crops[0]
    assert aligned.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]
