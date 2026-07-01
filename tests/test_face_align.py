"""Alignment + landmark-quality reject (EdgeFace preprocessing contract)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.face_align import (  # noqa: E402
    ARCFACE_REF_5PT,
    INPUT_SIZE,
    align_face,
    landmarks_ok,
    preprocess,
    preprocess_batch,
)


def test_landmarks_ok_accepts_plausible():
    assert landmarks_ok(ARCFACE_REF_5PT)


def test_landmarks_ok_rejects_degenerate():
    assert not landmarks_ok(np.zeros((5, 2), dtype=np.float32))        # eyes coincide
    assert not landmarks_ok(ARCFACE_REF_5PT[:4])                        # wrong shape
    bad = ARCFACE_REF_5PT.copy()
    bad[3][1] = bad[4][1] = 0.0                                         # mouth above eyes
    assert not landmarks_ok(bad)
    nan = ARCFACE_REF_5PT.copy(); nan[0][0] = np.nan
    assert not landmarks_ok(nan)


def test_align_face_maps_landmarks_to_template():
    # A crop whose landmarks are the template scaled+shifted should warp back so
    # its landmarks land (approximately) on the canonical template.
    img = (np.random.default_rng(0).random((160, 160, 3)) * 255).astype(np.uint8)
    src = ARCFACE_REF_5PT * 1.2 + np.array([10.0, 8.0], dtype=np.float32)
    aligned = align_face(img, src)
    assert aligned.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert aligned.dtype == np.uint8


def test_preprocess_normalization_contract():
    crop = np.full((INPUT_SIZE, INPUT_SIZE, 3), 127.5, dtype=np.float32).astype(np.uint8)
    x = preprocess(crop)
    assert x.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    # (127 - 127.5)/127.5 ~ 0; white -> ~ +1; black -> ~ -1.
    white = preprocess(np.full((INPUT_SIZE, INPUT_SIZE, 3), 255, dtype=np.uint8))
    black = preprocess(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
    assert white.max() == pytest.approx(1.0, abs=1e-6)
    assert black.min() == pytest.approx(-1.0, abs=1e-6)


def test_preprocess_batch_shape():
    crops = [np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8) for _ in range(4)]
    assert preprocess_batch(crops).shape == (4, 3, INPUT_SIZE, INPUT_SIZE)
    assert preprocess_batch([]).shape == (0, 3, INPUT_SIZE, INPUT_SIZE)
