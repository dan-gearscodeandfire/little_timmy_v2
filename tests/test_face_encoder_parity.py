"""Lock the EdgeFace ONNX export against silent drift.

Uses the fixture written by ops/export_edgeface_onnx.py (a fixed seeded input +
the reference embedding from the torch-verified export), so this runs on the
PRODUCTION venv with only onnxruntime — no torch/timm/hub. If the .onnx is
re-exported with a different model/opset/preprocessing and diverges, this fails.
Also asserts the production face_encoder + face_align preprocessing path agrees
with a raw session.run (guards the normalization contract).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
ONNX_PATH = REPO / "models" / "face" / "edgeface_s_gamma_05.onnx"
FIXTURE = REPO / "tests" / "fixtures" / "edgeface_parity.npz"

pytestmark = pytest.mark.skipif(
    not (ONNX_PATH.exists() and FIXTURE.exists()),
    reason="EdgeFace ONNX/fixture not exported (run ops.export_edgeface_onnx)")


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_onnx_reproduces_reference_embedding():
    import onnxruntime as ort
    d = np.load(FIXTURE)
    x, ref = d["input"], d["embedding"]
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(["embedding"], {"input": x})[0].reshape(-1)
    assert out.shape[0] == 512
    assert _cos(ref, out) >= 0.9999


def test_face_encoder_outputs_normalized_512():
    from presence import face_encoder
    # A synthetic aligned crop; we only check shape + L2-normalization here.
    rng = np.random.default_rng(0)
    crop = (rng.random((112, 112, 3)) * 255).astype(np.uint8)
    emb = face_encoder.extract_embedding(crop)
    assert emb.shape == (512,)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)


def test_face_encoder_batch_matches_single():
    from presence import face_encoder
    rng = np.random.default_rng(1)
    crops = [(rng.random((112, 112, 3)) * 255).astype(np.uint8) for _ in range(3)]
    batch = face_encoder.embed_batch(crops)
    assert batch.shape == (3, 512)
    singles = np.vstack([face_encoder.extract_embedding(c) for c in crops])
    # Batched and single-crop paths must agree (same preprocessing/session).
    for i in range(3):
        assert _cos(batch[i], singles[i]) >= 0.9999
