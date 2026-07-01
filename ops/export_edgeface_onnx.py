"""Export EdgeFace-S (gamma 0.5) to ONNX + pin a parity fixture.

Downloads the pretrained EdgeFace-S weights via torch.hub (otroshi/edgeface,
CPU), exports to ``models/face/edgeface_s_gamma_05.onnx``, then verifies the
exported ONNX reproduces the torch reference on a FIXED seeded input (cosine
>= 0.9999). Saves the fixed input + reference embedding to
``tests/fixtures/edgeface_parity.npz`` so ``tests/test_face_encoder_parity.py``
can lock the export WITHOUT needing torch/hub at test time.

Run once (network + torch): ``.venv/bin/python -m ops.export_edgeface_onnx``.
Re-run whenever the model variant or preprocessing contract changes.
"""

from pathlib import Path

import numpy as np
import torch

# Inlined (not imported from presence.face_align) so this export script can run
# in a THROWAWAY venv without pulling cv2 — the production venv must never get
# the export-time deps (timm/torchvision), which drag in a CUDA torch build.
# Keep in sync with presence.face_align.{INPUT_SIZE,EMBED_DIM}.
INPUT_SIZE = 112
EMBED_DIM = 512

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_OUT = REPO_ROOT / "models" / "face" / "edgeface_s_gamma_05.onnx"
FIXTURE_OUT = REPO_ROOT / "tests" / "fixtures" / "edgeface_parity.npz"
MODEL_NAME = "edgeface_s_gamma_05"
SEED = 20260630


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading {MODEL_NAME} via torch.hub (CPU)...")
    model = torch.hub.load("otroshi/edgeface", MODEL_NAME,
                           source="github", pretrained=True)
    model.eval()

    # Fixed, seeded, already-normalized input in the EdgeFace contract range.
    x = (np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32) - 0.5) / 0.5
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy().reshape(-1)
    assert ref.shape[0] == EMBED_DIM, f"expected {EMBED_DIM}-d, got {ref.shape}"

    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX -> {ONNX_OUT}")
    torch.onnx.export(
        model, torch.from_numpy(x), str(ONNX_OUT),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    # The dynamo exporter externalizes weights to a .onnx.data sidecar; re-save
    # as ONE self-contained file so the artifact can't lose its sidecar.
    import onnx
    _m = onnx.load(str(ONNX_OUT))
    onnx.save_model(_m, str(ONNX_OUT), save_as_external_data=False)
    _sidecar = ONNX_OUT.with_suffix(".onnx.data")
    if _sidecar.exists():
        _sidecar.unlink()

    # Parity check via onnxruntime CPU.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_OUT),
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["embedding"], {"input": x})[0].reshape(-1)

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos = _cos(ref, onnx_out)
    print(f"torch-vs-onnx cosine = {cos:.6f}")
    assert cos >= 0.9999, f"ONNX export diverged from torch reference (cos={cos})"

    FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE_OUT, input=x, embedding=onnx_out)
    print(f"Saved parity fixture -> {FIXTURE_OUT}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
