"""EdgeFace-S face embedder — okDemerzel-side, CPU, off the event loop.

Runs the EdgeFace-S (gamma 0.5) ONNX on onnxruntime's CPUExecutionProvider so it
never contends with the two 35B GPU servers (:8083 brain, :8084 vision) — the
same reason WeSpeaker voice ID runs on CPU. The ONNX is produced by
``ops/export_edgeface_onnx.py`` (torch/timm live in a throwaway venv; the
production venv only needs onnxruntime, which it already has). Mirrors
``speaker/encoder.py``: a single lazily-loaded, cached session shared by
enrollment and live matching so they never diverge.

Intra-op threads are pinned LOW (default 2) so a burst (e.g. an enroll pose
batch) can't starve the asyncio event loop or WeSpeaker's torch threads. Always
call from ``asyncio.to_thread`` on the hot path — embed a whole batch in ONE
call, never one ``to_thread`` per crop.
"""

import logging
import os
from pathlib import Path

import numpy as np

from presence.face_align import EMBED_DIM, preprocess, preprocess_batch

log = logging.getLogger(__name__)

ONNX_PATH = Path(os.getenv(
    "TIMMY_FACE_ENCODER_ONNX",
    str(Path(__file__).resolve().parent.parent / "models" / "face" /
        "edgeface_s_gamma_05.onnx")))

# Pinned low so an enroll pose batch can't starve the event loop / voice threads.
INTRA_OP_THREADS = int(os.getenv("TIMMY_FACE_ENCODER_THREADS", "2"))

_session = None
_input_name = None
_output_name = None


def _load_session():
    """Lazily build + cache the onnxruntime session (CPU, pinned threads)."""
    global _session, _input_name, _output_name
    if _session is not None:
        return _session
    import onnxruntime as ort
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"EdgeFace ONNX not found at {ONNX_PATH}; run "
            f"`python -m ops.export_edgeface_onnx` (in a throwaway venv).")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = INTRA_OP_THREADS
    opts.inter_op_num_threads = 1
    _session = ort.InferenceSession(str(ONNX_PATH),
                                    sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name
    log.info("EdgeFace encoder loaded: %s (%d-dim, intra_op=%d)",
             ONNX_PATH.name, EMBED_DIM, INTRA_OP_THREADS)
    return _session


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def extract_embedding(aligned_rgb_112: np.ndarray) -> np.ndarray:
    """Aligned 112x112x3 RGB uint8 crop -> (512,) L2-normalized embedding."""
    sess = _load_session()
    x = preprocess(aligned_rgb_112)
    out = sess.run([_output_name], {_input_name: x})[0]
    return _l2(out.reshape(-1))


def embed_batch(aligned_rgb_list: list) -> np.ndarray:
    """List of aligned 112x112x3 RGB crops -> (N, 512) L2-normalized, in ONE
    session.run (the batched form to use behind a single asyncio.to_thread)."""
    if not aligned_rgb_list:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    sess = _load_session()
    x = preprocess_batch(aligned_rgb_list)
    out = sess.run([_output_name], {_input_name: x})[0]
    return _l2(out.reshape(len(aligned_rgb_list), -1))
