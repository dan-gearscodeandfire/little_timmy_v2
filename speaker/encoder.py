"""WeSpeaker speaker-embedding backend (single source of truth).

Production speaker-ID, all enrollment scripts, and the ops sweeps must embed
audio THE SAME WAY or their distances aren't comparable. This module is that one
way. It wraps ``pyannote/wespeaker-voxceleb-resnet34-LM`` (a WeSpeaker ResNet34
voice encoder, loaded offline from the HF cache) behind a single
``extract_embedding(audio_16k)`` call.

History: LT shipped on Resemblyzer. The 2026-06-17 encoder A/B + through-chain
calibration (Obsidian ``lt-wespeaker-threshold-calibration-2026-06-17``) moved
the production encoder to WeSpeaker for its wider genuine/impostor separation
and to share one encoder with the diarization/anti-model path. WeSpeaker and
Resemblyzer are BOTH 256-dim but live on DIFFERENT, uncorrelated cosine scales
(measured cross-encoder sim ~0.006), so voiceprints, cohorts, and every tuned
threshold are encoder-specific — never mix files from the two spaces.

Contract (matches what identify() needs and what the calibration measured):
  - input  : float32/float64 mono waveform @ 16 kHz (the exact post-capture
             signal: 48k stereo -> ch0 -> ::3 decimate -> 80 Hz Butterworth HP,
             produced upstream in audio/capture.py). No extra VAD/preprocess —
             the calibration that set the WeSpeaker thresholds fed the whole
             waveform to ``Inference(window="whole")``, so we do the same.
  - output : (256,) float64 L2-normalized embedding. Cosine distance is then
             scale-invariant, but downstream code (_build_prototypes, open_set)
             assumes unit rows, so we normalize here once.

Loading the resnet34 costs ~2 s once; per-utterance embedding is ~10-22 ms on
okDemerzel CPU for 2-5 s clips (measured 2026-06-17) — comparable to Resemblyzer,
no real-time-budget regression on the live identify path.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

# pyannote/HF must not hit the network in the live service or headless ops runs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

WESPEAKER_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBED_DIM = 256

_EPS = 1e-8

# Process-wide cached Inference, guarded so concurrent first-touches (the live
# identify path can race a background warmup) only build the model once.
_inference = None
_lock = threading.Lock()


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + _EPS)


def get_inference():
    """Return the cached pyannote ``Inference`` (whole-window), loading the
    WeSpeaker model offline on first call. Heavy import is deferred so this
    module stays import-clean for unit tests that never embed real audio."""
    global _inference
    if _inference is None:
        with _lock:
            if _inference is None:
                import torch  # noqa: F401  (ensures torch present; pyannote needs it)
                from pyannote.audio import Inference, Model
                model = Model.from_pretrained(WESPEAKER_MODEL)
                _inference = Inference(model, window="whole")
                log.info("WeSpeaker encoder loaded (%s)", WESPEAKER_MODEL)
    return _inference


def extract_embedding(audio_16k: np.ndarray) -> np.ndarray:
    """Embed a 16 kHz mono waveform into a (256,) L2-normalized WeSpeaker vector.

    Whole-window inference (no VAD trim) to match the threshold calibration.
    """
    import torch
    inf = get_inference()
    x = np.asarray(audio_16k, dtype=np.float32).ravel()
    out = inf({"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": 16000})
    return _l2(out)
