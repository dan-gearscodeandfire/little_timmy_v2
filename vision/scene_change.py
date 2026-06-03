"""Scene-change detection for vision gating.

Compares consecutive JPEG frames to determine if the scene changed
enough to warrant a VLM analysis. Uses downsampled grayscale mean
absolute difference (MAD) — fast, no OpenCV dependency.

Typical thresholds:
  - Static scene (lighting flicker): MAD ~1-3
  - Person moves slightly: MAD ~5-10
  - Person enters/leaves frame: MAD ~15-30
  - Major scene change: MAD ~30+
"""

import io
import logging
import numpy as np
from PIL import Image
import config

log = logging.getLogger(__name__)

# Downsample resolution for comparison (fast, ignores noise)
_COMPARE_SIZE = (160, 90)

# Change threshold — MAD above this triggers VLM analysis
CHANGE_THRESHOLD = 12.0

# Minimum threshold for "something moved" (below this is just noise/flicker)
NOISE_FLOOR = 2.0


def jpeg_to_gray(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG to downsampled grayscale numpy array."""
    img = Image.open(io.BytesIO(jpeg_bytes))
    img = img.convert("L").resize(_COMPARE_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _frame_diff(frame_a: np.ndarray, frame_b: np.ndarray, illum_invariant: bool) -> np.ndarray:
    """Absolute per-pixel diff. When illum_invariant, subtract the spatial mean
    of the signed diff first so a uniform lighting shift cancels out."""
    diff = frame_b - frame_a
    if illum_invariant:
        diff = diff - float(diff.mean())
    return np.abs(diff)


def compute_change_score(frame_a: np.ndarray, frame_b: np.ndarray,
                         illum_invariant: bool = False) -> float:
    """Compute mean absolute difference between two grayscale frames.

    Returns a score from 0 (identical) to 255 (completely different).
    """
    return float(np.mean(_frame_diff(frame_a, frame_b, illum_invariant)))


def compute_localized_score(frame_a: np.ndarray, frame_b: np.ndarray,
                            rows: int, cols: int,
                            illum_invariant: bool = False) -> float:
    """Max per-cell MAD over a rows x cols grid.

    A small gesture confined to one corner produces a high MAD in its cell but
    a low whole-frame MAD; this surfaces that localized change so the gate can
    fire on it. np.array_split tolerates non-divisible frame dimensions.
    """
    diff = _frame_diff(frame_a, frame_b, illum_invariant)
    max_mad = 0.0
    for row_block in np.array_split(diff, max(1, rows), axis=0):
        for cell in np.array_split(row_block, max(1, cols), axis=1):
            if cell.size:
                m = float(cell.mean())
                if m > max_mad:
                    max_mad = m
    return max_mad


class SceneChangeDetector:
    """Stateful scene-change detector.

    Call `check(jpeg_bytes)` with each new frame. Returns True if the
    scene changed enough to warrant VLM analysis.
    """

    def __init__(self, threshold: float = CHANGE_THRESHOLD):
        self.threshold = threshold
        # Additive localized gate (2026-06-03): catches small/edge motion the
        # whole-frame MAD dilutes below threshold. Config-tunable.
        self.localized_threshold = config.VISION_SCENE_LOCALIZED_THRESHOLD
        self.grid_rows = config.VISION_SCENE_GRID_ROWS
        self.grid_cols = config.VISION_SCENE_GRID_COLS
        self.illum_invariant = config.VISION_SCENE_ILLUM_INVARIANT
        self._prev_frame: np.ndarray | None = None
        self._last_vlm_frame: np.ndarray | None = None
        self._frames_since_vlm: int = 0
        self._max_frames_without_vlm: int = 60  # force VLM every ~60s at 1fps

    def check(self, jpeg_bytes: bytes) -> tuple[bool, float]:
        """Check if the scene changed enough to trigger VLM.

        Args:
            jpeg_bytes: Raw JPEG frame data.

        Returns:
            (should_analyze, change_score) — True if VLM should run.
        """
        current = jpeg_to_gray(jpeg_bytes)
        self._frames_since_vlm += 1

        # First frame ever — always analyze
        if self._prev_frame is None:
            self._prev_frame = current
            self._last_vlm_frame = current
            self._frames_since_vlm = 0
            return True, 255.0

        # Compare against last VLM-analyzed frame (not just previous frame)
        # This prevents gradual drift from never triggering
        score = compute_change_score(self._last_vlm_frame, current, self.illum_invariant)
        local = compute_localized_score(
            self._last_vlm_frame, current,
            self.grid_rows, self.grid_cols, self.illum_invariant,
        )
        self._prev_frame = current

        # Force periodic VLM even if scene is static
        if self._frames_since_vlm >= self._max_frames_without_vlm:
            log.info("[GATE] Forced VLM after %d frames (score=%.1f, local=%.1f)",
                     self._frames_since_vlm, score, local)
            self._last_vlm_frame = current
            self._frames_since_vlm = 0
            return True, score

        # Global gate (existing) OR additive localized gate (catches small/edge
        # motion the whole-frame MAD dilutes below threshold).
        if score >= self.threshold or local >= self.localized_threshold:
            trigger = "global" if score >= self.threshold else "localized"
            log.info("[GATE] Scene changed (%s: score=%.1f/%.1f, local=%.1f/%.1f), triggering VLM",
                     trigger, score, self.threshold, local, self.localized_threshold)
            self._last_vlm_frame = current
            self._frames_since_vlm = 0
            return True, score

        if score > NOISE_FLOOR:
            log.debug("[GATE] Minor motion (score=%.1f, local=%.1f), below threshold",
                      score, local)

        return False, score

    def force_next(self):
        """Force the next check() to trigger VLM (e.g., on speech event)."""
        self._frames_since_vlm = self._max_frames_without_vlm
