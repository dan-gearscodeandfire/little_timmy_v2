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


def compute_change_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute mean absolute difference between two grayscale frames.

    Returns a score from 0 (identical) to 255 (completely different).
    """
    return float(np.mean(np.abs(frame_a - frame_b)))


class SceneChangeDetector:
    """Stateful scene-change detector.

    Call `check(jpeg_bytes)` with each new frame. Returns True if the
    scene changed enough to warrant VLM analysis.
    """

    def __init__(self, threshold: float = CHANGE_THRESHOLD):
        self.threshold = threshold
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
        score = compute_change_score(self._last_vlm_frame, current)
        self._prev_frame = current

        # Force periodic VLM even if scene is static
        if self._frames_since_vlm >= self._max_frames_without_vlm:
            log.info("[GATE] Forced VLM after %d frames (score=%.1f)",
                     self._frames_since_vlm, score)
            self._last_vlm_frame = current
            self._frames_since_vlm = 0
            return True, score

        if score >= self.threshold:
            log.info("[GATE] Scene changed (score=%.1f > %.1f), triggering VLM",
                     score, self.threshold)
            self._last_vlm_frame = current
            self._frames_since_vlm = 0
            return True, score

        if score > NOISE_FLOOR:
            log.debug("[GATE] Minor motion (score=%.1f), below threshold", score)

        return False, score

    def force_next(self):
        """Force the next check() to trigger VLM (e.g., on speech event)."""
        self._frames_since_vlm = self._max_frames_without_vlm
