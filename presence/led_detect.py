"""Green-LED blob detection for the LED-mic anchor (2026-07-06).

Finds the lit green LED on the handed booth mic in a /capture frame — pure
CPU OpenCV (~1ms), deliberately NOT a VLM query (the two-35B GPU-contention
finding: cross-process halving on the shared card).

Pipeline: HSV mask (green hue band + saturation/brightness floors) OR'd with
a specular-core rescue — a lit LED blooms WHITE at the center (saturation
collapses, so the green mask alone loses the hottest pixels) with a green
halo; near-white pixels whose small dilation overlaps the green mask are
folded back in. Morphological open kills speckle, then connected components
filtered by area. EXACTLY one surviving blob -> its centroid; zero or 2+ ->
None (abstain — two green lights is ambiguity, the same contract as the
face geometry).

Knobs live in runtime_toggles (anchor_led_*) so the booth's lighting can be
tuned live; callers may also pass explicit values (hermetic tests).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def find_green_led(frame_bgr: np.ndarray, *,
                   h_lo: Optional[int] = None, h_hi: Optional[int] = None,
                   s_min: Optional[int] = None, v_min: Optional[int] = None,
                   min_area: Optional[int] = None,
                   max_area: Optional[int] = None) -> Optional[tuple]:
    """(x, y) centroid of the single lit green LED in ``frame_bgr``, or None.

    Blocking (cv2); call behind asyncio.to_thread on the hot path. Any knob
    left None reads its anchor_led_* runtime toggle live."""
    import cv2
    from persistence import runtime_toggles

    if frame_bgr is None or frame_bgr.size == 0:
        return None
    if h_lo is None:
        h_lo = int(runtime_toggles.get("anchor_led_h_lo"))
    if h_hi is None:
        h_hi = int(runtime_toggles.get("anchor_led_h_hi"))
    if s_min is None:
        s_min = int(runtime_toggles.get("anchor_led_s_min"))
    if v_min is None:
        v_min = int(runtime_toggles.get("anchor_led_v_min"))
    if min_area is None:
        min_area = int(runtime_toggles.get("anchor_led_min_area_px"))
    if max_area is None:
        max_area = int(runtime_toggles.get("anchor_led_max_area_px"))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))

    # Specular-core rescue: the LED's blown-out white center (V near max,
    # S collapsed) fails the green mask; fold in near-white pixels that sit
    # within 3px of green ones so the blob doesn't split into a ring.
    white = cv2.inRange(hsv, (0, 0, 250), (179, 60, 255))
    if cv2.countNonZero(white):
        near_green = cv2.dilate(green, np.ones((7, 7), np.uint8))
        green = cv2.bitwise_or(green, cv2.bitwise_and(white, near_green))

    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(green)
    hits = [i for i in range(1, n)
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    if len(hits) != 1:
        if len(hits) > 1:
            log.debug("[LED] %d candidate blobs -> abstain", len(hits))
        return None
    cx, cy = centroids[hits[0]]
    return (float(cx), float(cy))
