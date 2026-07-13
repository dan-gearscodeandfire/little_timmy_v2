"""Green-LED blob detection for the LED-mic anchor (2026-07-06).

Finds the lit green LED on the handed booth mic in a /capture frame — pure
CPU OpenCV (~1ms), deliberately NOT a VLM query (the two-35B GPU-contention
finding: cross-process halving on the shared card).

Pipeline: HSV mask (green hue band + saturation/brightness floors) OR'd with
a specular-core rescue — a lit LED blooms WHITE at the center (saturation
collapses, so the green mask alone loses the hottest pixels) with a green
halo; near-white pixels whose small dilation overlaps the green mask are
folded back in. Morphological open kills speckle, then connected components
filtered by area, then single-link CLUSTERING by centroid distance
(anchor_led_cluster_px) — the real mic carries several LEDs with two visible
at any angle (2026-07-13 hardware), so nearby blobs are ONE beacon, not
ambiguity. EXACTLY one surviving cluster -> its area-weighted centroid;
zero or 2+ clusters -> None (abstain — two green lights across the room is
still ambiguity, the same contract as the face geometry).

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
                   max_area: Optional[int] = None,
                   cluster_px: Optional[int] = None) -> Optional[tuple]:
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
    if cluster_px is None:
        cluster_px = int(runtime_toggles.get("anchor_led_cluster_px"))

    green = green_mask(frame_bgr, h_lo, h_hi, s_min, v_min)

    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(green)
    hits = [i for i in range(1, n)
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    if not hits:
        return None

    # Single-link clustering: blobs whose centroids sit within cluster_px of
    # any blob already in the cluster merge (the mic's own LEDs, or one LED
    # fragmenting into core + halo). Union-find over the handful of hits.
    parent = list(range(len(hits)))

    def _root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(hits)):
        for b in range(a + 1, len(hits)):
            dx = centroids[hits[a]][0] - centroids[hits[b]][0]
            dy = centroids[hits[a]][1] - centroids[hits[b]][1]
            if dx * dx + dy * dy <= cluster_px * cluster_px:
                parent[_root(a)] = _root(b)

    clusters: dict = {}
    for a, i in enumerate(hits):
        clusters.setdefault(_root(a), []).append(i)
    if len(clusters) != 1:
        log.debug("[LED] %d candidate clusters -> abstain", len(clusters))
        return None

    members = next(iter(clusters.values()))
    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in members], dtype=float)
    cxs = np.array([centroids[i][0] for i in members])
    cys = np.array([centroids[i][1] for i in members])
    w = areas / areas.sum()
    return (float((cxs * w).sum()), float((cys * w).sum()))


def green_mask(frame_bgr: np.ndarray, h_lo: int, h_hi: int,
               s_min: int, v_min: int) -> np.ndarray:
    """The detector's binary green mask for ``frame_bgr`` — HSV band +
    specular-core rescue + morphological open, BEFORE the blob/area filter.

    Factored out (review 7-07) so ops/led_calibrate.py dumps candidates
    through the IDENTICAL pipeline: it used to copy-paste this block, and the
    hard-coded specular-rescue constants below could silently desync the
    tuning tool from the detector."""
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))

    # Specular-core rescue: the LED's blown-out white center (V near max,
    # S collapsed) fails the green mask; fold in near-white pixels that sit
    # within 3px of green ones so the blob doesn't split into a ring.
    white = cv2.inRange(hsv, (0, 0, 250), (179, 60, 255))
    if cv2.countNonZero(white):
        near_green = cv2.dilate(green, np.ones((7, 7), np.uint8))
        green = cv2.bitwise_or(green, cv2.bitwise_and(white, near_green))

    return cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
