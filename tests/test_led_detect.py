"""Hermetic tests for the green-LED blob detector (LED-mic anchor, 2026-07-06).

Synthetic BGR frames drawn with cv2 (already a test dependency via the face
pipeline). The contract: exactly ONE cluster of bright-green, LED-sized
blobs -> its area-weighted centroid (the mic carries several LEDs, two
visible at any angle — nearby blobs are one beacon); zero blobs, two FAR
blobs (two green lights across the room), dim green (unlit), or an
oversized green region (a shirt) -> None. Knobs passed explicitly so the
tests never read the live toggles file.

Run:
    .venv/bin/pytest tests/test_led_detect.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.led_detect import find_green_led

KNOBS = dict(h_lo=40, h_hi=85, s_min=80, v_min=200, min_area=4, max_area=400,
             cluster_px=60)


def _frame(w=640, h=480):
    """Dim grey room."""
    return np.full((h, w, 3), 40, dtype=np.uint8)


def _draw_led(frame, x, y, r=5, bgr=(40, 255, 40)):
    cv2.circle(frame, (x, y), r, bgr, -1)
    return frame


def test_single_led_centroid():
    f = _draw_led(_frame(), 320, 300)
    hit = find_green_led(f, **KNOBS)
    assert hit is not None
    assert abs(hit[0] - 320) <= 2 and abs(hit[1] - 300) <= 2


def test_two_far_leds_abstain():
    # Two green lights ACROSS THE ROOM (240px apart, way over cluster_px):
    # genuine ambiguity, abstain.
    f = _draw_led(_draw_led(_frame(), 200, 300), 440, 300)
    assert find_green_led(f, **KNOBS) is None


def test_two_near_leds_cluster_to_one_beacon():
    # The real mic shows TWO of its LEDs at any angle (~20px apart in frame,
    # 2026-07-13 hardware): one cluster, area-weighted centroid between them.
    f = _draw_led(_draw_led(_frame(), 310, 300), 330, 300)
    hit = find_green_led(f, **KNOBS)
    assert hit is not None
    assert abs(hit[0] - 320) <= 3 and abs(hit[1] - 300) <= 3


def test_near_pair_plus_far_led_abstains():
    # The mic's LED pair AND a stray green light elsewhere -> two clusters,
    # still ambiguous, abstain.
    f = _draw_led(_draw_led(_draw_led(_frame(), 310, 300), 330, 300), 560, 120)
    assert find_green_led(f, **KNOBS) is None


def test_dim_green_rejected():
    # Unlit green plastic: hue right, brightness far below the V floor.
    f = _draw_led(_frame(), 320, 300, bgr=(20, 120, 20))
    assert find_green_led(f, **KNOBS) is None


def test_oversized_green_region_rejected():
    # A green shirt: right color, way over max_area.
    f = _frame()
    cv2.rectangle(f, (200, 200), (400, 420), (40, 255, 40), -1)
    assert find_green_led(f, **KNOBS) is None


def test_speckle_rejected():
    # Single bright-green pixels under min_area (post-morph-open they vanish).
    f = _frame()
    for x, y in [(100, 100), (300, 240), (500, 400)]:
        f[y, x] = (40, 255, 40)
    assert find_green_led(f, **KNOBS) is None


def test_specular_core_still_one_blob():
    """A lit LED blooms white at the center — the rescue must keep it ONE
    blob (a green ring around a white core must not read as 2+ or lose the
    centroid)."""
    f = _draw_led(_frame(), 320, 300, r=6)
    cv2.circle(f, (320, 300), 2, (255, 255, 255), -1)   # blown-out core
    hit = find_green_led(f, **KNOBS)
    assert hit is not None
    assert abs(hit[0] - 320) <= 2 and abs(hit[1] - 300) <= 2


def test_empty_frame():
    assert find_green_led(_frame(), **KNOBS) is None


def test_none_frame():
    assert find_green_led(None, **KNOBS) is None
