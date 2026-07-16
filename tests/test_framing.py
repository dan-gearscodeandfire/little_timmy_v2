"""Hermetic tests for the booth framing core (presence/framing.py — Dan
2026-07-15, Open Sauce spec 2: centroid framing that NEVER loses the
green-LED mic, plus the LED-reacquire sweep)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presence.framing import (  # noqa: E402
    LedScan,
    clip_centroid_for_led,
    faces_centroid,
)

SIZE = (640, 480)


def test_centroid_averages_qualifying_faces():
    # Two faces at x-centers 160 and 480 -> centroid x = 0.5.
    bboxes = [(100, 100, 220, 260), (420, 120, 540, 280)]
    c = faces_centroid(bboxes, SIZE, min_height_frac=0.10)
    assert c is not None
    assert abs(c[0] - 0.5) < 1e-6
    assert 0.3 < c[1] < 0.5


def test_centroid_ignores_small_background_faces():
    big = (100, 100, 220, 260)            # height 160/480 = 0.33
    tiny = (600, 10, 620, 40)             # height 30/480 = 0.06 — passer-by
    c = faces_centroid([big, tiny], SIZE, min_height_frac=0.10)
    cx_big = ((100 + 220) / 2) / 640
    assert abs(c[0] - cx_big) < 1e-6      # tiny face did not drag the centroid


def test_centroid_none_when_no_faces_qualify():
    assert faces_centroid([], SIZE) is None
    assert faces_centroid([(0, 0, 10, 20)], SIZE, min_height_frac=0.10) is None


def test_led_clamp_keeps_led_in_frame():
    # LED near the left edge (x=32 -> 0.05 norm); centering far right would
    # push it out. Clamp must cap the target so LED' >= margin.
    led = (32, 240)
    target = clip_centroid_for_led((0.9, 0.5), led, SIZE, margin=0.15)
    # C_max = L + 0.5 - margin = 0.05 + 0.35 = 0.40
    assert target[0] <= 0.40 + 1e-9
    # y unconstrained here (LED at 0.5): C in [0.15, 0.85] -> 0.5 passes.
    assert abs(target[1] - 0.5) < 1e-9
    # And a target already safe is untouched.
    same = clip_centroid_for_led((0.3, 0.5), led, SIZE, margin=0.15)
    assert same == (0.3, 0.5)


def test_led_clamp_noop_without_led():
    assert clip_centroid_for_led((0.9, 0.1), None, SIZE) == (0.9, 0.1)


def test_led_scan_expands_and_alternates():
    s = LedScan(step=6.0, max_offset=30.0)
    offs = [s.next_offset() for _ in range(6)]
    assert offs[0] == 6.0 and offs[1] == -12.0 and offs[2] == 18.0
    assert all(abs(o) <= 30.0 for o in offs)     # capped
    s.reset()
    assert s.next_offset() == 6.0                # reset restarts the pattern
