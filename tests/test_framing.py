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


# --- xywh boundary conversion (7-16 live bug: raw Pi bboxes fed xyxy math) --

def test_bbox_xyxy_converts_pi_wire_format():
    from presence.framing import bbox_xyxy
    # Matt's live track from the 7-16 test: x=112 y=150 w=61 h=71.
    assert bbox_xyxy((112, 150, 61, 71)) == (112.0, 150.0, 173.0, 221.0)


def test_centroid_of_converted_pi_faces_is_sane():
    from presence.framing import bbox_xyxy
    # Raw xywh through the converter: face center must land mid-bbox, and a
    # normal booth face must QUALIFY (raw xywh made y1-y0 negative -> None).
    b = bbox_xyxy((300, 100, 80, 120))            # height 120/480 = 0.25
    c = faces_centroid([b], SIZE, min_height_frac=0.10)
    assert c is not None
    assert abs(c[0] - (340 / 640)) < 1e-6
    assert abs(c[1] - (160 / 480)) < 1e-6


# --- LedHolderProxy: virtual LED rides the holder's face (Dan 2026-07-16) --

def _proxy():
    from presence.framing import LedHolderProxy
    p = LedHolderProxy(ttl_s=45.0)
    # LED seen at (320, 300); holder face centered (316, 220) directly above
    # -> offset (+4, +80): a mouth-hold puts the LED just below the face.
    p.remember(7, (4.0, 80.0), now=100.0)
    return p


def test_holder_proxy_rides_the_moving_face():
    p = _proxy()
    # Holder drifted left/down; virtual LED keeps the remembered offset.
    v = p.resolve({7: (250.0, 240.0)}, now=110.0, image_size=SIZE)
    assert v == (254.0, 320.0)


def test_holder_proxy_none_when_track_gone():
    p = _proxy()
    assert p.resolve({9: (250.0, 240.0)}, now=110.0, image_size=SIZE) is None
    assert p.resolve({}, now=110.0, image_size=SIZE) is None


def test_holder_proxy_expires_after_ttl():
    p = _proxy()
    assert p.resolve({7: (250.0, 240.0)}, now=146.0, image_size=SIZE) is None
    # Fresh sighting re-arms it.
    p.remember(7, (4.0, 80.0), now=150.0)
    assert p.resolve({7: (250.0, 240.0)}, now=160.0, image_size=SIZE) \
        is not None


def test_holder_proxy_clamps_virtual_into_frame():
    p = _proxy()
    # Face near the bottom edge: center + offset would leave the frame; the
    # virtual point clamps so the clip interval never inverts.
    v = p.resolve({7: (630.0, 470.0)}, now=110.0, image_size=SIZE)
    assert v == (634.0, 479.0)                    # y clamped to h-1


def test_holder_proxy_ignores_none_track():
    from presence.framing import LedHolderProxy
    p = LedHolderProxy()
    p.remember(None, (4.0, 80.0), now=100.0)     # unpaired sighting: no-op
    assert p.resolve({None: (1.0, 1.0)}, now=101.0, image_size=SIZE) is None
