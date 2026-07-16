"""Booth framing controller core (Dan 2026-07-15, Open Sauce spec 2).

When multiple visitors stand at the booth, Timmy centers his view on the
AVERAGE CENTROID of the qualifying faces — but the green-LED mic must ALWAYS
stay in frame (the anchor is what binds face<->voice<->name; losing it blinds
enrollment). If the LED hasn't been seen for a while, he actively sweeps to
reacquire it.

This module is the PURE-LOGIC core (hermetic-testable, no I/O): centroid
computation, the keep-LED-in-frame clamp, and the expanding scan pattern.
The live loop (main.framing_monitor) owns fetching faces/servo state, the Pi
tracker stand-down lease, and the /servo/move POST. Ownership ruling (Dan
2026-07-15): the BRAIN owns head movement at the booth; the Pi's own
single-face tracker is held off via behavior 'hold' leases while this
controller drives.
"""

from __future__ import annotations


def faces_centroid(bboxes, image_size, min_height_frac: float = 0.10):
    """Normalized (bx, by) average center of qualifying faces, or None.

    ``bboxes`` are (x_min, y_min, x_max, y_max) px; a face qualifies when its
    bbox height is at least ``min_height_frac`` of the frame height (ignore
    background passers-by — same signal family as the proximity gate)."""
    if not bboxes or not image_size:
        return None
    w, h = float(image_size[0]), float(image_size[1])
    if w <= 0 or h <= 0:
        return None
    centers = []
    for b in bboxes:
        try:
            x0, y0, x1, y1 = b[:4]
        except Exception:
            continue
        if (y1 - y0) / h >= min_height_frac:
            centers.append((((x0 + x1) / 2.0) / w, ((y0 + y1) / 2.0) / h))
    if not centers:
        return None
    return (sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers))


def clip_centroid_for_led(centroid, led_xy, image_size,
                          margin: float = 0.15):
    """Clamp a recenter target so the LED stays inside the frame.

    Recentering the camera on normalized point C moves any point P to
    P' = P - (C - 0.5) per axis. Requiring the LED's new position L' to stay
    within [margin, 1-margin] bounds C to
    [L + 0.5 - (1 - margin), L + 0.5 - margin]. The clamp wins over perfect
    centroid centering (LED visibility is the harder requirement)."""
    if led_xy is None or not image_size:
        return centroid
    w, h = float(image_size[0]), float(image_size[1])
    if w <= 0 or h <= 0:
        return centroid
    led_norm = (led_xy[0] / w, led_xy[1] / h)
    out = []
    for i in (0, 1):
        lo = led_norm[i] + 0.5 - (1.0 - margin)
        hi = led_norm[i] + 0.5 - margin
        # A margin > 0.5 would invert the interval; guard degenerate input.
        if lo > hi:
            lo = hi = led_norm[i]
        out.append(min(max(centroid[i], lo), hi))
    return tuple(out)


class LedScan:
    """Expanding alternating pan sweep to reacquire a lost LED.

    Offsets from the anchor pan: +s, -2s, +3s, -4s ... capped at
    ``max_offset``; ``reset()`` on every fresh LED sighting. The tilt is left
    alone (the mic is roughly at speaking height already)."""

    def __init__(self, step: float = 6.0, max_offset: float = 30.0):
        self.step = float(step)
        self.max_offset = float(max_offset)
        self._n = 0

    def reset(self) -> None:
        self._n = 0

    def next_offset(self) -> float:
        self._n += 1
        mag = min(self.step * self._n, self.max_offset)
        return mag if self._n % 2 else -mag
