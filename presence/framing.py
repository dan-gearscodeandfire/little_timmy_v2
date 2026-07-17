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


def bbox_xyxy(bbox):
    """Convert a Pi /faces bbox (x, y, w, h px) to (x_min, y_min, x_max, y_max).

    The Pi wire format is xywh; every brain-side consumer converts at the
    boundary (cf. main.py FacePrediction ingestion). The framing controller
    originally fed raw xywh into ``faces_centroid``/``pick_anchored_face``
    (both xyxy) — garbage centroids, the 2026-07-16 "mic keeps leaving frame"
    live failure. Convert exactly once, here."""
    x, y, w, h = (float(v) for v in bbox[:4])
    return (x, y, x + w, y + h)


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


class LedHolderProxy:
    """Virtual LED while the mic's HOLDER is still on camera (Dan 2026-07-16).

    The CV LED detector drops out routinely (head-on the LEDs read v~50 —
    hardware; occlusion by the holder's hand) while the person holding the
    mic is plainly still there. Old behavior: lose the LED -> either recenter
    on faces with NO led clamp or blind-sweep (179s of scanning at the 7-16
    live test while the holder's face was tracked the whole time).

    New contract: on every FRESH LED sighting, remember which face track sat
    directly above it (``pick_anchored_face`` pairing — the anchor's own
    rule) plus the face-center->LED pixel offset. While the LED is stale but
    that track is still present, ``resolve()`` synthesizes a virtual LED at
    the track's current center + the remembered offset, and the caller keeps
    applying the keep-in-frame clamp to it instead of sweeping. Holder gone
    (track vanished / TTL expired) -> None -> the scan fallback resumes.

    Pure logic (hermetic-testable): callers pass ``now`` explicitly and a
    ``{track_id: (cx, cy)}`` map of current face centers in px."""

    def __init__(self, ttl_s: float = 45.0):
        self.ttl_s = float(ttl_s)
        self.track_id = None
        self._offset = None
        self._stamped = 0.0

    def remember(self, track_id, offset_xy, now: float) -> None:
        if track_id is None or offset_xy is None:
            return
        self.track_id = track_id
        self._offset = (float(offset_xy[0]), float(offset_xy[1]))
        self._stamped = float(now)

    def reset(self) -> None:
        self.track_id = None
        self._offset = None
        self._stamped = 0.0

    def resolve(self, centers, now: float, image_size):
        """Virtual LED (x, y) px, or None if the holder can't vouch for it.

        ``centers`` maps track_id -> face center (cx, cy) px. The virtual
        point is clamped into the frame so a low mouth-hold offset can't
        push it past the bottom edge and invert the clip interval."""
        if self.track_id is None or self._offset is None:
            return None
        if (now - self._stamped) > self.ttl_s:
            return None
        c = centers.get(self.track_id)
        if c is None or not image_size:
            return None
        w, h = float(image_size[0]), float(image_size[1])
        if w <= 0 or h <= 0:
            return None
        return (min(max(c[0] + self._offset[0], 0.0), w - 1.0),
                min(max(c[1] + self._offset[1], 0.0), h - 1.0))


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
