"""okDemerzel-authority face recognition (multi-frame), for the live hot path.

When identity authority is flipped to okDemerzel, this produces the
``FaceObservation`` that feeds ``IdentityFusion`` — recognizing BOTH the
household AND the enrolled makers (a superset of what the Pi's SFace knows).

Multi-frame by design: an off-center/wiggling subject intermittently dodges a
single frame (observed live 2026-06-30 — a burst caught Tomasz that lone frames
missed), so grab N frames in PARALLEL and keep the best match per identity. The
Pi still supplies the ``BehaviorSnapshot`` (mode/head-steady) that fuse_identity
gates on — only the identity source moves here. All failure paths return None so
the caller can fall back to the Pi observation; recognition never blocks a turn
beyond ``timeout_sec``.
"""

import asyncio
import logging
import time

import numpy as np

from presence.face_client_local import _safe_get_behavior
from presence.types import FaceObservation

log = logging.getLogger(__name__)


def recognize_frame(jpeg: bytes, led_xy=None, detect_led: bool = False):
    """JPEG bytes -> (list[FacePrediction], image_size, detected_face_count,
    sole_crop, anchored_crop).

    Blocking (cv2 + YuNet + EdgeFace); call behind asyncio.to_thread. Empty
    prediction list if no recognized face. detected_face_count is the number of
    DETECTED+alignable faces in the frame (recognized or not) — the input to the
    "sole face == speaker" rule; None if the frame failed to decode. sole_crop is
    the aligned 112x112 RGB crop when EXACTLY one face was detected (the
    unambiguous speaker, for Phase B co-sampling), else None. anchored_crop is
    the crop the LED-mic anchor geometry selected when ``led_xy`` is given
    (face directly above the lit LED — presence/anchor.pick_anchored_face,
    abstain on ambiguity), else None; no ==1-face requirement, the anchor is
    what disambiguates a crowd. ``detect_led`` (with led_xy=None) runs the CV
    green-LED detector on this frame (~1ms CPU) and, on an unambiguous
    LED+face pick, PUBLISHES the anchor (anchor.set_anchor source="cv") —
    the per-turn TTL refresh; LED gone -> no refresh -> the gate decays dark."""
    import cv2
    from presence.face_detect import aligned_crops
    from presence.face_identifier import get_shared_identifier
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return [], None, None, None, None
    fi = get_shared_identifier()
    crops = aligned_crops(frame)
    preds = []
    for crop, bbox in crops:
        p = fi.identify_crop(crop, bbox)
        if p is not None:
            preds.append(p)
    h, w = frame.shape[:2]
    sole_crop = crops[0][0] if len(crops) == 1 else None
    cv_led = False
    if led_xy is None and detect_led:
        from presence.led_detect import find_green_led
        led_xy = find_green_led(frame)
        cv_led = led_xy is not None
    anchored_crop = None
    if led_xy is not None and crops:
        from presence.anchor import pick_anchored_face
        idx = pick_anchored_face([bbox for _, bbox in crops], led_xy, (w, h))
        if idx is not None:
            anchored_crop = crops[idx][0]
            if cv_led:
                # CV-sourced anchors publish per unambiguous frame (a stub
                # anchor was declared by the operator — don't restamp it).
                from presence import anchor
                anchor.set_anchor(led_xy, crops[idx][1], source="cv")
    return preds, (w, h), len(crops), sole_crop, anchored_crop


def _recognize_many(jpegs: list, led_xy=None, detect_led: bool = False):
    """Recognize several frames, keep the BEST (highest-confidence) prediction
    per identity across them. Returns (tuple[FacePrediction], image_size,
    detected_face_count, sole_crops, anchored_crops). The count is the MAX
    detected across frames — the conservative choice for the sole-face gate:
    if ANY frame shows a second face, treat it as ambiguous (abstain) rather
    than risk mis-binding. A genuinely-lone speaker who dodges some frames
    still reads as 1."""
    best = {}
    size = None
    detected = 0
    sole_crops = []
    anchored_crops = []
    for jpeg in jpegs:
        preds, sz, n, sole, anchored = recognize_frame(
            jpeg, led_xy=led_xy, detect_led=detect_led)
        if sz is not None:
            size = sz
        if n is not None:
            detected = max(detected, n)
        if sole is not None:
            sole_crops.append(sole)
        if anchored is not None:
            anchored_crops.append(anchored)
        for p in preds:
            cur = best.get(p.user_id)
            if cur is None or p.confidence > cur.confidence:
                best[p.user_id] = p
    # Only surface co-sample crops when the WHOLE grab was unambiguously one
    # face (detected == 1). If any frame caught a second face, drop them all —
    # we won't risk co-sampling the wrong person. Multiple frames of a lone
    # speaker yield several crops (pose diversity for enrollment).
    # NOTE anchored_crops carry no such cross-frame rule: the anchor abstains
    # PER FRAME (pick_anchored_face returns None on ambiguity), and every crop
    # that survived was individually unambiguous under the LED.
    if detected != 1:
        sole_crops = []
    return tuple(best.values()), size, detected, sole_crops, anchored_crops


async def fetch_face_observation_okdemerzel(
    http_client, capture_url: str, behavior_url: str,
    *, frames: int = 2, timeout_sec: float = 1.5):
    """Multi-frame okDemerzel recognition -> FaceObservation, or None on failure.

    Grabs ``frames`` /capture JPEGs in parallel + the Pi behavior concurrently,
    recognizes them off the event loop, and merges the best match per identity.
    image_size is the /capture frame size (bbox space). None -> caller falls back
    to the Pi observation."""
    beh_task = asyncio.create_task(
        _safe_get_behavior(http_client, behavior_url, timeout_sec))

    async def _grab():
        try:
            r = await http_client.get(capture_url, timeout=timeout_sec)
            return r.content if r.status_code == 200 else None
        except Exception:
            return None

    try:
        jpegs = await asyncio.gather(*[_grab() for _ in range(max(1, frames))])
    except Exception:
        beh_task.cancel()
        return None
    jpegs = [j for j in jpegs if isinstance(j, (bytes, bytearray)) and j]
    behavior = await beh_task
    if not jpegs:
        return None
    # LED-mic anchor (2026-07-06): resolve the LED position for this grab.
    # A fresh STUB anchor's operator-declared led_xy is reused verbatim; in
    # every other enabled case the CV detector runs per frame (the mic moves —
    # a CV-sourced position must never be reused stale) and republishes the
    # anchor on an unambiguous pick. Feature off -> led_xy None, no detection,
    # byte-identical to before.
    led_xy = None
    detect_led = False
    from persistence import runtime_toggles
    if runtime_toggles.get("anchor_enabled"):
        from presence import anchor
        st = anchor.get_anchor()
        if (st is not None and st.source == "stub" and st.led_xy is not None
                and anchor.anchor_active()):
            led_xy = st.led_xy
        else:
            detect_led = True
    try:
        preds, size, detected, sole_crops, anchored_crops = \
            await asyncio.to_thread(_recognize_many, jpegs, led_xy, detect_led)
    except Exception:
        log.info("[FACE-AUTH] recognition failed", exc_info=True)
        return None
    return FaceObservation(
        captured_at=time.time(),
        predictions=preds,
        behavior=behavior,
        image_size=size,
        detected_face_count=detected,
        sole_face_crops=tuple(sole_crops),
        anchored_face_crops=tuple(anchored_crops),
    )
