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
    sole_crop, anchored_pick).

    Blocking (cv2 + YuNet + EdgeFace); call behind asyncio.to_thread. Empty
    prediction list if no recognized face. detected_face_count is the number of
    DETECTED+alignable faces in the frame (recognized or not) — the input to the
    "sole face == speaker" rule; None if the frame failed to decode. sole_crop is
    the aligned 112x112 RGB crop when EXACTLY one face was detected (the
    unambiguous speaker, for Phase B co-sampling), else None. anchored_pick is
    the LED-mic anchor selection when ``led_xy`` is given or ``detect_led``
    found one: {"crop", "bbox", "led_xy", "cv_led", "name"} for the face
    directly above the lit LED (presence/anchor.pick_anchored_face, abstain on
    ambiguity), else None; no ==1-face requirement, the anchor is what
    disambiguates a crowd. "name" is the crop's RECOGNIZED enrolled identity
    (None = unrecognized) — the voice<->anchor binding input (F1/F7 review
    7-07). This function no longer PUBLISHES the anchor; _resolve_anchored
    does, once per grab, after the cross-frame consistency check (F6)."""
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
    # Faceless-frame guard (review 7-07): an anchor pick needs a face, so the
    # LED mask on a frame with zero alignable faces is pure waste — skip it.
    if led_xy is None and detect_led and crops:
        from presence.led_detect import find_green_led
        led_xy = find_green_led(frame)
        cv_led = led_xy is not None
    anchored_pick = None
    if led_xy is not None and crops:
        from presence.anchor import pick_anchored_face
        idx = pick_anchored_face([bbox for _, bbox in crops], led_xy, (w, h))
        if idx is not None:
            a_bbox = crops[idx][1]
            a_name = next((p.user_id for p in preds if p.bbox == a_bbox), None)
            anchored_pick = {"crop": crops[idx][0], "bbox": a_bbox,
                             "led_xy": led_xy, "cv_led": cv_led, "name": a_name}
    return preds, (w, h), len(crops), sole_crop, anchored_pick


def _resolve_anchored(picks: list, size, publish_cv: bool = True):
    """Cross-frame consistency + single publish for a grab's anchor picks.

    F6 fix (review 7-07): pick_anchored_face abstains PER FRAME, but a mic
    handoff mid-grab could pass each frame individually and still mix two
    people's crops under one name. A grab's picks must agree with each other:
    bbox x-centers within anchor_x_tol_frac of frame width AND no two
    different recognized names. Disagreement -> drop ALL picks + no publish
    (abstain, the standard ambiguity contract).

    Publishes the anchor ONCE per consistent grab (was per frame — TTL-
    equivalent, minus the restamp/log churn), carrying the track's recognized
    name (a face IDed in ANY frame of a spatially-consistent track names the
    track). publish_cv=False suppresses the publish even for CV-found LEDs —
    the fresh-stub protection (F4): an operator-declared anchor must never be
    restamped source=cv (which would clobber its ttl_s).

    Returns (anchored_crops, anchored_name)."""
    if not picks:
        return [], None
    names = {p["name"] for p in picks if p["name"] is not None}
    if len(names) > 1:
        log.info("[ANCHOR] cross-frame identity disagreement %s -> abstain",
                 sorted(names))
        return [], None
    if size is not None and len(picks) > 1:
        from persistence import runtime_toggles
        x_tol = float(runtime_toggles.get("anchor_x_tol_frac")) * float(size[0])
        centers = [(p["bbox"][0] + p["bbox"][2]) / 2.0 for p in picks]
        if max(centers) - min(centers) > x_tol:
            log.info("[ANCHOR] cross-frame position disagreement "
                     "(x-spread %.0fpx > tol %.0fpx) -> abstain",
                     max(centers) - min(centers), x_tol)
            return [], None
    name = next(iter(names), None)
    last = picks[-1]
    if publish_cv and any(p["cv_led"] for p in picks):
        from presence import anchor
        anchor.set_anchor(last["led_xy"], last["bbox"], source="cv",
                          anchored_name=name)
    return [p["crop"] for p in picks], name


def _recognize_many(jpegs: list, led_xy=None, detect_led: bool = False,
                    publish_cv: bool = True):
    """Recognize several frames, keep the BEST (highest-confidence) prediction
    per identity across them. Returns (tuple[FacePrediction], image_size,
    detected_face_count, sole_crops, anchored_crops, anchored_name). The count
    is the MAX detected across frames — the conservative choice for the
    sole-face gate: if ANY frame shows a second face, treat it as ambiguous
    (abstain) rather than risk mis-binding. A genuinely-lone speaker who
    dodges some frames still reads as 1."""
    best = {}
    size = None
    detected = 0
    sole_crops = []
    anchored_picks = []
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
            anchored_picks.append(anchored)
        for p in preds:
            cur = best.get(p.user_id)
            if cur is None or p.confidence > cur.confidence:
                best[p.user_id] = p
    # Only surface co-sample crops when the WHOLE grab was unambiguously one
    # face (detected == 1). If any frame caught a second face, drop them all —
    # we won't risk co-sampling the wrong person. Multiple frames of a lone
    # speaker yield several crops (pose diversity for enrollment).
    if detected != 1:
        sole_crops = []
    anchored_crops, anchored_name = _resolve_anchored(
        anchored_picks, size, publish_cv=publish_cv)
    return (tuple(best.values()), size, detected, sole_crops,
            anchored_crops, anchored_name)


def poll_anchor_frame(jpeg: bytes) -> bool:
    """Single-frame LED detect + face pick + identify for the periodic anchor
    poll (Ruling A, Dan 2026-07-07): keep the anchor fresh BETWEEN turns so
    the identity-dialog gate is already open when a visitor's FIRST utterance
    arrives (the per-turn-only refresh opened it one turn late — the opening
    "hi, I'm X" leaked to the LLM as ordinary speech).

    Blocking (cv2 + ONNX CPU); call behind asyncio.to_thread. Returns True
    when an unambiguous LED+face pick published/refreshed the anchor. The
    caller (main.anchor_poll_monitor) owns the enabled/fresh-stub skip."""
    _, size, _, _, pick = recognize_frame(jpeg, detect_led=True)
    if pick is None or not pick["cv_led"]:
        return False
    _resolve_anchored([pick], size, publish_cv=True)
    return True


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
    # A fresh STUB anchor is the operator's declaration and is never restamped
    # by CV (F4 fix 7-07 — this now protects led_xy-LESS stubs too, which used
    # to fall through to detect+publish and get their ttl_s clobbered): its
    # led_xy is reused verbatim when declared, and when it declared none the
    # CV detector may still run for crop-PICKING but publish_cv stays False.
    # In every other enabled case the CV detector runs per frame (the mic
    # moves — a CV-sourced position must never be reused stale) and the grab
    # republishes the anchor once on a consistent pick. Feature off -> led_xy
    # None, no detection, byte-identical to before.
    led_xy = None
    detect_led = False
    publish_cv = True
    from persistence import runtime_toggles
    if runtime_toggles.get("anchor_enabled"):
        from presence import anchor
        st = anchor.get_anchor()
        if (st is not None and st.source == "stub" and anchor.anchor_active()):
            led_xy = st.led_xy            # may be None (position-less stub)
            detect_led = led_xy is None   # CV for picking only...
            publish_cv = False            # ...never a restamp over the stub
        else:
            detect_led = True
    try:
        preds, size, detected, sole_crops, anchored_crops, anchored_name = \
            await asyncio.to_thread(_recognize_many, jpegs, led_xy, detect_led,
                                    publish_cv)
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
        anchored_face_name=anchored_name,
    )
