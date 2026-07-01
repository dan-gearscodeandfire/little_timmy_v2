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


def recognize_frame(jpeg: bytes):
    """JPEG bytes -> (list[FacePrediction], image_size, detected_face_count,
    sole_crop).

    Blocking (cv2 + YuNet + EdgeFace); call behind asyncio.to_thread. Empty
    prediction list if no recognized face. detected_face_count is the number of
    DETECTED+alignable faces in the frame (recognized or not) — the input to the
    "sole face == speaker" rule; None if the frame failed to decode. sole_crop is
    the aligned 112x112 RGB crop when EXACTLY one face was detected (the
    unambiguous speaker, for Phase B co-sampling), else None."""
    import cv2
    from presence.face_detect import aligned_crops
    from presence.face_identifier import get_shared_identifier
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return [], None, None, None
    fi = get_shared_identifier()
    crops = aligned_crops(frame)
    preds = []
    for crop, bbox in crops:
        p = fi.identify_crop(crop, bbox)
        if p is not None:
            preds.append(p)
    h, w = frame.shape[:2]
    sole_crop = crops[0][0] if len(crops) == 1 else None
    return preds, (w, h), len(crops), sole_crop


def _recognize_many(jpegs: list):
    """Recognize several frames, keep the BEST (highest-confidence) prediction
    per identity across them. Returns (tuple[FacePrediction], image_size,
    detected_face_count). The count is the MAX detected across frames — the
    conservative choice for the sole-face gate: if ANY frame shows a second
    face, treat it as ambiguous (abstain) rather than risk mis-binding. A
    genuinely-lone speaker who dodges some frames still reads as 1."""
    best = {}
    size = None
    detected = 0
    sole_crops = []
    for jpeg in jpegs:
        preds, sz, n, sole = recognize_frame(jpeg)
        if sz is not None:
            size = sz
        if n is not None:
            detected = max(detected, n)
        if sole is not None:
            sole_crops.append(sole)
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
    return tuple(best.values()), size, detected, sole_crops


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
    try:
        preds, size, detected, sole_crops = await asyncio.to_thread(
            _recognize_many, jpegs)
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
    )
