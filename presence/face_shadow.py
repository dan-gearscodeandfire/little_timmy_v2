"""Face-recognition shadow mode: okDemerzel EdgeFace vs the Pi's SFace.

Runs the okDemerzel-side recognition pipeline (self-served from a /capture grab
-> YuNet detect -> align -> EdgeFace -> FaceIdentifier) alongside the live turn
and logs how its identity call compares to the Pi's `/faces` result, WITHOUT
touching fusion. This is the drop-in proof for the plan's Phase A: watch real
agreement on live traffic before flipping identity authority to okDemerzel.

Fire-and-forget (never awaited by the turn) and the CPU work (decode + detect +
align + embed) runs in one asyncio.to_thread, so it adds zero reply latency and
can never break a turn (all failures swallowed). Gated by the
`face_shadow_enabled` runtime toggle (default OFF), read live per-turn.
"""

import logging

import httpx
import numpy as np

import config

log = logging.getLogger(__name__)

_identifier = None


def _get_identifier():
    """Lazily load the FaceIdentifier once (shared across turns)."""
    global _identifier
    if _identifier is None:
        from presence.face_identifier import FaceIdentifier
        fi = FaceIdentifier()
        fi.load()
        _identifier = fi
    return _identifier


def _recognize(jpeg: bytes) -> list:
    """Blocking: JPEG bytes -> [FacePrediction] via okDemerzel recognition.
    Runs behind asyncio.to_thread (cv2 decode + YuNet + EdgeFace are CPU)."""
    import cv2
    from presence.face_detect import aligned_crops
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return []
    fi = _get_identifier()
    obs = fi.observe(aligned_crops(frame))
    return list(obs.predictions)


async def shadow_compare(pi_face_obs) -> None:
    """Grab a frame, recognize on okDemerzel, log agreement vs the Pi obs.
    Caller gates on the toggle and launches this via asyncio.create_task."""
    import asyncio
    try:
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            r = await client.get(config.STREAMERPI_CAPTURE_URL)
        if r.status_code != 200:
            return
        preds = await asyncio.to_thread(_recognize, r.content)
    except Exception:
        log.debug("[FACE-SHADOW] recognition failed", exc_info=True)
        return

    ok_names = sorted(p.user_id for p in preds)
    pi_names = sorted(
        p.user_id for p in (pi_face_obs.predictions if pi_face_obs else ()))
    top = max(preds, key=lambda p: p.confidence, default=None)
    detail = (f" top={top.user_id}({top.confidence:.2f},{top.band})"
              if top is not None else "")
    verdict = "AGREE" if ok_names == pi_names else "DIFFER"
    log.info("[FACE-SHADOW] %s okDemerzel=%s pi=%s%s",
             verdict, ok_names or "-", pi_names or "-", detail)
