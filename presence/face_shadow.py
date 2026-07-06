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

import config

log = logging.getLogger(__name__)

def _recognize(jpeg: bytes) -> list:
    """Blocking: JPEG bytes -> [FacePrediction] via okDemerzel recognition.
    Runs behind asyncio.to_thread. Delegates to the shared recognizer."""
    from presence.face_recognize import recognize_frame
    # First element only — recognize_frame's tuple has grown (sole_crop
    # Phase B, anchored_crop 2026-07-06) and the 3-way unpack here had been
    # silently broken since the 4th element landed (shadow is default-off).
    preds = recognize_frame(jpeg)[0]
    return preds


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
