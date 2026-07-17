"""EdgeFace identity backfeed okDemerzel -> streamerpi (2026-07-16).

okDemerzel is the identity authority (EdgeFace superset of the household +
enrolled makers); the Pi keeps YuNet detection + servo tracking but its SFace
recognizer is retired. Every successful okDemerzel recognition pushes its
name+bbox predictions to ``POST /faces/backfeed`` on the Pi, whose identity
stabilizer latches the names onto the spatially-nearest live tracks. That is
what now names the Pi's ``/faces`` entries, ``behavior.params.face_identity``
(the booth reticle tag), and the engage speaker-lock in multi-face scenes —
closing the "EdgeFace backfeed not implemented" gap (camera.py C12 warning).

Fire-and-forget: a failed push only delays naming until the next turn's
recognition; it must never block or fail the hot path.
"""

import logging
import time

import config

log = logging.getLogger(__name__)

# One push per turn is the natural cadence; the floor just guards against a
# burst of call sites (per-turn fetch + future anchor-poll pushes) hammering
# the Pi's aiohttp loop.
MIN_PUSH_INTERVAL_S = 1.0

_last_push_ts = 0.0
_last_fail_log_ts = 0.0


def _payload(predictions, image_size, captured_at: float):
    """FacePrediction tuple -> backfeed JSON, or None if nothing named.

    FacePrediction.bbox is corner-format (x_min, y_min, x_max, y_max);
    the Pi speaks [x, y, w, h] everywhere, so convert here."""
    identities = []
    for p in predictions or ():
        if not p.user_id or not p.bbox:
            continue
        x0, y0, x1, y1 = p.bbox
        identities.append({
            "name": str(p.user_id).lower(),
            "bbox": [float(x0), float(y0),
                     float(x1) - float(x0), float(y1) - float(y0)],
            "confidence": float(p.confidence),
        })
    if not identities:
        return None
    return {"identities": identities,
            "image_size": list(image_size) if image_size else None,
            "captured_at": captured_at}


async def push_identities(http_client, predictions, image_size,
                          captured_at: float | None = None,
                          timeout_sec: float = 1.0) -> bool:
    """POST recognized identities to the Pi. Returns True on a 200.

    All failure paths log (rate-limited) and return False — never raises."""
    global _last_push_ts, _last_fail_log_ts
    now = time.time()
    if now - _last_push_ts < MIN_PUSH_INTERVAL_S:
        return False
    body = _payload(predictions, image_size, captured_at or now)
    if body is None:
        return False
    _last_push_ts = now
    try:
        r = await http_client.post(config.STREAMERPI_FACE_BACKFEED_URL,
                                   json=body, timeout=timeout_sec)
        if r.status_code == 200:
            log.debug("[BACKFEED] pushed %s",
                      [i["name"] for i in body["identities"]])
            return True
        raise RuntimeError(f"HTTP {r.status_code}")
    except Exception as e:
        if now - _last_fail_log_ts > 60.0:
            _last_fail_log_ts = now
            log.info("[BACKFEED] push failed (%s) — Pi tracks stay unnamed "
                     "until the next successful push", e)
        return False
