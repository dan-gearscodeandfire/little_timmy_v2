"""Async client for streamerpi capture + demerzel-vision face recognize + behavior status."""

import asyncio
import hashlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from .types import BehaviorSnapshot, FaceObservation, FacePrediction


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceClientConfig:
    capture_url: str
    face_url: str  # base; /v1/vision/face/recognize is appended
    behavior_url: str
    capture_verify_tls: bool = False  # streamerpi uses self-signed
    timeout_sec: float = 1.5
    min_recognize_confidence: float = 0.45  # demerzel-vision form param


def _embedding_hash(bbox: tuple, ts: float) -> str:
    """Stable short hash for a not-yet-recognized face.

    Without access to the actual SFace embedding (the :8895 API doesn't
    return it), we approximate continuity by bbox center bucketed to ~5%
    of frame width. Same person re-detected nearby gets the same hash.
    """
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    bucket = (cx // 32, cy // 32)
    h = hashlib.sha1(f"{bucket}".encode()).hexdigest()
    return h[:8]


def _parse_behavior(payload: dict) -> Optional[BehaviorSnapshot]:
    if not isinstance(payload, dict):
        return None
    try:
        return BehaviorSnapshot(
            mode=str(payload.get("mode", "")),
            face_visible=bool(payload.get("face_visible", False)),
            elapsed_ms=int(payload.get("elapsed_ms", 0)),
            last_face_pan=payload.get("last_face_pan"),
            last_face_tilt=payload.get("last_face_tilt"),
        )
    except (TypeError, ValueError) as e:
        log.warning("Bad behavior payload: %s", e)
        return None


def _parse_predictions(payload: dict, now_ts: float) -> tuple:
    if not isinstance(payload, dict):
        return tuple()
    if not payload.get("success"):
        return tuple()
    preds = payload.get("predictions") or []
    out = []
    for p in preds:
        try:
            bbox = (
                int(p.get("x_min", 0)),
                int(p.get("y_min", 0)),
                int(p.get("x_max", 0)),
                int(p.get("y_max", 0)),
            )
            uid = str(p.get("userid", "")) or "unknown"
            conf = float(p.get("confidence", 0.0))
            out.append(FacePrediction(
                user_id=uid,
                confidence=conf,
                bbox=bbox,
                embedding_hash=_embedding_hash(bbox, now_ts) if uid.lower().startswith("unknown") else None,
            ))
        except (TypeError, ValueError) as e:
            log.warning("Bad prediction entry %r: %s", p, e)
    return tuple(out)


def _image_size_from_bytes(data: bytes) -> Optional[tuple]:
    """Best-effort JPEG size parse without importing Pillow.

    Returns (width, height) or None.
    """
    try:
        idx = 0
        if data[:3] != b"\xff\xd8\xff":
            return None
        idx = 2
        while idx < len(data):
            if data[idx] != 0xFF:
                return None
            marker = data[idx + 1]
            idx += 2
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                # SOFn frame start
                height = (data[idx + 3] << 8) | data[idx + 4]
                width = (data[idx + 5] << 8) | data[idx + 6]
                return (width, height)
            seg_len = (data[idx] << 8) | data[idx + 1]
            idx += seg_len
    except (IndexError, ValueError):
        return None
    return None


async def fetch_face_observation(
    client: httpx.AsyncClient,
    config: FaceClientConfig,
) -> Optional[FaceObservation]:
    """One round-trip: behavior + capture in parallel, then recognize.

    Returns None if capture or recognize fails. Behavior may be None on
    its own without invalidating the observation.
    """
    now_ts = time.time()
    recognize_url = config.face_url.rstrip("/") + "/v1/vision/face/recognize"

    cap_task = client.get(
        config.capture_url,
        timeout=config.timeout_sec,
    )
    beh_task = client.get(
        config.behavior_url,
        timeout=config.timeout_sec,
    )

    try:
        cap_resp, beh_resp = await asyncio.gather(
            cap_task, beh_task, return_exceptions=True,
        )
    except Exception as e:
        log.warning("face_client gather failed: %s", e)
        return None

    behavior = None
    if not isinstance(beh_resp, Exception):
        try:
            if beh_resp.status_code == 200:
                behavior = _parse_behavior(beh_resp.json())
        except Exception as e:
            log.warning("behavior parse failed: %s", e)

    if isinstance(cap_resp, Exception):
        log.info("capture fetch failed: %s", cap_resp)
        return None
    if cap_resp.status_code != 200:
        log.info("capture HTTP %d", cap_resp.status_code)
        return None

    img_bytes = cap_resp.content
    if not img_bytes:
        return None

    img_size = _image_size_from_bytes(img_bytes)

    try:
        rec_resp = await client.post(
            recognize_url,
            files={"image": ("frame.jpg", img_bytes, "image/jpeg")},
            data={"min_confidence": str(config.min_recognize_confidence)},
            timeout=config.timeout_sec,
        )
    except Exception as e:
        log.info("recognize failed: %s", e)
        return None

    if rec_resp.status_code != 200:
        log.info("recognize HTTP %d", rec_resp.status_code)
        return None

    try:
        payload = rec_resp.json()
    except Exception as e:
        log.warning("recognize JSON decode failed: %s", e)
        return None

    predictions = _parse_predictions(payload, now_ts)

    return FaceObservation(
        captured_at=now_ts,
        predictions=predictions,
        behavior=behavior,
        image_size=img_size,
    )
