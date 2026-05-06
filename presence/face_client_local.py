"""In-process face observation client.

Uses the in-tree vision pipeline's FaceID + FrameCapture instead of the
demerzel-vision :8895 HTTP service. Same FaceObservation shape returned
to the caller. Behavior status still fetched over HTTP from streamerpi
(no in-process equivalent).

Benefits over the HTTP path:
  - No HTTP latency or stale-cache risk for face recognition
  - Stable unknown tracking via in-tree FaceID._assign_unknown
  - Single face DB load shared with the periodic vision enrichment
  - LT no longer requires demerzel-vision.service to be running

The caller (Orchestrator) injects vision_context (must have non-None
_face_id_ready, _face_id, _capture).
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

from .face_client import _image_size_from_bytes, _parse_behavior
from .types import BehaviorSnapshot, FaceObservation, FacePrediction


log = logging.getLogger(__name__)


def _convert_in_tree_results(results) -> tuple:
    """Translate in-tree FaceID output to FacePrediction tuple.

    In-tree shape: list of {name, distance, confidence, bbox} where bbox is
    [x, y, w, h] (xywh) and confidence is a label string ("high"|"medium"|
    "low"|"unknown").

    Filtered: low/unknown confidence and "unidentified person" are dropped
    here (mirrors the existing _on_passive_face_id gate). Unknown stable
    IDs from in-tree (e.g. "unknown_1") are also dropped at this layer
    for v1; they remain available via the in-tree pipeline's own state.
    """
    out = []
    for r in results:
        name = (r.get("name") or "").strip()
        conf_label = r.get("confidence", "")
        if conf_label not in ("high", "medium"):
            continue
        if not name or name.lower().startswith("unknown") or name == "unidentified person":
            continue

        bbox_xywh = r.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]
        bbox_xyxy = (int(x), int(y), int(x + w), int(y + h))
        distance = float(r.get("distance", 1.0))
        conf_float = max(0.0, 1.0 - distance)

        out.append(FacePrediction(
            user_id=name,
            confidence=conf_float,
            bbox=bbox_xyxy,
            embedding_hash=None,
        ))
    return tuple(out)


async def _safe_get_behavior(
    client: httpx.AsyncClient,
    url: str,
    timeout_sec: float,
) -> Optional[BehaviorSnapshot]:
    try:
        resp = await client.get(url, timeout=timeout_sec)
        if resp.status_code == 200:
            return _parse_behavior(resp.json())
    except Exception as e:
        log.info("[face_client_local] behavior fetch failed: %s", e)
    return None


async def fetch_face_observation_local(
    vision_context,
    http_client: httpx.AsyncClient,
    behavior_url: str,
    timeout_sec: float = 1.5,
) -> Optional[FaceObservation]:
    """Speech-turn observation: in-process face recognition + remote behavior.

    Returns None if the vision pipeline isn't ready, capture fails, or
    face recognition raises. Behavior may be None on its own.
    """
    if vision_context is None:
        return None
    if not getattr(vision_context, "_face_id_ready", False):
        return None
    if not getattr(vision_context, "_capture", None):
        return None
    if not getattr(vision_context, "_face_id", None):
        return None

    now_ts = time.time()

    # Kick off behavior fetch in parallel with capture+identify
    beh_task = asyncio.create_task(
        _safe_get_behavior(http_client, behavior_url, timeout_sec)
    )

    # Capture fresh frame
    try:
        jpeg = await asyncio.wait_for(
            vision_context._capture._fetch_frame("presence"),
            timeout=timeout_sec,
        )
    except Exception as e:
        log.info("[face_client_local] capture failed: %s", e)
        beh_task.cancel()
        return None

    if not jpeg:
        beh_task.cancel()
        return None

    img_size = _image_size_from_bytes(jpeg)

    # Run face_id off the event loop (cv2 + onnx is synchronous)
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(vision_context._face_id.identify_from_jpeg, jpeg),
            timeout=timeout_sec,
        )
    except Exception as e:
        log.info("[face_client_local] face_id failed: %s", e)
        beh_task.cancel()
        return None

    behavior = await beh_task

    predictions = _convert_in_tree_results(results)

    return FaceObservation(
        captured_at=now_ts,
        predictions=predictions,
        behavior=behavior,
        image_size=img_size,
    )
