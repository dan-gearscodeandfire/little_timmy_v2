"""HTTP-format helpers retained from the legacy demerzel-vision client.

face_client_local.py reuses _image_size_from_bytes (best-effort JPEG SOF
parser) and _parse_behavior (streamerpi /behavior/status JSON -> dataclass).
Everything else from the old HTTP path was removed when LT migrated to the
in-process face client (commit 5a1ee2e).
"""

import logging
from typing import Optional

from .types import BehaviorSnapshot


log = logging.getLogger(__name__)


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

