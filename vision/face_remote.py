"""Remote face state client.

Fetches face detection + identity results from streamerpi's /faces HTTP
endpoint instead of running YuNet+SFace locally on okdemerzel. streamerpi's
2 Hz tracking thread is the single source of truth for face state.

This eliminates ~100 ms/frame of redundant cv2+ONNX work that was previously
running synchronously on okdemerzel's asyncio event loop, and removes the
secondary face DB at ~/.face_db/embeddings.json that drifted from
streamerpi's authoritative DB at ~/little_timmy_motor_raspi/face_db/.
"""

import logging
from typing import Optional

import httpx

import config

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(2.0, connect=1.0)


class RemoteFaceClient:
    """HTTP client for streamerpi's /faces endpoint."""

    def __init__(self, url: Optional[str] = None, max_age_s: float = 5.0):
        self.url = url or config.STREAMERPI_FACES_URL
        self.max_age_s = max_age_s
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, verify=False)

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self) -> Optional[dict]:
        """Fetch most recent face state dict, or None on failure.

        Format: {timestamp, age_s, image_size [w, h], faces [{name, distance, confidence, bbox}]}
        """
        await self._ensure_client()
        try:
            resp = await self._client.get(self.url)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            log.debug("streamerpi /faces not reachable at %s", self.url)
            return None
        except Exception:
            log.exception("Failed to fetch /faces")
            return None

    async def fetch_fresh_results(self) -> list[dict]:
        """Fetch faces, return [] if stale or unavailable.

        Returns list of {name, distance, confidence, bbox} dicts in the same
        shape: list of {name, distance, confidence, bbox} dicts (legacy in-tree YuNet+SFace path was retired 2026-05-14).
        """
        data = await self.fetch()
        if data is None:
            return []
        age = data.get("age_s")
        if age is None or age > self.max_age_s:
            log.debug("face data stale (age=%s, max=%.1fs)", age, self.max_age_s)
            return []
        return data.get("faces", [])

    async def fetch_full(self) -> Optional[dict]:
        """Fetch faces + image_size, applying staleness filter to faces only.

        Returns: {faces: [...], image_size: (w, h), age_s: float}
        Returns None only if the connection failed entirely.
        """
        data = await self.fetch()
        if data is None:
            return None
        age = data.get("age_s")
        faces = []
        if age is not None and age <= self.max_age_s:
            faces = data.get("faces", [])
        size = data.get("image_size") or [0, 0]
        return {
            "faces": faces,
            "image_size": (int(size[0]), int(size[1])),
            "age_s": age,
        }
