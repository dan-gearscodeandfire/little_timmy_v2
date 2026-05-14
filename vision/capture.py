"""Frame capture from streamerpi with scene-change gating.

Pulls JPEG frames from streamerpi's /capture endpoint at ~1fps.
Uses scene-change detection to decide when to trigger VLM analysis,
rather than blindly analyzing every frame.

Flow:
  1fps poll -> scene_change check -> if changed: on_frame callback (VLM)
"""

import asyncio
import logging
import time
from typing import Callable, Awaitable

import httpx
import config
from vision.scene_change import SceneChangeDetector

log = logging.getLogger(__name__)


# Single capture resolution for all paths (passive polls + visual
# questions). Owner does not need OCR (decided 2026-05-08), so the prior
# adaptive LOW/HIGH split was collapsed. Streamerpi /capture supports
# ?w=&h= since 2026-05-07.
LOW_RES = (320, 180)


class FrameCapture:
    """Async frame capture with scene-change gating."""

    def __init__(self):
        self.capture_url = config.STREAMERPI_CAPTURE_URL
        self._client: httpx.AsyncClient | None = None
        self._last_capture_time: float = 0.0
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task | None = None
        self._running = False
        self._on_frame: Callable[[bytes], Awaitable[None]] | None = None
        self._detector = SceneChangeDetector()

        # Cooldown: minimum seconds between VLM calls
        self._vlm_cooldown: float = 10.0
        self._last_vlm_time: float = 0.0

        # Stats
        self._frames_polled: int = 0
        self._frames_analyzed: int = 0
        self._last_change_score: float = 0.0

        # Pause control: counted so overlapping pauses (e.g. concurrent
        # speech turns) do not race the resume back to running prematurely.
        # Used by the orchestrator to gate the poll loop during the
        # speech->TTS window so the conversation-tier LLM does not contend
        # for GPU with a periodic VLM call. See Orchestrator main loop.
        self._pause_count: int = 0

        # User-facing auto-poll toggle, independent of the orchestrator's
        # transient pause counter above. When False the periodic poll loop
        # skips entirely (no streamerpi fetch, no VLM call). Event-driven
        # trigger() calls still fire so a speech event can still grab a
        # snapshot. Exposed via LT-OS so the user can save GPU/electricity
        # while the conversation tier and event-driven vision still work.
        # Bundle D 2026-05-14: read persisted state instead of
        # hard-coding True so a manual disable survives a reboot.
        from persistence import runtime_toggles as _toggles
        self._auto_poll_enabled: bool = _toggles.get("vision_auto_poll_enabled")

    def pause(self):
        """Increment pause counter. Poll loop skips while count > 0.

        Idempotent and re-entrant safe via counting; matched calls to
        pause()/resume() compose correctly with overlapping pause windows.
        """
        self._pause_count += 1

    def resume(self):
        """Decrement pause counter. Poll loop resumes when count reaches 0."""
        self._pause_count = max(0, self._pause_count - 1)

    @property
    def is_paused(self) -> bool:
        return self._pause_count > 0

    def set_auto_poll(self, enabled: bool):
        """Toggle the periodic poll loop. Event-driven trigger() unaffected."""
        prev = self._auto_poll_enabled
        self._auto_poll_enabled = bool(enabled)
        if prev != self._auto_poll_enabled:
            log.info("Vision auto-poll %s", "enabled" if self._auto_poll_enabled else "disabled")
            # Bundle D 2026-05-14: persist the change so a reboot doesn't
            # silently reset the toggle. Read-back at next FrameCapture init.
            try:
                from persistence import runtime_toggles as _toggles
                _toggles.set("vision_auto_poll_enabled", self._auto_poll_enabled)
            except Exception as e:
                log.warning("auto_poll persist failed: %s", e)

    @property
    def is_auto_poll_enabled(self) -> bool:
        return self._auto_poll_enabled

    async def start(self, on_frame: Callable[[bytes], Awaitable[None]]):
        """Start polling at ~1fps with scene-change gating."""
        self._on_frame = on_frame
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            verify=False,  # streamerpi uses self-signed certs
        )
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("Frame capture started (poll=1fps, gate=scene-change, url=%s)",
                 self.capture_url)

    async def stop(self):
        """Stop polling and close HTTP client."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None
        log.info("Frame capture stopped (polled=%d, analyzed=%d, ratio=%.0f%%)",
                 self._frames_polled, self._frames_analyzed,
                 (self._frames_analyzed / max(self._frames_polled, 1)) * 100)

    async def trigger(self, reason: str = "event", resolution: tuple | None = None) -> bytes | None:
        """Force an immediate capture + VLM analysis (speech event, etc).

        Bypasses scene-change gating AND cooldown. Optional resolution lets
        callers escalate to HIGH_RES for visual questions.
        """
        self._detector.force_next()
        self._last_vlm_time = 0.0  # bypass cooldown for forced triggers
        return await self._fetch_frame(reason, resolution)

    async def _poll_loop(self):
        """Poll at ~1fps, only trigger VLM callback on scene change.

        Skips while is_paused (set by orchestrator during speech->TTS turn
        to keep the periodic VLM off the GPU while conversation-tier LLM
        is generating).
        """
        while self._running:
            try:
                await asyncio.sleep(1.0)
                if not self._running:
                    break
                if self.is_paused:
                    continue
                if not self._auto_poll_enabled:
                    continue

                jpeg = await self._fetch_frame("poll", LOW_RES)
                if jpeg is None:
                    continue

                self._frames_polled += 1

                # Scene-change gate
                should_analyze, score = self._detector.check(jpeg)
                self._last_change_score = score

                if should_analyze and self._on_frame:
                    now = time.monotonic()
                    elapsed = now - self._last_vlm_time
                    if elapsed < self._vlm_cooldown:
                        log.debug("[CAPTURE] VLM cooldown (%.0fs remaining)",
                                  self._vlm_cooldown - elapsed)
                        continue
                    self._frames_analyzed += 1
                    self._last_vlm_time = now
                    log.info("[CAPTURE] Triggering VLM (score=%.1f, analyzed %d/%d frames)",
                             score, self._frames_analyzed, self._frames_polled)
                    await self._on_frame(jpeg)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Error in poll loop")
                await asyncio.sleep(5.0)

    async def _fetch_frame(self, reason: str, resolution: tuple | None = None) -> bytes | None:
        """Fetch a single JPEG frame from streamerpi.

        If resolution is given, append ?w=&h= so streamerpi serves a
        downsampled JPEG (saves prompt tokens + LAN bytes).
        """
        if not self._client:
            return None

        url = self.capture_url
        if resolution and len(resolution) == 2:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}w={int(resolution[0])}&h={int(resolution[1])}"

        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            jpeg_bytes = resp.content

            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                log.warning("Unexpected content-type from /capture: %s", content_type)
                return None

            return jpeg_bytes

        except httpx.ConnectError:
            log.debug("streamerpi not reachable (%s)", reason)
            return None
        except httpx.HTTPStatusError as e:
            log.warning("Capture HTTP error: %s", e)
            return None
        except Exception:
            log.exception("Capture failed (%s)", reason)
            return None
