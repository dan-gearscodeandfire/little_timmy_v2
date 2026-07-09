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
from persistence import runtime_toggles
from vision.face_remote import RemoteFaceClient
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

        # Face-proximity gate (EXPO): reuses the Pi's /faces YuNet bbox as the
        # VLM trigger instead of pixel-MAD scene-change. Client created in
        # start(); gate logic in _poll_proximity(). Enabled live via the
        # vision_proximity_gate_enabled toggle (default OFF).
        self._face_client: RemoteFaceClient | None = None
        self._prox_window: list[bool] = []   # recent polls: was a close face present?
        self._prox_engaged: bool = False     # hysteresis latch for rising-edge firing
        self._last_face_height_frac: float = 0.0  # observability (HUD/stats)

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
        self._face_client = RemoteFaceClient()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("Frame capture started (poll=1fps, gate=%s, url=%s)",
                 "proximity" if runtime_toggles.get("vision_proximity_gate_enabled")
                 else "scene-change", self.capture_url)

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
        if self._face_client:
            await self._face_client.close()
            self._face_client = None
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
        """Poll at ~1fps, trigger the VLM callback only when the active gate
        fires.

        Two gates, selected live per poll by the vision_proximity_gate_enabled
        toggle:
          - scene-change (default): pixel-MAD frame diff (vision/scene_change).
          - proximity (EXPO): a close face in the Pi's /faces YuNet bbox.

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

                if runtime_toggles.get("vision_proximity_gate_enabled"):
                    await self._poll_proximity()
                else:
                    await self._poll_scene_change()

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Error in poll loop")
                await asyncio.sleep(5.0)

    async def _poll_scene_change(self):
        """Default gate: fetch a frame, fire the VLM on pixel-MAD scene change."""
        jpeg = await self._fetch_frame("poll", LOW_RES)
        if jpeg is None:
            return

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
                return
            self._frames_analyzed += 1
            self._last_vlm_time = now
            log.info("[CAPTURE] Triggering VLM (score=%.1f, analyzed %d/%d frames)",
                     score, self._frames_analyzed, self._frames_polled)
            await self._on_frame(jpeg)

    async def _poll_proximity(self):
        """EXPO gate: fire the VLM when a face gets close to the booth.

        Reads the Pi's /faces YuNet bbox (cheap JSON — no frame transfer until
        we actually fire) and computes the largest face's height as a fraction
        of frame height. A qualifying poll = height >= threshold. Engage when
        >= N of the last M polls qualified (debounced — YuNet is intermittent
        and dark during servo motion); fire the VLM on the RISING edge (a
        visitor stepping up), then hold until the window empties (hysteresis)
        so we don't re-fire every poll. Identity is ignored on purpose — a
        stranger walking up is exactly who we want to greet.
        """
        thr = runtime_toggles.get("vision_proximity_height_frac")
        m = max(1, int(runtime_toggles.get("vision_proximity_debounce_m")))
        n = max(1, int(runtime_toggles.get("vision_proximity_debounce_n")))
        refresh_s = runtime_toggles.get("vision_proximity_refresh_s")

        self._frames_polled += 1

        # Largest face height fraction from the Pi's authoritative /faces state.
        max_hf = 0.0
        if self._face_client is not None:
            state = await self._face_client.fetch_full()
            if state:
                fh = state["image_size"][1]
                if fh > 0:
                    for f in state["faces"]:
                        b = f.get("bbox")
                        if b and len(b) >= 4:
                            max_hf = max(max_hf, b[3] / fh)
        self._last_face_height_frac = max_hf

        # Debounced engagement over the last M polls.
        self._prox_window.append(max_hf >= thr)
        if len(self._prox_window) > m:
            self._prox_window = self._prox_window[-m:]
        count = sum(self._prox_window)
        engaged_now = count >= n

        if engaged_now:
            was_engaged = self._prox_engaged
            self._prox_engaged = True
            if not was_engaged:
                await self._fire_proximity("arrival", max_hf, count, m)
            elif refresh_s and refresh_s > 0 and \
                    (time.monotonic() - self._last_vlm_time) >= refresh_s:
                await self._fire_proximity("refresh", max_hf, count, m)
        elif count == 0:
            # Window fully clear of close faces -> visitor left; re-arm.
            if self._prox_engaged:
                log.info("[CAPTURE] proximity disengaged (face gone)")
            self._prox_engaged = False

    async def _fire_proximity(self, reason: str, max_hf: float, count: int, m: int):
        """Fetch the current frame and fire the VLM for a proximity event,
        honoring the shared VLM cooldown. The rising-edge latch is already set
        by the caller, so a cooldown block just skips this fire (we won't
        re-fire until disengage/re-engage or the refresh timer)."""
        if not self._on_frame:
            return
        now = time.monotonic()
        elapsed = now - self._last_vlm_time
        if elapsed < self._vlm_cooldown:
            log.debug("[CAPTURE] proximity %s but VLM cooldown (%.0fs remaining)",
                      reason, self._vlm_cooldown - elapsed)
            return
        jpeg = await self._fetch_frame("proximity", LOW_RES)
        if jpeg is None:
            return
        self._frames_analyzed += 1
        self._last_vlm_time = now
        log.info("[CAPTURE] Triggering VLM (proximity %s, height=%.0f%%, %d/%d window, "
                 "analyzed %d/%d)", reason, max_hf * 100, count, m,
                 self._frames_analyzed, self._frames_polled)
        await self._on_frame(jpeg)

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
