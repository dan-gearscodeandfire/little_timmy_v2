"""Behavioral supervisor — the 'director' for Little Timmy's body.

Runs on okDemerzel alongside the conversation pipeline. Watches
conversation state and vision context, then sends high-level mode
commands to the behavioral state machine on the Pi (streamerpi).

This is the bridge between:
  - What Timmy hears (speech events, speaker ID)
  - What Timmy sees (vision pipeline, SceneRecords)
  - What Timmy's body does (servo states on the Pi)

Design principle: the supervisor decides WHAT to do,
the Pi's state machine decides HOW to do it.
"""

import asyncio
import logging
import time

import httpx

log = logging.getLogger(__name__)

# streamerpi behavioral API
BEHAVIOR_URL = "https://192.168.1.110:8080/behavior"
EVENT_URL = "https://192.168.1.110:8080/behavior/event"
STATUS_URL = "https://192.168.1.110:8080/behavior/status"

# Timing thresholds
IDLE_AFTER_SILENCE = 15.0    # seconds of no speech → idle
SCAN_AFTER_IDLE = 60.0       # seconds of idle → occasional scan
LOOK_AROUND_CHANCE = 0.3     # probability of look_around vs idle after conversation


class BehaviorSupervisor:
    """Watches conversation + vision and directs the Pi's body behavior."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._running = False
        self._monitor_task: asyncio.Task | None = None

        # Conversation state
        self._last_speech_time: float = 0.0
        self._last_tts_end_time: float = 0.0
        self._in_conversation: bool = False
        self._current_speaker: str | None = None

        # Vision state
        self._people_visible: bool = False
        self._last_people_time: float = 0.0

        # Body state (last known from Pi)
        self._last_mode: str = "idle"
        self._last_mode_time: float = 0.0

        # Prevent command spam
        self._last_command_time: float = 0.0
        self._min_command_interval: float = 1.0

    async def start(self):
        """Start the supervisor."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=3.0),
            verify=False,  # streamerpi uses self-signed certs
        )
        self._running = True
        # Wake Pi from any prior sleep state
        await self._send_command("idle", priority="critical")
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("[SUPERVISOR] Started — monitoring conversation + vision")

    async def stop(self):
        """Stop the supervisor. Sends sleep to Pi so it doesn't scan aimlessly."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        # Tell Pi to sleep on shutdown
        await self._send_command("sleep", priority="critical")
        if self._client:
            await self._client.aclose()
            self._client = None
        log.info("[SUPERVISOR] Stopped (Pi set to sleep)")

    # --- Event hooks (called by orchestrator) ---

    async def on_speech_detected(self, speaker_name: str):
        """Someone started speaking to Timmy."""
        self._last_speech_time = time.time()
        self._in_conversation = True
        self._current_speaker = speaker_name

        # Tell Pi: speech started, engage mode
        await self._send_event("speech_start")
        await self._send_command("engage", priority="high", timeout_ms=30000)

        log.info("[SUPERVISOR] Speech from %s → engage", speaker_name)

    async def on_tts_start(self):
        """Timmy started speaking (TTS playing)."""
        # Keep engage mode while Timmy talks
        if self._last_mode != "engage":
            await self._send_command("engage", priority="high", timeout_ms=30000)

    async def on_tts_end(self):
        """Timmy finished speaking."""
        self._last_tts_end_time = time.time()
        # Stay in track for a bit — conversation may continue
        await self._send_command("track", priority="high", timeout_ms=15000)

    async def on_conversation_idle(self):
        """No speech for a while — wind down body behavior."""
        self._in_conversation = False
        self._current_speaker = None

        if self._people_visible:
            # Someone is still there, just not talking — scan to find them
            await self._send_command("scan", priority="normal",
                                     timeout_ms=30000,
                                     params={"pattern": "room_sweep", "loops": 2})
        else:
            # Nobody visible — scan then idle
            await self._send_command("scan", priority="normal",
                                     timeout_ms=20000,
                                     params={"pattern": "room_sweep"})

        log.info("[SUPERVISOR] Conversation idle → scan")

    async def on_vision_people_changed(self, people: list[str], was_empty: bool):
        """Vision pipeline detected a change in people present."""
        had_people = self._people_visible
        self._people_visible = len(people) > 0

        if self._people_visible:
            self._last_people_time = time.time()

        if not had_people and self._people_visible:
            # Someone appeared — look at them
            log.info("[SUPERVISOR] Person appeared: %s → track", people)
            await self._send_event("face_detected", pan=0, tilt=0)
            if not self._in_conversation:
                await self._send_command("track", priority="high", timeout_ms=15000)

        elif had_people and not self._people_visible:
            # Everyone left
            log.info("[SUPERVISOR] No people visible → scan")
            if not self._in_conversation:
                await self._send_command("scan", priority="normal",
                                         timeout_ms=20000,
                                         params={"pattern": "room_sweep"})

    # --- Monitor loop ---

    async def _monitor_loop(self):
        """Background loop checking for idle timeout and periodic behaviors."""
        while self._running:
            try:
                await asyncio.sleep(3.0)
                now = time.time()

                # Check for conversation timeout
                if self._in_conversation:
                    silence = now - max(self._last_speech_time, self._last_tts_end_time)
                    if silence > IDLE_AFTER_SILENCE:
                        await self.on_conversation_idle()

                # Periodic scan when idle for a long time
                if (not self._in_conversation
                        and self._last_mode == "idle"
                        and now - self._last_mode_time > SCAN_AFTER_IDLE):
                    log.info("[SUPERVISOR] Long idle → look_around")
                    await self._send_command("look_around", priority="normal",
                                             timeout_ms=20000)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("[SUPERVISOR] Error in monitor loop")

    # --- Pi communication ---

    async def _send_command(self, mode: str, priority: str = "normal",
                            timeout_ms: int = 0, params: dict = None):
        """Send a mode command to the Pi's behavioral state machine."""
        now = time.time()
        if now - self._last_command_time < self._min_command_interval:
            return  # throttle

        if not self._client:
            return

        body = {
            "mode": mode,
            "priority": priority,
            "timeout_ms": timeout_ms,
        }
        if params:
            body["params"] = params

        try:
            resp = await self._client.post(BEHAVIOR_URL, json=body)
            if resp.status_code == 200:
                self._last_mode = mode
                self._last_mode_time = now
                self._last_command_time = now
                log.debug("[SUPERVISOR] → Pi: %s (pri=%s)", mode, priority)
            else:
                log.warning("[SUPERVISOR] Pi returned %d: %s",
                            resp.status_code, resp.text[:100])
        except httpx.ConnectError:
            log.debug("[SUPERVISOR] Pi not reachable")
        except Exception:
            log.exception("[SUPERVISOR] Failed to send command")

    async def _send_event(self, event: str, **kwargs):
        """Send an event notification to the Pi."""
        if not self._client:
            return

        body = {"event": event}
        body.update(kwargs)

        try:
            await self._client.post(EVENT_URL, json=body)
        except httpx.ConnectError:
            log.debug("[SUPERVISOR] Pi not reachable for event")
        except Exception:
            log.exception("[SUPERVISOR] Failed to send event")

    async def get_pi_status(self) -> dict | None:
        """Query the Pi's current behavioral state."""
        if not self._client:
            return None
        try:
            resp = await self._client.get(STATUS_URL)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None
