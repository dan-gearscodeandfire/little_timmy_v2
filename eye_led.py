"""Fire-and-forget Eye LED state notifications to streamerpi -> ESP32.

The streamerpi forwards arbitrary text over UART to the ESP32 eye firmware
via POST /esp32/write. The firmware recognizes the v1 protocol:

    "THINKING"       sent at STT-finalize  (Dan stopped speaking, LLM thinking)
    "SPEAKING"       sent at TTS-start     (Timmy begins replying)
    "AI_CONNECTED"   sent at TTS-end       (back to listening / normal pulse)

Recovered 2026-05-13 from v1 repo dan-gearscodeandfire/little_timmy
(stt-server-v17/timmy_hears.py::notify_eye + tts-server/timmy_speaks_cuda.py::post_indicator_text).
See Obsidian: reference-eye-led-endpoint-v1-archaeology.
"""
import logging
from typing import Optional

import httpx

import config

log = logging.getLogger("eye_led")

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(verify=False, timeout=1.0)
    return _client


async def notify(state: str) -> None:
    """Fire-and-forget Eye LED state signal. Swallows all errors."""
    url = getattr(config, "STREAMERPI_EYE_LED_URL", "")
    if not url:
        return
    try:
        await _get_client().post(
            url,
            json={"text": state},
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        log.debug("eye_led.notify(%r) failed: %s", state, e)
