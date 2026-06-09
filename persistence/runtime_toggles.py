"""LT-side runtime toggle persistence.

Backed by a single JSON file at ~/little_timmy/data/lt_runtime_toggles.json.
Two flags live here today; add more by appending to _DEFAULTS and exposing
get/set helpers. Concurrent writes are guarded by a process-local lock.

Stays separate from persona/state.py (mood state) and from the streamerpi
face_tracking_state.json (which is owned by streamerpi).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)

STATE_PATH = Path.home() / "little_timmy" / "data" / "lt_runtime_toggles.json"

_DEFAULTS: dict = {
    "vision_auto_poll_enabled": True,         # 1fps VLM poll loop
    "hearing_enabled": True,                  # mic frames -> STT enqueue
    # Phase 2 Qwen3.6 conversation-tier swap (2026-05-14): when set to a
    # non-empty URL, llm.client.stream_conversation routes there instead
    # of config.LLM_CONVERSATION_URL. LT-OS writes this when the operator
    # picks an external-service model from the dropdown (qwen36 today).
    # Empty string == use the static default (Llama 3B :8081).
    "conversation_url_override": "",
    # Persisted dropdown choice. LT-OS reads on startup to restore the
    # last-selected conversation model and applies it: stops/starts the
    # llama-3b systemd unit + sets/clears conversation_url_override as
    # appropriate. Default matches the LT-OS config.py module default.
    "conversation_model_id": "llama3.2-3b",
    # Proactive (unprompted) speech (2026-06-03). Live kill/enable switch for
    # Timmy reacting verbally to high-urgency visual events without being
    # addressed first. Gated jointly with config.PROACTIVE_SPEECH_ENABLED (the
    # static master switch) -- BOTH must be true. Default False (opt-in).
    "proactive_speech_enabled": False,
    # --- Near-field / party-capture knobs (2026-06-09) ---------------------
    # Foundation only at this stage: persisted + readable, but the capture loop
    # does not consume them yet (that's the Phase 1/2 wiring in
    # docs/plans/party-mode-near-field-capture.md). Defaults reproduce current
    # behaviour exactly, so adding them is a no-op until something reads them.
    "capture_vad_threshold": 0.4,        # Silero onset prob floor (seeds config.VAD_THRESHOLD)
    "capture_energy_floor": 0.0,         # peak-amplitude floor for onset; 0.0 == disabled
    "party_mode_enabled": False,         # master switch for near-field + allowlist gating
    "speaker_allowlist": ["dan"],        # names allowed to get a reply when party mode is on; [] == allow all
}

_lock = threading.Lock()


def _load() -> dict:
    """Read the on-disk state, merging over defaults so missing keys fall
    back to the design default (not crash)."""
    try:
        raw = json.loads(STATE_PATH.read_text())
    except FileNotFoundError:
        return dict(_DEFAULTS)
    except Exception as e:
        log.warning("lt_runtime_toggles load failed (%s); using defaults", e)
        return dict(_DEFAULTS)
    merged = dict(_DEFAULTS)
    for k, default in _DEFAULTS.items():
        if k in raw and isinstance(raw[k], type(default)):
            merged[k] = raw[k]
    return merged


def _save(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.warning("lt_runtime_toggles save failed: %s", e)


def get(key: str):
    """Return the current value (disk-backed). On first call this reads
    the file; afterwards still re-reads so a manual edit takes effect
    without service restart. Return type matches the default's type
    (bool / str / etc.).
    """
    with _lock:
        return _load().get(key, _DEFAULTS.get(key))


def set(key: str, value) -> None:
    """Persist a toggle value. Idempotent if value matches current.

    Accepts bool or str (or any JSON-serializable type matching the
    default's type). Type coercion is left to the caller; runtime_toggles
    only writes whatever it gets and re-reads it on next get().
    """
    with _lock:
        state = _load()
        if state.get(key) == value:
            return
        state[key] = value
        _save(state)
        log.info("lt_runtime_toggles: %s = %r", key, value)
