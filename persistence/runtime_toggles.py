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
    # Mouth-mute (2026-06-12). When True, Timmy's CONVERSATIONAL voice (replies
    # + THINKING fillers) is silenced — speak()/speak_filler() skip the playback
    # enqueue, so capture.suppressed never sets and the mic stays fully open.
    # His EARS + the speaker matcher keep running (unlike hearing_muted, which
    # gates input). The supervisor /api/announce channel bypasses this (force=
    # True) so Claude can still talk to Dan. Purpose: a clean bench for two-voice
    # attribution tests and for enrolling guests without Timmy talking over the
    # cues. Default False. Read live by the TTS engine — no restart to flip.
    "tts_muted": False,
    # --- Near-field capture knobs (2026-06-09) ------------------------------
    # Always-on, mode-independent (Dan 2026-06-10: no "party/presenter mode" —
    # the knob's value IS the switch). Consumed live by audio/capture.py;
    # exposed via the LT-OS VU meter + sliders.
    "capture_vad_threshold": 0.4,        # Silero onset prob floor (seeds config.VAD_THRESHOLD)
    "capture_energy_floor": 0.0,         # peak-amplitude floor for onset; 0.0 == disabled
    # --- P4 face-ID flap debounce knobs (2026-06-11) ------------------------
    # B3: a person must appear in >= ceil(this * lookback-5) recent scene
    # records before they count as "new" for novelty / proactive rising-edge.
    # 0.4 -> 2 of last 5. 0.0 disables the gate (legacy per-frame behavior).
    # Consumed live by vision/relevance.py classify().
    "people_novelty_min_persistence": 0.4,
    # C5: enroll-candidate hardening (presence/face_enroller.py trigger).
    # Candidate samples must span >= this many seconds...
    "enroll_candidate_min_span_s": 6.0,
    # ...and the track must be a CONFIDENT stranger: min_dist to every known
    # identity > this on most samples (shares A2's release-threshold value).
    "enroll_candidate_min_dist": 0.60,
    # --- Slice A: manual situational-awareness regime knob (2026-06-12) ------
    # Human-set operating regime that injects an NL [SITUATION] line into the
    # ephemeral prompt (llm/prompt_builder.build_ephemeral_block), framing the
    # WHO-IS-SPEAKING / WHO-IS-PRESENT lines below it. Empty string == OFF ==
    # no line emitted (changes nothing until Dan sets it). Whitelist enforced
    # at the web/app.py /api/situation boundary, NOT here. Re-read per turn.
    "situation_regime": "",
    # --- Slice B: symmetric + temporal identity fusion (2026-06-12, DARK) -----
    # All default-OFF / today's-behavior. Prototype — enable only after Dan's
    # live review. Read live per turn by presence.identity.IdentityFusion.
    #   symmetric: a CONFIDENT voice synthesizes a face-identity HINT (presence/
    #     proactive path only — voice still always wins final_name) when the face
    #     is absent/low-conf and was dropped upstream. Kills the P4 "did you
    #     bring home a stray" misfire when Dan's face flaps but his voice is sure.
    "identity_fusion_symmetric_enabled": False,
    #   continuity: carry a recent face-identity across a 1-2 frame dropout.
    "identity_continuity_enabled": False,
    #   continuity window (s): how long a held identity stays fresh. Kept shorter
    #     than identifier.py's 15s so the two windows don't stack a stale tail.
    "identity_continuity_window_s": 2.5,
    #   regime: "party" short-circuits BOTH new blocks to today's exact behavior
    #     (at a party a wrong bind is worse than an abstain).
    "identity_regime": "normal",
    # --- Short-audio speaker-continuity hardening (2026-06-12, party-prep) ----
    # Live-tunable caps for speaker/identifier.py's short-audio continuity
    # fallback (stamps a brief, non-confident utterance as whoever JUST spoke).
    # Live test 2026-06-11 showed a stranger clip at 0.52 from Dan on 3s audio
    # sitting inside the OLD 0.55 cap (Dan self-matches ~0.13-0.21); only an
    # expired 60s timer saved it. Tightened: cap 0.40, window 15s. Additionally
    # disabled outright when situation_regime is PARTY/EXPO (see identifier.py).
    "speaker_continuity_dist_cap": 0.40,    # was hardcoded 0.55
    "speaker_continuity_window_s": 15.0,    # was hardcoded 60.0
    "speaker_continuity_margin": 0.10,      # last speaker must beat 2nd-nearest known by this to "continue" (anti-latch, 2026-06-15)
    # Retired 2026-06-10: "party_mode_enabled" + "speaker_allowlist" (Phase 2
    # reply gating). Speaker-ID isn't reliable enough to gate replies on; the
    # predicate lives on as main.speaker_allowlist_drop (gate commented out in
    # process_speech). Re-add both keys here when re-enabling. Stale keys in
    # the on-disk JSON are ignored by _load()'s defaults-merge.
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
