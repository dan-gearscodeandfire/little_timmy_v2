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
    # Empty string == use the static default config.LLM_CONVERSATION_URL
    # (now :8083 -- the qwen36 brain; the old Llama-3B :8081 was ceased 2026-06-22).
    "conversation_url_override": "",
    # Persisted dropdown choice. LT-OS reads on startup to restore the
    # last-selected conversation model and applies it: stops/starts the
    # llama-3b systemd unit + sets/clears conversation_url_override as
    # appropriate. Default matches the LT-OS config.py module default.
    # 2026-06-22 (Dan): default flipped llama3.2-3b -> qwen36 so a fresh/reset
    # toggle state never tries to spawn the disabled :8081 server.
    "conversation_model_id": "qwen36",
    # Proactive (unprompted) speech (2026-06-03). Live kill/enable switch for
    # Timmy reacting verbally to high-urgency visual events without being
    # addressed first. Gated jointly with config.PROACTIVE_SPEECH_ENABLED (the
    # static master switch) -- BOTH must be true. Default False (opt-in).
    "proactive_speech_enabled": False,
    # Face-recognition SHADOW mode (2026-06-30). When True, each speech turn also
    # runs okDemerzel-side EdgeFace recognition (self-served from a /capture grab)
    # and logs [FACE-SHADOW] okDemerzel-vs-Pi agreement — WITHOUT changing fusion
    # (the Pi's SFace stays authoritative). Lets us watch real accuracy before
    # flipping authority to okDemerzel. Fire-and-forget + off the event loop, so
    # it never adds reply latency. Read live per-turn; default False (opt-in).
    "face_shadow_enabled": False,
    # Identity AUTHORITY source (2026-07-01). "pi" = the Pi's SFace /faces result
    # feeds IdentityFusion (legacy). "okdemerzel" = okDemerzel EdgeFace (multi-
    # frame /capture grab) feeds fusion instead, recognizing the enrolled makers
    # too (superset of the Pi). The Pi still supplies the BehaviorSnapshot either
    # way. Falls back to the Pi observation on any okDemerzel error/timeout. Read
    # live per-turn; default "pi".
    "face_authority": "pi",
    # Frames grabbed per turn when face_authority == "okdemerzel" (best match per
    # identity across them; an off-center subject dodges a single frame).
    "face_authority_frames": 3,
    # Live face-recognition accept cutoff (cosine distance). A probe within this
    # of an enrolled prototype is recognized; bands scale with it (high=0.8x,
    # medium=this). Default is the calibrated KNOWN_FACE_THRESHOLD. Tunable on the
    # day at OpenSauce (different lighting) with NO restart: lower = stricter
    # (fewer false hits), higher = looser (catches more, watch false accepts).
    "face_threshold": 0.50,
    # Phase B unified dual-modality enroll (2026-07-01). When True, "enroll me /
    # remember my face / remember my voice as X" routes through
    # presence.identity_commit.commit_identity (okDemerzel voiceprint + EdgeFace +
    # shared id-map + Postgres) instead of main._handle_enrollment's RETIRED Pi
    # SFace POST, and passively co-sampled sole-face crops enroll the face with no
    # separate capture dialog. Wins over config.UNIFIED_ENROLL_ENABLED (the env
    # default). Read live per-turn at the doorway; default False (opt-in) — flip
    # here to enable without a restart once validated. See
    # docs/enroll-unification-phase-bc.md.
    "unified_enroll_enabled": False,
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
    # --- EXPO identity-dialog gate override (2026-07-06) ---------------------
    # The identity-dialog CLASS (enroll latches, misID-correction protest,
    # introductions name-ask, face auto-enroll consent — every multi-turn
    # identity FSM that seizes turns and can end in a store write) is gated
    # OFF when situation_regime is a crowd regime (see
    # identity_dialogs_allowed() below): booth triggers are ordinary speech
    # ("my name is Bob" = every self-introduction), the off-mic chain defeats
    # the contradiction gate, and confirm-yes is an identity-MUTATION surface.
    # At the show: mutation surfaces dark, recognition read-only. This
    # override forces dialogs back ON under a crowd regime for a SUPERVISED
    # enroll mid-show (webui-flippable live, read per turn). No effect when
    # the regime is Shop ('') — dialogs are already allowed there.
    "identity_dialogs_override": False,
    # --- LED-mic anchor (2026-07-06, Dan's booth engagement token) ------------
    # A green-LED handheld mic marks the person Timmy is engaged with at EXPO;
    # the face directly above the lit LED is the speaker's. State (who/when)
    # lives IN-PROCESS in presence/anchor.py — a restart wipes it back to the
    # dark gate on purpose. These are just the live-tunable knobs.
    # Master switch: gates the identity-dialog disjunct (anchor.gate_disjunct),
    # anchored-crop selection, and the CV LED detector. Default OFF — every
    # deploy is behavior-neutral until flipped at the booth.
    "anchor_enabled": False,
    # Freshness window (s): the anchor is "active" only while refreshed within
    # this. A racked mic / occluded LED decays back to the dark gate.
    "anchor_ttl_s": 30.0,
    # Horizontal tolerance for "face directly above the LED", as a fraction of
    # frame width. Two candidate faces inside the tolerance = ABSTAIN (never
    # guess which face holds the mic — sole-face rule's ambiguity contract).
    "anchor_x_tol_frac": 0.25,
    # Name-tell writes the full triple (2026-07-06): when True, a confirmed
    # introductions name-tell ALSO commits the co-sampled face crops through
    # commit_identity (name<->voiceprint<->faceprint), closing the voice-only
    # gap in the introductions path. At EXPO the crops are the LED-anchored
    # face (implied consent, Dan's ruling); in Shop they're the sole-face
    # crops. Default OFF — flip with the anchor at the booth (or in Shop to
    # close the gap there too). Read live at the confirm; no restart.
    "intro_face_commit_enabled": False,
    # Passive self-intro (2026-07-06): an UNKNOWN speaker volunteering
    # "my name is X" (framed only — never bare tokens or "I'm X") arms the
    # introductions confirm flow without Timmy asking first. Runs AFTER the
    # enroll-keyword and misID detectors (they win) and only when the
    # identity dialogs are effectively allowed (regime/override/anchor).
    # Default OFF. Read live per turn.
    "passive_self_intro_enabled": False,
    # CV green-LED detection knobs (presence/led_detect.py) — HSV mask +
    # blob-size bounds, all live-tunable at the booth (lighting varies).
    # OpenCV hue is 0-179; 40-85 spans green. The V floor is high on purpose:
    # a LIT LED is one of the brightest things in frame. Area bounds are in
    # px at /capture resolution — reject a green shirt (huge) and sensor
    # speckle (tiny). Exactly ONE surviving blob anchors; zero or 2+ abstain
    # (two green lights = no anchor, same ambiguity contract as the faces).
    "anchor_led_h_lo": 40,
    "anchor_led_h_hi": 85,
    "anchor_led_s_min": 80,
    "anchor_led_v_min": 200,
    "anchor_led_min_area_px": 4,
    "anchor_led_max_area_px": 400,
    # Periodic anchor-poll cadence (s) — main.anchor_poll_monitor grabs one
    # /capture frame and runs the CPU LED detect + face pick + identify so the
    # anchor is fresh BEFORE a visitor's first utterance (Ruling A, Dan
    # 2026-07-07; the per-turn-only refresh opened the gate one turn late).
    # Keep well under anchor_ttl_s. Only ticks while anchor_enabled.
    "anchor_poll_interval_s": 2.0,
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
    "speaker_continuity_dist_cap": 0.60,    # WeSpeaker-calibrated 2026-06-17 (was 0.40 Resemblyzer); below the ~0.70 impostor floor
    "speaker_continuity_window_s": 15.0,    # was hardcoded 60.0
    "speaker_continuity_margin": 0.12,      # last speaker must beat 2nd-nearest known by this to "continue" (anti-latch, 2026-06-15; WeSpeaker-rescaled 2026-06-17)
    # --- Open-set rejection guard (WeSpeaker anti-model + s-norm, 2026-06-17) --
    # Layered on top of the raw known-speaker accept in speaker/identifier.py.
    # When True, an otherwise-accepted known match must ALSO clear the anti-model
    # / s-norm bar (snorm >= min_snorm AND am_margin >= min_am_margin) or it falls
    # through to continuity -> unknown. Fixes the P1 noise-collapse false accept
    # (degraded capture stamped onto the enrolled print nearest the noise centroid,
    # measured = Erin). Default OFF; min_* are PROVISIONAL until calibrated live by
    # ops/open_set_calibrate.py against real genuine/impostor WeSpeaker captures.
    # Re-read per identify() so it's live-flippable without restart.
    "open_set_reject_enabled": False,
    "open_set_min_snorm": 0.0,              # s-norm at/above this required to accept
    "open_set_min_am_margin": 0.0,          # anti-model margin (s_raw - max_cohort_sim) at/above this
    # --- First-pass tool-call classifier (Qwen3-4B :8092, 2026-06-18) ---------
    # Master gate for the intent router that runs BEFORE the conversation brain.
    # When True, each utterance is first classified by the :8092 server; a
    # recognized intent (today: store_fact) routes to a tool instead of the LLM
    # reply, and any non-tool utterance falls through to the normal pipeline.
    # Default OFF (opt-in). Read live per turn by conversation/tool_router.py;
    # also the LT-OS toggle target. Classifier-server outage degrades gracefully
    # (classify returns None -> fall through), so flipping this ON is safe even
    # if :8092 is down.
    "classifier_enabled": False,
    # --- Elliptical-query coreference resolution (2026-06-18) -----------------
    # When True, retrieval rewrites a deictic/elliptical utterance ("what's its
    # name again?") into a standalone semantic query via the :8092 classifier
    # server BEFORE embedding, replacing the role-tagged context blend. Gated on
    # a cheap Python deixis check (memory/retrieval._needs_resolution) so the
    # extra ~170ms call only fires on turns that actually contain a pronoun/
    # reference; clean queries skip it and pay nothing. Falls back to the current
    # _build_semantic_query blend on resolver error or empty output, so a :8092
    # outage degrades gracefully. Measured 2026-06-18: MRR 0.71->0.85 over the
    # blend on elliptical follow-ups; no effect on self-contained queries.
    # Default OFF (opt-in A/B). Read live per retrieve() -- no restart to flip.
    "query_resolution_enabled": False,
    # --- Speculative coref: resolve in PARALLEL with the classifier (2026-06-22) -
    # When True, the doorway (main.process_speech) launches the coref resolution
    # (:8093, via memory.retrieval.resolve_for_retrieval) as a task BEFORE awaiting
    # the tool-call classifier (:8092), then hands the result into retrieve()
    # (query_pre_resolved=True) so the brain path doesn't pay :8093 serially AFTER
    # the classifier returns. The intent was to overlap the two Qwen3-4B servers
    # and hide ~min(classifier, resolver) on deictic-brain-path turns.
    # ⚠️ DEFAULT OFF (live-mic A/B verdict 2026-06-23, Dan): it LOSES on this box.
    # The single Strix Halo Vulkan GPU does NOT timeslice the two llama.cpp server
    # processes cleanly -- running the resolver concurrently with the classifier
    # TRIPLED its decode (serial 220ms -> parallel 612-778ms), so parallel is
    # slower end-to-end than serial (classifier 200 + resolver 220 ~= 420ms vs
    # 612-778ms). Cross-process extension of feedback_strix_halo_vulkan_np_no_parallel
    # (originally an IN-server -np 2 finding). Kept behind the toggle (not ripped
    # out) for future hardware -- a second GPU, or the resolver on separate compute,
    # would change the verdict; re-A/B before re-enabling. OFF reverts to inline
    # resolution inside retrieve() (byte-identical to the pre-2026-06-22 path).
    # Read live per turn -- no restart to flip.
    "speculative_coref_enabled": False,
    # --- Privacy / guest gate (2026-06-18, Dan) -------------------------------
    # When True, facts classified sensitive (memory/pii.py: contact, location,
    # financial, health/credentials, family_minor) are dropped from prompt
    # injection in conversation/turn.py so Timmy can't speak them via TTS in
    # front of guests. Manual toggle (LT-OS target). Read live per turn. The
    # presence-driven AUTO layer will OR into the gate later -- manual wins.
    # Default OFF.
    "guest_mode": False,
    # --- S4 read-path gate: needs_retrieval (2026-06-20) ----------------------
    # When True, conversation/turn.py:LiveMemory.gather() SKIPS the vector
    # retrieve() over the `memories` store on confidently-banter turns (no
    # question / recall verb / possessive referent -- see turn._RETRIEVAL_RE),
    # saving a DB round-trip + ~202 prompt tokens per skipped turn. Recall /
    # question turns retrieve exactly as before; the facts-about-speaker (and
    # facts-about-subject) lookups ALWAYS run, gate or no gate. Conservative by
    # design -- biased toward retrieving, so a misread costs latency, never a
    # dropped recall. Independent of the episodic work; recall_temporal handles
    # temporal recall upstream. Read live per turn -- no restart to flip.
    # Default OFF (opt-in A/B). No config master (cf. query_resolution_enabled).
    "needs_retrieval_gate": False,
    # --- "Conversation active" window for the mail-defer gate (2026-06-20) -----
    # Seconds after the last conversation stream that /api/active still reports
    # active=true. The demerzel-mail ingest loop polls /api/active and defers
    # email fetches while active, so this knob = "how long after Dan stops
    # talking before email polling resumes." Live-tunable via the LT-OS slider
    # (/api/conversation/idle_gate); 0.0 = defer only while a stream is literally
    # in flight. NOTE: originally this ALSO widened llm/client._wait_for_
    # conversation_idle to defer extraction past inter-turn gaps, but that path
    # was RETIRED 2026-06-20 when extraction moved to the :8084 vision server
    # (the :8083/:8084 split solves contention physically). This toggle now
    # drives ONLY the mail-active window.
    "conversation_idle_gate_seconds": 20.0,
    # --- EXPO face-proximity vision-poll gate (2026-07-09, Dan) ---------------
    # Replaces the pixel-MAD scene-change gate (vision/scene_change.py) as the
    # VLM trigger when True. Instead of firing on raw frame motion (which pins
    # OPEN in a crowd — scores 28-65 vs threshold 12 with 2 people, and self-
    # triggers on servo head-moves), it fires on a FACE getting close: max
    # /faces YuNet bbox height >= vision_proximity_height_frac of frame,
    # sustained N-of-M polls, on the RISING edge (someone steps up to the
    # booth/mic). Gate on DETECTION + size, NOT identity (SFace can't ID
    # strangers/far faces; the expo audience is unenrolled). Event-driven
    # trigger() (visual questions) stays independent — "what do you see?"
    # always works. Calibrated live in-shop 2026-07-09: standing-back baseline
    # = 13-14% height, at-mic close = 45-65%; 20% clears background-standers by
    # ~6pts. See project_expo_vision_poll_gate. Default OFF (opt-in A/B vs the
    # MAD gate). Read live per poll — no restart to flip.
    "vision_proximity_gate_enabled": False,
    # Height threshold: a face bbox taller than this fraction of the frame
    # height counts as "close / at the booth". Dan set 20% (0.20) 2026-07-09
    # (I proposed 30%; he chose 20% for earlier/more-sensitive firing). Live-
    # tunable at the booth (camera height / crowd depth vary). Higher = must be
    # closer to fire; lower = fires from farther back (watch background-standers).
    "vision_proximity_height_frac": 0.20,
    # Debounce window: engage when >= _debounce_n of the last _debounce_m polls
    # (at ~1 poll/s) had a qualifying close face. Load-bearing because YuNet
    # detection is intermittent and goes DARK during servo head-motion
    # (camera.py:1080 skips frames while servos move) — a consecutive-frame rule
    # would flap. Disengage (re-arm the rising edge) only when the window is
    # fully empty of qualifiers, giving hysteresis. Defaults: 2-of-6.
    "vision_proximity_debounce_m": 6,
    "vision_proximity_debounce_n": 2,
    # While a visitor stays engaged, optionally re-poll the VLM every N seconds
    # so the scene description refreshes (they pick up an object, etc). 0.0 =
    # OFF = rising-edge only (leanest; avoids reintroducing a duty cycle). The
    # 10s VLM cooldown still floors it. Default 0.0.
    "vision_proximity_refresh_s": 0.0,
    # Retired 2026-06-10: "party_mode_enabled" + "speaker_allowlist" (Phase 2
    # reply gating). Speaker-ID isn't reliable enough to gate replies on; the
    # predicate lives on as main.speaker_allowlist_drop (gate commented out in
    # process_speech). Re-add both keys here when re-enabling. Stale keys in
    # the on-disk JSON are ignored by _load()'s defaults-merge.
}

_lock = threading.Lock()

# mtime-keyed parse cache (review 7-07): get() re-read AND re-parsed the JSON
# on every call — ~7x per frame on an anchored turn via the led_detect knob
# reads. The stat() is kept (manual edits + /api writers must still apply
# live, the design contract); only the parse+merge is skipped when the file
# is unchanged. Guarded by _lock (callers hold it).
_cache_stamp: tuple | None = None
_cache_state: dict | None = None


def _load() -> dict:
    """Read the on-disk state, merging over defaults so missing keys fall
    back to the design default (not crash)."""
    global _cache_stamp, _cache_state
    try:
        st = STATE_PATH.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return dict(_DEFAULTS)
    except Exception as e:
        log.warning("lt_runtime_toggles stat failed (%s); using defaults", e)
        return dict(_DEFAULTS)
    if _cache_state is not None and stamp == _cache_stamp:
        return dict(_cache_state)
    try:
        raw = json.loads(STATE_PATH.read_text())
    except FileNotFoundError:
        return dict(_DEFAULTS)
    except Exception as e:
        log.warning("lt_runtime_toggles load failed (%s); using defaults", e)
        return dict(_DEFAULTS)
    merged = dict(_DEFAULTS)
    for k, default in _DEFAULTS.items():
        if k not in raw:
            continue
        v = raw[k]
        # Type-checked merge with numeric coercion (F10, review 7-07): a
        # hand-edited int for a float knob (ttl_s: 60) used to be SILENTLY
        # dropped back to the default (30.0) — the booth-tuning trap. int<->
        # float now coerce; any other mismatch is dropped LOUDLY. bool is an
        # int subclass in Python, so it's matched exactly, never coerced.
        if isinstance(v, bool) == isinstance(default, bool) and isinstance(v, type(default)):
            merged[k] = v
        elif (isinstance(default, float) and isinstance(v, int)
              and not isinstance(v, bool)):
            merged[k] = float(v)
        elif (isinstance(default, int) and not isinstance(default, bool)
              and isinstance(v, float) and v.is_integer()):
            merged[k] = int(v)
        else:
            log.warning(
                "lt_runtime_toggles: %s=%r has wrong type (%s, want %s) — "
                "IGNORED, using default %r", k, v,
                type(v).__name__, type(default).__name__, default)
    _cache_stamp, _cache_state = stamp, dict(merged)
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


# Crowd regimes in which identity dialogs are gated. Mirrors
# speaker/identifier.py CONTINUITY_DISABLED_REGIMES (same crowd rationale,
# same legacy-value tolerance — the /api/situation whitelist is binary
# ''/'EXPO' since 2026-07-05, but a stale on-disk 'PARTY' must still gate).
IDENTITY_DIALOG_GATED_REGIMES = frozenset({"PARTY", "EXPO"})


def identity_dialogs_allowed() -> bool:
    """THE one gate for the identity-dialog class (Dan 2026-07-06).

    True when multi-turn identity dialogs (enroll latches, misID-correction
    protest, introductions name-ask, face auto-enroll consent) may start or
    continue. False under a crowd regime unless the supervised-enroll
    override is on. Gated behavior is FULLY SILENT: consumers skip their
    detectors/triggers so the utterance falls through to the LLM as ordinary
    speech — no deflection line, no signal the machinery exists.

    Read live per call (like the regime itself) so a webui flip applies to
    the very next turn without a restart. Consumers must also drop any
    dialog state armed BEFORE the flip (latches, FSM pendings) — a latch
    must not survive the flip.
    """
    if get("situation_regime") not in IDENTITY_DIALOG_GATED_REGIMES:
        return True
    return bool(get("identity_dialogs_override"))


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
