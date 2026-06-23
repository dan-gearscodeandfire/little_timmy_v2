"""Little Timmy — Voice Assistant Orchestrator.

Main event loop tying together:
  Audio capture → Speaker ID → STT → Memory retrieval → Prompt assembly → LLM → TTS → Async memory
"""

import asyncio
import logging
import sys
import os
import re
import time
import httpx
import numpy as np
from collections import deque
from persistence import runtime_toggles

import config
import eye_led
import json as _json
from pathlib import Path
from db import migrate
from db.connection import get_pool, close_pool
from audio.capture import AudioCapture
from stt.client import transcribe
from tts.engine import TTSEngine
from audio import fillers as audio_fillers
from llm.client import stream_conversation, set_reasoning_tap
from llm.prompt_builder import build_ephemeral_block, build_messages, build_proactive_messages
from memory.retrieval import retrieve
from memory.facts import get_all_facts_for_prompt, get_facts_about_speaker, resolve_entity


async def _empty_facts():
    """Awaitable that resolves to an empty fact list; used when there are no
    non-speaker 'my X' subjects to query so asyncio.gather still has a slot."""
    return []
from memory.extraction import extract_and_store
from feedback.detector import maybe_capture_feedback
from speaker.voice_commands import detect_reenroll_intent
from conversation.manager import ConversationManager
from conversation.turn import (
    ConversationTurn, LiveLLM, LiveMemory, SpeakerIdentity, TurnContext, TurnSettings,
)
from conversation.introductions import Introductions
from speaker.identifier import SpeakerIdentifier
from web.app import (app, init as web_init, broadcast_event, update_metrics,
                     record_turn_stats, latency_stats_snapshot)
from vision.context import VisionContext
from vision.visual_question import is_visual_question, is_self_referential_visual_question
from conversation.enroll_intent import detect_enroll_intent
from conversation import tool_router
from vision.supervisor import BehaviorSupervisor
from presence import (
    RoomLedger,
    anyone_present,
    fuse_identity,
    IdentityFusion,
    FaceHintStreak,
    LookAtPolicy,
)
from presence.face_enroller import FaceEnroller
from presence.new_face_trigger import NewFaceTrigger, TriggerConfig
from vision.face_remote import RemoteFaceClient
from presence.face_client_local import fetch_face_observation_local

logging.basicConfig(
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/little_timmy.log"),
    ],
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timmy")


# The reply-hygiene post-filter now lives in conversation.reply_filter
# (canonical copy, used inside ConversationTurn). main.py only needs these
# two symbols for the finalized-turn snapshot's sentence-cap recompute.
from conversation.reply_filter import (
    user_invites_longer_reply,
    _REPLY_LONGER_SENTENCES,
)


def speaker_allowlist_drop(name: str, allowlist) -> bool:
    """Speaker-allowlist reply-gate predicate (pure, unit-testable).

    True == drop this speaker. An empty/None allowlist allows everyone.
    Unknown speakers are always dropped when an allowlist is set — guests
    should get neither a reply nor a name prompt.

    SHELVED 2026-06-10 (Dan): speaker-ID isn't reliable enough to gate
    replies on; the call site in process_speech is commented out and the
    runtime-toggle keys were removed. Kept (with tests) for when it is.
    """
    allow = [n.lower() for n in (allowlist or [])]
    if not allow:
        return False
    n = name.lower()
    return n.startswith("unknown") or n not in allow


class Orchestrator:
    def __init__(self):
        self.conversation = ConversationManager()
        self.tts = TTSEngine(config.PIPER_MODEL)
        self.capture = AudioCapture()
        self.speaker_id_module = SpeakerIdentifier()
        self.vision = VisionContext()
        self.supervisor = BehaviorSupervisor()
        self.room_ledger = RoomLedger(
            presence_ttl_sec=config.PRESENCE_TTL_SEC,
            unknown_voice_ttl_sec=config.UNKNOWN_VOICE_TTL_SEC,
            camera_pan_fov_steps=config.CAMERA_PAN_FOV_STEPS,
            camera_tilt_fov_steps=config.CAMERA_TILT_FOV_STEPS,
            on_camera_fresh_threshold_sec=config.ON_CAMERA_FRESH_SEC,
            face_confirm_min=config.FACE_CONFIRM_MIN,
            unconfirmed_face_ttl_sec=config.UNCONFIRMED_FACE_TTL_SEC,
            face_reconfirm_gap_sec=config.FACE_RECONFIRM_GAP_SEC,
            save_path=config.LEDGER_SAVE_PATH,
        )
        self._face_http = httpx.AsyncClient(verify=False, timeout=1.5)
        self._presence_enabled = config.PRESENCE_ENABLED
        self._face_hint_streak = FaceHintStreak(
            threshold=config.FACE_HINT_AUTO_ENROLL_TURNS,
        )
        self._look_at_policy = LookAtPolicy(
            cooldown_sec=config.LOOK_AT_COOLDOWN_SEC,
            max_pose_age_sec=config.LOOK_AT_MAX_POSE_AGE_SEC,
            fresh_face_age_sec=config.LOOK_AT_FRESH_FACE_AGE_SEC,
        )
        self._look_at_enabled = config.LOOK_AT_ENABLED
        # Slice B (DARK): stateful identity fusion (symmetric voice-stabilize +
        # temporal hold). Reads its toggles live; all default-OFF so resolve()
        # is identical to today's fuse_identity() until Dan enables it.
        self._identity_fusion = IdentityFusion()
        # Mutual-exclusion for ALL spoken turns (reactive + proactive). A
        # reactive turn (process_speech / process_text_input) holds this for
        # its whole duration; the proactive path try-acquires non-blocking and
        # drops if a turn is in flight, so it never talks over the user.
        self._turn_lock = asyncio.Lock()
        # Proactive-speech debounce state.
        self._last_proactive_time = 0.0
        self._proactive_times: deque[float] = deque(maxlen=16)  # for per-minute rate cap

        # The deep module that owns a conversation turn (retrieve -> prompt ->
        # stream -> filter -> per-sentence TTS -> save). The Orchestrator is now
        # the doorway: it identifies the speaker, handles presence side-effects,
        # then delegates the turn here. See CONTEXT.md. Collaborators are the
        # real adapters; tests build ConversationTurn with fakes instead.
        self._turn = ConversationTurn(
            speaker=self.tts,
            llm=LiveLLM(),
            memory=LiveMemory(top_k=config.RETRIEVAL_TOP_K),
            history=self.conversation,
            settings=TurnSettings.from_config(),
            on_event=broadcast_event,
        )

        # The "what's your name?" sub-dialog owns its own cross-turn state and
        # speaks via the turn's say(). The doorway consults it each turn.
        self._introductions = Introductions(
            speaker_id_module=self.speaker_id_module,
            turn=self._turn,
        )

        # Interactive auto-enrollment for *faces* (distinct from FaceHintStreak,
        # which is voiceprint->known-face binding). Owns the consent/name/guided-
        # capture/verify FSM; consulted from the conversation doorway (handle) and
        # from a dedicated /faces poll loop (observe_faces). Default OFF until armed
        # via TIMMY_AUTO_ENROLL_ENABLED. See presence/face_enroller.py.
        self._faces_client = RemoteFaceClient(max_age_s=2.0)
        self._last_unknown_speech_ts = 0.0  # engagement signal: last unknown-voice turn
        self._last_speech_ts = 0.0          # any-voice turn (test-only engagement relax)
        self._face_enroller = FaceEnroller(
            say=self._turn.say,
            speak=self.tts.speak,
            enroll_stream=self._enroll_stream,
            verify_faces=self._faces_client.fetch_fresh_results,
            turn_lock=self._turn_lock,
            on_enrolled=self._record_auto_enroll,
            hold_head=self._hold_head_for_enroll,
            record_action=self.conversation.add_system_action_turn,
            # C5 knobs (enroll_candidate_min_span_s / _min_dist) read live from
            # runtime_toggles so LT-OS sliders take effect without a restart.
            trigger=NewFaceTrigger(TriggerConfig(), knobs=runtime_toggles.get),
        )

    async def _fetch_face_safe(self):
        # Wrapper that never raises; returns None on any failure or timeout.
        # Uses in-tree FaceID via face_client_local (no HTTP to :8895).
        if not self._presence_enabled:
            return None
        try:
            return await asyncio.wait_for(
                fetch_face_observation_local(
                    self.vision,
                    self._face_http,
                    config.STREAMERPI_BEHAVIOR_URL,
                    timeout_sec=1.5,
                ),
                timeout=1.5,
            )
        except Exception as e:
            log.info("[PRESENCE] face fetch failed: %s", e)
            return None

    def _on_passive_face_id(self, results, image_size):
        """Callback from VisionContext: feed periodic face-id hits into the room ledger."""
        if not self._presence_enabled or not results:
            return
        try:
            from presence.types import FacePrediction, FaceObservation
            preds = []
            for r in results:
                name = r.get("name", "")
                conf_label = r.get("confidence", "")
                # Only feed enrolled high/medium-confidence matches into the ledger.
                # Low/unknown confidence detections are too noisy to track per-frame
                # and would accumulate as unknown_face_<hash> records.
                if conf_label not in ("high", "medium"):
                    continue
                if not name or name.lower().startswith("unknown") or name == "unidentified person":
                    continue
                bbox_xywh = r.get("bbox", [0, 0, 0, 0])
                x, y, w, h = bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3]
                bbox = (int(x), int(y), int(x + w), int(y + h))
                distance = float(r.get("distance", 1.0))
                conf = max(0.0, 1.0 - distance)
                preds.append(FacePrediction(
                    user_id=name,
                    confidence=conf,
                    bbox=bbox,
                    embedding_hash=None,
                ))
            if not preds:
                return
            obs = FaceObservation(
                captured_at=time.time(),
                predictions=tuple(preds),
                behavior=None,
                image_size=image_size,
            )
            self.room_ledger.update_from_face(obs)
        except Exception:
            log.exception("[PRESENCE] passive face callback failed")

    async def _fire_look_at(self, name: str, target_pan: float, target_tilt: float) -> None:
        # Send /servo/move to streamerpi to point head at target pose. Fire-and-forget.
        try:
            r = await self._face_http.post(
                config.STREAMERPI_SERVO_MOVE_URL,
                json={
                    "pan": float(target_pan),
                    "tilt": float(target_tilt),
                    "speed": float(config.LOOK_AT_SPEED),
                },
                timeout=2.0,
            )
            if r.status_code == 200:
                log.info("[PRESENCE] look_at: %s -> pan=%.1f tilt=%.1f",
                         name, target_pan, target_tilt)
            else:
                log.warning("[PRESENCE] look_at HTTP %d for %s", r.status_code, name)
        except Exception as e:
            log.info("[PRESENCE] look_at failed for %s: %s", name, e)

    async def _handle_enrollment(self, name: str, used_speaker_fallback: bool) -> None:
        """Voice-triggered face enrollment.

        Speaks acknowledgment, calls streamerpi /face_db/enroll over HTTP,
        speaks the result. Returns when TTS finishes; the caller is expected
        to early-return from process_speech to skip the normal LLM/memory path.
        """
        log.info("[ENROLL] voice-triggered for '%s' (speaker_fallback=%s)",
                 name, used_speaker_fallback)
        await self.tts.speak(
            f"Sure thing. Hold still and look at me for about ten seconds, {name}."
        )

        try:
            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                resp = await client.post(
                    config.STREAMERPI_FACE_ENROLL_URL,
                    json={"name": name, "count": 15, "interval_s": 0.7},
                )
                try:
                    data = resp.json()
                except Exception:
                    data = {"error": resp.text[:200]}
        except Exception:
            log.exception("[ENROLL] HTTP call failed")
            await self.tts.speak(
                "Sorry, something went wrong with my camera. Try again later."
            )
            return

        if resp.status_code == 200 and data.get("saved"):
            captured = data.get("samples_captured", 0)
            skipped = data.get("samples_skipped", 0)
            log.info("[ENROLL] saved %s (captured=%d, skipped=%d, total=%d)",
                     name, captured, skipped, len(data.get("enrolled", [])))
            await self.tts.speak(f"Got it. I'll remember you, {name}.")
        else:
            err = data.get("error", "I couldn't get a clear look at your face.")
            log.warning("[ENROLL] failed for %s: %s", name, err)
            await self.tts.speak(
                f"Sorry, I couldn't get a clear look. Try again with better lighting?"
            )

    async def _enroll_stream(self, name: str, count: int, interval_s: float, mode: str):
        """Async generator over streamerpi's SSE /face_db/enroll/stream.

        Yields (event_type, payload) tuples — 'started' | 'progress' | 'complete'
        | 'error' — so FaceEnroller can pace pose cues in real time and abort if
        the person leaves. Fixed enrollment (/face_db/enroll) stays the voice-
        triggered path; this streaming variant is for interactive auto-enroll."""
        payload = {"name": name, "count": count, "interval_s": interval_s, "mode": mode}
        timeout = httpx.Timeout(60.0, connect=5.0)
        async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
            async with client.stream(
                "POST", config.STREAMERPI_FACE_ENROLL_STREAM_URL, json=payload
            ) as resp:
                resp.raise_for_status()
                event_type = None
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        try:
                            data = _json.loads(line[5:].strip())
                        except Exception:
                            data = {}
                        if event_type:
                            yield (event_type, data)
                            event_type = None
                    elif not line.strip():
                        event_type = None

    async def _hold_head_for_enroll(self, timeout_ms: int) -> bool:
        """Freeze-frame seam for the auto-enroller (FaceEnroller._maybe_hold).

        Leases a Pi behavior 'hold' so servo moves stop starving the face
        thread's detection passes (it skips passes while servos move; measured
        21s blackout 2026-06-10). Priority must be critical: track/engage run
        at HIGH and an equal-priority command only queues. Never holds over
        sleep — check status first and back off. Returns True if posted."""
        try:
            r = await self._face_http.get(config.STREAMERPI_BEHAVIOR_URL)
            if r.status_code != 200 or r.json().get("mode") == "sleep":
                return False
        except Exception:
            return False  # can't see the Pi -> don't push servo commands blind
        try:
            r = await self._face_http.post(
                config.STREAMERPI_BEHAVIOR_MODE_URL,
                json={"mode": "hold", "priority": "critical",
                      "timeout_ms": int(timeout_ms)},
            )
            return r.status_code == 200
        except Exception:
            log.debug("[AUTOENROLL] behavior hold post failed", exc_info=True)
            return False

    def _record_auto_enroll(self, name: str, meta: dict) -> None:
        """Append an auto-enrolled identity to the provenance file (source:auto)
        for audit / pruning / a future 'forget me' command. Best-effort."""
        path = config.FACE_ENROLL_PROVENANCE_PATH
        try:
            records = []
            if os.path.exists(path):
                try:
                    records = _json.loads(open(path).read()).get("records", [])
                except Exception:
                    records = []
            records.append({"name": name, **meta})
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(_json.dumps({"version": 1, "records": records}, indent=2))
            os.replace(tmp, path)
            log.info("[AUTOENROLL] provenance recorded: %s (%s)", name, meta.get("source"))
        except Exception:
            log.exception("[AUTOENROLL] provenance write failed for %s", name)

    async def process_speech(self, audio: np.ndarray):
        """Process a speech segment through the full pipeline."""
        t_start = time.time()

        # --- Speaker Identification (runs on raw 16kHz audio) ---
        t_spk = time.time()
        # Run speaker ID in thread to not block event loop
        speaker_result = await asyncio.to_thread(
            self.speaker_id_module.identify, audio, ""
        )
        spk_ms = int((time.time() - t_spk) * 1000)

        # Discard Timmy's own voice
        if speaker_result.is_timmy:
            log.debug("Discarding Timmy's own voice (dist confidence=%.2f, %dms)",
                      speaker_result.confidence, spk_ms)
            return

        # Speaker-allowlist reply gate: only allowlisted speakers get past
        # here. Sits before STT so non-allowlisted guests get neither a reply
        # nor a name-solicitation prompt, and their voices don't accumulate
        # into unknown-speaker clusters.
        # SHELVED 2026-06-10 (Dan): speaker recognition not reliable enough
        # yet to gate replies on — false rejects would silently drop Dan.
        # To re-enable: uncomment, and re-add "speaker_allowlist_enabled" +
        # "speaker_allowlist" to persistence/runtime_toggles._DEFAULTS.
        # Predicate (speaker_allowlist_drop) + tests stay live.
        # if runtime_toggles.get("speaker_allowlist_enabled") and speaker_allowlist_drop(
        #         speaker_result.name, runtime_toggles.get("speaker_allowlist")):
        #     if speaker_result.is_new or speaker_result.name.startswith("unknown_"):
        #         self.speaker_id_module.undo_last_observation(speaker_result.name)
        #     log.info("Allowlist gate: dropping non-allowlisted speaker %s (conf=%.2f)",
        #              speaker_result.name, speaker_result.confidence)
        #     return

        # --- STT ---
        t0 = time.time()
        transcription = await transcribe(audio)
        user_text = transcription.text
        stt_words = transcription.words           # per-word probs for value-confidence
        stt_ms = int((time.time() - t0) * 1000)

        if not user_text:
            # Roll back unknown speaker accumulation — this was noise, not speech
            if speaker_result.is_new or speaker_result.name.startswith("unknown_"):
                self.speaker_id_module.undo_last_observation(speaker_result.name)
            log.debug("Empty transcription, skipping")
            return

        # Eye LED: signal thinking state — eye flashes during LLM processing
        asyncio.create_task(eye_led.notify("THINKING"))

        # Store transcribed text on unknown speaker for name solicitation prompt
        if speaker_result.is_new or speaker_result.name.startswith("unknown_"):
            self.speaker_id_module.set_last_text(speaker_result.name, user_text)
            # Engagement signal for face auto-enroll: an unrecognised voice is
            # actively talking to LT right now. The face-poll loop pairs this
            # with a new-face CANDIDATE before offering to remember a face.
            self._last_unknown_speech_ts = time.time()

        speaker_name = speaker_result.name
        speaker_db_id = speaker_result.speaker_id
        self._last_speech_ts = time.time()  # any-voice engagement (test relax)

        # No vision capture is kicked here (or at speech-onset anymore, removed
        # 2026-06-23 for GPU contention -- see main()). The reply uses the
        # cached scene from the gated periodic poll; visual questions force
        # their own fresh capture downstream.

        # Notify behavioral supervisor of speech
        asyncio.create_task(self.supervisor.on_speech_detected(speaker_name))

        log.info("[%s] %s (STT: %dms, SPK: %dms)",
                 speaker_name.upper(), user_text, stt_ms, spk_ms)
        await broadcast_event("turn", {
            "role": "user",
            "content": user_text,
            "speaker": speaker_name,
        })
        await self.conversation.add_user_turn(user_text, speaker=speaker_name)

        # A face auto-enroll consent dialog in flight OWNS this turn. Route it to
        # the FSM BEFORE the legacy voice enroll-intent / name-ask below, which
        # would otherwise hijack natural consent phrases. STT audits out a bare
        # "yes", so consent replies are necessarily long ("yes, you can remember
        # my face") and keyword-laden — exactly what trips detect_enroll_intent.
        if self._face_enroller.awaiting:
            ae = await self._face_enroller.handle(user_text, speaker_name)
            if ae.handled:
                return

        # voice-enroll-shortcut
        enroll = detect_enroll_intent(user_text, speaker_name)
        if enroll.matched:
            await self._handle_enrollment(enroll.name, enroll.used_speaker_fallback)
            return

        # --- Handle name solicitation for unknown speakers ---
        if speaker_result.should_ask_name:
            unknown_info = self.speaker_id_module.get_unknown_for_name_ask(
                speaker_result.name
            )
            if unknown_info:
                self.speaker_id_module.mark_name_asked(speaker_result.name)
                await self._introductions.ask_name(unknown_info)
                return

        # Speculative coref (2026-06-22, "speculative_coref_enabled", default OFF):
        # kick the resolver (:8093) off NOW so it overlaps the classifier (:8092)
        # below instead of running serially INSIDE retrieve() after the classifier
        # returns -- hiding up to min(classifier, resolver) (~150ms measured) on
        # deictic-brain-path turns. resolve_for_retrieval is self-gating
        # (query_resolution_enabled + _needs_resolution), so it returns None
        # cheaply on non-deictic turns; we launch the task regardless so even the
        # gate check stays off the critical path. The window matches _gather's
        # (the WIDER of the two). Cancelled on a classifier hit (a wasted :8093
        # call on the minority tool-hit turns). Toggle OFF -> task is None and
        # retrieve() resolves inline, byte-identical to the pre-2026-06-22 path.
        coref_task = None
        if runtime_toggles.get("speculative_coref_enabled"):
            from memory.retrieval import resolve_for_retrieval
            _resolve_ctx = self.conversation.recent_turns_excluding_current(
                max(config.CONTEXT_TURNS, config.RESOLVE_CONTEXT_TURNS))
            coref_task = asyncio.create_task(
                resolve_for_retrieval(user_text, _resolve_ctx))

        # First-pass tool-call classifier (gated by runtime_toggles
        # "classifier_enabled", default OFF). On a recognized intent it executes
        # the tool, speaks the canned ACK, injects the assistant turn, and OWNS
        # this turn -> early return, skipping the brain. Any non-tool utterance
        # (or a classifier-server outage) returns False and falls through to the
        # normal pipeline unchanged. The user turn was already added/broadcast above.
        outcome = await tool_router.maybe_handle_tool_call(
                user_text, speaker_name, speaker_db_id, self.conversation, self.tts,
                t_start=t_start, stt_words=stt_words)
        if outcome.handled:
            if coref_task is not None:
                coref_task.cancel()  # tool turn owns the reply; resolved query unused
            return

        # Collect the speculative resolution (already running concurrently with the
        # classifier above). resolve_query swallows its own errors -> None, so this
        # await is effectively non-throwing; the guard is belt-and-suspenders.
        # query_pre_resolved=True tells retrieve() the doorway owns resolution --
        # don't call :8093 again (None -> fall back to the embedding blend).
        resolved_query, query_pre_resolved = None, False
        if coref_task is not None:
            try:
                resolved_query = await coref_task
            except Exception:
                log.debug("[RESOLVE] speculative coref failed (non-fatal)", exc_info=True)
            query_pre_resolved = True

        await self._generate_response(
            user_text, stt_ms, t_start,
            speaker_name=speaker_name,
            speaker_db_id=speaker_db_id,
            spk_ms=spk_ms,
            voice_confidence=speaker_result.confidence,
            recall_block=outcome.recall_block,
            stt_words=stt_words,
            resolved_query=resolved_query,
            query_pre_resolved=query_pre_resolved,
        )

    # Snapshot of the most-recent finalized turn so /api/feedback/manual_flag
    # can capture the same (user, assistant, full system prompt) tuple the
    # verbal-feedback path captures.
    _last_finalized_turn: dict | None = None

    async def process_text_input(self, text: str):
        """Process typed text input (from web UI)."""
        # Hold the turn lock so a proactive remark can't overlap this turn.
        async with self._turn_lock:
            t_start = time.time()
            log.info("[USER:text] %s", text)
            await broadcast_event("turn", {"role": "user", "content": text})
            await self.conversation.add_user_turn(text)
            # First-pass tool-call classifier on the text path too (default OFF).
            # speaker is the operator/Dan on the typed path (matches the
            # _generate_response defaults below).
            outcome = await tool_router.maybe_handle_tool_call(
                    text, "dan", 1, self.conversation, self.tts, t_start=t_start)
            if outcome.handled:
                return
            await self._generate_response(text, stt_ms=0, t_start=t_start,
                                           speaker_name="dan", speaker_db_id=1, spk_ms=0,
                                           recall_block=outcome.recall_block)

    async def maybe_speak_proactively(self, record, is_new_arrival: bool) -> None:
        """Maybe emit a short unprompted remark in reaction to a visual event.

        Called from vision_people_monitor every ~2s with the current scene
        record and whether new people just appeared (rising edge). Heavily
        gated: master switch + runtime toggle + hearing on + urgency/speak_now
        + rising edge + cooldown + per-minute cap + turn-lock (drops, never
        queues, if a reactive turn is in flight). Generates via the same
        LLM->TTS path as a reactive turn, so echo suppression and the
        conversation-priority gate are inherited.
        """
        # --- cheap gates first ---
        if not config.PROACTIVE_SPEECH_ENABLED:
            return
        if not runtime_toggles.get("proactive_speech_enabled"):
            return
        if self.capture.hearing_muted:
            return  # a muted Timmy that still talks is surprising
        # Turn-taking / barge-in guard. The reactive _turn_lock (acquired in the
        # main loop only when a *finalized* segment lands on speech_queue) does
        # NOT cover an in-progress utterance, so without this the proactive path
        # talks right over the user mid-sentence. Bail if they're speaking now,
        # or spoke within the grace window (covers the finalize->turn-lock handoff
        # gap and natural mid-thought pauses VAD may endpoint). See config knob.
        if self.capture.user_speaking:
            return
        if time.time() - self.capture.last_voice_ts < config.PROACTIVE_USER_SPEECH_GRACE_SEC:
            return
        if record is None:
            return

        # Occupancy gate (theme D): don't monologue into an empty room. The room
        # ledger is the authoritative TTL-windowed presence signal; proactive
        # otherwise fires off VLM person-detection flapping (is_new_arrival) even
        # when nobody is actually present -- 89 remarks into an empty workshop
        # overnight (2026-06-13) while the ledger had aged everyone out. When
        # presence tracking is on and the ledger shows no one, stay silent.
        # (Presence off -> we can't tell -> preserve prior behaviour.) Snapshot
        # is reused below to flavour the prompt, so we take it once here.
        presence_state = (
            self.room_ledger.current_state() if self._presence_enabled else None
        )
        if self._presence_enabled and not anyone_present(presence_state):
            return

        relevance = self.vision.get_last_relevance()
        urgent = bool(record.speak_now) or (
            relevance is not None
            and relevance.urgency_score >= config.PROACTIVE_URGENCY_THRESHOLD
        )
        # Rising edge only: a fresh arrival, or a newly-urgent scene. Sustained
        # urgency must not re-fire (the cooldown is the backstop, this is the
        # primary debounce).
        if not (is_new_arrival or urgent):
            return

        now = time.time()
        if now - self._last_proactive_time < config.PROACTIVE_COOLDOWN_SEC:
            return
        # Per-minute hard cap (belt + suspenders over the cooldown).
        recent = [t for t in self._proactive_times if now - t < 60.0]
        if len(recent) >= config.PROACTIVE_MAX_PER_MIN:
            return

        # --- turn-lock: drop (don't queue) if a reactive turn is in flight ---
        # No await between this check and acquire(), so in asyncio's single
        # thread the lock can't be taken in between -- acquire() returns
        # immediately without blocking. We never queue: a stale "someone
        # entered" remark must not fire late behind a real conversation.
        if self._turn_lock.locked():
            return
        await self._turn_lock.acquire()

        # Record cooldown immediately so a long generation can't let a second
        # trigger slip through before we update it.
        self._last_proactive_time = now
        self._proactive_times.append(now)

        self.vision.pause_polling()  # free the GPU while we stream (mirrors main loop)
        try:
            # presence_state was snapshotted at the occupancy gate above.
            ephemeral = build_ephemeral_block(
                memories=[],
                facts=[],
                vision_description=self.vision.get_description(),
                visual_question=False,
                presence_state=presence_state,
            )

            asyncio.create_task(self.supervisor.on_tts_start())
            asyncio.create_task(eye_led.notify("SPEAKING"))

            # speak_proactively owns the LLM->TTS engine and — only if something
            # was actually said — the proactive "turn" broadcast and the
            # assistant-turn persistence. There is no user turn.
            result = await self._turn.speak_proactively(ephemeral)

            asyncio.create_task(self.supervisor.on_tts_end())
            asyncio.create_task(eye_led.notify("AI_CONNECTED"))

            if result.text and result.text.strip():
                log.info("[PROACTIVE] %s", result.text)
        except Exception:
            log.exception("[PROACTIVE] generation failed")
        finally:
            self.vision.resume_polling()
            self._turn_lock.release()

    async def _generate_response(self, user_text: str, stt_ms: int, t_start: float,
                                  speaker_name: str = "dan",
                                  speaker_db_id: int | None = 1,
                                  spk_ms: int = 0,
                                  voice_confidence: float | None = None,
                                  recall_block: str | None = None,
                                  stt_words: list | None = None,
                                  resolved_query: str | None = None,
                                  query_pre_resolved: bool = False):
        """Core response pipeline: presence doorway → delegate to the turn.

        `resolved_query`/`query_pre_resolved` carry the doorway's speculative coref
        result (resolved in parallel with the classifier) into the turn's
        retrieval. Both stay at their OFF defaults on the text path."""

        # --- Name-confirmation sub-dialog (Introductions owns the state) ---
        # If we're mid-introduction, this may speak a follow-up (handled=True ->
        # we're done) or promote the speaker to a just-confirmed name and fall
        # through to a normal turn.
        intro = await self._introductions.handle(user_text, speaker_name)
        if intro.handled:
            return
        speaker_name = intro.speaker_name

        # --- Face auto-enroll consent dialog (FaceEnroller owns the state) ---
        # If we're mid-offer ("mind if I remember your face?"), this consumes the
        # yes/no/name reply and speaks the follow-up; a no-op when not in flight.
        enroll_outcome = await self._face_enroller.handle(user_text, speaker_name)
        if enroll_outcome.handled:
            return

        # --- Presence face fetch (doorway) ---
        # Memory retrieval (memories + facts, with coreference + "my X" subject
        # scoping and fact dedup) now lives inside ConversationTurn. The doorway
        # only fetches the face observation that identity fusion needs below;
        # _fetch_face_safe returns None when presence is disabled.
        face_obs = await self._fetch_face_safe()

        # --- Presence: voice + face fusion ---
        fusion_source = None
        face_hint_name = None
        presence_state = None
        if self._presence_enabled:
            self.room_ledger.update_from_voice(speaker_name)
            if face_obs is not None:
                self.room_ledger.update_from_face(face_obs)
            # Slice B: stateful resolve() (symmetric/temporal fusion). All
            # toggles default OFF -> identical to fuse_identity() today. Threads
            # voice confidence so a sure voice can stabilize an absent face.
            verdict = self._identity_fusion.resolve(
                voice_name=speaker_name,
                voice_is_unknown=speaker_name.startswith("unknown_"),
                face=face_obs,
                voice_confidence=voice_confidence,
                face_conf_threshold=config.FACE_CONF_THRESHOLD,
                head_steady_min_ms=config.HEAD_STEADY_MS,
            )
            # Track face_hint streak for auto voice-enrollment.
            # Use pre-override speaker_name (the unknown_N temp_id).
            streak_temp_id = speaker_name if speaker_name.startswith("unknown_") else None
            # Gate on face_hint_source=='face': a synthesized/held hint (Slice B
            # 'voice'/'temporal') must NEVER train a voiceprint — that's the
            # "calls everyone Dan" corruption vector.
            streak_face_name = (
                verdict.face_hint_name
                if (verdict.resolution_source == "face_hint"
                    and verdict.face_hint_source == "face")
                else None
            )
            streak_hit = self._face_hint_streak.observe(streak_face_name, streak_temp_id)
            if streak_hit is not None:
                if config.AUTO_ENROLL_KILL:
                    # Emergency kill switch: suppress voiceprint auto-enroll. A
                    # crowd makes face_hint streaks unreliable (false-accepts),
                    # and binding a voiceprint off a bad streak corrupts speaker
                    # attribution at scale. Still reset so the streak doesn't
                    # accumulate stale across turns.
                    log.info(
                        "[PRESENCE] auto-enroll SUPPRESSED (AUTO_ENROLL_KILL): %s -> %s (%d-turn streak)",
                        streak_hit.voice_temp_id, streak_hit.face_hint_name, streak_hit.count,
                    )
                else:
                    ok = self.speaker_id_module.assign_name(
                        streak_hit.voice_temp_id, streak_hit.face_hint_name
                    )
                    if ok:
                        log.info(
                            "[PRESENCE] AUTO-ENROLL: voiceprint for %s trained from %d-turn face_hint streak (was %s)",
                            streak_hit.face_hint_name, streak_hit.count, streak_hit.voice_temp_id,
                        )
                    else:
                        log.warning(
                            "[PRESENCE] auto-enroll declined for %s -> %s (name reserved or taken)",
                            streak_hit.voice_temp_id, streak_hit.face_hint_name,
                        )
                self._face_hint_streak.reset()

            # Look-at-speaker: voice-confident off-camera speaker w/ fresh pose -> pan head
            if (
                self._look_at_enabled
                and verdict.resolution_source == "voice"
                and not speaker_name.startswith("unknown_")
            ):
                _present_state = self.room_ledger.current_state()
                _present_record = next(
                    (p for p in _present_state["present"] if p["name"] == speaker_name),
                    None,
                )
                _last_pose = self.room_ledger.find_pose_for(speaker_name)
                _beh_mode = (
                    face_obs.behavior.mode if face_obs and face_obs.behavior else None
                )
                _now = time.time()
                _v = self._look_at_policy.evaluate(
                    speaker_name, _present_record, _last_pose, _beh_mode, _now,
                )
                if _v.should_look:
                    self._look_at_policy.record_look_at(speaker_name, _now)
                    asyncio.create_task(
                        self._fire_look_at(speaker_name, _v.target_pan, _v.target_tilt)
                    )
            if verdict.resolution_source == "face_hint":
                log.info(
                    "[PRESENCE] face_hint promoted: voice=%s -> %s (face_conf=%.2f)",
                    speaker_name, verdict.face_hint_name, verdict.face_hint_confidence or 0.0,
                )
                speaker_name = verdict.final_name
            fusion_source = verdict.resolution_source
            face_hint_name = verdict.face_hint_name
            presence_state = self.room_ledger.current_state()

        # --- Vision context for the prompt (doorway-resolved, passed in) ---
        # Self-referential visual questions ("what's on my shoulder?", "how do
        # I look?") are genuine visual questions that is_visual_question misses
        # (the C6 utterance was one), so fold them in -- otherwise they'd take
        # the background-awareness vision branch and confabulate.
        self_ref_q = is_self_referential_visual_question(user_text)
        visual_q = is_visual_question(user_text) or self_ref_q
        subject_not_in_view = False

        # Block-on-fresh (2026-06-07): a visual question about a just-presented
        # object can't be answered from a cached frame predating the gesture.
        # Historically we never awaited a capture here because the HIGH_RES path
        # cost ~9s e2e; that path is retired and LOW_RES runs ~2-4s. The
        # background speech-onset capture also races this turn and loses (it
        # logged "teal water bottle" microseconds after we snapshotted the stale
        # "empty hands" record -> confabulation). So when the question is visual
        # and the cached frame is stale, await our own fresh capture first; the
        # description AND the averted-gaze guard below then read that frame.
        if visual_q:
            age = self.vision.scene_age()
            if age is None or age > config.VISION_VISUAL_Q_MAX_AGE_S:
                log.info(
                    "[VISION] visual question + stale frame (age=%s) "
                    "-> blocking on fresh capture before answering",
                    f"{age:.1f}s" if age is not None else "none",
                )
                await self.vision.trigger_capture("visual_question")
            # Direct visual question: answer from the raw frame, bypassing the
            # relevance filter (which only gates UNSOLICITED observations). Using
            # the filtered get_description() here returns None for low-novelty
            # scenes -> no [WHAT YOU SEE] block -> the brain confabulates.
            vision_desc = self.vision.get_raw_description()
        else:
            vision_desc = self.vision.get_description()
        if visual_q:
            log.info("[VISION] Visual question detected (frame age=%s)",
                     f"{self.vision.scene_age():.1f}s"
                     if self.vision.scene_age() is not None else "none")
            # Averted-gaze guard (C6): self-referential visual questions
            # presuppose the user is in frame. If the frame we'd answer from
            # contains no person AND streamerpi reports no live face, the head is
            # aimed away -- deflect honestly instead of confabulating, and fire a
            # delayed background recapture so the next turn answers from an aimed
            # frame (look-at pans the head toward the off-camera voice meanwhile).
            if config.VISION_AVERTED_GAZE_GUARD and self_ref_q:
                rec = self.vision.get_scene_record()
                frame_has_person = bool(rec and rec.people)
                face_live = (
                    face_obs.behavior.face_visible
                    if face_obs and face_obs.behavior else None
                )
                if not frame_has_person and face_live is not True:
                    subject_not_in_view = True
                    log.info(
                        "[VISION] averted-gaze guard: self-ref visual Q %r but "
                        "subject not in view (frame_people=%s, face_visible=%s) "
                        "-> deflecting + background recapture",
                        user_text[:60], rec.people if rec else None, face_live,
                    )

                    async def _delayed_recapture():
                        # Let the look-at pan land before grabbing a fresh frame.
                        await asyncio.sleep(config.VISION_RECAPTURE_DELAY_S)
                        await self.vision.trigger_capture("visual_question_recapture")
                    asyncio.create_task(_delayed_recapture())

        # Filler-word gate: queue a short pre-rendered filler ahead of the real
        # reply so the user hears Timmy start immediately. Fired before the turn
        # streams so it queues naturally ahead of the first real sentence.
        if audio_fillers.should_fire(user_text):
            asyncio.create_task(self.tts.speak_filler(audio_fillers.pick()))

        async def _on_first_sentence():
            # Eye LED + supervisor cues fire the moment the first sentence is
            # about to hit TTS. Fire-and-forget so they don't delay speak().
            asyncio.create_task(self.supervisor.on_tts_start())
            asyncio.create_task(eye_led.notify("SPEAKING"))

        # --- Delegate the turn to the ConversationTurn module ---
        # It owns: memory retrieval (+ the "retrieval" broadcast), prompt
        # assembly (vision + presence passed in), LLM stream, narration/length
        # filter, per-sentence TTS, the "turn" broadcast, assistant-turn
        # persistence, and the memory save. The sentence cap is computed inside
        # from user_invites_longer_reply(user_text).
        result = await self._turn.respond(
            user_text,
            SpeakerIdentity(name=speaker_name, db_id=speaker_db_id),
            TurnContext(
                stt_ms=stt_ms, spk_ms=spk_ms, t_start=t_start,
                vision_description=vision_desc, visual_question=visual_q,
                subject_not_in_view=subject_not_in_view,
                presence_state=presence_state, fusion_source=fusion_source,
                face_hint_name=face_hint_name,
                # Slice A: live regime knob, read once here per turn (re-reads
                # disk so an LT-OS change takes effect without restart).
                situation_regime=runtime_toggles.get("situation_regime"),
                recall_block=recall_block,
                on_first_sentence=_on_first_sentence,
                stt_words=stt_words,
                resolved_query=resolved_query,
                query_pre_resolved=query_pre_resolved,
            ),
        )
        full_response = result.text
        ephemeral = result.ephemeral
        messages = result.messages

        # Notify supervisor that TTS is done + Eye LED back to listening.
        asyncio.create_task(self.supervisor.on_tts_end())
        asyncio.create_task(eye_led.notify("AI_CONNECTED"))

        llm_first_token_ms = result.timings["first_token_ms"] or 0
        llm_total_ms = result.timings["total_ms"]
        tts_ms = result.timings["first_tts_ms"] or 0
        retrieval_ms = result.retrieval_ms
        e2e_ms = int((time.time() - t_start) * 1000)

        log.info("[TIMMY] %s", full_response)
        log.info("[PERF] stt=%dms spk=%dms retrieval=%dms llm_ft=%dms llm=%dms tts=%dms e2e=%dms",
                 stt_ms, spk_ms, retrieval_ms, llm_first_token_ms, llm_total_ms, tts_ms, e2e_ms)

        # The turn already broadcast the assistant "turn" event and persisted
        # the assistant turn; the doorway only assembles the metrics report.
        # est_prompt/completion tokens (~4 chars/token) come back on the result.
        est_prompt_tokens = result.est_prompt_tokens
        est_completion_tokens = result.est_completion_tokens

        await broadcast_event("metrics", {
            "stt_ms": stt_ms,
            "spk_ms": spk_ms,
            "retrieval_ms": retrieval_ms,
            "llm_first_token_ms": llm_first_token_ms,
            "llm_total_ms": llm_total_ms,
            "tts_ms": tts_ms,
            "e2e_ms": e2e_ms,
            "turns": self.conversation.turn_count,
            "speaker": speaker_name,
            "est_prompt_tokens": est_prompt_tokens,
            "est_completion_tokens": est_completion_tokens,
        })
        update_metrics(
            last_stt_ms=stt_ms,
            last_retrieval_ms=retrieval_ms,
            last_llm_first_token_ms=llm_first_token_ms,
            last_llm_total_ms=llm_total_ms,
            last_tts_ms=tts_ms,
            last_e2e_ms=e2e_ms,
            turns=self.conversation.turn_count,
            last_est_prompt_tokens=est_prompt_tokens,
            last_est_completion_tokens=est_completion_tokens,
        )
        # Rolling distributions for the "conversation" (full-brain) turn class.
        # Tool-routed turns record themselves in tool_router (early-return path);
        # classifier_route / resolution land in their own cross-cutting series.
        record_turn_stats("conversation", {
            "stt": stt_ms,
            "retrieval": retrieval_ms,
            "llm_ft": llm_first_token_ms,
            "llm_total": llm_total_ms,
            "tts": tts_ms,
            "e2e": e2e_ms,
        }, flags={"long_reply": user_invites_longer_reply(user_text)})
        # Periodic aggregate so the distribution is greppable in the log, not only
        # via /api/latency_stats. Every 10th turn keeps it cheap and skimmable.
        if self.conversation.turn_count % 10 == 0:
            snap = latency_stats_snapshot()
            ce2e = snap["series"].get("conversation:e2e", {})
            cft = snap["series"].get("conversation:llm_ft", {})
            cr = snap["series"].get("stage:classifier_route", {})
            te2e = snap["series"].get("tool:e2e", {})
            log.info("[PERF-AGG] turns=%s | conv_e2e p50=%s p95=%s (n=%s) | "
                     "conv_llm_ft p50=%s p95=%s | classifier_route p50=%s p95=%s (n=%s) | "
                     "tool_e2e p50=%s (n=%s)",
                     snap["counts"].get("turns"),
                     ce2e.get("p50"), ce2e.get("p95"), ce2e.get("n"),
                     cft.get("p50"), cft.get("p95"),
                     cr.get("p50"), cr.get("p95"), cr.get("n"),
                     te2e.get("p50"), te2e.get("n"))

        # (the assistant turn was already persisted inside ConversationTurn)

        # Recompute the sentence cap for the finalized-turn snapshot below; the
        # turn computes it internally, mirror it here for the LoRA payload.
        cap = _REPLY_LONGER_SENTENCES if user_invites_longer_reply(user_text) else None

        # --- Compliment detection for persona tuning (fire-and-forget) ---
        asyncio.create_task(
            self._check_compliment(user_text, full_response, ephemeral, messages)
        )

        # --- Meta-feedback capture for Claude-Code-side review (fire-and-forget) ---
        # Snapshot the just-finalized turn so the manual-flag endpoint can
        # read the exact (user, assistant, ephemeral) tuple the LLM saw.
        # conversation_history is a frozen copy of hot_turns taken AT
        # finalization (not at flag time) so a flag clicked 30s later still
        # surfaces the conversation the LLM actually had in context, not a
        # slice that has since rolled new turns in or out.
        self._last_finalized_turn = {
            "ts": time.time(),
            "user_text": user_text,
            "assistant_response": full_response,
            "ephemeral": ephemeral,
            "speaker_name": speaker_name,
            "conversation_history": [
                {"role": t.role, "content": t.content, "speaker": t.speaker,
                 "timestamp": t.timestamp}
                for t in self.conversation.hot_turns
            ],
            # 2026-05-15: full LLM payload for LoRA tuning. messages is the
            # exact list sent to llm-server (history + system + current user);
            # hyperparameters carries the model URL + sampling config so a
            # training pipeline can reproduce / replay the call.
            "messages": list(messages),
            "hyperparameters": {
                "conversation_url": config.LLM_CONVERSATION_URL,
                "temperature": config.CONVERSATION_TEMPERATURE,
                "max_tokens": cap if cap else config.CONVERSATION_MAX_TOKENS,
                "model_id_hint": "llama3.2-3b",
            },
        }
        asyncio.create_task(
            maybe_capture_feedback(user_text, full_response, messages, speaker_name, ephemeral)
        )

        # --- Voice-command re-enrollment (Trigger 2) -----------------------
        # Detect mid-conversation requests like "re-enroll my voice" and open
        # a 60s collection window on the speaker_id_module. Confident matches
        # during that window blend into the persisted voiceprint at finalize.
        try:
            target = detect_reenroll_intent(user_text, default_speaker=speaker_name)
            if target:
                ok = self.speaker_id_module.start_reenrollment(target, duration_s=60.0)
                if ok:
                    log.info("[T2] re-enrollment opened from voice command for %s (60s)", target)
                else:
                    log.info("[T2] re-enrollment intent detected for %r but refused", target)
        except Exception as _e:
            log.warning("[T2] voice-command detection failed: %s", _e)

        # (memory formation / extract_and_store already ran as the turn's
        # final real step inside ConversationTurn — see CONTEXT.md decision 4)

        # --- Async Mood Update (fire-and-forget) ---
        # Updates the deterministic 2-axis mood state (engagement, warmth)
        # off the hot path. Runs VADER on user_text + Ollama embedding for
        # topic-progression signal. The next turn's ephemeral block reads
        # the new state; latency cost on this turn is zero.
        try:
            from persona.updater import schedule as _schedule_mood_update
            _schedule_mood_update(user_text, full_response)
        except Exception as _e:
            log.warning("mood updater schedule failed: %s", _e)

    _COMPLIMENT_PATTERNS = {
        "good one", "nice one", "good job", "well done", "impressive",
        "that was funny", "that was good", "that was great", "love it",
        "perfect", "nailed it", "brilliant", "hilarious", "great response",
        "impressive response", "good answer", "nice answer", "attaboy",
        "proud of you", "keep it up",
    }

    async def _check_compliment(self, user_text: str, response: str,
                                ephemeral: str, messages: list[dict]):
        """Log compliment examples for future LoRA fine-tuning. File I/O
        runs in a worker thread so the response path's event loop never
        stalls on disk."""
        lower = user_text.lower().strip().rstrip(".!?,")
        is_compliment = any(p in lower for p in self._COMPLIMENT_PATTERNS)
        if not is_compliment:
            return

        log_dir = Path(os.path.expanduser("~/little_timmy/persona_tuning"))

        # Find penultimate user message from history
        prev_user = ""
        user_turns = [m for m in messages if m.get("role") == "user"]
        if len(user_turns) >= 2:
            prev_user = user_turns[-2]["content"]

        entry = {
            "timestamp": time.time(),
            "penultimate_user": prev_user,
            "system_prompt": ephemeral,
            "compliment": user_text,
            "response": response,
            # 2026-05-15: full LLM payload for LoRA. messages is the exact
            # input list the LLM saw; hyperparameters records temperature,
            # max_tokens, model_id, URL for reproducibility.
            "messages": list(messages),
            "hyperparameters": {
                "conversation_url": config.LLM_CONVERSATION_URL,
                "temperature": config.CONVERSATION_TEMPERATURE,
                "max_tokens": config.CONVERSATION_MAX_TOKENS,
                "model_id_hint": "llama3.2-3b",
            },
        }

        filename = log_dir / f"example_{int(time.time())}.json"
        try:
            await asyncio.to_thread(self._write_compliment_log, log_dir, filename, entry)
            log.info("[PERSONA] Logged compliment example: %s", filename.name)
            try:
                from feedback.storage import append_flagged
                await asyncio.to_thread(append_flagged, "good", {
                    "ts": entry["timestamp"],
                    "source": "compliment",
                    "speaker": None,
                    "user_prompt": prev_user,
                    "response": response,
                    "comment": user_text,
                    "system_prompt": ephemeral,
                    "persona_tuning_file": filename.name,
                })
            except Exception as fe:
                log.warning("append_flagged(good) failed: %s", fe)
        except Exception as e:
            log.warning("Failed to log compliment example: %s", e)

    @staticmethod
    def _write_compliment_log(log_dir: Path, filename: Path, entry: dict):
        log_dir.mkdir(exist_ok=True)
        with open(filename, "w") as f:
            _json.dump(entry, f, indent=2)


async def main():
    log.info("=== Little Timmy starting ===")

    # Initialize database
    await migrate.run()
    pool = await get_pool()
    log.info("Database connected")

    # Initialize orchestrator
    orch = Orchestrator()
    web_init(orch.conversation, orch)

    # Tap Qwen3.6 thinking-on reasoning_content into the WS fanout. Booth-display
    # /visitor's "ghost reasoning" panel renders these deltas so visitors see
    # Timmy's actual thought process from real production work (memory
    # extraction + rollup) instead of a dedicated narrator service.
    async def _broadcast_reasoning(source: str, reasoning: str, content: str) -> None:
        await broadcast_event("reasoning", {"source": source, "reasoning": reasoning})
    set_reasoning_tap(_broadcast_reasoning)

    # Load speaker voiceprints
    await asyncio.to_thread(orch.speaker_id_module.load_voiceprints)
    log.info("Speaker identification ready")

    # Start vision pipeline
    await orch.vision.start()
    orch.vision.set_passive_face_callback(orch._on_passive_face_id)
    log.info("Vision pipeline ready (enabled=%s)", orch.vision.enabled)

    # Start behavioral supervisor
    await orch.supervisor.start()
    log.info("Behavioral supervisor ready")

    # Wire TTS suppression to audio capture
    orch.tts._capture = orch.capture

    # Start TTS engine
    await orch.tts.start()
    # Fire-and-forget prewarm of the filler-word cache. ~10 calls to
    # Piper at ~200 ms each, off the hot path. The first conversational
    # turn might race the cache and fall through to live synthesis
    # (still works; just slightly slower for that one filler).
    import asyncio as _asyncio
    _asyncio.create_task(orch.tts.prewarm_fillers(audio_fillers.FILLERS))
    log.info("TTS engine ready")

    # Start audio capture
    loop = asyncio.get_running_loop()
    await orch.capture.start(loop)
    log.info("Audio capture started (always listening)")

    # Set up live transcription callback for hybrid endpointing
    def live_transcribe(audio_np):
        """Synchronous whisper.cpp call for endpointing (runs in capture thread)."""
        import io, wave
        audio_clipped = np.clip(audio_np, -1.0, 1.0)
        pcm = (audio_clipped * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{config.WHISPER_URL}/inference",
                method="POST",
            )
            boundary = "----TimmmyBoundary"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode() + buf.getvalue() + (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
                f"json\r\n"
                f"--{boundary}--\r\n"
            ).encode()
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            req.data = body
            import json as _json
            with urllib.request.urlopen(req, timeout=5) as resp:
                return _json.loads(resp.read()).get("text", "").strip()
        except Exception:
            return ""

    orch.capture.set_live_transcribe(live_transcribe)

    # speech_onset vision trigger: REMOVED 2026-06-23 (latency).
    # It used to kick a fresh VLM capture at VAD speech-onset on the premise
    # that :8084 (vision) "never contends" with the :8083 brain because they
    # are separate processes. That premise is FALSE: both 35B models share the
    # one Strix Halo Vulkan GPU and the driver serializes their compute, so a
    # concurrent VLM call roughly HALVES the brain (measured 71.6->35.9 tok/s
    # decode, 972->694 tok/s prefill). The onset trigger bypassed the poll
    # pause + scene-change gate + 10s cooldown, so it fired a full 35B VLM call
    # right on the brain's prefill + first token -- the worst moment for
    # first_token_ms. Direct A/B: pulling it cut median brain TTFT 6744->5352 ms
    # (-1.4s, -21%) on a ~4800-tok context; bigger contexts saw bigger spikes.
    # Ambient scene context still comes from the gated periodic poll (paused
    # during the turn), and genuine visual questions still force their own
    # fresh capture (trigger_capture("visual_question")). So no capability loss.
    # If ever re-added, defer it to AFTER the turn completes, never concurrent.

    # Start web server
    import uvicorn
    server_config = uvicorn.Config(
        app, host=config.WEB_HOST, port=config.WEB_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)

    web_task = asyncio.create_task(server.serve())

    # Bridge: watch vision records for people changes → supervisor
    async def vision_people_monitor():
        last_people = []
        while True:
            try:
                await asyncio.sleep(2.0)
                record = orch.vision.get_scene_record()
                if record is None:
                    if last_people:
                        await orch.supervisor.on_vision_people_changed([], was_empty=False)
                        last_people = []
                    continue
                # B3 persistence gate (P4 fix 2026-06-11): diff the debounced
                # confirmed-people set, not raw record.people -- a single-frame
                # face-ID flap ('unidentified person' while Dan turns his head)
                # must not register as a new arrival and fire proactive speech,
                # nor must a single-frame all-faces miss read as "everyone
                # left". Falls back to raw people before the first relevance
                # result (or with the gate toggled off, where they're equal).
                relevance = orch.vision.get_last_relevance()
                current_people = (relevance.confirmed_people if relevance is not None
                                  else (record.people or []))
                new_names = (set(p.lower() for p in current_people)
                             - set(p.lower() for p in last_people))
                if set(p.lower() for p in current_people) != set(p.lower() for p in last_people):
                    was_empty = len(last_people) == 0
                    await orch.supervisor.on_vision_people_changed(current_people, was_empty)
                    last_people = current_people
                # Proactive verbal reaction. Heavily gated inside (master switch
                # + runtime toggle + cooldown + turn-lock); a no-op unless
                # enabled. Rising edge = a person present now who wasn't before.
                await orch.maybe_speak_proactively(record, is_new_arrival=bool(new_names))
            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("Vision people monitor error")
                await asyncio.sleep(5.0)

    vision_monitor_task = asyncio.create_task(vision_people_monitor())

    # Dedicated fast /faces poll feeding the new-face trigger -> interactive
    # auto-enrollment. Separate from vision_people_monitor (2s, VLM-based) because
    # the trigger window needs >= MIN_SAMPLES in WINDOW_S (~0.4s cadence). Engaged
    # = an unrecognised voice spoke recently AND hearing is on AND nobody is
    # mid-utterance (don't barge in). No-op unless TIMMY_AUTO_ENROLL_ENABLED.
    async def face_enroll_monitor():
        if not orch._face_enroller.cfg.enabled:
            # cfg.enabled = TIMMY_AUTO_ENROLL_ENABLED AND NOT TIMMY_AUTO_ENROLL_KILL
            # (face_enroller.py). Name the actual reason so future debugging
            # doesn't chase an "unset" var that's really set-but-overridden.
            if os.environ.get("TIMMY_AUTO_ENROLL_KILL", "").strip().lower() in ("1", "true", "yes", "on"):
                log.info("[AUTOENROLL] disabled (overridden by TIMMY_AUTO_ENROLL_KILL)")
            else:
                log.info("[AUTOENROLL] disabled (TIMMY_AUTO_ENROLL_ENABLED unset)")
            return
        log.info("[AUTOENROLL] armed: polling /faces every %.2fs",
                 config.AUTO_ENROLL_POLL_INTERVAL_S)
        while True:
            try:
                await asyncio.sleep(config.AUTO_ENROLL_POLL_INTERVAL_S)
                full = await orch._faces_client.fetch_full()
                if full is None:
                    continue
                now = time.time()
                # Engagement: an unrecognised voice spoke recently (production), or
                # ANY voice when the test relax is set (solo single-person test).
                speech_ts = (
                    orch._last_speech_ts
                    if config.AUTO_ENROLL_ENGAGE_ANY_SPEECH
                    else orch._last_unknown_speech_ts
                )
                engaged = (
                    not orch.capture.hearing_muted
                    and not orch.capture.user_speaking
                    and (now - speech_ts) < orch._face_enroller.cfg.engagement_window_s
                )
                await orch._face_enroller.observe_faces(
                    full["faces"], full["image_size"], engaged=engaged,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("[AUTOENROLL] monitor tick error", exc_info=True)
                await asyncio.sleep(2.0)

    face_enroll_task = asyncio.create_task(face_enroll_monitor())

    log.info("=== Little Timmy ready on http://0.0.0.0:%d ===", config.WEB_PORT)

    try:
        while True:
            audio_segment = await orch.capture.speech_queue.get()
            # Hold the turn lock so an in-flight proactive remark finishes (or
            # is excluded) before this reactive turn speaks -- never overlap.
            async with orch._turn_lock:
                orch.vision.pause_polling()
                try:
                    await orch.process_speech(audio_segment)
                finally:
                    orch.vision.resume_polling()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        face_enroll_task.cancel()
        await orch.supervisor.stop()
        await orch.vision.stop()
        await orch.capture.stop()
        await orch.tts.stop()
        await orch._faces_client.close()
        await close_pool()
        server.should_exit = True


if __name__ == "__main__":
    asyncio.run(main())
