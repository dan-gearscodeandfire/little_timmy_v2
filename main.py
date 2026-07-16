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
from conversation.enroll_intent import (
    detect_enroll_intent, detect_identity_correction, detect_self_intro)
from presence.cosample import CoSampleBuffer
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
from presence import anchor
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


def _speaker_has_enrolled_face(name: str) -> bool:
    """True when ``name`` already has an on-disk face prototype — the same
    source /api/identity/list reports as the identity's ``face`` flag
    (``face_store.path_for(name).exists()``). Used by the anchor co-sample
    guard to tell a voice-only-promotion bootstrap (known voice, no face yet ->
    OK to bind the mic-holder's unrecognized face) from an off-mic known
    speaker (already has a face -> skip). Best-effort: any failure -> False so
    the guard treats the speaker as face-less and allows the bootstrap bind."""
    try:
        from presence.face_identifier import get_shared_identifier
        return get_shared_identifier()._store.path_for(name).exists()
    except Exception:
        return False


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

        # Phase B: passive dual-modality co-sampling. Buffers the SOLE in-frame
        # face crop each speaking turn (keyed by the turn's speaker) so a later
        # "enroll me" — or a confirmed name-tell (2026-07-06) — can bind the
        # face that has been talking without a separate capture dialog.
        # Bounded ring; crops embed only at commit. Consumed by
        # _handle_unified_enroll and Introductions._maybe_commit_face.
        # (Constructed before Introductions, which takes it.)
        self._cosample = CoSampleBuffer()

        # The "what's your name?" sub-dialog owns its own cross-turn state and
        # speaks via the turn's say(). The doorway consults it each turn.
        self._introductions = Introductions(
            speaker_id_module=self.speaker_id_module,
            turn=self._turn,
            cosample=self._cosample,
        )
        # When the name-ask was last armed. Introductions has no expiry of
        # its own, so _dialog_owns_turn bounds its term with this stamp
        # (review 7-05) — a walked-away visitor must not mute proactive
        # speech forever.
        self._introductions_asked_ts: float = 0.0

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

        # A two-turn latch: armed when the user says "enroll me" without a
        # name, so the next turn's name completes the enroll. Dict while
        # waiting, None otherwise:
        #   {scope, speaker_key, voice_embs, face_crops}
        # It carries the REQUESTER's identity and an arm-time biometric
        # snapshot (code review R2): the bare-name reply that completes this
        # latch is a short, easily mis-attributed turn, so both the speaker
        # key and the samples must come from the arming "enroll me" turn and
        # survive every re-ask/correction loop.
        self._pending_enroll: dict | None = None
        # Confirm-before-commit latch (2026-07-02, tests E/F/G): a parsed
        # enroll name is spoken back and committed only after an explicit yes
        # on the NEXT turn. Dict while waiting, None otherwise:
        #   {name, scope, speaker_key, voice_embs, face_crops}
        # The biometric snapshot is captured AT ASK TIME from the enroll
        # turn's speaker key (code review C2): the confirm turn's short
        # "yes, that's right" is exactly the utterance most likely to mint a
        # fresh unknown_N or mis-attribute to a bystander, so samples resolved
        # on the yes-turn could be thin or the wrong person's.
        self._pending_enroll_confirm: dict | None = None
        # Both latches share one armed-at stamp and expire after TTL (code
        # review C4): without expiry, "enroll me" + walking away wedges the
        # latch (and, via the proactive gate, mutes proactive speech) until
        # some future turn happens to clear it.
        self._enroll_latch_ts: float = 0.0

    ENROLL_LATCH_TTL_SEC = 120.0

    def _enroll_latch_pending(self) -> bool:
        """True while an enroll latch is armed and fresh; expires stale ones."""
        if self._pending_enroll is None and self._pending_enroll_confirm is None:
            return False
        if time.time() - self._enroll_latch_ts > self.ENROLL_LATCH_TTL_SEC:
            log.info("[ENROLL] latch expired after %.0fs — dropping",
                     self.ENROLL_LATCH_TTL_SEC)
            self._pending_enroll = None
            self._pending_enroll_confirm = None
            return False
        return True

    def _dialog_owns_turn(self) -> bool:
        """A multi-turn dialog is mid-flight: enroll latches, the face
        auto-enroll consent FSM, or the introductions name-ask (review
        finding #5, 2026-07-05 — introductions was left out of this gate the
        same way the FSM originally was, code review C8). Used by the
        proactive gate so an unprompted remark never interjects into a
        pending dialog.

        The introductions term is TIME-BOUNDED here (review 7-05): unlike the
        enroll latches (TTL) and the FSM (deadline), Introductions has no
        expiry of its own — its pending state only clears when the same
        unknown_* speaker talks again, so a walked-away visitor would mute
        proactive speech for the process lifetime."""
        intro_fresh = (
            self._introductions.awaiting and
            time.time() - self._introductions_asked_ts
            <= self.ENROLL_LATCH_TTL_SEC)
        return (self._enroll_latch_pending() or self._face_enroller.awaiting
                or intro_fresh)

    async def _speak_direct(self, text: str) -> None:
        """Speak a doorway/handler reply with full turn visibility (2026-07-02).

        Raw ``tts.speak`` from the enroll doorway was invisible — no [TIMMY]
        log line, no chat-history turn, no WS broadcast — and the eye LED
        stayed stuck on THINKING because only the normal reply flow resets it
        (every early-return path leaked it during the 2026-07-02 test run).
        Every doorway/handler reply routes here instead."""
        log.info("[TIMMY] %s", text)
        try:
            await broadcast_event("turn", {"role": "assistant", "content": text})
            await self.conversation.add_system_action_turn(text)
        except Exception:
            log.exception("[SPEAK-DIRECT] visibility bookkeeping failed")
        asyncio.create_task(eye_led.notify("SPEAKING"))
        try:
            await self.tts.speak(text)
        finally:
            # Parity with the normal reply flow: AI_CONNECTED after enqueue.
            asyncio.create_task(eye_led.notify("AI_CONNECTED"))

    @staticmethod
    def _display_name(name: str) -> str:
        """Canonical (lowercase, underscore-joined) -> spoken form."""
        return (name or "").replace("_", " ").title()

    async def _fetch_face_safe(self):
        # Wrapper that never raises; returns None on any failure or timeout.
        # face_authority toggle picks the IDENTITY source: "okdemerzel" runs the
        # local EdgeFace multi-frame recognizer (recognizes the enrolled makers,
        # a superset of the Pi's SFace) and falls back to the Pi observation on
        # any miss; "pi" (default) uses the Pi's /faces via face_client_local.
        # Either way the Pi supplies the BehaviorSnapshot fuse_identity gates on.
        if not self._presence_enabled:
            return None
        try:
            if runtime_toggles.get("face_authority") == "okdemerzel":
                from presence.face_recognize import fetch_face_observation_okdemerzel
                frames = int(runtime_toggles.get("face_authority_frames") or 2)
                obs = await asyncio.wait_for(
                    fetch_face_observation_okdemerzel(
                        self._face_http,
                        config.STREAMERPI_CAPTURE_URL,
                        config.STREAMERPI_BEHAVIOR_URL,
                        frames=frames,
                        timeout_sec=1.5,
                    ),
                    timeout=2.5,
                )
                if obs is not None:
                    return obs
                # okDemerzel recognized nothing / grab failed -> Pi fallback below.
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

    async def _handle_enrollment(self, name: str) -> None:
        """Voice-triggered face enrollment.

        Speaks acknowledgment, calls streamerpi /face_db/enroll over HTTP,
        speaks the result. Returns when TTS finishes; the caller is expected
        to early-return from process_speech to skip the normal LLM/memory path.
        """
        log.info("[ENROLL] voice-triggered for '%s'", name)
        await self._speak_direct(
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
            await self._speak_direct(
                "Sorry, something went wrong with my camera. Try again later."
            )
            return

        if resp.status_code == 200 and data.get("saved"):
            captured = data.get("samples_captured", 0)
            skipped = data.get("samples_skipped", 0)
            log.info("[ENROLL] saved %s (captured=%d, skipped=%d, total=%d)",
                     name, captured, skipped, len(data.get("enrolled", [])))
            await self._speak_direct(f"Got it. I'll remember you, {name}.")
        else:
            err = data.get("error", "I couldn't get a clear look at your face.")
            log.warning("[ENROLL] failed for %s: %s", name, err)
            await self._speak_direct(
                f"Sorry, I couldn't get a clear look. Try again with better lighting?"
            )

    async def _gather_enroll_samples(
            self, speaker_name: str, scope: str,
            newest_voice_only: bool = False) -> tuple[list, list | None]:
        """Resolve (face_crops, voice_embs) for ``speaker_name`` per scope.

        Face: the passively co-sampled sole-face crops, with a live grab
        fallback (crop-buffer starvation fix, 2026-07-02 — the buffer only
        fills on NORMAL turns and clears on success, so back-to-back enrolls
        found it empty). Voice: the tracked unknown buffer for unknown_N, or
        the rolling confident-match buffer for a recognized speaker (test A
        fix). Shared by ask-time snapshotting and commit-time fallback."""
        want_face = scope in ("both", "face")
        want_voice = scope in ("both", "voice")

        face_crops = self._cosample.crops_for(speaker_name) if want_face else []
        if want_face and not face_crops:
            obs = await self._fetch_face_safe()
            if obs is not None:
                # F3 fix (review 7-07): mirror the cosample feed — anchored
                # crops (bound to this speaker, same rule as the buffer) win
                # over sole-face, so an anchored enroll in a CROWD (the exact
                # scenario the anchor exists for) no longer grabs zero crops.
                if obs.anchored_face_crops:
                    _n = obs.anchored_face_name
                    if (_n == speaker_name if _n is not None
                            else speaker_name.startswith("unknown_")):
                        face_crops = list(obs.anchored_face_crops)
                elif obs.sole_face_crops:
                    face_crops = list(obs.sole_face_crops)

        voice_embs = None
        if want_voice:
            if speaker_name.startswith("unknown_"):
                us = self.speaker_id_module.get_unknown_for_name_ask(speaker_name)
                if us is not None and us.embeddings:
                    voice_embs = list(us.embeddings)
            else:
                recent = self.speaker_id_module.recent_embeddings_for(speaker_name)
                # The rolling buffer is per-NAME, session-long: after a misID
                # it holds the real owner's earlier embeddings alongside this
                # turn's (code review 7-06). Correction snapshots take only
                # the newest entry — appended by THIS turn's identify, so it
                # is the protester's own voice (and still the rich protest
                # sentence, C2).
                if newest_voice_only and recent:
                    recent = recent[-1:]
                if recent:
                    voice_embs = recent
        return face_crops, voice_embs

    async def _ask_enroll_confirm(self, name: str, scope: str,
                                  speaker_name: str,
                                  snapshot: dict | None = None) -> None:
        """Speak the parsed name back and latch for a yes/no next turn
        (2026-07-02, tests E/F/G): STT homophones ("Jon"->"John") and garbled
        names must never bind silently. One extra turn, max.

        Captures the biometric snapshot from the ENROLL turn's speaker key
        (code review C2): the confirm turn's short reply is the utterance
        most likely to be mis-attributed, so samples must come from the turn
        where the requester actually spoke a full sentence. When the enroll
        turn was earlier in the dialog (name-ask latch, correction loop) the
        caller passes its latch as ``snapshot`` and we reuse it verbatim
        (code review R2) — re-gathering here would source from the short
        reply turn, exactly what C2 forbids."""
        name = (name or "").strip().lower()
        if snapshot is None:
            # Direct path: THIS turn is the requester's rich enroll sentence.
            face_crops, voice_embs = await self._gather_enroll_samples(
                speaker_name, scope)
            snapshot = {"scope": scope, "speaker_key": speaker_name,
                        "voice_embs": voice_embs, "face_crops": face_crops}
        # Reserved/malformed name check BEFORE latching (2026-07-02 P1): test 1
        # burned a confirm turn on "Timmy" then refused. Refuse up front so we
        # never latch on a name we'd reject anyway. _handle_unified_enroll keeps
        # its own copy as belt-and-suspenders.
        from presence.prototype_base import is_valid_enroll_name
        if not is_valid_enroll_name(name):
            # Re-arm the name ask (code review C3): the refusal invites
            # "pick another name", so the next turn must be read as a name
            # reply — without this the dialog dead-ended after e.g. "Timmy".
            # Carry the requester's snapshot through the re-ask (R2).
            self._pending_enroll = {**snapshot, "scope": scope}
            self._enroll_latch_ts = time.time()
            await self._speak_direct(
                f"I can't enroll anyone as {self._display_name(name)} — "
                f"pick another name.")
            return
        self._pending_enroll_confirm = {**snapshot, "name": name,
                                        "scope": scope}
        self._enroll_latch_ts = time.time()
        # Coach multi-word phrasing from turn one (Dan 2026-07-05): STT
        # filters one-word turns, so an uncoached "Yes." never even arrives —
        # every uncoached confirm reply is a paraphrase the verdict may miss.
        # A misID-correction latch owns the mistake out loud (Dan 7-06).
        _prefix = "My mistake. " if snapshot.get("correction") else ""
        await self._speak_direct(
            f"{_prefix}{self._display_name(name)} — did I get that right? "
            f"Say 'yes that is right' or 'no that is wrong'.")

    async def _handle_unified_enroll(self, name: str, scope: str,
                                     speaker_name: str, speaker_result,
                                     snapshot: dict | None = None) -> None:
        """Phase B dual-modality enroll via commit_identity (okDemerzel stores).

        Binds ``name`` to a face and/or voice under one shared speaker_id.
        Samples come from the ask-time ``snapshot`` when the confirm latch
        provides one (code review C2 — the requester's own turn, not the
        possibly-misattributed confirm turn), with a live re-resolve as
        fallback for any modality the snapshot lacks. Scoped by what the user
        asked (``face`` / ``voice`` / ``both``). Speaks the result; the caller
        early-returns to skip the normal turn.
        """
        name = (name or "").strip().lower()
        disp = self._display_name(name)
        # Prefer the enroll-turn's speaker key for buffer bookkeeping.
        key = (snapshot or {}).get("speaker_key") or speaker_name
        want_face = scope in ("both", "face")
        want_voice = scope in ("both", "voice")

        # Cheap name validation BEFORE the "hold on" ack (test B: the old flow
        # said "Okay Timmy, hold on" and then masked invalid_name behind the
        # generic look/listen apology).
        from presence.prototype_base import is_valid_enroll_name
        if not is_valid_enroll_name(name):
            await self._speak_direct(
                f"I can't enroll anyone as {disp} — pick another name.")
            return

        face_crops = list((snapshot or {}).get("face_crops") or [])
        voice_embs = (snapshot or {}).get("voice_embs") or None
        if (want_face and not face_crops) or (want_voice and not voice_embs):
            live_crops, live_embs = await self._gather_enroll_samples(key, scope)
            face_crops = face_crops or live_crops
            voice_embs = voice_embs or live_embs
        if not want_face:
            face_crops = []
        if not want_voice:
            voice_embs = None

        if not face_crops and not voice_embs:
            # Nothing available for the requested scope.
            if want_face:
                await self._speak_direct(
                    "I can't get a clear look at you just now — face me and say that again.")
            else:
                await self._speak_direct(
                    "I haven't heard enough of your voice yet — say a bit more and try again.")
            return

        await self._speak_direct(f"Okay {disp}, hold on while I remember you.")
        try:
            from presence.identity_commit import commit_identity
            res = await commit_identity(
                name,
                voice_embeddings=voice_embs,
                face_crops=(face_crops or None),
                speaker_identifier=self.speaker_id_module,
            )
        except Exception:
            log.exception("[ENROLL] unified commit failed for %s", name)
            await self._speak_direct(
                "Sorry, something went wrong saving that. Try again in a moment.")
            return

        if res.status == "mismatch":
            await self._speak_direct(
                f"Hmm — you don't quite match the {disp} I already know, so I'll "
                "hold off for now.")
            return
        if res.status == "lookalike":
            who = self._display_name(res.lookalike_of or "someone I know")
            await self._speak_direct(
                f"Hold on — you look and sound like {who} to me, so I won't file "
                f"you as {disp}. If I've got that wrong, tell Dan.")
            return
        if res.status == "invalid_name":
            await self._speak_direct(
                f"I can't enroll anyone as {disp} — pick another name.")
            return
        if res.status == "retired_name":
            # Tombstoned identity (code review 7-06): the generic sensor
            # apology coached an impossible retry — the refusal is by name,
            # not by sample quality, and only revive_identity clears it.
            await self._speak_direct(
                f"The name {disp} is retired on my end — Dan has to bring "
                "it back before I can use it.")
            return
        if not (res.voice_committed or res.face_committed):
            await self._speak_direct(
                "Sorry, I couldn't get a good enough look or listen just then. Try again?")
            return

        self._cosample.clear_speaker(key)
        got = []
        if res.face_committed:
            got.append("face")
        if res.voice_committed:
            got.append("voice")
        log.info("[ENROLL] unified %s -> id=%s (%s) status=%s warnings=%s",
                 res.name, res.speaker_id, "+".join(got), res.status, res.warnings)
        await self._speak_direct(
            f"Got it — I'll recognize your {' and '.join(got)} now, {disp}.")

    async def _enroll_stream(self, name: str, count: int, interval_s: float, mode: str):
        """Async generator over streamerpi's SSE /face_db/enroll/stream.

        Yields (event_type, payload) tuples — 'started' | 'progress' | 'complete'
        | 'error' — so FaceEnroller can track capture progress and abort if the
        person leaves. Fixed enrollment (/face_db/enroll) stays the voice-
        triggered path; this streaming variant is for interactive auto-enroll.
        Capture is frontal hold-still only (pose-cue coaching retired 7-15)."""
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

    async def process_speech(self, audio: np.ndarray,
                             vad_onset_ts: float | None = None,
                             vad_offset_ts: float | None = None,
                             vad_eou_ts: float | None = None,
                             dequeue_ts: float | None = None):
        """Process a speech segment through the full pipeline.

        vad_onset_ts / vad_offset_ts are the wall-clock of this utterance's
        speech onset and last-voiced-chunk (offset), carried from capture so we
        can report a true speech-end -> first-reply-audio latency that includes
        the endpointing-silence delay.

        vad_eou_ts (segment finalize / enqueue) and dequeue_ts (main loop
        dequeue) split two formerly-invisible slices out of the booth's WAIT
        remainder: endpoint_ms = eou - offset (the endpointing-silence wait, the
        dominant felt lag) and queue_ms = dequeue - eou (queue/handoff gap,
        usually ~0, spikes under shared-GPU contention)."""
        t_start = time.time()
        # Attribute the endpointing-silence window and the queue-handoff gap.
        # Both are genuine sequential sub-intervals of reply_lag (which is
        # measured first_audio - vad_offset), so surfacing them shrinks the
        # booth's ghosted WAIT bar by exactly their sum. None on any path that
        # lacks the capture timestamps (text / inject / legacy producers).
        endpoint_ms = (int(max(0.0, vad_eou_ts - vad_offset_ts) * 1000)
                       if vad_eou_ts is not None and vad_offset_ts is not None else None)
        queue_ms = (int(max(0.0, dequeue_ts - vad_eou_ts) * 1000)
                    if dequeue_ts is not None and vad_eou_ts is not None else None)

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

        # Phase B unified enroll (flag-gated). Enabled by EITHER the live runtime
        # toggle (flip without restart, UI-visible) OR the static env master
        # (config.UNIFIED_ENROLL_ENABLED) — mirrors the proactive-speech gating
        # pattern. Default OFF on both. All reactive-enroll routing keys off this.
        _unified = bool(runtime_toggles.get("unified_enroll_enabled")) or \
            config.UNIFIED_ENROLL_ENABLED

        # EXPO identity-dialog gate (Dan 2026-07-06): ONE gate for the whole
        # dialog class (enroll latches, misID correction, introductions
        # name-ask, face consent FSM — every multi-turn identity FSM that
        # seizes turns and can end in a store write). Gated == FULLY SILENT:
        # the blocks below are skipped wholesale so the utterance falls
        # through to the LLM as ordinary speech. Read live per turn, same as
        # _unified. See runtime_toggles.identity_dialogs_allowed().
        #
        # LED-mic anchor split (Dan 2026-07-06, second ruling): a fresh anchor
        # (lit mic in a visitor's hand — presence/anchor.py) un-darks the
        # SPEECH identity dialogs for the mic-holder (_dialogs_ok), but NOT
        # the face-consent FSM (_consent_ok stays pure regime+override):
        # mic-in-hand is IMPLIED consent, so the anchored face is stored via
        # the cosample->commit path on the name-tell — offering a consent
        # dialog whose answer changes nothing would be noise. No anchor (or
        # feature off) -> both predicates are identical to the shipped gate.
        # F7 binding (review 7-07): the anchor disjunct now also requires the
        # turn's VOICE attribution to bind to the ANCHORED face (anchor.
        # binding_ok) — a fresh anchor used to un-dark the mutation surfaces
        # for EVERY speaker in the TTL window.
        _consent_ok = anchor.consent_allowed()
        _dialogs_ok = anchor.speech_dialogs_allowed(speaker_name)

        # Latch lifecycle guards (code review C4): expire stale latches (TTL —
        # _enroll_latch_pending clears them as a side effect), and clear both
        # when the unified toggle was flipped OFF — or the identity-dialog
        # gate closed — mid-dialog. The clearing blocks below are gated on
        # both flags, so without this a latch armed before the flip could
        # never be cleared (and would permanently mute proactive speech via
        # _dialog_owns_turn).
        _latch_live = self._enroll_latch_pending()
        if _latch_live and (not _unified or not _dialogs_ok):
            log.info("[ENROLL] %s with latch pending — dropping",
                     "unified toggle off" if not _unified
                     else "identity-dialog gate closed")
            self._pending_enroll = None
            self._pending_enroll_confirm = None
            _latch_live = False

        # Confirm-before-commit latch (2026-07-02, tests E/F/G): a parsed name
        # was spoken back last turn; an explicit yes commits it, an explicit
        # no re-opens the name ask, a cancel aborts. Anything else RE-ASKS
        # with escalating scripted prompts (Dan 2026-07-05) — the old
        # silent-drop handed the turn to the LLM, which confabulated
        # enrollment success with zero [COMMIT] (live-proven 2x). STT filters
        # one-word turns, so a paraphrased confirm is the DEFAULT reply, not
        # an edge case. The dialog owns every turn until resolution; the only
        # no-answer exit is the walk-away TTL.
        if _unified and _latch_live and self._pending_enroll_confirm is not None:
            from conversation.enroll_intent import (
                confirm_verdict, is_enroll_cancel, confirm_reask_line)
            _latch = self._pending_enroll_confirm
            self._pending_enroll_confirm = None
            _verdict = confirm_verdict(user_text)
            # Order matters (review 7-05): yes BEFORE cancel ("yeah, don't
            # bother re-asking" must commit, not abort), cancel BEFORE no
            # ("never mind" contains a negation cue and would otherwise
            # re-open the name ask instead of aborting).
            if _verdict == "yes":
                if _latch.get("correction"):
                    # MisID protest confirmed (Dan 7-06). Commit mechanics are
                    # identical — commit_identity augments a matching known Y
                    # (the augmented prototypes ARE the re-bind: next turn's
                    # nearest-prototype match flips to Y) and its mismatch/
                    # lookalike guards refuse a claim the biometrics reject.
                    log.info("[ENROLL] identity correction confirmed: "
                             "%s -> %s (attributed %s)",
                             _latch.get("denied"), _latch["name"], speaker_name)
                await self._handle_unified_enroll(
                    _latch["name"], _latch["scope"], speaker_name,
                    speaker_result, snapshot=_latch)
                return
            if is_enroll_cancel(user_text):
                self._pending_enroll = None
                await self._speak_direct("Okay, dropping it.")
                return
            if _verdict == "no":
                # Re-arm carrying the ORIGINAL snapshot + speaker key (code
                # review R2): the correction reply is another short turn —
                # re-snapshotting from it is the C2 hole all over again.
                self._pending_enroll = {
                    k: _latch[k] for k in
                    ("scope", "speaker_key", "voice_embs", "face_crops",
                     "correction", "denied") if k in _latch}
                self._enroll_latch_ts = time.time()
                await self._speak_direct(
                    "Okay, scratch that — what name should I go with?")
                return
            # A restated enroll intent is a CORRECTION, not an unclear reply
            # (review 7-05): "Actually, enroll me as Sarah" has no yes/no cue
            # and used to loop 'is the name Joe?' with no escape short of a
            # cancel. Re-route to confirm the new name/scope, keeping the
            # original arm-time snapshot (R2).
            _restated = detect_enroll_intent(user_text, speaker_name)
            if _restated.matched:
                await self._ask_enroll_confirm(
                    _restated.name, _restated.scope, speaker_name,
                    snapshot=_latch)
                return
            # Unclear -> re-arm and escalate. Refreshing the TTL stamp keeps
            # "walk-away" semantics: expiry measures silence since the last
            # exchange, not since the original arm.
            _tries = _latch.get("confirm_attempts", 0) + 1
            self._pending_enroll_confirm = {**_latch,
                                            "confirm_attempts": _tries}
            self._enroll_latch_ts = time.time()
            await self._speak_direct(confirm_reask_line(
                self._display_name(_latch["name"]), _tries))
            return

        # Two-turn latch: a prior "enroll me" without a name is waiting for one.
        # This turn supplies it -> speak it back for confirmation (NOT a direct
        # commit — a bare name utterance is where STT homophones bite hardest).
        if _unified and _latch_live and self._pending_enroll is not None:
            from conversation.enroll_intent import (
                extract_reply_name, is_enroll_cancel, name_reask_line)
            # Multi-word-capable, evasive-aware reply parsing (code review
            # C5/C6): the old inline extractor-first order truncated "My name
            # is Mary Jane" to 'mary' and canonicalized "not telling".
            _latch = self._pending_enroll
            self._pending_enroll = None
            if is_enroll_cancel(user_text):
                await self._speak_direct("Okay, dropping it.")
                return
            _nm = extract_reply_name(user_text)
            if _nm:
                # Pass the arm-time latch as the snapshot (code review R2):
                # samples + speaker key come from the "enroll me" turn, not
                # this bare-name reply.
                await self._ask_enroll_confirm(_nm, _latch["scope"],
                                               speaker_name, snapshot=_latch)
                return
            # No name heard -> NEVER abandon silently (Dan 2026-07-05):
            # re-arm and coach the phrasing until a name or a cancel; the
            # only no-answer exit is the walk-away TTL (refreshed here, same
            # semantics as the confirm latch above).
            _tries = _latch.get("name_attempts", 0) + 1
            self._pending_enroll = {**_latch, "name_attempts": _tries}
            self._enroll_latch_ts = time.time()
            await self._speak_direct(name_reask_line(_tries))
            return

        # A face auto-enroll consent dialog in flight OWNS this turn. Route it to
        # the FSM BEFORE the legacy voice enroll-intent / name-ask below, which
        # would otherwise hijack natural consent phrases. STT audits out a bare
        # "yes", so consent replies are necessarily long ("yes, you can remember
        # my face") and keyword-laden — exactly what trips detect_enroll_intent.
        if self._face_enroller.awaiting:
            if not _consent_ok:
                # Identity-dialog gate closed with a consent dialog armed
                # pre-flip: drop it silently and let this turn fall through
                # to the normal pipeline as ordinary speech. Keyed on
                # _consent_ok, NOT _dialogs_ok — the LED anchor never
                # un-darks the consent FSM (implied consent).
                self._face_enroller.drop_gated()
            else:
                ae = await self._face_enroller.handle(user_text, speaker_name)
                if ae.handled:
                    # The FSM speaks through its own machinery, which never
                    # resets the eye — clear THINKING here (2026-07-02).
                    asyncio.create_task(eye_led.notify("AI_CONNECTED"))
                    return

        # voice-enroll-shortcut — skipped wholesale (detector included) under
        # the identity-dialog gate: "enroll me as Bob" at the booth becomes
        # ordinary speech for the LLM, no latch, no reply about enrollment.
        if _dialogs_ok:
            enroll = detect_enroll_intent(user_text, speaker_name)
            if _unified:
                # Phase B: route through commit_identity (okDemerzel stores),
                # scoped to what the user asked (face / voice / both). A parsed
                # name is confirmed on the next turn before committing; a
                # keyword with no name latches a one-turn name ask instead of
                # the dead Pi POST.
                if enroll.matched:
                    await self._ask_enroll_confirm(enroll.name, enroll.scope,
                                                   speaker_name)
                    return
                if enroll.keyword_present:
                    # Capture the requester's biometric snapshot NOW (code
                    # review R2): the "enroll me" sentence is the rich,
                    # reliably-attributed turn; the upcoming bare-name reply
                    # is not.
                    _crops, _embs = await self._gather_enroll_samples(
                        speaker_name, enroll.scope)
                    self._pending_enroll = {
                        "scope": enroll.scope, "speaker_key": speaker_name,
                        "voice_embs": _embs, "face_crops": _crops}
                    self._enroll_latch_ts = time.time()
                    await self._speak_direct(
                        "Sure — what name should I remember you by?")
                    return
            elif enroll.matched:
                # Legacy path (flag OFF): the Pi SFace gallery enroll. NOTE
                # this gallery is retired (Pi recognition disabled) so this is
                # effectively a no-op until the unified flag is flipped —
                # Phase B is the fix.
                await self._handle_enrollment(enroll.name)
                return

        # Identity-correction protest (Dan 2026-07-06): a misidentified user
        # says "No, my name is not Walter, my name is Flynn" — or a bare
        # "My name is Flynn" contradicting an enrolled attribution. Routes
        # into the SAME never-silent confirm FSM as enroll (scope=voice —
        # misID is a close-talk voice problem; face has its own consent FSM).
        # commit_identity's mismatch/lookalike guards distance-gate the claim
        # at commit, so a stranger can't talk their way into an enrolled
        # identity. Runs AFTER detect_enroll_intent (enroll keywords win) and
        # only under the unified flag. Bare claims from unknown_N speakers
        # stay with introductions below. Skipped (detector included) under
        # the identity-dialog gate — at the booth "my name is Bob" is every
        # visitor's self-intro, and the off-mic chain collapses visitors
        # onto enrolled voiceprints, so a normal self-intro reads as a misID
        # protest; worse, confirm-yes is an identity-MUTATION surface.
        if _unified and _dialogs_ok:
            corr = detect_identity_correction(
                user_text, speaker_name,
                speaker_enrolled=self.speaker_id_module.is_known_speaker(
                    speaker_name))
            if corr.matched:
                log.info("[ENROLL] misID correction: denied=%s claim=%s "
                         "(attributed %s)", corr.denied, corr.name,
                         speaker_name)
                # Snapshot THIS turn (the protest sentence is the rich,
                # reliably-voiced turn — C2): its embeddings are both the
                # confirm-time commit samples and, on a known-Y augment,
                # the effective re-bind. newest_voice_only: the rolling
                # buffer under the DENIED name may hold the real owner's
                # earlier voice (code review 7-06) — only this turn's entry
                # is provably the protester's.
                _crops, _embs = await self._gather_enroll_samples(
                    speaker_name, "voice", newest_voice_only=True)
                _snap = {"scope": "voice", "speaker_key": speaker_name,
                         "voice_embs": _embs, "face_crops": _crops,
                         "correction": True,
                         "denied": corr.denied or speaker_name}
                if corr.name:
                    await self._ask_enroll_confirm(
                        corr.name, "voice", speaker_name, snapshot=_snap)
                else:
                    # Denial without a claim ("that's not my name") ->
                    # ask-name latch, same never-silent semantics.
                    self._pending_enroll = _snap
                    self._enroll_latch_ts = time.time()
                    await self._speak_direct(
                        "My mistake — what name should I go with? "
                        "Say 'my name is', then the name.")
                return

        # Passive self-intro (LED-mic anchor, 2026-07-06): an UNKNOWN speaker
        # volunteering "my name is Flynn" arms the introductions confirm flow
        # without an ask. Runs AFTER enroll keywords and misID (both return
        # on match, so they keep priority); enrolled speakers never reach it
        # (unknown_ guard — a known speaker's bare claim is misID's turf).
        # Toggle-gated (default OFF) + the same _dialogs_ok as the rest of
        # the dialog class, so at EXPO it fires only for the anchored
        # mic-holder.
        if (_dialogs_ok and speaker_name.startswith("unknown_")
                and not self._introductions.awaiting
                and runtime_toggles.get("passive_self_intro_enabled")):
            _intro_name = detect_self_intro(user_text)
            if _intro_name:
                log.info("[INTRO] passive self-intro: %r from %s",
                         _intro_name, speaker_name)
                await self._introductions.offer_confirm(
                    speaker_name, _intro_name)
                self._introductions_asked_ts = time.time()
                return

        # --- Handle name solicitation for unknown speakers ---
        # Gated with the rest of the identity-dialog class: at the booth
        # every visitor is a stable unknown, and the name-ask both seizes
        # the turn and feeds assign_name (a store write). The unknown stays
        # un-marked so the ask can still happen post-show.
        if speaker_result.should_ask_name and _dialogs_ok:
            unknown_info = self.speaker_id_module.get_unknown_for_name_ask(
                speaker_result.name
            )
            if unknown_info:
                self.speaker_id_module.mark_name_asked(speaker_result.name)
                await self._introductions.ask_name(unknown_info)
                self._introductions_asked_ts = time.time()
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
            vad_onset_ts=vad_onset_ts,
            vad_offset_ts=vad_offset_ts,
            endpoint_ms=endpoint_ms,
            queue_ms=queue_ms,
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

    async def inject_couples_therapist_turn(self, text: str):
        """Inject a couples-therapist utterance DIRECTLY into the conversation as
        a turn from speaker 'couples_therapist', bypassing the mic / VAD / close-
        talk addressee gate entirely. Timmy processes it and responds.

        This is the reliable path for the therapist to address Timmy: the room-
        speaker→close-talk-mic level is too low to clear the close-talk gate, and
        the service already knows it's emitting the persona, so we assert the
        identity here instead of round-tripping through acoustics. The enrolled
        voiceprint (speaker_id 6) stays the durable identity / FK target and the
        fallback if the persona ever arrives via an external mic.

        NOT the default: /api/announce only calls this when inject=true. By
        default the therapist is gated (spoken for Dan, not processed by Timmy)."""
        from speaker.identifier import SpeakerIdentifier
        ct_id = SpeakerIdentifier().enrolled_speaker_ids().get("couples_therapist", 6)
        async with self._turn_lock:
            t_start = time.time()
            log.info("[INJECT:couples_therapist] %s", text)
            await broadcast_event("turn", {"role": "user", "content": text,
                                           "speaker": "couples_therapist"})
            await self.conversation.add_user_turn(text, speaker="couples_therapist")
            outcome = await tool_router.maybe_handle_tool_call(
                text, "couples_therapist", ct_id, self.conversation, self.tts,
                t_start=t_start)
            if outcome.handled:
                return
            await self._generate_response(
                text, stt_ms=0, t_start=t_start,
                speaker_name="couples_therapist", speaker_db_id=ct_id, spk_ms=0,
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
        # Dialog gate (2026-07-02 P1, generalized by code review C8): a pending
        # enroll latch OR a face-consent FSM dialog owns the next turn.
        # [PROACTIVE] interjected 50s into a pending confirm (01:47), derailing
        # the enroll dialog. Stay silent until the dialog resolves (latches
        # self-expire after ENROLL_LATCH_TTL_SEC, so a walked-away user can't
        # wedge this gate — code review C4).
        if self._dialog_owns_turn():
            return
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
        # A finalized user utterance is already waiting on the speech queue but
        # the main loop hasn't dequeued + turn-locked it yet (the high-latency /
        # busy handoff gap that the fixed last_voice_ts grace can outlast). The
        # user HAS given verbal input; it's just unprocessed. Yield — never let a
        # proactive vision remark grab the turn-lock ahead of a pending real turn
        # and speak over / delay it. Root of the 2026-06-24 "vision overrode my
        # long prompt" report: proactive must always defer to pending user input.
        if self.capture.speech_queue.qsize() > 0:
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
                                  query_pre_resolved: bool = False,
                                  vad_onset_ts: float | None = None,
                                  vad_offset_ts: float | None = None,
                                  endpoint_ms: int | None = None,
                                  queue_ms: int | None = None):
        """Core response pipeline: presence doorway → delegate to the turn.

        `resolved_query`/`query_pre_resolved` carry the doorway's speculative coref
        result (resolved in parallel with the classifier) into the turn's
        retrieval. Both stay at their OFF defaults on the text path."""

        # EXPO identity-dialog gate (Dan 2026-07-06): recomputed here (not
        # threaded from process_speech) because this is also the text-input /
        # injected-turn entry point. When closed, any dialog state armed
        # pre-flip is dropped SILENTLY and the turn proceeds as ordinary
        # speech — a latch must not survive the flip. Same anchor split as
        # the doorway: a fresh LED-mic anchor un-darks the speech dialogs
        # (_dialogs_ok) but never the face-consent FSM (_consent_ok —
        # mic-in-hand is implied consent, the offer would be noise), and the
        # anchor disjunct is speaker-BOUND (F7, review 7-07).
        _consent_ok = anchor.consent_allowed()
        _dialogs_ok = anchor.speech_dialogs_allowed(speaker_name)
        if not _dialogs_ok:
            self._introductions.drop_pending()
        if not _consent_ok:
            self._face_enroller.drop_gated()

        # --- Name-confirmation sub-dialog (Introductions owns the state) ---
        # If we're mid-introduction, this may speak a follow-up (handled=True ->
        # we're done) or promote the speaker to a just-confirmed name and fall
        # through to a normal turn.
        if _dialogs_ok:
            intro = await self._introductions.handle(user_text, speaker_name)
            if intro.handled:
                return
            speaker_name = intro.speaker_name

        # --- Face auto-enroll consent dialog (FaceEnroller owns state) ---
        # If we're mid-offer ("mind if I remember your face?"), this
        # consumes the yes/no/name reply and speaks the follow-up; a no-op
        # when not in flight. Keyed on _consent_ok (see split above).
        if _consent_ok:
            enroll_outcome = await self._face_enroller.handle(
                user_text, speaker_name)
            if enroll_outcome.handled:
                return

        # --- THINKING window (filler + eye-pulse + body wobble) ---
        # We've committed to a brain reply (past the Timmy-voice/empty/tool/
        # intro/enroll drop-gates above). Fire the thinking cues HERE, as early
        # as is safe, rather than after retrieval/vision:
        #   - filler: queue a spoken thinking-beat so Timmy starts almost
        #     instantly. Previously fired just before _turn.respond(), AFTER the
        #     face fetch + presence fusion + the whole vision block (which can
        #     even BLOCK on a fresh VLM capture), so the "instant" filler
        #     actually trailed seconds of silence (Dan's "fire way earlier").
        #     Gated 50% / curt-prompt; speak_filler() respects tts_muted.
        #   - body: put the Pi into the gentle 'thinking' wobble so the head
        #     looks like it's pondering during the LLM run (Dan 2026-06-25).
        #     Fires every brain turn, matching the THINKING eye-LED (sent at STT
        #     finalize above). Cleared when the first sentence re-asserts engage.
        # Per-turn audio-onset scratch for the filler-latency instrumentation
        # (2026-06-30, measurement only). filler_text is the picked clip;
        # filler_play is stamped by the TTS playback loop the instant the filler
        # actually starts sounding. Read later in _on_first_audio to compute the
        # overrun (how long the ready reply waited behind the filler).
        audio_ts: dict = {}
        # Filler length is chosen by how slow we expect the turn to be. The one
        # high-precision "slow" signal free at fire time: a visual question
        # landing on a stale frame forces a blocking VLM capture (+2.4-2.85s
        # measured), where a 2-3s clip can't outrun the answer. Everything else
        # defaults SHORT (~1.1s) and fails safe — under-fill never delays the
        # reply, over-fill would. (is_visual_question is a cheap pure-text check;
        # it's re-evaluated in the vision block below — same result.) (Dan 6-30)
        _vq = (is_visual_question(user_text)
               or is_self_referential_visual_question(user_text))
        _age = self.vision.scene_age()
        long_think = _vq and (_age is None or _age > config.VISION_VISUAL_Q_MAX_AGE_S)
        if audio_fillers.should_fire(user_text):
            # Register-aware pick: declarative vs interrogative phrasing (free
            # lexical check). Tool/command turns route through tool_router and
            # don't reach here, so register() only splits question vs statement.
            filler_text = audio_fillers.pick(
                long=long_think, reg=audio_fillers.register(user_text))
            audio_ts["filler_text"] = filler_text
            asyncio.create_task(self.tts.speak_filler(
                filler_text,
                on_play_start=lambda ts: audio_ts.__setitem__("filler_play", ts),
            ))
        asyncio.create_task(self.supervisor.on_thinking_start())

        # --- Presence face fetch (doorway) ---
        # Memory retrieval (memories + facts, with coreference + "my X" subject
        # scoping and fact dedup) now lives inside ConversationTurn. The doorway
        # only fetches the face observation that identity fusion needs below;
        # _fetch_face_safe returns None when presence is disabled.
        face_obs = await self._fetch_face_safe()

        # Face-recognition SHADOW mode (2026-06-30): also run okDemerzel-side
        # EdgeFace recognition (self-served from a /capture grab) and log how it
        # compares to the Pi's SFace, WITHOUT touching fusion. Fire-and-forget +
        # off the event loop -> zero reply latency. Gated by face_shadow_enabled
        # (default OFF). Lets us watch real agreement before flipping identity
        # authority to okDemerzel (plan Phase A).
        if runtime_toggles.get("face_shadow_enabled"):
            from presence.face_shadow import shadow_compare
            asyncio.create_task(shadow_compare(face_obs))

        # Phase B passive co-sampling: the recognizer only fills sole_face_crops
        # when EXACTLY one face was in frame (the sole-face==speaker rule), so a
        # buffered crop is unambiguously this turn's speaker. Key by the pre-fusion
        # voice name (temp_id for an unknown) so crops stay per-person. Cheap
        # (bounded ring, no embedding here) so we buffer even with the flag off,
        # keeping the buffer warm for the moment it's flipped on.
        # LED-mic anchor (2026-07-06): anchored_face_crops (the face directly
        # above the lit mic — unambiguous even in a crowd) win over the
        # sole-face rule when present; the sole-face predicate stays intact
        # for Shop. F1 binding (review 7-07): the anchored crops are the
        # MIC-HOLDER's face, but speaker_name is VOICE-attributed — an
        # off-mic bystander's turn used to buffer the mic-holder's face under
        # the bystander's key (wrong-face commit). Buffer anchored crops only
        # when the two sensors agree: the anchored face recognized AS this
        # speaker, or an unrecognized face with an unknown_N voice (the
        # ordinary visitor). Unbound -> skip (never fall back to sole crops:
        # a frame with an anchored pick had a face, so ==1-face co-sampling
        # of somebody ELSE is exactly the mis-bind to avoid).
        if face_obs is not None:
            if face_obs.anchored_face_crops:
                _n = face_obs.anchored_face_name
                # F1 binding: bind the mic-holder's anchored face to this
                # speaker when the two sensors agree. Recognized anchored face
                # -> must match the voice. UNRECOGNIZED anchored face -> bind for
                # a fresh visitor (unknown_N) OR a known speaker who has no face
                # enrolled yet (voice-only-promotion bootstrap: the name-tell
                # minted a voiceprint before a face ever bound, so speaker_name
                # is a known name but the face is still unseen). Keep SKIPPING a
                # known speaker who ALREADY has an enrolled face: an unrecognized
                # mic-holder face then means they are off-mic while someone else
                # holds the lit mic (the wrong-face commit this guard exists to
                # avoid). Fixes the catch-22 where a voice-known/face-unknown
                # person could never bootstrap their face (Tushar, 2026-07-15).
                if _n is not None:
                    _bound = (_n == speaker_name)
                else:
                    _bound = (speaker_name.startswith("unknown_")
                              or not _speaker_has_enrolled_face(speaker_name))
                if _bound:
                    self._cosample.add(speaker_name,
                                       list(face_obs.anchored_face_crops))
                else:
                    log.info("[ANCHOR] anchored crops NOT bound to speaker "
                             "%s (anchored face=%s) — skip co-sample",
                             speaker_name, _n or "unrecognized")
            elif face_obs.sole_face_crops:
                self._cosample.add(speaker_name,
                                   list(face_obs.sole_face_crops))

        # --- Presence: voice + face fusion ---
        fusion_source = None
        face_hint_name = None
        face_trust_name = None
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
                streak_high_conf=config.FACE_STREAK_HIGH_CONF,
                head_steady_min_ms=config.HEAD_STEADY_MS,
            )
            # Track face_hint streak for auto voice-enrollment.
            # Use pre-override speaker_name (the unknown_N temp_id).
            streak_temp_id = speaker_name if speaker_name.startswith("unknown_") else None
            # Gate on streak_eligible (stricter than attribution: high OR a
            # sticky-held medium) AND face_hint_source=='face' (a synthesized/held
            # Slice B 'voice'/'temporal' hint must NEVER train a voiceprint —
            # the "calls everyone Dan" corruption vector). A medium-non-sticky
            # face still attributes the turn (face_hint) but does not bind here.
            streak_face_name = (
                verdict.face_hint_name
                if (verdict.streak_eligible
                    and verdict.face_hint_source == "face")
                else None
            )
            streak_hit = self._face_hint_streak.observe(streak_face_name, streak_temp_id)
            if streak_hit is not None:
                # Suppress the voiceprint auto-enroll when EITHER the env kill
                # switch (config.AUTO_ENROLL_KILL, hard master-off) OR the live
                # LT-OS operator toggle (auto_enroll_enabled=False) is set. A
                # crowd makes face_hint streaks unreliable (false-accepts), and
                # binding a voiceprint off a bad streak corrupts speaker
                # attribution at scale. Read live per turn so a booth flip
                # applies immediately. Still reset either way so the streak
                # doesn't accumulate stale across turns.
                _ae_toggle_on = bool(runtime_toggles.get("auto_enroll_enabled"))
                if config.AUTO_ENROLL_KILL or not _ae_toggle_on:
                    _reason = ("AUTO_ENROLL_KILL" if config.AUTO_ENROLL_KILL
                               else "auto_enroll_enabled=False")
                    log.info(
                        "[PRESENCE] auto-enroll SUPPRESSED (%s): %s -> %s (%d-turn streak)",
                        _reason,
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
                    "[PRESENCE] face_hint promoted: voice=%s -> %s (face_conf=%.2f, streak_eligible=%s)",
                    speaker_name, verdict.face_hint_name, verdict.face_hint_confidence or 0.0,
                    verdict.streak_eligible,
                )
                speaker_name = verdict.final_name
            elif (
                speaker_name.startswith("unknown_")
                and verdict.gates.get("face_present")
                and not verdict.gates.get("single_face")
            ):
                # No-silent-caps: an unknown voice with MULTIPLE faces in frame
                # is ambiguous — the "sole face == speaker" rule only fires on
                # exactly one detected face. With 2+ we abstain (respond as
                # guest, bind nothing) rather than guess which face is talking.
                # Make that visible, not silent.
                _n_faces = (
                    face_obs.detected_face_count
                    if (face_obs and face_obs.detected_face_count is not None)
                    else (len(face_obs.predictions) if face_obs else 0)
                )
                log.info(
                    "[PRESENCE] attribution ABSTAINED: unknown voice %s + %d faces "
                    "in frame (need exactly 1 for sole-face attribution) -> staying guest (no bind)",
                    speaker_name, _n_faces,
                )
            fusion_source = verdict.resolution_source
            face_hint_name = verdict.face_hint_name
            # PARTY-2 face-trust (2026-07-09, OpenSauce-critical): the voice is
            # unknown_N and full attribution ABSTAINED (fusion_source stayed
            # 'voice' — e.g. head not steady 2s yet, or no behavior snapshot),
            # so speaker_name is still unknown and the guest's on-file facts
            # would never surface ("face recognized, but Timmy says he doesn't
            # know you"). If the same fusion confidently sees a RECOGNISED SOLE
            # face (single_face + face_above_threshold — a strict subset of the
            # promote gate, minus the head-steady/behavior binding conditions),
            # trust that face for READ-ONLY purposes: fact retrieval + the
            # WHO-IS-SPEAKING addressing hypothesis. NOTHING is bound or
            # persisted — speaker_name stays unknown_N, no voiceprint streak, no
            # attribution write; the READ tier sits strictly below attribution,
            # which itself sits below the voiceprint streak. Multiple faces ->
            # single_face is False -> abstain (same ambiguity contract as the
            # attribution-abstain branch above): never guess which face speaks.
            if (speaker_name.startswith("unknown_")
                    and fusion_source != "face_hint"
                    and verdict.gates.get("single_face")
                    and verdict.gates.get("face_above_threshold")
                    and verdict.face_hint_name
                    and not verdict.face_hint_name.startswith("unknown")):
                face_trust_name = verdict.face_hint_name
                log.info(
                    "[PARTY-2] voice %s unknown + recognized sole face %r "
                    "(conf=%.2f, promotion abstained) -> trust face for facts + "
                    "addressing (READ-only; no voiceprint bind)",
                    speaker_name, face_trust_name,
                    verdict.face_hint_confidence or 0.0,
                )
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
                        # Never contend with the in-flight turn (2026-07-15
                        # double-VLM diagnosis): trigger() bypasses the poll
                        # pause, so a recapture here ran concurrently with the
                        # reply's generation+TTS and both halved. Wait for the
                        # turn to release the pause; past the cap, skip -- the
                        # next visual question's block-on-fresh captures anyway.
                        waited = 0.0
                        while (self.vision.is_polling_paused
                                and waited < config.VISION_RECAPTURE_MAX_WAIT_S):
                            await asyncio.sleep(0.25)
                            waited += 0.25
                        if self.vision.is_polling_paused:
                            log.info(
                                "[VISION] recapture skipped (turns still "
                                "holding the GPU after %.1fs)", waited)
                            return
                        # Detection-not-ID gate (proximity-gate idiom): the
                        # recapture exists to catch the subject once the pan
                        # lands -- if the Pi still sees no face, the VLM would
                        # burn GPU on another empty frame. Fail open (None =
                        # /faces unreachable) to preserve the C6 behavior.
                        face_now = await self.vision.face_currently_visible()
                        if face_now is False:
                            log.info(
                                "[VISION] recapture skipped (no face visible "
                                "after look-at pan)")
                            return
                        await self.vision.trigger_capture("visual_question_recapture")
                    asyncio.create_task(_delayed_recapture())

        # Wall-clock of the first REAL reply sentence hitting TTS. Stamped in the
        # callback below so the speech-onset/end -> first-reply latency is exact
        # (the turn's own first_tts_ms is relative to the turn's local t_start, so
        # it can't be combined with the capture timestamps). Measures the real
        # answer, not the filler stall.
        first_audio_wall: float | None = None

        async def _on_first_sentence():
            # Eye LED + supervisor cues fire the moment the first sentence is
            # about to hit TTS. Fire-and-forget so they don't delay speak().
            nonlocal first_audio_wall
            if first_audio_wall is None:
                first_audio_wall = time.time()
            asyncio.create_task(self.supervisor.on_tts_start())
            asyncio.create_task(eye_led.notify("SPEAKING"))

        def _on_first_audio(play_ts: float):
            # Fires from the TTS playback loop the instant the FIRST real reply
            # sentence actually starts sounding (true audible onset). The gap
            # from first_audio_wall (the ENQUEUE stamp) is the filler OVERRUN:
            # how long the ready answer sat behind the THINKING filler in the
            # serial playback queue — latency the booth's reply_lag can't see,
            # because reply_lag is stamped at enqueue. Measurement only; no
            # behavior change. (2026-06-30, filler-latency study.)
            enq = first_audio_wall
            overrun_ms = (int(max(0.0, play_ts - enq) * 1000)
                          if enq is not None else None)
            true_reply_lag_ms = (int(max(0.0, play_ts - vad_offset_ts) * 1000)
                                 if vad_offset_ts is not None else None)
            fplay = audio_ts.get("filler_play")
            ftext = audio_ts.get("filler_text")
            filler_fired = fplay is not None
            # filler_lead = silence from user-stop to the filler sounding (the
            # early gap a filler fired post-endpointing CANNOT mask).
            filler_lead_ms = (int((fplay - vad_offset_ts) * 1000)
                              if (fplay is not None and vad_offset_ts is not None)
                              else None)
            filler_dur_ms = self.tts.filler_duration_ms(ftext) if ftext else None
            log.info("[PERF-AUDIO] filler_fired=%s filler_dur=%sms "
                     "filler_lead=%sms overrun=%sms true_reply_lag=%sms",
                     filler_fired, filler_dur_ms, filler_lead_ms,
                     overrun_ms, true_reply_lag_ms)
            asyncio.create_task(broadcast_event("audio_onset", {
                "filler_fired": filler_fired,
                "filler_dur_ms": filler_dur_ms,
                "filler_lead_ms": filler_lead_ms,
                "overrun_ms": overrun_ms,
                "true_reply_lag_ms": true_reply_lag_ms,
            }))

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
                face_trust_name=face_trust_name,
                # Slice A: live regime knob, read once here per turn (re-reads
                # disk so an LT-OS change takes effect without restart).
                situation_regime=runtime_toggles.get("situation_regime"),
                recall_block=recall_block,
                on_first_sentence=_on_first_sentence,
                on_first_audio=_on_first_audio,
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
        build_ms = result.build_ms
        e2e_ms = int((time.time() - t_start) * 1000)

        # True user-perceived latency, computed from the capture timestamps and
        # the wall-clock of the first real reply sentence. reply_lag_ms is the
        # headline number Dan wants on the booth: from when the user STOPPED
        # talking (incl. the endpointing-silence delay) to Timmy's first reply
        # audio. speech_to_reply_ms is the full span from speech onset (incl. the
        # user's own utterance duration). Both None on the text path / if no real
        # sentence was spoken (e.g. a deflection with no TTS).
        reply_lag_ms = None
        speech_to_reply_ms = None
        if first_audio_wall is not None:
            if vad_offset_ts is not None:
                reply_lag_ms = max(0, int((first_audio_wall - vad_offset_ts) * 1000))
            if vad_onset_ts is not None:
                speech_to_reply_ms = max(0, int((first_audio_wall - vad_onset_ts) * 1000))

        log.info("[TIMMY] %s", full_response)
        log.info("[PERF] endpoint=%sms queue=%sms spk=%dms stt=%dms retrieval=%dms build=%dms "
                 "llm_ft=%dms llm=%dms tts=%dms e2e=%dms reply_lag=%sms speech_to_reply=%sms",
                 endpoint_ms, queue_ms, spk_ms, stt_ms, retrieval_ms, build_ms,
                 llm_first_token_ms, llm_total_ms, tts_ms, e2e_ms,
                 reply_lag_ms, speech_to_reply_ms)

        # The turn already broadcast the assistant "turn" event and persisted
        # the assistant turn; the doorway only assembles the metrics report.
        # est_prompt/completion tokens (~4 chars/token) come back on the result.
        est_prompt_tokens = result.est_prompt_tokens
        est_completion_tokens = result.est_completion_tokens

        await broadcast_event("metrics", {
            "endpoint_ms": endpoint_ms,
            "queue_ms": queue_ms,
            "stt_ms": stt_ms,
            "spk_ms": spk_ms,
            "retrieval_ms": retrieval_ms,
            "build_ms": build_ms,
            "llm_first_token_ms": llm_first_token_ms,
            "llm_total_ms": llm_total_ms,
            "tts_ms": tts_ms,
            "e2e_ms": e2e_ms,
            "reply_lag_ms": reply_lag_ms,
            "speech_to_reply_ms": speech_to_reply_ms,
            "turns": self.conversation.turn_count,
            "speaker": speaker_name,
            "est_prompt_tokens": est_prompt_tokens,
            "est_completion_tokens": est_completion_tokens,
        })
        update_metrics(
            last_reply_lag_ms=reply_lag_ms,
            last_speech_to_reply_ms=speech_to_reply_ms,
            last_endpoint_ms=endpoint_ms,
            last_queue_ms=queue_ms,
            last_spk_ms=spk_ms,
            last_stt_ms=stt_ms,
            last_retrieval_ms=retrieval_ms,
            last_build_ms=build_ms,
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
    # Reconcile the postgres speakers table with the freshly loaded id-map so a
    # newly enrolled voiceprint can never FK-fail a facts/memories insert. Runs
    # on every boot, which is also when a new voiceprint goes live. See
    # db/speakers.py.
    from db.speakers import sync_speakers_from_id_map
    await sync_speakers_from_id_map()
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
                # EXPO identity-dialog gate (Dan 2026-07-06): at the booth a
                # stable stranger is EVERY visitor — no new consent offers,
                # and an offer armed pre-flip is dropped silently. Read per
                # tick so a webui flip applies without a restart.
                # Deliberately the PURE predicate (no anchor disjunct): the
                # LED-mic anchor never un-darks the consent FSM — mic-in-hand
                # is implied consent, stored via the name-tell commit instead.
                if not runtime_toggles.identity_dialogs_allowed():
                    orch._face_enroller.drop_gated()
                    continue
                # Live auto-enroll operator switch (LT-OS Services, 2026-07-09):
                # the same toggle that gates the voiceprint streak also gates the
                # interactive face-enroll consent FSM. Read per tick so a booth
                # flip applies without a restart; drop any armed candidate so a
                # pre-flip offer can't complete after the operator turned it off.
                if not runtime_toggles.get("auto_enroll_enabled"):
                    orch._face_enroller.drop_gated()
                    continue
                await orch._face_enroller.observe_faces(
                    full["faces"], full["image_size"], engaged=engaged,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("[AUTOENROLL] monitor tick error", exc_info=True)
                await asyncio.sleep(2.0)

    face_enroll_task = asyncio.create_task(face_enroll_monitor())

    # LED-mic anchor periodic poll (Ruling A, Dan 2026-07-07). The per-turn CV
    # refresh armed the anchor only DURING a turn's face grab, so the gate
    # computed at the TOP of the next turn was the first to see it — a
    # visitor's opening "hi, I'm X" ran dark and leaked to the LLM as
    # ordinary speech (gate-lag finding, review 7-07). Polling keeps the
    # anchor (and its recognized-face binding) fresh BEFORE the first
    # utterance, and fixes the per-turn-only refresh (a silent booth let the
    # anchor decay mid-engagement). CPU-only work (~ms: JPEG decode + YuNet +
    # EdgeFace) — never the VLM (the two-35B GPU-contention finding). Toggles
    # read per tick so booth flips apply without a restart; a fresh STUB
    # anchor is the operator's declaration and skips the tick entirely (F4 —
    # never restamped).
    async def anchor_poll_monitor():
        from presence.face_recognize import poll_anchor_frame
        while True:
            try:
                await asyncio.sleep(
                    float(runtime_toggles.get("anchor_poll_interval_s")))
                if not runtime_toggles.get("anchor_enabled"):
                    continue
                st = anchor.get_anchor()
                if (st is not None and st.source == "stub"
                        and anchor.anchor_active()):
                    continue
                try:
                    r = await orch._face_http.get(
                        config.STREAMERPI_CAPTURE_URL, timeout=1.5)
                    jpeg = r.content if r.status_code == 200 else None
                except Exception:
                    jpeg = None
                if jpeg:
                    await asyncio.to_thread(poll_anchor_frame, jpeg)
            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("[ANCHOR] poll tick error", exc_info=True)
                await asyncio.sleep(2.0)

    anchor_poll_task = asyncio.create_task(anchor_poll_monitor())

    log.info("=== Little Timmy ready on http://0.0.0.0:%d ===", config.WEB_PORT)

    try:
        while True:
            item = await orch.capture.speech_queue.get()
            dequeue_ts = time.time()  # handoff: segment leaves the queue here
            # Queue items are (audio, onset_ts, offset_ts, eou_ts) since
            # 2026-06-29 (eou_ts added to split endpoint_ms/queue_ms out of the
            # booth WAIT remainder). Tolerate the older 3-tuple and a bare
            # ndarray (tests / any legacy producer) -> missing timestamps.
            if isinstance(item, tuple):
                if len(item) >= 4:
                    audio_segment, vad_onset_ts, vad_offset_ts, vad_eou_ts = item[:4]
                else:
                    audio_segment, vad_onset_ts, vad_offset_ts = item[:3]
                    vad_eou_ts = None
            else:
                audio_segment, vad_onset_ts, vad_offset_ts, vad_eou_ts = item, None, None, None
            # Hold the turn lock so an in-flight proactive remark finishes (or
            # is excluded) before this reactive turn speaks -- never overlap.
            async with orch._turn_lock:
                orch.vision.pause_polling()
                try:
                    await orch.process_speech(
                        audio_segment,
                        vad_onset_ts=vad_onset_ts,
                        vad_offset_ts=vad_offset_ts,
                        vad_eou_ts=vad_eou_ts,
                        dequeue_ts=dequeue_ts,
                    )
                finally:
                    orch.vision.resume_polling()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        anchor_poll_task.cancel()
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
