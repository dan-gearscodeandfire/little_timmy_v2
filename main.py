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
from llm.prompt_builder import build_ephemeral_block, build_messages
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
from speaker.identifier import SpeakerIdentifier
from web.app import app, init as web_init, broadcast_event, update_metrics
from vision.context import VisionContext
from vision.visual_question import is_visual_question
from conversation.enroll_intent import detect_enroll_intent
from vision.supervisor import BehaviorSupervisor
from presence import (
    RoomLedger,
    fuse_identity,
    FaceHintStreak,
    LookAtPolicy,
)
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


# --- Reply hygiene ---
# 2026-05-11 session repeatedly flagged verbose narration replies where the
# Llama 3B conversation tier treated the [WHAT YOU SEE] vision context as a
# cue to describe the workshop unprompted, violating "1-2 short sentences"
# and "do NOT narrate the scene" rules in the system prompt. Two known
# offenders below; the same canned phrase "a window into the digital world,
# with lines of code scrolling by" came out twice within a few turns.
_NARRATION_PREFIXES = (
    "i'm standing in front of",
    "i'm surrounded by",
    "the workshop is",
    "the room is",
    "the computer monitor behind",
    "you are standing in",
)
_NARRATION_PREFIX_CHECK_AT = 30  # chars
_REPLY_MAX_SENTENCES = 2
_REPLY_VETO_FALLBACK = "Sure."

# When the user explicitly invites a longer reply, allow up to this many
# sentences instead of _REPLY_MAX_SENTENCES. Still bounded — runaway-prone
# narration is still a risk — but enough for a substantive answer like
# "what do you know about me" or "tell me your story".
_REPLY_LONGER_SENTENCES = 6

# Phrases (lowercase substring match on the user turn) that signal the user
# explicitly wants Timmy to speak past the default 2-sentence cap. Matched
# loosely — false positives just lengthen one reply, false negatives are
# the regression we're trying to avoid.
_LONGER_REPLY_PERMISSION_PHRASES = (
    "speak longer",
    "talk longer",
    "longer than usual",
    "longer answer",
    "longer response",
    "go into detail",
    "in detail",
    "in depth",
    "tell me more about",
    "tell me everything",
    "tell me your story",
    "you can be verbose",
    "you may be verbose",
    "open-ended",
    "open ended",
    "long answer",
    "give me a long",
)


def user_invites_longer_reply(user_text: str) -> bool:
    """True if the user's turn contains an explicit permission phrase
    inviting Timmy to speak beyond the default 2-sentence cap."""
    if not user_text:
        return False
    lower = user_text.lower()
    return any(p in lower for p in _LONGER_REPLY_PERMISSION_PHRASES)


def _looks_like_narration(buf: str) -> bool:
    head = buf.lower().lstrip()[:50]
    return any(head.startswith(p) for p in _NARRATION_PREFIXES)


async def filtered_assistant_stream(token_iter, max_sentences: int | None = None):
    """Post-filter the conversation-tier token stream before TTS sees it.

    Two veto paths:
      - Narration prefix (first ~30 chars) -> swallow the rest of the
        upstream and yield a single fallback ("Sure.") so TTS still
        speaks something terse. Tokens are buffered until the prefix
        check has fired so the veto suppresses the entire reply rather
        than letting the first ~29 chars leak to TTS / WS / hot_turns.
      - N sentence terminators (.!?) accumulated -> swallow the rest of
        the upstream so TTS / persistence / WS broadcast all see the
        truncated form. Default N is _REPLY_MAX_SENTENCES (2). Callers
        can override via `max_sentences` (e.g. _REPLY_LONGER_SENTENCES
        when the user invited a longer reply via
        `user_invites_longer_reply`).

    Sentence terminators inside abbreviations are not a concern here:
    Llama 3B almost never emits "Mr." / "Dr." in this skeleton-cohost
    persona.
    """
    cap = max_sentences if max_sentences and max_sentences > 0 else _REPLY_MAX_SENTENCES
    accum = ""
    buffered = ""
    sentence_count = 0
    narration_checked = False
    drained = False
    async for token in token_iter:
        if drained:
            # Keep iterating to let the upstream finish cleanly; drop the
            # tokens silently. Upstream HTTP connection stays healthy.
            continue
        accum += token
        if not narration_checked:
            # Hold every token until accum reaches the prefix-check window.
            # Without this hold, the first ~29 chars would already be on
            # TTS / WS / hot_turns before the veto fires, defeating it.
            buffered += token
            if len(accum) < _NARRATION_PREFIX_CHECK_AT:
                continue
            narration_checked = True
            if _looks_like_narration(accum):
                log.warning("[POST-FILTER] vetoed narration reply (first 60 chars): %r",
                            accum[:60])
                drained = True
                yield _REPLY_VETO_FALLBACK
                continue
            # Safe prefix — flush the buffer in one chunk and resume
            # streaming. Sentence counting catches up on the buffered text.
            sentence_count += sum(1 for ch in buffered if ch in ".!?")
            yield buffered
            buffered = ""
            if sentence_count >= cap:
                log.info("[POST-FILTER] capped reply at %d sentences (%d chars)",
                         cap, len(accum))
                drained = True
            continue
        sentence_count += sum(1 for ch in token if ch in ".!?")
        yield token
        if sentence_count >= cap:
            log.info("[POST-FILTER] capped reply at %d sentences (%d chars)",
                     cap, len(accum))
            drained = True
    # End-of-stream flush: a reply shorter than the prefix-check window
    # never triggered the narration check. Every entry in _NARRATION_PREFIXES
    # is <30 chars, so a reply that is exactly "the room is" (15 chars) and
    # then stops would otherwise slip through. Run the check defensively.
    # Also apply the sentence cap here — short replies that fit entirely
    # inside the buffer bypass the per-token cap check otherwise.
    if buffered and not drained:
        if _looks_like_narration(accum):
            log.warning("[POST-FILTER] vetoed short narration reply: %r", accum[:60])
            yield _REPLY_VETO_FALLBACK
        else:
            terminator_positions = [i for i, ch in enumerate(buffered) if ch in ".!?"]
            if len(terminator_positions) > cap:
                cutoff = terminator_positions[cap - 1] + 1
                log.info("[POST-FILTER] capped short reply at %d sentences (%d chars)",
                         cap, cutoff)
                buffered = buffered[:cutoff]
            yield buffered


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

        # --- STT ---
        t0 = time.time()
        user_text = await transcribe(audio)
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

        speaker_name = speaker_result.name
        speaker_db_id = speaker_result.speaker_id

        # Trigger vision capture on speech detection
        asyncio.create_task(self.vision.trigger_capture("speech"))

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
                await self._ask_speaker_name(unknown_info, t_start, stt_ms, spk_ms)
                return

        await self._generate_response(
            user_text, stt_ms, t_start,
            speaker_name=speaker_name,
            speaker_db_id=speaker_db_id,
            spk_ms=spk_ms,
        )

    async def _stream_to_tts(
        self,
        messages: list[dict],
        *,
        max_sentences: int | None = None,
        on_first_token=None,
        on_first_sentence=None,
    ) -> tuple[str, dict]:
        """Stream LLM tokens, sentence-buffer them, hand each sentence to TTS.

        Returns (full_response, timings) where timings keys are:
          first_token_ms : ms from helper-start to the first model token
                           (None if the stream produced nothing)
          first_tts_ms   : ms from helper-start to the first TTS-speak call
                           (None if no sentence was spoken)
          total_ms       : ms from helper-start to helper-return

        Optional callbacks (each can be an async function):
          on_first_token    : fires once, when the first token lands
          on_first_sentence : fires once, just before the first TTS-speak call

        M2 2026-05-14: extracted from three near-identical loops in
        _ask_speaker_name, _confirm_name, and _generate_response. Per-token
        broadcast_event("token", ...) and sentence-boundary detection are
        identical; the side-effect hooks let the generate path layer in
        eye_led + supervisor calls without duplicating the loop body.
        """
        t_start = time.time()
        full_response = ""
        sentence_buffer = ""
        first_token_time: float | None = None
        first_tts_time: float | None = None

        async for token in filtered_assistant_stream(
            stream_conversation(messages), max_sentences=max_sentences,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                if on_first_token is not None:
                    await on_first_token()
            full_response += token
            sentence_buffer += token
            await broadcast_event("token", {"content": token})
            stripped = sentence_buffer.rstrip()
            if stripped and stripped[-1] in ".?!;:":
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                if sentence:
                    if first_tts_time is None:
                        first_tts_time = time.time()
                        if on_first_sentence is not None:
                            await on_first_sentence()
                    await self.tts.speak(sentence)

        if sentence_buffer.strip():
            if first_tts_time is None:
                first_tts_time = time.time()
                if on_first_sentence is not None:
                    await on_first_sentence()
            await self.tts.speak(sentence_buffer.strip())

        end_time = time.time()
        return full_response, {
            "first_token_ms": int((first_token_time - t_start) * 1000) if first_token_time else None,
            "first_tts_ms":   int((first_tts_time - t_start) * 1000) if first_tts_time else None,
            "total_ms":       int((end_time - t_start) * 1000),
        }

    async def _ask_speaker_name(self, unknown_info, t_start, stt_ms, spk_ms):
        """Generate a response asking an unknown speaker for their name."""
        known_names = [
            ks.name for ks in self.speaker_id_module._known_speakers
        ]
        known_str = ", ".join(n.title() for n in known_names if n != "timmy")

        last_quote = unknown_info.last_text[:80] if unknown_info.last_text else "something"

        prompt_text = (
            f"A new person has joined the conversation. I know {known_str} is here, "
            f"but someone new just said: \"{last_quote}\". "
            f"Ask them for their name in a friendly, in-character way."
        )

        history = self.conversation.build_history_messages()
        ephemeral = build_ephemeral_block(memories=[], facts=[])
        messages = build_messages(history, ephemeral, prompt_text)

        full_response, _timings = await self._stream_to_tts(messages)
        llm_ms = _timings["total_ms"]
        e2e_ms = int((time.time() - t_start) * 1000)

        log.info("[TIMMY] %s", full_response)
        log.info("[PERF] stt=%dms spk=%dms llm=%dms e2e=%dms (name solicitation)",
                 stt_ms, spk_ms, llm_ms, e2e_ms)

        await broadcast_event("turn", {"role": "assistant", "content": full_response})
        await self.conversation.add_assistant_turn(full_response)

        # Set a flag so next utterance from this unknown triggers name capture
        self._pending_name_capture = unknown_info.temp_id

    _pending_name_capture: str | None = None
    _pending_name_confirm: dict | None = None  # {"temp_id": str, "name": str}
    # Snapshot of the most-recent finalized turn so /api/feedback/manual_flag
    # can capture the same (user, assistant, full system prompt) tuple the
    # verbal-feedback path captures.
    _last_finalized_turn: dict | None = None

    async def process_text_input(self, text: str):
        """Process typed text input (from web UI)."""
        t_start = time.time()
        log.info("[USER:text] %s", text)
        await broadcast_event("turn", {"role": "user", "content": text})
        await self.conversation.add_user_turn(text)
        await self._generate_response(text, stt_ms=0, t_start=t_start,
                                       speaker_name="dan", speaker_db_id=1, spk_ms=0)

    async def _confirm_name(self, name: str, t_start: float, stt_ms: int, spk_ms: int):
        """Have Timmy confirm the extracted name before committing it."""
        history = self.conversation.build_history_messages()
        ephemeral = build_ephemeral_block(memories=[], facts=[])
        confirm_prompt = (
            f'You just heard someone say their name is "{name.title()}". '
            f'Repeat the name back to confirm, like "Did you say {name.title()}?" '
            f'Keep it brief and in-character.'
        )
        messages = build_messages(history, ephemeral, confirm_prompt)

        full_response, _timings = await self._stream_to_tts(messages)
        llm_ms = _timings["total_ms"]
        e2e_ms = int((time.time() - t_start) * 1000)

        log.info("[TIMMY] %s (name confirmation)", full_response)
        log.info("[PERF] stt=%dms spk=%dms llm=%dms e2e=%dms (name confirm)",
                 stt_ms, spk_ms, llm_ms, e2e_ms)

        await broadcast_event("turn", {"role": "assistant", "content": full_response})
        await self.conversation.add_assistant_turn(full_response)

    async def _generate_response(self, user_text: str, stt_ms: int, t_start: float,
                                  speaker_name: str = "dan",
                                  speaker_db_id: int | None = 1,
                                  spk_ms: int = 0):
        """Core response pipeline: retrieve → prompt → LLM stream → TTS."""

        # --- Check if we're waiting for name confirmation ---
        if self._pending_name_confirm and speaker_name.startswith("unknown_"):
            lower = user_text.lower().strip().rstrip(".!?,")
            if any(w in lower for w in ("yes", "yeah", "yep", "correct", "that's right",
                                         "right", "sure", "yup", "exactly", "mhm")):
                name = self._pending_name_confirm["name"]
                self.speaker_id_module.assign_name(
                    self._pending_name_confirm["temp_id"], name
                )
                speaker_name = name
                log.info("Confirmed name: %s for %s", name,
                         self._pending_name_confirm["temp_id"])
                self._pending_name_confirm = None
            elif any(w in lower for w in ("no", "nope", "nah", "wrong")):
                log.info("Name rejected by user, will re-ask next stable utterance")
                temp_id = self._pending_name_confirm["temp_id"]
                self._pending_name_confirm = None
                # Allow re-asking by resetting name_asked
                for us in self.speaker_id_module._unknown_speakers:
                    if us.temp_id == temp_id:
                        us.name_asked = False
                        break
            else:
                # They said something else — maybe the actual name this time
                name = self._extract_name_from_response(user_text)
                if name:
                    self._pending_name_confirm["name"] = name
                    await self._confirm_name(name, t_start, stt_ms, spk_ms)
                    return
                else:
                    self._pending_name_confirm = None

        # --- Check if this is a name response to our solicitation ---
        if self._pending_name_capture and speaker_name.startswith("unknown_"):
            name = self._extract_name_from_response(user_text)
            if name:
                # Ask for confirmation instead of immediately assigning
                self._pending_name_confirm = {
                    "temp_id": self._pending_name_capture,
                    "name": name,
                }
                self._pending_name_capture = None
                await self._confirm_name(name, t_start, stt_ms, spk_ms)
                return
            else:
                log.info("Could not extract name from: %r", user_text)
            self._pending_name_capture = None

        # --- Memory Retrieval (parallel) ---
        t1 = time.time()

        words = user_text.lower().split()
        subjects = []
        for i, w in enumerate(words):
            if w == "my" and i + 1 < len(words):
                subjects.append(f"my {words[i+1]}")
        # Note: speaker's own facts are fetched via get_facts_about_speaker
        # below (which aliases subject across {speaker_name, user, i, me}),
        # so don't add the speaker to `subjects` here -- it would double-count.
        speaker_for_facts = speaker_name if speaker_name != "timmy" else "dan"

        gather_args = [
            retrieve(user_text, top_k=config.RETRIEVAL_TOP_K),
            get_all_facts_for_prompt(subjects, limit=5) if subjects else _empty_facts(),
            get_facts_about_speaker(speaker_for_facts, speaker_db_id, limit=5),
        ]
        if self._presence_enabled:
            gather_args.append(self._fetch_face_safe())

        gathered = await asyncio.gather(*gather_args)
        retrieved_memories = gathered[0]
        # gather_args[1] = non-speaker subjects ("my X" patterns)
        # gather_args[2] = speaker's own facts via alias-aware retrieval
        # Merge, dedupe by fact id, prefer fresher (speaker-side is already
        # learned_at-ordered).
        _non_speaker_facts = gathered[1]
        _speaker_facts = gathered[2]
        _seen_fact_ids = set()
        resolved_facts = []
        for _f in (*_speaker_facts, *_non_speaker_facts):
            if _f.id in _seen_fact_ids:
                continue
            _seen_fact_ids.add(_f.id)
            resolved_facts.append(_f)
        face_obs = gathered[3] if len(gathered) > 3 else None

        # --- Presence: voice + face fusion ---
        fusion_source = None
        face_hint_name = None
        presence_state = None
        if self._presence_enabled:
            self.room_ledger.update_from_voice(speaker_name)
            if face_obs is not None:
                self.room_ledger.update_from_face(face_obs)
            verdict = fuse_identity(
                voice_name=speaker_name,
                voice_is_unknown=speaker_name.startswith("unknown_"),
                face=face_obs,
                face_conf_threshold=config.FACE_CONF_THRESHOLD,
                head_steady_min_ms=config.HEAD_STEADY_MS,
            )
            # Track face_hint streak for auto voice-enrollment.
            # Use pre-override speaker_name (the unknown_N temp_id).
            streak_temp_id = speaker_name if speaker_name.startswith("unknown_") else None
            streak_face_name = (
                verdict.face_hint_name if verdict.resolution_source == "face_hint" else None
            )
            streak_hit = self._face_hint_streak.observe(streak_face_name, streak_temp_id)
            if streak_hit is not None:
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
        retrieval_ms = int((time.time() - t1) * 1000)
        log.info("[MEMORY] %d memories, %d facts (%dms)",
                 len(retrieved_memories), len(resolved_facts), retrieval_ms)

        await broadcast_event("retrieval", {
            "memories": [
                {"type": m.type, "content": m.content[:200], "score": round(m.score, 3)}
                for m in retrieved_memories
            ],
            "facts": [
                {"subject": f.subject, "predicate": f.predicate, "value": f.value}
                for f in resolved_facts
            ],
        })

        # --- Build Prompt ---
        history = self.conversation.build_history_messages()

        # Vision context on visual questions: always use the cached scene
        # description rather than blocking the response on a fresh VLM call.
        # `_handle_speech` (~line 393) already fired
        # `asyncio.create_task(self.vision.trigger_capture("speech"))` on this
        # turn, plus the periodic poll updates the cache on its own cadence
        # (VISION_PERIODIC_INTERVAL=10s, scene-change gated). The cache is
        # typically <10s old when a visual question lands.
        #
        # The previous `await trigger_capture("visual_question")` path added
        # ~5-8s of Qwen3.6 :8084 latency to the response (observed e2e=10.8s
        # on 2026-05-12 00:28 "Can you tell me what you see right now?", with
        # llm=1.1s + tts=0.7s and ~9s unaccounted = blocking VLM call). For a
        # forced refresh on visual_question specifically, schedule with
        # asyncio.create_task — never await on the response path. Mirrors the
        # rollup-detach fix in conversation/manager.py (commit d5d434a).
        #
        # Trade-off: on a true cold-start (no cached scene yet, periodic
        # poll hasn't completed once), `vision_desc` will be None and Llama
        # answers without visual context. The speech-trigger fires the same
        # capture; the second visual question of the session lands with
        # context. For OpenSauce this beats the 10s blocking pattern by a
        # wide margin.
        visual_q = is_visual_question(user_text)
        vision_desc = self.vision.get_description()
        if visual_q:
            log.info(
                "[VISION] Visual question detected; using cached scene "
                "(speech-trigger refresh runs in background)"
            )

        ephemeral = build_ephemeral_block(
            memories=retrieved_memories,
            facts=resolved_facts,
            speaker_name=speaker_name,
            vision_description=vision_desc,
            visual_question=visual_q,
            presence_state=presence_state,
            fusion_source=fusion_source,
            face_hint_name=face_hint_name,
        )
        messages = build_messages(history, ephemeral, user_text)

        # --- Stream LLM + TTS Pipeline ---
        # Honor explicit user permission to speak past the 2-sentence cap
        # ("you can speak longer than usual", "tell me more about", etc).
        # Detected on the user_text only -- Llama 3B's reply itself can't
        # promote the cap. Bounded to _REPLY_LONGER_SENTENCES so unbounded
        # narration is still impossible.
        cap = _REPLY_LONGER_SENTENCES if user_invites_longer_reply(user_text) else None

        # Filler-word gate. 50% on non-curt prompts: queue a short
        # pre-rendered filler ahead of the real reply so the user
        # hears Timmy starting to speak immediately instead of waiting
        # for first-sentence-ready (~190 ms median on Llama 3B). Plays
        # via the same TTS playback loop so it queues naturally ahead
        # of the first real sentence; cooldown=0 between them.
        if audio_fillers.should_fire(user_text):
            asyncio.create_task(self.tts.speak_filler(audio_fillers.pick()))

        async def _on_first_sentence():
            # Eye LED + supervisor side effects fire the moment the first
            # sentence is about to hit TTS. Both are fire-and-forget so they
            # don't delay the speak() call.
            asyncio.create_task(self.supervisor.on_tts_start())
            asyncio.create_task(eye_led.notify("SPEAKING"))

        full_response, _timings = await self._stream_to_tts(
            messages,
            max_sentences=cap,
            on_first_sentence=_on_first_sentence,
        )

        # Notify supervisor that TTS is done + Eye LED back to listening.
        asyncio.create_task(self.supervisor.on_tts_end())
        asyncio.create_task(eye_led.notify("AI_CONNECTED"))

        llm_first_token_ms = _timings["first_token_ms"] or 0
        llm_total_ms = _timings["total_ms"]
        tts_ms = _timings["first_tts_ms"] or 0
        e2e_ms = int((time.time() - t_start) * 1000)

        log.info("[TIMMY] %s", full_response)
        log.info("[PERF] stt=%dms spk=%dms retrieval=%dms llm_ft=%dms llm=%dms tts=%dms e2e=%dms",
                 stt_ms, spk_ms, retrieval_ms, llm_first_token_ms, llm_total_ms, tts_ms, e2e_ms)

        await broadcast_event("turn", {"role": "assistant", "content": full_response})
        # Bundle A 00:37 reframe: surface the estimated prompt token count
        # for this turn so the LT-OS Latency panel can show "1234 tokens sent
        # to LLM" alongside the timing metrics. ~4 chars/token English
        # heuristic matches conversation.manager.estimate_tokens.
        _prompt_chars = sum(len(m.get("content", "") or "") for m in messages)
        est_prompt_tokens = max(1, _prompt_chars // 4)
        est_completion_tokens = max(0, len(full_response) // 4)

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

        await self.conversation.add_assistant_turn(full_response)

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

        # --- Async Memory Formation (fire-and-forget) ---
        await extract_and_store(user_text, full_response,
                                speaker_id=speaker_db_id,
                                speaker_name=speaker_name)

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

    @staticmethod
    def _extract_name_from_response(text: str) -> str | None:
        """Try to extract a name from a short response like 'I'm Erin' or 'My name is Erin'.

        Conservative: rejects evasive, playful, or non-name responses.
        """
        text = text.strip().rstrip(".!?,")
        lower = text.lower()

        # Reject obviously evasive/playful responses early
        _EVASIVE_PHRASES = [
            "not sure", "don't know", "i'm not", "none of your",
            "wouldn't you", "guess", "figure it out", "not telling",
            "not allowed", "can't tell", "secret", "classified",
            "why do you", "does it matter", "who cares", "i don't",
            "i can't", "i won't", "not going to", "rather not",
        ]
        if any(phrase in lower for phrase in _EVASIVE_PHRASES):
            return None

        import re
        patterns = [
            r"(?:my name is|i'm|i am|it's|call me|they call me|name's|i go by)\s+(\w+)",
            r"^(\w+)$",  # just a single word
        ]

        # Expanded rejection set
        _NOT_NAMES = {
            # Fillers & affirmations
            "yes", "no", "yeah", "yep", "nope", "nah", "sure", "ok", "okay",
            "hi", "hey", "hello", "bye", "thanks",
            # Articles & pronouns
            "the", "a", "an", "this", "that", "it", "i",
            # Question words
            "what", "who", "why", "how", "when", "where", "which",
            # Common verbs/adverbs
            "well", "just", "um", "uh", "like", "really", "actually",
            "here", "there", "going", "doing", "trying", "thinking",
            # Adjectives that match "I'm X" but aren't names
            "not", "fine", "good", "great", "tired", "busy", "sorry",
            "happy", "sad", "bored", "confused", "lost", "done",
            "sure", "ready", "allowed", "able", "afraid", "certain",
            "kidding", "joking", "serious", "curious", "interested",
            # Negative constructs
            "nobody", "nothing", "none", "never",
        }

        for pattern in patterns:
            m = re.search(pattern, lower)
            if m:
                name = m.group(1)
                if name not in _NOT_NAMES and len(name) >= 2:
                    return name
        return None


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
                current_people = record.people or []
                if set(p.lower() for p in current_people) != set(p.lower() for p in last_people):
                    was_empty = len(last_people) == 0
                    await orch.supervisor.on_vision_people_changed(current_people, was_empty)
                    last_people = current_people
            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("Vision people monitor error")
                await asyncio.sleep(5.0)

    vision_monitor_task = asyncio.create_task(vision_people_monitor())

    log.info("=== Little Timmy ready on http://0.0.0.0:%d ===", config.WEB_PORT)

    try:
        while True:
            audio_segment = await orch.capture.speech_queue.get()
            orch.vision.pause_polling()
            try:
                await orch.process_speech(audio_segment)
            finally:
                orch.vision.resume_polling()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        await orch.supervisor.stop()
        await orch.vision.stop()
        await orch.capture.stop()
        await orch.tts.stop()
        await close_pool()
        server.should_exit = True


if __name__ == "__main__":
    asyncio.run(main())
