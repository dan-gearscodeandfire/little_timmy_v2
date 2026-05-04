"""Little Timmy — Voice Assistant Orchestrator.

Main event loop tying together:
  Audio capture → Speaker ID → STT → Memory retrieval → Prompt assembly → LLM → TTS → Async memory
"""

import asyncio
import logging
import sys
import os
import time
import httpx
import numpy as np

import config
import json as _json
from pathlib import Path
from db import migrate
from db.connection import get_pool, close_pool
from audio.capture import AudioCapture
from stt.client import transcribe
from tts.engine import TTSEngine
from llm.client import stream_conversation
from llm.prompt_builder import build_ephemeral_block, build_messages
from memory.retrieval import retrieve
from memory.facts import get_all_facts_for_prompt, resolve_entity
from memory.extraction import extract_and_store
from conversation.manager import ConversationManager
from speaker.identifier import SpeakerIdentifier
from web.app import app, init as web_init, broadcast_event, update_metrics
from vision.context import VisionContext
from vision.visual_question import is_visual_question
from vision.supervisor import BehaviorSupervisor

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


class Orchestrator:
    def __init__(self):
        self.conversation = ConversationManager()
        self.tts = TTSEngine(config.PIPER_MODEL)
        self.capture = AudioCapture()
        self.speaker_id_module = SpeakerIdentifier()
        self.vision = VisionContext()
        self.supervisor = BehaviorSupervisor()

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

        t2 = time.time()
        full_response = ""
        sentence_buffer = ""

        async for token in stream_conversation(messages):
            full_response += token
            sentence_buffer += token
            stripped = sentence_buffer.rstrip()
            if stripped and stripped[-1] in ".?!;:":
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                if sentence:
                    await self.tts.speak(sentence)

        if sentence_buffer.strip():
            await self.tts.speak(sentence_buffer.strip())

        llm_ms = int((time.time() - t2) * 1000)
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

        t2 = time.time()
        full_response = ""
        sentence_buffer = ""

        async for token in stream_conversation(messages):
            full_response += token
            sentence_buffer += token
            stripped = sentence_buffer.rstrip()
            if stripped and stripped[-1] in ".?!;:":
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                if sentence:
                    await self.tts.speak(sentence)

        if sentence_buffer.strip():
            await self.tts.speak(sentence_buffer.strip())

        llm_ms = int((time.time() - t2) * 1000)
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
        subjects.append(speaker_name if speaker_name != "timmy" else "dan")

        retrieved_memories, resolved_facts = await asyncio.gather(
            retrieve(user_text, top_k=config.RETRIEVAL_TOP_K),
            get_all_facts_for_prompt(subjects, limit=5),
        )
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

        # On-demand vision: if the user is asking a visual question,
        # force a fresh VLM capture and WAIT for the result
        visual_q = is_visual_question(user_text)
        if visual_q and self.vision.enabled:
            log.info("[VISION] Visual question detected, triggering fresh capture")
            fresh_record = await self.vision.trigger_capture("visual_question")
            if fresh_record:
                vision_desc = fresh_record.summary()
                log.info("[VISION] Fresh capture: %s", vision_desc[:100])
            else:
                vision_desc = self.vision.get_description()
        else:
            vision_desc = self.vision.get_description()

        ephemeral = build_ephemeral_block(
            memories=retrieved_memories,
            facts=resolved_facts,
            speaker_name=speaker_name,
            vision_description=vision_desc,
            visual_question=visual_q,
        )
        messages = build_messages(history, ephemeral, user_text)

        # --- Stream LLM + TTS Pipeline ---
        t2 = time.time()
        full_response = ""
        sentence_buffer = ""
        first_token_time = None
        first_tts_time = None

        async for token in stream_conversation(messages):
            if first_token_time is None:
                first_token_time = time.time()

            full_response += token
            sentence_buffer += token

            stripped = sentence_buffer.rstrip()
            if stripped and stripped[-1] in ".?!;:":
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                if sentence:
                    if first_tts_time is None:
                        first_tts_time = time.time()
                        asyncio.create_task(self.supervisor.on_tts_start())
                    await self.tts.speak(sentence)

        if sentence_buffer.strip():
            if first_tts_time is None:
                first_tts_time = time.time()
            await self.tts.speak(sentence_buffer.strip())

        # Notify supervisor that TTS is done
        asyncio.create_task(self.supervisor.on_tts_end())

        llm_first_token_ms = int((first_token_time - t2) * 1000) if first_token_time else 0
        llm_total_ms = int((time.time() - t2) * 1000)
        tts_ms = int((first_tts_time - t2) * 1000) if first_tts_time else 0
        e2e_ms = int((time.time() - t_start) * 1000)

        log.info("[TIMMY] %s", full_response)
        log.info("[PERF] stt=%dms spk=%dms retrieval=%dms llm_ft=%dms llm=%dms tts=%dms e2e=%dms",
                 stt_ms, spk_ms, retrieval_ms, llm_first_token_ms, llm_total_ms, tts_ms, e2e_ms)

        await broadcast_event("turn", {"role": "assistant", "content": full_response})
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
        })
        update_metrics(
            last_stt_ms=stt_ms,
            last_retrieval_ms=retrieval_ms,
            last_llm_first_token_ms=llm_first_token_ms,
            last_llm_total_ms=llm_total_ms,
            last_tts_ms=tts_ms,
            last_e2e_ms=e2e_ms,
            turns=self.conversation.turn_count,
        )

        await self.conversation.add_assistant_turn(full_response)

        # --- Compliment detection for persona tuning (fire-and-forget) ---
        asyncio.create_task(
            self._check_compliment(user_text, full_response, ephemeral, messages)
        )

        # --- Async Memory Formation (fire-and-forget) ---
        await extract_and_store(user_text, full_response,
                                speaker_id=speaker_db_id)

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

    # Load speaker voiceprints
    await asyncio.to_thread(orch.speaker_id_module.load_voiceprints)
    log.info("Speaker identification ready")

    # Start vision pipeline
    await orch.vision.start()
    log.info("Vision pipeline ready (enabled=%s)", orch.vision.enabled)

    # Start behavioral supervisor
    await orch.supervisor.start()
    log.info("Behavioral supervisor ready")

    # Wire TTS suppression to audio capture
    orch.tts._capture = orch.capture

    # Start TTS engine
    await orch.tts.start()
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
            await orch.process_speech(audio_segment)
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
