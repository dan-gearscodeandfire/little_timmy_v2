"""Piper TTS engine — in-process ONNX with raw PCM streaming to audio output."""

import asyncio
import re
import time
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

log = logging.getLogger(__name__)


def _apply_pronunciations(text: str) -> str:
    """Respell name/word manglings before espeak phonemization.

    Whole-word, case-insensitive substitution from config.TTS_PRONUNCIATIONS.
    Whole-word (`\\b`) is essential: a bare replace would rewrite "erin" inside
    "gathering"/"engineering". Fails open (returns text unchanged) so a config
    glitch can never break synthesis."""
    try:
        import config as cfg
        overrides = getattr(cfg, "TTS_PRONUNCIATIONS", None) or {}
        for word, say_as in overrides.items():
            text = re.sub(rf"\b{re.escape(word)}\b", say_as, text, flags=re.IGNORECASE)
    except Exception as e:
        log.warning("pronunciation substitution skipped: %s", e)
    return text


@dataclass
class PlaybackItem:
    """One unit of audio queued for the playback loop.

    kind distinguishes a 'real' utterance (a spoken reply / announce /
    deflection) from a 'filler' (a THINKING beat). on_play_start, if set, is
    called with time.time() the instant this item's sd.play() begins — its TRUE
    audible onset — so the doorway can measure how long a ready reply waited
    behind a filler in this serial queue (the "overrun"), vs the enqueue-time
    stamp it has today. The hook runs on the event loop right before playback,
    so it must be cheap and non-throwing; defer any real work with create_task.
    (2026-06-30, filler-latency instrumentation — measurement only.)"""
    audio: np.ndarray
    sr: int
    cooldown_s: float
    suppress_mic: bool = True
    kind: str = "real"
    on_play_start: Optional[Callable[[float], None]] = None
    # Booth VOX caption (2026-08-11): the human-readable text this clip says,
    # broadcast on the /ws `speaking` event at TRUE audible onset so the booth
    # overlay shows exactly the sentence currently coming out of the speaker.
    # Held here (not derived at the call site) because the caller's string is
    # mangled for synthesis — commas become em dashes and TTS_PRONUNCIATIONS
    # rewrites words — and the display wants the ORIGINAL. None = no caption
    # (fillers, and the supervisor/announce channel, which must never surface
    # its private coaching text on the attendee-facing screen).
    caption: Optional[str] = None
    # Stop generation this clip was ENQUEUED under (2026-08-11). The playback
    # loop drops the clip if the engine's generation has moved on since. Taken
    # at enqueue rather than at dequeue so a Stop cancels everything already in
    # flight, including a clip that was queued a few ms before the Stop and
    # only reaches the front of the queue afterwards.
    gen: int = 0
    # Puppet Mode (2026-08-04): a caller that needs to AWAIT until this clip has
    # finished playing (incl. cooldown) passes an Event; the playback loop sets
    # it in its finally so the awaiter never hangs, even on error/stop.
    done: Optional["asyncio.Event"] = None

# Loaded Piper voices, cached per model_path. Timmy's conversational voice
# (config.PIPER_MODEL) plus any persona voices (e.g. the couples-therapist voice
# the supervisor/announce channel speaks in) coexist here, each loaded once.
_piper_voices: dict[str, object] = {}


async def _broadcast_speech(event_type: str, data: dict) -> None:
    """Fire a booth VOX event on the LT websocket, best-effort.

    Lazy import: tts.engine is imported by the orchestrator well before
    web.app finishes wiring, and a caption is never worth breaking playback
    over — any failure is swallowed so the speaker keeps talking."""
    try:
        from web.app import broadcast_event
        await broadcast_event(event_type, data)
    except Exception as e:
        log.debug("speech broadcast (%s) skipped: %s", event_type, e)


def _tts_muted() -> bool:
    """Live read of the mouth-mute toggle. Lazy import keeps this module
    import-clean; any failure fails OPEN (not muted) so a persistence glitch
    can never silence Timmy."""
    try:
        from persistence import runtime_toggles
        return bool(runtime_toggles.get("tts_muted"))
    except Exception:
        return False


def _load_voice(model_path: str):
    """Load a Piper voice model (lazy, cached per model_path).

    Keying the cache by path lets a second persona voice (the couples-therapist
    voice) load alongside Timmy's conversational voice instead of the old
    singleton returning whichever loaded first."""
    voice = _piper_voices.get(model_path)
    if voice is not None:
        return voice

    from piper import PiperVoice
    voice = PiperVoice.load(model_path)
    _piper_voices[model_path] = voice
    log.info("Loaded Piper voice from %s (sample_rate=%d)",
             model_path, voice.config.sample_rate)
    return voice


def _synthesize_raw(text: str, model_path: str) -> tuple[np.ndarray, int]:
    """Synthesize text to raw float32 numpy array. Returns (audio, sample_rate)."""
    from piper.config import SynthesisConfig
    import config as cfg
    voice = _load_voice(model_path)
    text = _apply_pronunciations(text)
    syn_config = SynthesisConfig(length_scale=cfg.TTS_LENGTH_SCALE)
    chunks = []
    sr = voice.config.sample_rate
    for chunk in voice.synthesize(text, syn_config=syn_config):
        chunks.append(chunk.audio_float_array)
        sr = chunk.sample_rate
    if not chunks:
        return np.array([], dtype=np.float32), sr
    return np.concatenate(chunks), sr


class TTSEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Queue items are (audio, sample_rate, post_cooldown_s). real speak()
        # calls use 0.5 s reverb-die-down; filler calls use 0.0 so the next
        # real sentence plays back-to-back against the filler (no extra pause).
        self._playback_queue: asyncio.Queue[PlaybackItem | None] = asyncio.Queue()
        self._capture = None  # set by orchestrator for TTS suppression
        self._playback_task: asyncio.Task | None = None
        # Pre-rendered filler audio cache, keyed by text. Populated by
        # prewarm_fillers() at startup; speak_filler() consults this before
        # falling back to live Piper synthesis.
        self._filler_cache: dict[str, tuple[np.ndarray, int]] = {}
        # Stop generation, bumped by stop_playback (2026-08-11). Every clip
        # snapshots it at dequeue; a mismatch means "a Stop happened after you
        # were handed this clip", which cancels it whether it is still waiting
        # to start or already playing. Needed because Puppet lines are now
        # several clips, so Stop routinely lands between two of them.
        #
        # INVARIANT this replaced a lock with: ALL sounddevice stream calls
        # (play / stop) happen on the event-loop thread — see _await_clip.
        # Two threads touching one PortAudio stream core-dumped the service.
        self._stop_gen = 0

    async def start(self):
        """Pre-load the model and start the playback loop."""
        await asyncio.to_thread(_load_voice, self.model_path)
        self._playback_task = asyncio.create_task(self._playback_loop())
        log.info("TTS engine started")

    async def speak(self, text: str, force: bool = False,
                    voice_model: str | None = None,
                    suppress_mic: bool = True,
                    on_play_start: Callable[[float], None] | None = None,
                    caption: bool = True):
        """Synthesize text and queue raw PCM for playback. Non-blocking.

        force=True bypasses the mouth-mute (tts_muted) — used by the supervisor
        /api/announce channel so Claude can still speak to Dan while Timmy's own
        conversational voice is muted. Muted speak() skips the enqueue entirely,
        so capture.suppressed never fires and the mic stays open.

        voice_model: path to an alternate Piper voice to synthesize THIS
        utterance in (default = Timmy's conversational voice, self.model_path).
        Used by the supervisor/couples-therapist channel so it speaks in its own
        distinct voice.

        suppress_mic: when True (default) _playback_loop gates the mic
        (capture.suppressed) for the duration, so Timmy never hears this via STT
        (no loopback) — the normal, safe behavior for every voice. When False,
        the mic stays OPEN during playback so Timmy DOES hear and transcribe it
        as an incoming turn (e.g. a test where the couples-therapist speaks TO
        Timmy). The persona voice comes in as an unknown speaker unless its
        voiceprint is enrolled.

        caption: surface this line on the booth VOX overlay the instant it
        starts playing. Default True — Timmy's conversational voice IS what
        the booth captions. The supervisor/announce channel passes False."""
        if not force and _tts_muted():
            return
        display_text = text.strip()
        # Replace commas with em dashes for shorter TTS pauses
        text = text.replace(",", " —")
        if not text.strip():
            return
        audio, sr = await asyncio.to_thread(
            _synthesize_raw, text, voice_model or self.model_path)
        if len(audio) > 0:
            await self._playback_queue.put(PlaybackItem(
                audio=audio, sr=sr, cooldown_s=0.5, suppress_mic=suppress_mic,
                kind="real", on_play_start=on_play_start, gen=self._stop_gen,
                caption=display_text if caption else None))

    async def speak_blocking(self, text: str,
                             voice_model: str | None = None,
                             suppress_mic: bool = True) -> float:
        """Speak `text` and AWAIT until playback (incl. cooldown) finishes;
        return the clip duration in seconds (0.0 if nothing was spoken).

        Puppet Mode's browser sequencer needs each line to FINISH before it
        advances to the pause + next line, but speak() is fire-and-forget. This
        variant enqueues with a done-Event and awaits it. It also intentionally
        does NOT honor the mouth-mute (tts_muted) — a stray mute toggle must
        never silently eat a scripted puppet line (same rationale as announce's
        force=True). voice_model=None => Timmy's own conversational voice.

        Single-clip case of speak_sequence_blocking; see there for the
        pipelining and captioning contract."""
        return await self.speak_sequence_blocking(
            [text], voice_model=voice_model, suppress_mic=suppress_mic)

    async def speak_sequence_blocking(self, sentences: list[str],
                                      voice_model: str | None = None,
                                      suppress_mic: bool = True,
                                      abort: Callable[[], bool] | None = None
                                      ) -> float:
        """Speak `sentences` back-to-back as SEPARATE clips and await the last.

        This is what makes a scripted Puppet line behave like a generated
        reply. The live engine (conversation.turn._stream_and_speak) hands TTS
        one sentence at a time as the stream crosses each boundary, so the
        booth's VOX band advances sentence by sentence and the mic gate stays
        closed across the whole run. Speaking a typed multi-sentence line as
        ONE clip captioned the entire paragraph at once — visibly not what the
        real path does.

        Synthesis is deliberately interleaved with playback rather than done
        up front: sentence N+1 is rendered while N is already coming out of the
        speaker, exactly as the live path's fire-and-forget speak() calls
        pipeline. Rendering everything first would delay first audio by the
        whole line's synthesis time and open a gap the live path never has.

        abort: polled between sentences; return True to stop enqueuing the rest
        (Puppet Stop). Anything already queued is cut by stop_playback().

        Returns the total audible seconds enqueued (excluding cooldowns)."""
        done: asyncio.Event | None = None
        total_s = 0.0
        for sentence in sentences:
            if abort is not None and abort():
                break
            display_text = sentence.strip()
            spoken = sentence.replace(",", " —")
            if not spoken.strip():
                continue
            audio, sr = await asyncio.to_thread(
                _synthesize_raw, spoken, voice_model or self.model_path)
            if len(audio) == 0:
                continue
            # Re-check AFTER synthesis: Piper takes a few hundred ms, and a
            # Stop pressed inside that window would otherwise be beaten by the
            # clip it was meant to cancel — the pre-synthesis check has already
            # passed by then, so this sentence would enqueue under the NEW
            # generation and play in full.
            if abort is not None and abort():
                break
            done = asyncio.Event()
            await self._playback_queue.put(PlaybackItem(
                audio=audio, sr=sr, cooldown_s=0.5, suppress_mic=suppress_mic,
                kind="real", done=done, caption=display_text,
                gen=self._stop_gen))
            total_s += len(audio) / float(sr)
        # Only the LAST clip is awaited — the queue is serial, so its done
        # Event firing means every earlier one has already played out.
        if done is not None:
            await done.wait()
        return total_s

    def stop_playback(self) -> None:
        """Cut any in-flight playback immediately (Puppet Mode Stop / exit).

        Aborts the current sd.play, drains queued items — signaling their done
        Events so blocking awaiters unblock rather than hang — and reopens the
        mic gate. Best-effort: safe to call when nothing is playing."""
        # Bump the generation FIRST. That cancels both a clip already dequeued
        # but not yet started, and the _await_clip that is timing the clip
        # currently playing — which is what actually cuts the audio, since
        # that helper owns every sounddevice call on the event-loop thread.
        # This runs on the event loop too (called from the HTTP handlers), so
        # the sd.stop below cannot overlap the loop's own sd.play/sd.stop.
        self._stop_gen += 1
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        drained = []
        while True:
            try:
                drained.append(self._playback_queue.get_nowait())
            except Exception:
                break
        for it in drained:
            if it is not None and getattr(it, "done", None) is not None:
                it.done.set()
        if self._capture:
            self._capture.suppressed = False

    async def prewarm_fillers(self, texts) -> None:
        """Load the frozen filler .wav assets into _filler_cache.

        Fillers are pre-rendered to committed .wav files (audio/fillers_wav,
        produced by audio.render_fillers) rather than synthesized at startup,
        so the clips are locked/curated and identical across boots and Piper
        voice changes (Dan 2026-06-27). This just reads ~10 small WAVs off
        disk into RAM — no Piper inference on the boot path.

        A missing clip (e.g. the FILLERS tuple was edited but render_fillers
        wasn't re-run) falls back to live synthesis for that one entry and
        logs a warning, so a stale asset set degrades rather than going
        silent.
        """
        import soundfile as sf
        from audio import fillers as _fillers

        loaded = synthesized = 0
        for text in texts:
            if text in self._filler_cache:
                continue
            path = _fillers.wav_path(text)
            if path.exists():
                audio, sr = await asyncio.to_thread(sf.read, str(path), dtype="float32")
                self._filler_cache[text] = (audio, sr)
                loaded += 1
            else:
                log.warning("filler .wav missing for %r (%s); synthesizing live. "
                            "Run: python -m audio.render_fillers", text, path.name)
                audio, sr = await asyncio.to_thread(_synthesize_raw, text, self.model_path)
                if len(audio) > 0:
                    self._filler_cache[text] = (audio, sr)
                    synthesized += 1
        log.info("TTS filler cache ready (%d entries: %d from .wav, %d synthesized)",
                 len(self._filler_cache), loaded, synthesized)

    def filler_duration_ms(self, text: str) -> int | None:
        """Audible length (ms) of a cached filler, or None if not cached.

        Cheap len/sr read off the prewarmed cache — used by the doorway's
        latency instrumentation to log filler busy-time without touching disk."""
        cached = self._filler_cache.get(text)
        if cached is None:
            return None
        audio, sr = cached
        return int(len(audio) / sr * 1000) if sr else None

    async def speak_filler(self, text: str,
                           on_play_start: Callable[[float], None] | None = None) -> None:
        """Queue a pre-rendered filler. Falls through to speak() on miss.

        2026-05-15: cooldown bumped 0.0 → 0.4 s to cover the reverb tail of
        the filler word. With cooldown=0.0, mic suppression released the
        moment `sd.wait()` returned, but the speaker's reverb still hit the
        mic ~50–200 ms later. Whisper then transcribed the echo as a user
        turn (observed: phantom `[Dan]: Wow.` and `[Dan]: my name.` during
        the 2026-05-15 session). 0.4 s is shorter than the main-speech
        cooldown (0.5 s) so the back-to-back-with-main-TTS goal is mostly
        preserved; LLM warm-up usually covers the small gap anyway.
        """
        if _tts_muted():
            return
        cached = self._filler_cache.get(text)
        if cached is None:
            await self.speak(text, on_play_start=on_play_start)
            return
        audio, sr = cached
        await self._playback_queue.put(PlaybackItem(
            audio=audio, sr=sr, cooldown_s=0.4, kind="filler",
            on_play_start=on_play_start, gen=self._stop_gen))

    async def _await_clip(self, sd, clip_s: float, gen: int,
                          until=None) -> bool:
        """Wait out a playing clip. Returns True if it was cut short.

        Deliberately NOT sounddevice's sd.wait(). sd.wait() closes the stream
        in its own finally (sounddevice._CallbackContext.wait), so calling it
        from a worker thread while stop_playback() calls sd.stop() on the event
        loop means two threads tearing down the SAME PortAudio stream. That
        double-teardown is not theoretical: it core-dumped the service outright
        (`free(): corrupted unsorted chunks` -> SIGABRT, 2026-08-11) and, when
        it didn't crash, hung a worker inside Stream.close() forever — the
        playback loop then wedged silently while the rest of LT looked healthy.

        Instead the clip is timed out on its own known duration and every
        sounddevice call (play / stop) is made from the event loop, so stream
        lifecycle is single-threaded by construction. The 20 ms tick is the
        same cadence the filler preemption loop always used.

        gen: the stop generation this clip belongs to; a Stop bumps it and
        cuts playback here. until: optional extra predicate to cut on (filler
        preemption when a real reply lands behind it).
        """
        def _cancelled() -> bool:
            return gen != self._stop_gen or (until is not None and until())

        deadline = time.monotonic() + clip_s
        cut = False
        while time.monotonic() < deadline:
            if _cancelled():
                cut = True
                break
            await asyncio.sleep(0.02)

        # Drain phase. len(audio)/sr is when the last sample was HANDED to
        # PortAudio, not when it leaves the speaker — the device buffer is
        # still playing out for another few tens of ms. sd.stop() aborts
        # rather than drains, so stopping right on the deadline would shave
        # the tail off every single sentence. Wait for the stream to report
        # itself inactive, with a hard grace cap so a device that never
        # reports inactive can't wedge the loop (the failure mode this whole
        # helper exists to prevent).
        if not cut:
            grace = time.monotonic() + 0.75
            while time.monotonic() < grace:
                if _cancelled():
                    cut = True
                    break
                try:
                    stream = sd.get_stream()
                    if stream is None or not stream.active:
                        break
                except Exception:
                    break       # no stream to inspect: treat as finished
                await asyncio.sleep(0.02)

        # Close the stream out from THIS (event-loop) context either way;
        # sd.play() would otherwise leave the last one open until the next.
        try:
            sd.stop()
        except Exception as e:
            log.error("sd.stop after clip failed: %s", e)
        return cut

    async def _playback_loop(self):
        """Continuously play queued raw PCM audio."""
        try:
            import sounddevice as sd
        except Exception as e:
            log.error("sounddevice not available: %s", e)
            while True:
                item = await self._playback_queue.get()
                if item is None:
                    break
            return

        while True:
            item = await self._playback_queue.get()
            if item is None:
                break
            suppress_mic = item.suppress_mic
            # The generation this clip was ENQUEUED under; stale => a Stop
            # happened while it sat in the queue, so it must not be played.
            item_gen = item.gen
            try:
                audio, sr, cooldown_s = item.audio, item.sr, item.cooldown_s
                # suppress_mic=False leaves the mic OPEN so Timmy hears this
                # playback as an incoming turn (test channel). Default True =
                # gated (no loopback), the normal behavior for every voice.
                if self._capture and suppress_mic:
                    self._capture.suppressed = True

                # Cancelled by a Stop that landed between dequeue and start.
                if item_gen != self._stop_gen:
                    continue    # finally still runs (done set, gate released)
                sd.play(audio, sr)

                # True audible onset hook: fires the instant playback starts,
                # so the doorway can measure how long a ready reply waited
                # behind a filler in this serial queue. Cheap + non-throwing by
                # contract; guarded so a bad hook can never break playback.
                if item.on_play_start is not None:
                    try:
                        item.on_play_start(time.time())
                    except Exception as e:
                        log.error("on_play_start hook error: %s", e)
                # Booth VOX caption, same onset seam: emitted HERE rather than
                # at enqueue because the queue is serial — the LLM has usually
                # streamed (and this loop has synthesized) two or three
                # sentences ahead of the speaker, so an enqueue-time caption
                # would run visibly ahead of the voice. After the start check,
                # so a Stop-cancelled clip never flashes on the booth.
                if item.caption:
                    await _broadcast_speech("speaking", {"text": item.caption})

                clip_s = len(audio) / float(sr)

                if item.kind == "filler":
                    # Preemptible filler (2026-06-30): a filler must NEVER delay
                    # the real reply. The instant a real sentence lands in the
                    # queue behind us, cut the filler short AND skip its
                    # post-roll cooldown, so the answer plays with ~0 added
                    # latency. If no answer arrives, the filler finishes and
                    # serves its normal cooldown (the echo guard that stops the
                    # reverb tail from being transcribed) exactly as before.
                    # Both the audio and the cooldown are watched, so the answer
                    # can barge in during either phase.
                    preempted = await self._await_clip(
                        sd, clip_s, item_gen,
                        until=lambda: not self._playback_queue.empty())
                    if not preempted:
                        cooldown_until = time.time() + cooldown_s
                        while time.time() < cooldown_until:
                            if not self._playback_queue.empty():
                                break
                            await asyncio.sleep(0.02)
                else:
                    await self._await_clip(sd, clip_s, item_gen)
                    if cooldown_s > 0 and item_gen == self._stop_gen:
                        await asyncio.sleep(cooldown_s)

                # Release the mic gate only when nothing more is queued. This
                # keeps the gate continuously closed across back-to-back playback
                # (filler->reply, sentence->sentence) so a reverb tail can never
                # leak into STT in the microsecond between two clips.
                if self._capture and suppress_mic and self._playback_queue.empty():
                    self._capture.suppressed = False
            except Exception as e:
                log.error("TTS playback error: %s", e)
            finally:
                if self._capture and suppress_mic and self._playback_queue.empty():
                    self._capture.suppressed = False
                # Booth VOX: the caption clears only when Timmy has actually
                # stopped talking, i.e. this captioned clip finished and nothing
                # is queued behind it. Mid-reply, the next sentence's `speaking`
                # replaces the text instead, so the band never blinks between
                # sentences. In finally (not the try) so a playback error or a
                # Puppet-Mode stop_playback can't strand a stale line on screen.
                if item.caption and self._playback_queue.empty():
                    await _broadcast_speech("speech_end", {})
                # Signal any awaiter (Puppet Mode's blocking speak) that this
                # clip is done — in finally so a playback error or sd.stop()
                # can never leave the sequencer hung.
                if item.done is not None:
                    item.done.set()

    async def stop(self):
        """Signal playback loop to stop."""
        await self._playback_queue.put(None)
        if self._playback_task:
            await self._playback_task
