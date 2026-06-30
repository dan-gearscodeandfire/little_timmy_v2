"""Piper TTS engine — in-process ONNX with raw PCM streaming to audio output."""

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

log = logging.getLogger(__name__)


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

# Loaded Piper voices, cached per model_path. Timmy's conversational voice
# (config.PIPER_MODEL) plus any persona voices (e.g. the couples-therapist voice
# the supervisor/announce channel speaks in) coexist here, each loaded once.
_piper_voices: dict[str, object] = {}


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

    async def start(self):
        """Pre-load the model and start the playback loop."""
        await asyncio.to_thread(_load_voice, self.model_path)
        self._playback_task = asyncio.create_task(self._playback_loop())
        log.info("TTS engine started")

    async def speak(self, text: str, force: bool = False,
                    voice_model: str | None = None,
                    suppress_mic: bool = True,
                    on_play_start: Callable[[float], None] | None = None):
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
        voiceprint is enrolled."""
        if not force and _tts_muted():
            return
        # Replace commas with em dashes for shorter TTS pauses
        text = text.replace(",", " —")
        if not text.strip():
            return
        audio, sr = await asyncio.to_thread(
            _synthesize_raw, text, voice_model or self.model_path)
        if len(audio) > 0:
            await self._playback_queue.put(PlaybackItem(
                audio=audio, sr=sr, cooldown_s=0.5, suppress_mic=suppress_mic,
                kind="real", on_play_start=on_play_start))

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
            on_play_start=on_play_start))

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
            try:
                audio, sr, cooldown_s = item.audio, item.sr, item.cooldown_s
                # suppress_mic=False leaves the mic OPEN so Timmy hears this
                # playback as an incoming turn (test channel). Default True =
                # gated (no loopback), the normal behavior for every voice.
                if self._capture and suppress_mic:
                    self._capture.suppressed = True
                # True audible onset hook: fires the instant playback starts,
                # so the doorway can measure how long a ready reply waited
                # behind a filler in this serial queue. Cheap + non-throwing by
                # contract; guarded so a bad hook can never break playback.
                if item.on_play_start is not None:
                    try:
                        item.on_play_start(time.time())
                    except Exception as e:
                        log.error("on_play_start hook error: %s", e)
                await asyncio.to_thread(sd.play, audio, sr)

                if item.kind == "filler":
                    # Preemptible filler (2026-06-30): a filler must NEVER delay
                    # the real reply. The instant a real sentence lands in the
                    # queue behind us, cut the filler short (sd.stop) AND skip its
                    # post-roll cooldown, so the answer plays with ~0 added
                    # latency. If no answer arrives, the filler finishes and
                    # serves its normal cooldown (the echo guard that stops the
                    # reverb tail from being transcribed) exactly as before. Both
                    # the audio and the cooldown are watched, so the answer can
                    # barge in during either phase.
                    play_done = asyncio.create_task(asyncio.to_thread(sd.wait))
                    cooldown_until = None
                    while True:
                        if not self._playback_queue.empty():   # real reply waiting
                            if not play_done.done():
                                sd.stop()                       # abort filler audio
                            break
                        if play_done.done():
                            if cooldown_until is None:
                                cooldown_until = time.time() + cooldown_s
                            if time.time() >= cooldown_until:
                                break
                        await asyncio.sleep(0.02)
                    if not play_done.done():
                        await play_done
                else:
                    await asyncio.to_thread(sd.wait)
                    if cooldown_s > 0:
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

    async def stop(self):
        """Signal playback loop to stop."""
        await self._playback_queue.put(None)
        if self._playback_task:
            await self._playback_task
