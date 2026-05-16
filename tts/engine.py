"""Piper TTS engine — in-process ONNX with raw PCM streaming to audio output."""

import asyncio
import time
import logging
import numpy as np

log = logging.getLogger(__name__)

_piper_voice = None


def _load_voice(model_path: str):
    """Load Piper voice model (lazy, once)."""
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice

    from piper import PiperVoice
    _piper_voice = PiperVoice.load(model_path)
    log.info("Loaded Piper voice from %s (sample_rate=%d)",
             model_path, _piper_voice.config.sample_rate)
    return _piper_voice


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
        self._playback_queue: asyncio.Queue[tuple[np.ndarray, int, float] | None] = asyncio.Queue()
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

    async def speak(self, text: str):
        """Synthesize text and queue raw PCM for playback. Non-blocking."""
        # Replace commas with em dashes for shorter TTS pauses
        text = text.replace(",", " —")
        if not text.strip():
            return
        audio, sr = await asyncio.to_thread(_synthesize_raw, text, self.model_path)
        if len(audio) > 0:
            await self._playback_queue.put((audio, sr, 0.5))

    async def prewarm_fillers(self, texts) -> None:
        """Synthesize each filler once and stash in _filler_cache.

        Run this at startup so the first conversational turn doesn't pay a
        Piper-inference cost for the filler. ~10 calls * ~200 ms each on
        boot; LT process is long-lived so it amortizes to zero across the
        session.
        """
        for text in texts:
            if text in self._filler_cache:
                continue
            audio, sr = await asyncio.to_thread(_synthesize_raw, text, self.model_path)
            if len(audio) > 0:
                self._filler_cache[text] = (audio, sr)
        log.info("TTS filler cache prewarmed (%d entries)", len(self._filler_cache))

    async def speak_filler(self, text: str) -> None:
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
        cached = self._filler_cache.get(text)
        if cached is None:
            await self.speak(text)
            return
        audio, sr = cached
        await self._playback_queue.put((audio, sr, 0.4))

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
            try:
                # Tuple shape evolved 2026-05-14 to include per-item cooldown.
                # Tolerate any stragglers still using the 2-tuple form.
                if len(item) == 3:
                    audio, sr, cooldown_s = item
                else:
                    audio, sr = item
                    cooldown_s = 0.5
                if self._capture:
                    self._capture.suppressed = True
                await asyncio.to_thread(sd.play, audio, sr)
                await asyncio.to_thread(sd.wait)
                if cooldown_s > 0:
                    await asyncio.sleep(cooldown_s)
                if self._capture:
                    self._capture.suppressed = False
            except Exception as e:
                log.error("TTS playback error: %s", e)
            finally:
                if self._capture:
                    self._capture.suppressed = False

    async def stop(self):
        """Signal playback loop to stop."""
        await self._playback_queue.put(None)
        if self._playback_task:
            await self._playback_task
