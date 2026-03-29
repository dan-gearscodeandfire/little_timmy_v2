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
        self._playback_queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue()
        self._capture = None  # set by orchestrator for TTS suppression
        self._playback_task: asyncio.Task | None = None

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
            await self._playback_queue.put((audio, sr))

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
                audio, sr = item
                if self._capture:
                    self._capture.suppressed = True
                await asyncio.to_thread(sd.play, audio, sr)
                await asyncio.to_thread(sd.wait)
                # Brief cooldown for room reverb to die down
                await asyncio.sleep(0.5)
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
