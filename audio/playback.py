"""Audio playback utilities."""

import asyncio
import logging
import numpy as np

log = logging.getLogger(__name__)


async def play_wav_bytes(wav_bytes: bytes):
    """Play WAV audio bytes through the default output device."""
    import io
    import wave
    import sounddevice as sd

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    await asyncio.to_thread(sd.play, audio, sr)
    await asyncio.to_thread(sd.wait)
