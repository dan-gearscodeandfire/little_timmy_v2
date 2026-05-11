"""Async HTTP client for whisper.cpp on port 8891."""

import io
import re
import wave
import logging
import numpy as np
import httpx
import config

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None

# Short phrases that whisper commonly hallucinates from noise/silence
_HALLUCINATION_PATTERNS = {
    "yeah", "yes", "no", "oh", "okay", "ok", "uh", "um", "hmm", "huh",
    "ah", "both", "and", "but", "so", "if", "the", "right", "sure",
    "thanks", "thank you", "bye", "hey", "hi", "well", "actually",
    "i'm gonna", "you know",
}

# Max word count for hallucination check — longer texts are probably real
_HALLUCINATION_MAX_WORDS = 4

# Minimum no_speech_prob threshold — above this, likely not real speech
_NO_SPEECH_THRESHOLD = 0.6

# Known STT corrections: whisper frequently mishears these names/words.
# Keys are lowercase, applied as whole-word replacements (case-preserving).
_STT_CORRECTIONS = {
    "aaron": "Erin",
    "arron": "Erin",
    "erin's": "Erin's",  # possessive
    "tammy": "Timmy",
    "tommy": "Timmy",
}


def _apply_stt_corrections(text: str) -> str:
    """Apply known name/word corrections to STT output."""
    words = text.split()
    corrected = []
    for word in words:
        stripped = word.strip(".,!?;:'\"")
        lower = stripped.lower()
        if lower in _STT_CORRECTIONS:
            # Preserve surrounding punctuation
            prefix = word[:word.index(stripped[0])] if stripped else ""
            suffix = word[word.rindex(stripped[-1]) + 1:] if stripped else ""
            corrected.append(prefix + _STT_CORRECTIONS[lower] + suffix)
        else:
            corrected.append(word)
    return " ".join(corrected)


def _is_likely_hallucination(text: str, no_speech_prob: float) -> bool:
    """Check if transcription is likely a whisper hallucination."""
    clean = text.strip().lower().rstrip(".!?,;:")

    # High no_speech_prob = probably not real speech
    if no_speech_prob > _NO_SPEECH_THRESHOLD:
        log.debug("Filtered hallucination (no_speech_prob=%.2f): %r", no_speech_prob, text)
        return True

    # Very short text with common filler words
    words = clean.split()
    if len(words) <= _HALLUCINATION_MAX_WORDS:
        # Check if all words are in the hallucination set
        if all(w.strip(".,!?;:") in _HALLUCINATION_PATTERNS for w in words):
            log.debug("Filtered hallucination (pattern match): %r", text)
            return True

        # Repetitive patterns like "yes, yes" or "no, no" or "ten, ten, ten"
        unique_words = set(w.strip(".,!?;:-") for w in words)
        if len(unique_words) == 1 and len(words) > 1:
            log.debug("Filtered hallucination (repetitive): %r", text)
            return True

    return False


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy audio to WAV bytes (16kHz mono int16)."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


async def transcribe(audio: np.ndarray) -> str:
    """Send audio to whisper.cpp /inference endpoint and return text.

    Uses verbose_json to get no_speech_prob for hallucination filtering.
    """
    wav_bytes = _audio_to_wav_bytes(audio)
    client = await _get_client()

    try:
        resp = await client.post(
            f"{config.WHISPER_URL}/inference",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"response_format": "verbose_json", "temperature": "0.0"},
        )
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException,
            httpx.NetworkError, httpx.HTTPStatusError) as e:
        # Whisper is down or unreachable. Returning "" here makes the
        # capture pipeline skip the turn the same way it skips
        # hallucinations and empty transcripts — no crash-loop on LT
        # while whisper-server is stopped.
        log.warning("STT unavailable, dropping turn: %s", e)
        return ""
    result = resp.json()

    text = result.get("text", "").strip()
    if not text:
        return ""

    # Remove bracketed/parenthesized sound effects
    text = re.sub(r"\[.*?\]", "", text).strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    if not text:
        return ""

    # Get no_speech_prob from segments (use max across segments)
    segments = result.get("segments", [])
    no_speech_prob = max((s.get("no_speech_prob", 0.0) for s in segments), default=0.0)

    if _is_likely_hallucination(text, no_speech_prob):
        return ""

    # Apply known name corrections (e.g., Aaron → Erin)
    corrected = _apply_stt_corrections(text)
    if corrected != text:
        log.info("STT correction: %r -> %r", text, corrected)
        text = corrected

    return text
