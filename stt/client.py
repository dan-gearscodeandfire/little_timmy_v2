"""Async HTTP client for whisper.cpp on port 8891."""

import io
import re
import math
import wave
import logging
from dataclasses import dataclass, field
import numpy as np
import httpx
import config

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


@dataclass
class Transcription:
    """STT result carrying confidence, not just text. `confidence` is the
    utterance-level acoustic confidence in [0,1] (exp of mean segment
    avg_logprob). `words` is the per-word [(word, probability)] list from
    whisper.cpp word timestamps — used to score the confidence of a SPECIFIC
    extracted value (a misheard name) rather than the whole sentence. A bare
    Transcription is falsy/truthy on its text and stringifies to it, so existing
    `if not user_text` / logging call sites keep working."""
    text: str = ""
    confidence: float = 1.0
    words: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:
        return self.text


def _norm_word(w: str) -> str:
    return re.sub(r"[^\w']", "", w).strip().lower()


def value_confidence(words: list, value: str) -> float | None:
    """Acoustic confidence of the specific VALUE string within an utterance.

    Returns the WEAKEST probability among the whisper word-pieces that spell the
    value (a value is only as trustworthy as its least-certain piece). This
    targets the dominant FALSE source found 2026-06-21 acoustic testing: a
    misheard fact value (Bolt->Volt, Blaze->Blazed) silently committing as a
    confident "verified fact".

    Whisper tokenizes uncommon proper nouns into SUB-WORD pieces -- "Onyx" ->
    'On'+'yx', "Zoltan" -> 'Z'+'olt'+'an' -- which are exactly the names most
    likely to be misheard. So we don't match whole words; we slide a window over
    the word list, concatenate normalized pieces (letters only, punctuation
    dropped), and find the contiguous run whose concatenation equals the value's
    letters (spaces removed). Handles sub-word splits AND multi-word values.
    Returns None only if the value's letters can't be assembled from any run
    (caller falls back to utterance confidence).
    """
    if not words or not value:
        return None
    target = "".join(_norm_word(t) for t in value.split())
    if not target:
        return None
    norm = [(_norm_word(w), p) for (w, p) in words]
    n = len(norm)
    for i in range(n):
        if not norm[i][0]:
            continue
        acc = ""
        probs = []
        for j in range(i, n):
            piece, p = norm[j]
            if not piece:
                continue  # punctuation token: spans across it, ignore its prob
            acc += piece
            probs.append(p)
            if acc == target:
                return min(probs)
            if not target.startswith(acc):
                break  # this run diverged; advance the window start
    return None


# --- proper-noun read-back lever (2026-06-21) -------------------------------
# value_confidence catches mishears whisper was UNSURE of, but a *confident*
# homophone -- Thorne->Thorn (0.721), Praxx->Prax (0.804) -- clears the gate and
# commits as a verified, never-hedged fact. Acoustic test 2026-06-21: both were
# FALSE, while the CORRECT value Onyx scored 0.683 -- BELOW both -- so the scalar
# gate cannot rank-order correctness at the boundary and no threshold fixes it.
# Names are acoustically unverifiable; the fix is to read the VALUE back for
# confirmation whenever it's a name/proper noun, regardless of confidence.
# Confirming a name once is cheap UX; common-word values (teal, pizza) under
# non-name predicates stay breezy.
_DICT_PATH = "/usr/share/dict/words"
_COMMON_WORDS = None  # lazy-loaded lowercased dictionary set (None until tried)


def _common_words() -> frozenset:
    """Lowercased system word list, loaded once. Empty set if unavailable, which
    disables the proper-noun branch (the name-slot branch needs no dictionary)."""
    global _COMMON_WORDS
    if _COMMON_WORDS is None:
        try:
            with open(_DICT_PATH, encoding="utf-8", errors="ignore") as fh:
                _COMMON_WORDS = frozenset(
                    w.strip().lower() for w in fh if w.strip() and "'" not in w)
        except OSError:
            _COMMON_WORDS = frozenset()
    return _COMMON_WORDS


def is_name_like_value(predicate: str, value: str,
                       common: "frozenset | None" = None) -> bool:
    """True when a fact's VALUE should be read back for confirmation regardless
    of acoustic confidence -- it's a name or an unfamiliar proper noun, the class
    a *confident* homophone mishear hides in (see note above).

    (a) name slot: predicate carries a "name" token ("name", "robot.name",
        "first name") -- the user said "my X is NAMED Y".
    (b) proper-noun value under a non-name predicate: a single capitalized token
        that is NOT a common dictionary word (Acme, Praxton -- but not Blue).
    """
    if not value:
        return False
    pred_l = (predicate or "").lower()
    if any(tok == "name" for tok in re.split(r"[^a-z]+", pred_l)):
        return True
    tokens = value.split()
    if len(tokens) == 1:
        core = tokens[0].strip(".,!?;:'\"")
        if core[:1].isupper() and core.isalpha():
            if common is None:
                common = _common_words()
            if common and core.lower() not in common:
                return True
    return False


# --- query-side mishear guard (2026-06-22) ----------------------------------
# The store-side gate (value_confidence/is_name_like_value) protects WRITES. But
# a misheard word in a QUESTION corrupts RETRIEVAL: "what's my mail" -> "my male"
# keys the lookup on the wrong term, so Timmy answers about the wrong thing or
# wrongly says he doesn't know. We can't fix the retrieval blindly, but we can
# detect that a CONTENT word came through unclearly and tell the brain to confirm
# what was asked instead of guessing/denying. Function words ("my" routinely
# ~0.5) carry no retrieval meaning -- their low prob is acoustic noise, not a
# misheard query term -- so they're excluded; only a low-confidence content word
# is a signal.
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "my", "your", "his", "her", "its", "our", "their", "this",
    "that", "these", "those", "i", "me", "mine", "you", "we", "they", "he", "she",
    "it", "is", "are", "was", "were", "be", "am", "been", "do", "does", "did",
    "have", "has", "had", "what", "what's", "whats", "who", "who's", "whos",
    "where", "when", "which", "whose", "how", "why", "of", "to", "in", "on", "at",
    "and", "or", "but", "so", "if", "for", "with", "about", "tell", "say", "said",
    "name", "named", "call", "called", "please", "can", "could", "would", "will",
    "me", "us", "again", "just", "really", "do", "remember", "know",
})


def _group_into_words(words: list):
    """Reassemble whisper sub-word PIECES into whitespace-delimited words.

    Whisper splits uncommon nouns into pieces ("micro-santhemums" -> ' micro',
    '-', 's', 'anth', 'emums'); a piece beginning with whitespace starts a new
    word, pieces without a leading space (BPE continuations, punctuation) attach
    to the current one. Without this, scoring per-piece flags a meaningless
    fragment ('s') as the misheard "word". Yields (display_word, min_letter_prob)
    -- a word is only as trustworthy as its weakest LETTER piece (punctuation
    probs ignored), mirroring value_confidence."""
    cur_text, cur_probs = "", []
    for w, p in words:
        starts_word = bool(w) and w[:1].isspace()
        if starts_word and cur_text:
            yield (cur_text.strip(), min(cur_probs) if cur_probs else 1.0)
            cur_text, cur_probs = "", []
        cur_text += w
        if _norm_word(w):  # letter/digit piece -> counts toward the word's min
            cur_probs.append(p)
    if cur_text.strip():
        yield (cur_text.strip(), min(cur_probs) if cur_probs else 1.0)


def low_confidence_query_term(words: list, threshold: float) -> str | None:
    """The lowest-confidence CONTENT word in an utterance heard below threshold,
    or None. Reassembles sub-word pieces into whole words first, skips function/
    stop words and short fragments (<3 letters). Query-side companion to
    value_confidence: a misheard noun in a QUESTION corrupts retrieval, so surface
    the whole heard word to the brain to CONFIRM rather than answer/deny. Returns
    None when every content word cleared the threshold."""
    if not words:
        return None
    worst, worst_p = None, threshold
    for display, mp in _group_into_words(words):
        norm = _norm_word(display)
        if len(norm) < 3 or norm in _QUERY_STOPWORDS:
            continue
        if mp < worst_p:
            worst_p, worst = mp, display.strip()
    return worst


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


async def transcribe(audio: np.ndarray) -> "Transcription":
    """Send audio to whisper.cpp /inference and return a Transcription
    (text + acoustic confidence + per-word probabilities).

    Uses verbose_json to get no_speech_prob (hallucination filter), segment
    avg_logprob (utterance confidence) and word probabilities (value-level
    confidence). On any failure returns an empty Transcription (falsy), so the
    capture pipeline skips the turn exactly as before.
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
        return Transcription()
    result = resp.json()

    text = result.get("text", "").strip()
    if not text:
        return Transcription()

    # Remove bracketed/parenthesized sound effects
    text = re.sub(r"\[.*?\]", "", text).strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    if not text:
        return Transcription()

    segments = result.get("segments", [])
    # Get no_speech_prob from segments (use max across segments)
    no_speech_prob = max((s.get("no_speech_prob", 0.0) for s in segments), default=0.0)

    if _is_likely_hallucination(text, no_speech_prob):
        return Transcription()

    # Utterance confidence: exp(mean segment avg_logprob), clamped to [0,1].
    logps = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    confidence = max(0.0, min(1.0, math.exp(sum(logps) / len(logps)))) if logps else 1.0
    # Per-word probabilities (flatten across segments) for value-level scoring.
    words = [(w.get("word", ""), float(w.get("probability", 1.0)))
             for s in segments for w in (s.get("words") or [])]

    # Apply known name corrections (e.g., Aaron → Erin)
    corrected = _apply_stt_corrections(text)
    if corrected != text:
        log.info("STT correction: %r -> %r", text, corrected)
        text = corrected

    return Transcription(text=text, confidence=confidence, words=words)
