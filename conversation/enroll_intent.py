"""Regex-based intent detection for face enrollment voice commands.

Detects utterances like:
    "learn my face, my name is Dan"
    "remember my face. I'm Dan."
    "enroll my face as Dan"
    "save my face"  (uses speaker_name fallback if voice-known)

Returns an EnrollIntent. matched=True only if BOTH the keyword phrase is
present AND a usable name is found (either explicit or speaker-fallback).
This is a deliberate pre-LLM short-circuit; no tool-calling LLM required.
"""
import re
from dataclasses import dataclass
from typing import Optional

# Trigger phrase: "learn|remember|save|... my|this face"
_ENROLL_KEYWORDS = re.compile(
    r"\b(?:learn|remember|save|store|enroll|memori[sz]e|recogni[sz]e)\s+(?:my|this)\s+face\b",
    re.IGNORECASE,
)

# Name-extraction patterns. Most specific first.
_NAME_PATTERNS = [
    re.compile(r"\bmy\s+name\s+is\s+([A-Za-z][a-zA-Z']{1,20})\b"),
    re.compile(r"\bcall\s+me\s+([A-Za-z][a-zA-Z']{1,20})\b"),
    re.compile(r"\benroll\s+(?:my\s+face\s+)?as\s+([A-Za-z][a-zA-Z']{1,20})\b"),
    re.compile(r"\bI(?:'m|\s+am)\s+([A-Za-z][a-zA-Z']{1,20})\b"),
    re.compile(r"\bthis\s+is\s+([A-Za-z][a-zA-Z']{1,20})\b"),
]

# Words that look like names but aren't.
_NON_NAMES = frozenset({
    "face", "name", "person", "this", "that", "here", "sorry", "fine",
    "okay", "ok", "yes", "no", "sure", "back", "ready",
})


@dataclass
class EnrollIntent:
    matched: bool
    name: Optional[str] = None
    used_speaker_fallback: bool = False


def _normalize_name(raw: str) -> str:
    """Title-case a single token name (Whisper transcripts vary on case)."""
    s = raw.strip()
    if not s:
        return s
    return s[0].upper() + s[1:].lower()


def detect_enroll_intent(text: str, speaker_name: Optional[str] = None) -> EnrollIntent:
    """Detect 'learn my face' style intents and extract a name.

    Args:
        text: ASR transcript of the user turn.
        speaker_name: Currently identified voice-print name. Used only as a
            fallback if the keyword matched but no explicit name appeared and
            the speaker is voice-known (not "unknown_*").

    Returns:
        EnrollIntent. matched=True only if keyword + usable name both present.
    """
    if not text:
        return EnrollIntent(matched=False)
    if not _ENROLL_KEYWORDS.search(text):
        return EnrollIntent(matched=False)

    name = None
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip()
            if cand.lower() not in _NON_NAMES:
                name = _normalize_name(cand)
                break

    if name is None:
        if speaker_name and not speaker_name.startswith("unknown_"):
            return EnrollIntent(
                matched=True,
                name=_normalize_name(speaker_name),
                used_speaker_fallback=True,
            )
        return EnrollIntent(matched=False)

    return EnrollIntent(matched=True, name=name)
