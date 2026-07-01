"""Regex intent detection for enrollment voice commands (dual-modality).

Detects "enroll me" style utterances and classifies the SCOPE:
    "learn my face, my name is Dan"        -> scope=face
    "remember my voice, I'm Dan"           -> scope=voice
    "enroll me as Dan" / "remember me"     -> scope=both  (face + voice)
    "save my face"                         -> scope=face  (speaker fallback name)

Returns an EnrollIntent. ``matched`` stays back-compatible: True only when a
keyword phrase AND a usable name are both present (the pre-LLM short-circuit the
old face-only path relied on). ``keyword_present`` + ``scope`` are the richer
signals the unified enroll FSM uses to start an ASK_NAME dialog when the phrase
appeared without a name.

Casing: names are canonicalized to LOWERCASE here (the single id-space
convention shared with ``identity_commit`` / the id-map). Display code Title-cases.
"""
import re
from dataclasses import dataclass
from typing import Optional

# Shared enroll verbs.
_VERB = r"(?:learn|remember|save|store|enroll|memori[sz]e|recogni[sz]e)"

# Scope triggers.
_FACE_RE = re.compile(rf"\b{_VERB}\s+(?:my|this)\s+face\b", re.IGNORECASE)
_VOICE_RE = re.compile(rf"\b{_VERB}\s+(?:my|this)\s+voice\b", re.IGNORECASE)
# Whole-person: "enroll me", "remember me", "remember who I am".
_BOTH_RE = re.compile(
    rf"\b{_VERB}\s+me\b|\bremember\s+who\s+i\s+am\b", re.IGNORECASE)

# Name-extraction patterns (specific first). Kept as a fallback/superset over
# the shared conversational extractor (which doesn't parse "... as X").
_NAME_PATTERNS = [
    re.compile(r"\bmy\s+name\s+is\s+([A-Za-z][a-zA-Z']{1,20})\b", re.IGNORECASE),
    re.compile(r"\bcall\s+me\s+([A-Za-z][a-zA-Z']{1,20})\b", re.IGNORECASE),
    re.compile(r"\b(?:enroll|remember|learn|save)\s+(?:me\s+|my\s+(?:face|voice)\s+)?as\s+([A-Za-z][a-zA-Z']{1,20})\b", re.IGNORECASE),
    re.compile(r"\bI(?:'m|\s+am)\s+([A-Za-z][a-zA-Z']{1,20})\b", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+([A-Za-z][a-zA-Z']{1,20})\b", re.IGNORECASE),
]

# Words that look like names but aren't.
_NON_NAMES = frozenset({
    "face", "voice", "name", "person", "this", "that", "here", "sorry", "fine",
    "okay", "ok", "yes", "no", "sure", "back", "ready", "me",
})

# The shared conversational name extractor (returns lowercase). Falls back to
# the local patterns if it can't be imported (e.g. in isolated tests).
try:  # pragma: no cover - import wiring
    from conversation.introductions import _extract_name_from_response
except Exception:  # pragma: no cover
    _extract_name_from_response = None


@dataclass
class EnrollIntent:
    matched: bool
    name: Optional[str] = None
    scope: str = "face"                 # "both" | "face" | "voice"
    keyword_present: bool = False
    used_speaker_fallback: bool = False


def _extract_name(text: str) -> Optional[str]:
    """Return a canonical (lowercase) name from ``text`` or None. Tries the
    local enroll-specific patterns first (they parse "... as X"), then the
    shared conversational extractor."""
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip().lower()
            if cand not in _NON_NAMES:
                return cand
    if _extract_name_from_response is not None:
        cand = _extract_name_from_response(text)
        if cand and cand.lower() not in _NON_NAMES:
            return cand.lower()
    return None


def detect_enroll_intent(text: str, speaker_name: Optional[str] = None) -> EnrollIntent:
    """Detect an enrollment intent, its scope, and a name.

    Args:
        text: ASR transcript of the user turn.
        speaker_name: currently identified voiceprint name. Used only as a name
            fallback when a keyword matched but no explicit name appeared and the
            speaker is voice-known (not ``unknown_*``).

    Returns:
        EnrollIntent. ``matched`` is True only when keyword + usable name are
        both present (back-compat). ``keyword_present`` + ``scope`` are set
        whenever an enroll phrase is detected, even without a name.
    """
    if not text:
        return EnrollIntent(matched=False)

    has_face = bool(_FACE_RE.search(text))      # verb + "my/this face" adjacent
    has_voice = bool(_VOICE_RE.search(text))    # verb + "my/this voice" adjacent
    has_both = bool(_BOTH_RE.search(text))       # "enroll me" / "remember who I am"

    if not (has_face or has_voice or has_both):
        return EnrollIntent(matched=False)

    # A bare mention of the OTHER modality upgrades a single-modality match to
    # "both" ("learn my face and my voice" -> both).
    face_mention = bool(re.search(r"\b(?:my|this)\s+face\b", text, re.IGNORECASE))
    voice_mention = bool(re.search(r"\b(?:my|this)\s+voice\b", text, re.IGNORECASE))
    if has_both or (face_mention and voice_mention):
        scope = "both"
    elif has_voice:
        scope = "voice"
    else:
        scope = "face"

    name = _extract_name(text)

    if name is None:
        if speaker_name and not speaker_name.startswith("unknown_"):
            return EnrollIntent(
                matched=True, name=speaker_name.strip().lower(), scope=scope,
                keyword_present=True, used_speaker_fallback=True)
        # Keyword present but no resolvable name -> the FSM should ask.
        return EnrollIntent(matched=False, scope=scope, keyword_present=True)

    return EnrollIntent(matched=True, name=name, scope=scope, keyword_present=True)
