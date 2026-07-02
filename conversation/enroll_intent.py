"""Regex intent detection for enrollment voice commands (dual-modality).

Detects "enroll me" style utterances and classifies the SCOPE:
    "learn my face, my name is Dan"        -> scope=face
    "remember my voice, I'm Dan"           -> scope=voice
    "enroll me as Dan" / "remember me"     -> scope=both  (face + voice)
    "save my face"                         -> scope=face  (no name -> ask)

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
# Each captures up to THREE word tokens ("Mary Jane", "Billy Bob Thornton") —
# the single-token version silently truncated multi-word names (test F,
# 2026-07-02). Trailing non-name tokens are trimmed in _clean_name_tokens.
_NAME_TOKEN = r"[A-Za-z][a-zA-Z']{1,20}"
_NAME_SPAN = rf"({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})"
_NAME_PATTERNS = [
    re.compile(rf"\bmy\s+name\s+is\s+{_NAME_SPAN}\b", re.IGNORECASE),
    re.compile(rf"\bcall\s+me\s+{_NAME_SPAN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:enroll|remember|learn|save)\s+(?:me\s+|my\s+(?:face|voice)\s+)?as\s+{_NAME_SPAN}\b", re.IGNORECASE),
    re.compile(rf"\bI(?:'m|\s+am)\s+{_NAME_SPAN}\b", re.IGNORECASE),
    re.compile(rf"\bthis\s+is\s+{_NAME_SPAN}\b", re.IGNORECASE),
]

# Words that look like names but aren't.
_NON_NAMES = frozenset({
    "face", "voice", "name", "person", "this", "that", "here", "sorry", "fine",
    "okay", "ok", "yes", "no", "sure", "back", "ready", "me",
    # trailing filler that the multi-word span can drag in
    "please", "now", "again", "though", "thanks", "buddy", "man",
    # connectives/verbs: stop the 3-token span from fusing identities
    # ("I'm Dan and Sarah is here" must parse 'dan', not 'dan_and_sarah' —
    # code review C7) and from canonicalizing evasive replies ("not telling",
    # "never mind" — code review C5).
    "and", "or", "but", "with", "plus", "also", "is", "are", "was", "were",
    "the", "a", "an", "not", "never", "telling", "mind", "nothing", "nobody",
    "none", "guys", "everyone", "everybody",
})


def _clean_name_tokens(candidate: str) -> Optional[str]:
    """Validate a captured span token-by-token: keep leading name-like tokens,
    stop at the first non-name word. Returns the canonical form (lowercase,
    spaces -> underscores, apostrophes dropped so "O'Brien" -> "obrien" passes
    NAME_RE) or None if the FIRST token is already a non-name (e.g. "enroll me
    as here")."""
    kept = []
    for tok in candidate.strip().lower().split():
        if tok in _NON_NAMES:
            break
        tok = tok.replace("'", "")
        if tok:
            kept.append(tok)
    if not kept:
        return None
    return "_".join(kept)

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


def _extract_name(text: str) -> Optional[str]:
    """Return a canonical (lowercase, underscore-joined) name from ``text`` or
    None. Tries the local enroll-specific patterns first (they parse
    "... as X" and multi-word names), then the shared conversational
    extractor."""
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = _clean_name_tokens(m.group(1))
            if cand:
                return cand
    if _extract_name_from_response is not None:
        cand = _extract_name_from_response(text)
        if cand:
            cand = _clean_name_tokens(cand)
            if cand:
                return cand
    return None


# Evasive replies to "what name should I remember you by?" — never names.
# The bare-token fallback below bypasses the shared extractor's own evasive
# list, so this module needs one (code review C5).
_EVASIVE_RE = re.compile(
    r"\b(?:not\s+telling|never\s*mind|no\s+name|none\s+of\s+your|"
    r"rather\s+not|forget\s+(?:about\s+)?(?:it|that|this)|skip\s+it|"
    r"drop\s+it|no\s+thanks|doesn'?t\s+matter|"
    r"don'?t\s+(?:want|worry|care)|nothing|nobody)\b", re.IGNORECASE)


def extract_reply_name(text: str) -> Optional[str]:
    """Extract a canonical name from a reply to the ask-name latch.

    Unlike _extract_name (full enroll utterances), this handles bare replies:
    "Mary Jane." / "It's Bob" / "My name is Mary Jane". Runs the local
    multi-word-capable patterns FIRST (the shared extractor captures a single
    \\w+ and would truncate "My name is Mary Jane" to 'mary' — code review
    C6), rejects evasive replies (C5), and only then falls back to treating a
    short 1-3-token utterance as the name itself.
    """
    if not text or _EVASIVE_RE.search(text):
        return None
    cand = _extract_name(text)
    if cand:
        return cand
    bare = re.sub(r"[^\w\s']", " ", text).strip()
    # Conversational lead-ins the patterns above don't cover ("It's Bob").
    bare = re.sub(r"^(?:it'?s|i'?m|i\s+am|the\s+name'?s?(?:\s+is)?)\s+",
                  "", bare, flags=re.IGNORECASE).strip()
    if bare and len(bare.split()) <= 3:
        return _clean_name_tokens(bare)
    return None


# Confirm-turn parsing (2026-07-02): the doorway speaks the parsed name back
# and commits only on an explicit yes. Kept here so it's hermetically testable
# next to the intent detection it completes.
#
# Reworked same day (code review C1): the first cut matched bare
# sure/correct/right/exactly ANYWHERE, so "I'm not sure", "not exactly",
# "that is not correct", and "sure is loud in here" all read as yes — the
# exact wrong-name commit the confirm turn exists to prevent. The context is
# always a reply to "NAME — did I get that right?", so:
#   1. ANY negation cue (incl. a bare "not") -> no. A false "no" merely
#      re-asks the name; a false "yes" commits wrong biometrics. Bias hard.
#   2. Strong yes-anchors ("yes", "that's right", ...) -> yes anywhere.
#   3. Weak yes-words ("sure", "right", "correct", ...) -> yes only when they
#      LEAD a short utterance ("Sure." / "Exactly, correct") — never when
#      buried in an aside ("sure is loud in here", "turn right up there").
#   4. Anything else -> unclear (caller drops the latch silently).
_NO_RE = re.compile(
    r"\b(?:no|nope|nah|not|never|wrong|incorrect|negative|"
    r"don'?t|isn'?t|didn'?t|wasn'?t)\b", re.IGNORECASE)
_YES_STRONG_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|affirmative|"
    r"that'?s\s+(?:right|it|me|correct)|that\s+is\s+(?:right|it|me|correct)|"
    r"you\s+got\s+(?:it|that)(?:\s+right)?|spot\s+on)\b", re.IGNORECASE)
_YES_WEAK = frozenset({
    "correct", "right", "exactly", "sure", "ok", "okay", "alright", "fine",
    "absolutely", "indeed", "affirmative", "precisely", "bingo",
})
_YES_WEAK_MAX_TOKENS = 4


def confirm_verdict(text: str) -> str:
    """Classify a reply to "NAME — did I get that right?" as 'yes' | 'no' |
    'unclear'. Negation always wins; see ordering rationale above."""
    if not text:
        return "unclear"
    lower = text.strip().lower()
    if _NO_RE.search(lower):
        return "no"
    if _YES_STRONG_RE.search(lower):
        return "yes"
    tokens = re.sub(r"[^\w\s']", " ", lower).split()
    if tokens and tokens[0] in _YES_WEAK and len(tokens) <= _YES_WEAK_MAX_TOKENS:
        return "yes"
    return "unclear"


def is_affirmation(text: str) -> bool:
    """True when the turn reads as an explicit yes (negation wins ties)."""
    return confirm_verdict(text) == "yes"


def is_negation(text: str) -> bool:
    return confirm_verdict(text) == "no"


def detect_enroll_intent(text: str, speaker_name: Optional[str] = None) -> EnrollIntent:
    """Detect an enrollment intent, its scope, and a name.

    Args:
        text: ASR transcript of the user turn.
        speaker_name: currently identified voiceprint name. Retained for
            back-compat; NO LONGER used as a name fallback (2026-07-02) — a
            keyword without a usable name always routes to the ask-name latch.

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
        # NO speaker-name fallback (removed 2026-07-02, test E): a garbled or
        # reject-list "name" ("enroll me as here") used to silently commit
        # under the CURRENT SPEAKER's identity for voice-known speakers. A
        # missing/unusable name must always fall to the ask-name latch, for
        # knowns and unknowns alike.
        return EnrollIntent(matched=False, scope=scope, keyword_present=True)

    return EnrollIntent(matched=True, name=name, scope=scope, keyword_present=True)
