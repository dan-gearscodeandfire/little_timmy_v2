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
# Each captures up to FOUR word tokens ("Mary Jane", "Dan the Barbarian") —
# the single-token version silently truncated multi-word names (test F,
# 2026-07-02); epithet names need a connective slot (Dan 2026-07-05).
# Trailing non-name tokens are trimmed in _clean_name_tokens.
_NAME_TOKEN = r"[A-Za-z][a-zA-Z']{1,20}"
_NAME_SPAN = rf"({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}})"
# Each entry is (pattern, strong). ``strong`` marks non-vocative frames where
# the capture is unambiguously in name position ("my name is X", "... as X") —
# only those may lead with a soft filler token ("My name is Buddy"). Vocative-
# capable frames ("call me, buddy" / "I'm... man") stay weak (review 7-05).
# NOTE "it is X" is handled ONLY in extract_reply_name's bare-reply lead-in —
# as a full-utterance pattern it turned incidental clauses into names
# ("remember my face when it is dark in here" -> 'dark_in', review 7-05).
_NAME_PATTERNS = [
    # "my name is X" / "my name's X" / STT-flattened "my names X" — the
    # contraction miss truncated "My name's Mary Jane" to 'mary' via the
    # shared single-token extractor (name-turn miss #3, 2026-07-02).
    (re.compile(rf"\bmy\s+name(?:\s+is|\s+was|'?s)\s+{_NAME_SPAN}\b", re.IGNORECASE), True),
    (re.compile(rf"\bcall\s+me\s+{_NAME_SPAN}\b", re.IGNORECASE), False),
    (re.compile(rf"\b(?:enroll|remember|learn|save)\s+(?:me\s+|my\s+(?:face|voice)\s+)?as\s+{_NAME_SPAN}\b", re.IGNORECASE), True),
    (re.compile(rf"\bI(?:'m|\s+am)\s+{_NAME_SPAN}\b", re.IGNORECASE), False),
    (re.compile(rf"\bthis\s+is\s+{_NAME_SPAN}\b", re.IGNORECASE), False),
]

# Words that look like names but aren't.
_NON_NAMES = frozenset({
    "face", "voice", "name", "person", "this", "that", "here", "sorry", "fine",
    "okay", "ok", "yes", "no", "sure", "back", "ready", "me", "it",
    # trailing filler that the multi-word span can drag in
    "please", "now", "again", "though", "thanks",
    # stall words: "hang on" canonicalized to 'hang_on' and went to confirm
    # (name-turn miss #3, 2026-07-02)
    "hang", "wait", "hold", "um", "uh", "hmm", "well", "so", "just",
    "second", "minute", "moment",
    # verbs the bare-reply fallback can lead with ("leave it" -> 'leave',
    # "call me buddy" -> 'call' once the vocative frame is rejected — 7-05)
    "leave", "call",
    # connectives/verbs: stop the multi-token span from fusing identities
    # ("I'm Dan and Sarah is here" must parse 'dan', not 'dan_and_sarah' —
    # code review C7) and from canonicalizing evasive replies ("not telling",
    # "never mind" — code review C5).
    "and", "or", "but", "with", "plus", "also", "is", "are", "was", "were",
    "the", "a", "an", "not", "never", "telling", "mind", "nothing", "nobody",
    "none", "guys", "everyone", "everybody",
})

# Filler words that ARE plausible names ("Buddy"): rejected in bare replies
# (where they're almost always vocative filler) but accepted as the FIRST
# token of an explicit pattern — "My name is Buddy" is unambiguous, and the
# re-ask coaching funnels real Buddies to exactly that phrasing (Dan 7-05).
_SOFT_NON_NAMES = frozenset({"buddy", "man"})

# Mid-name connectives, kept only BETWEEN name tokens ("Dan the Barbarian" ->
# dan_the_barbarian, Dan 2026-07-05). Leading or trailing they break/trim as
# before ("the door" -> None, "I'm Dan the" -> dan).
_NAME_CONNECTIVES = frozenset({"the"})


def _clean_name_tokens(candidate: str, explicit: bool = False) -> Optional[str]:
    """Validate a captured span token-by-token: keep leading name-like tokens,
    stop at the first non-name word. ``explicit`` marks a capture from an
    explicit name pattern ("my name is X"), where a soft non-name may lead.
    Returns the canonical form (lowercase, spaces -> underscores, apostrophes
    dropped so "O'Brien" -> "obrien" passes NAME_RE) or None if the FIRST
    token is already a non-name (e.g. "enroll me as here")."""
    kept: list[str] = []
    pending: list[str] = []          # connectives held until a name follows
    for tok in candidate.strip().lower().split():
        if tok in _NAME_CONNECTIVES and kept:
            pending.append(tok)
            continue
        if tok in _SOFT_NON_NAMES:
            if not (explicit and not kept):
                break
        elif tok in _NON_NAMES:
            break
        tok = tok.replace("'", "")
        if tok:
            kept.extend(pending)
            pending = []
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
    for pat, strong in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = _clean_name_tokens(m.group(1), explicit=strong)
            if cand:
                return cand
    if _extract_name_from_response is not None:
        cand = _extract_name_from_response(text)
        if cand:
            # NOT explicit: the shared extractor also fires on bare tokens,
            # which carry no name-position evidence — a soft filler ("Buddy")
            # must stay rejected here and enroll only via the coached
            # "my name is X" patterns above.
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
    # Conversational lead-ins the patterns above don't cover ("It's Bob",
    # "It is Bob" — the latter parsed to 'it' pre-2026-07-05).
    bare = re.sub(r"^(?:it'?s|it\s+is|i'?m|i\s+am|"
                  r"(?:the|my)\s+name(?:'?s|\s+is)?)\s+",
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
    r"that(?:'?s|\s+is)\s+(?:right|it|me|correct|my\s+name)|"
    # paraphrased confirms, live-proven misses 2026-07-02 ("You did get that
    # right" -> unclear -> silent drop -> LLM confabulated the commit). STT
    # drops one-word turns, so EVERY confirm reply is a paraphrase.
    # DECLARATIVE forms only (review 7-05): present-tense/inverted forms with
    # optional tails read questions as confirms ("where did you get that
    # name", "can you get it right this time", "do you have it saved" all
    # classified yes — the wrong-biometrics commit C1 exists to prevent).
    # "you did get X" requires the right/correct tail; interrogative
    # inversion ("did you get...") never matches the declarative order.
    # _NO_RE still runs first, so "you did NOT get that right" is safe.
    r"you\s+did\s+get\s+(?:it|that|me)\s+(?:right|correct)|"
    r"you\s+got\s+(?:it|that)(?:\s+(?:right|correct))?|"
    r"you\s+(?:got|heard)\s+me\s+right|you\s+have\s+it\s+right|"
    r"nailed\s+it|spot\s+on)\b", re.IGNORECASE)
_YES_WEAK = frozenset({
    "correct", "right", "exactly", "sure", "ok", "okay", "alright", "fine",
    "absolutely", "indeed", "affirmative", "precisely", "bingo",
})
_YES_WEAK_MAX_TOKENS = 4


# Positive idioms whose surface contains a negation cue ("no worries, you got
# it right" read as 'no' and re-opened the name ask — review 7-05). Stripped
# before the verdict scan so the rest of the utterance decides.
_POSITIVE_IDIOM_RE = re.compile(
    r"\b(?:no\s+worries|no\s+problem|no\s+doubt)\b", re.IGNORECASE)


def confirm_verdict(text: str) -> str:
    """Classify a reply to "NAME — did I get that right?" as 'yes' | 'no' |
    'unclear'. Negation always wins; see ordering rationale above."""
    if not text:
        return "unclear"
    lower = _POSITIVE_IDIOM_RE.sub(" ", text.strip().lower())
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


# Never-silent latch dialogs (Dan 2026-07-05): an unclear confirm reply or a
# missed name NEVER drops the latch silently — the dialog re-asks with
# progressively blunter scripted prompts until a clear yes (commit), a clear
# no (correction loop), or a cancel (abort). The only no-answer exit is the
# walk-away TTL. Silent drops handed the turn to the LLM mid-dialog, which
# then confabulated enrollment success ("Fine. Joe it is." — zero [COMMIT],
# live-proven 2x on 2026-07-02).

# Explicit abort of the whole enroll dialog. Distinct from a "no" verdict —
# "no" re-opens the name ask, cancel drops everything. Checked BEFORE
# confirm_verdict because "never mind" contains a _NO_RE cue.
# NO "leave it": "yes, leave it as it is" is a CONFIRM, not an abort
# (review 7-05). The caller must also check the yes-verdict BEFORE cancel so
# "yeah, don't bother re-asking" commits instead of aborting.
_CANCEL_RE = re.compile(
    r"\b(?:never\s*mind|forget\s+(?:about\s+)?(?:it|that|this)|cancel|abort|"
    r"skip\s+it|drop\s+it|stop\s+(?:it|that|asking)|"
    r"don'?t\s+bother|no\s+thanks)\b", re.IGNORECASE)


def is_enroll_cancel(text: str) -> bool:
    """True when the turn explicitly aborts a pending enroll dialog."""
    return bool(text and _CANCEL_RE.search(text))


def confirm_reask_line(display_name: str, attempt: int) -> str:
    """Scripted re-ask after an unclear confirm verdict. ``attempt`` counts
    unclear replies so far (1-based); the script escalates in bluntness and
    coaches multi-word phrasing (STT filters one-word turns, so a coached
    "yes that is correct" survives where a bare "yes" never arrives)."""
    if attempt <= 1:
        return f"Yes or no — is the name {display_name}?"
    return (f"Say YES THAT IS CORRECT, or NO THAT IS WRONG. "
            f"Is the name {display_name}?")


def name_reask_line(attempt: int) -> str:
    """Scripted re-ask after a name-turn miss. Coaches the explicit phrasing
    that the extractor (and STT) parse most reliably."""
    if attempt <= 1:
        return "I didn't catch a name. Say 'my name is', then the name."
    return "I still didn't catch it. Say: MY NAME IS, then your name."


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
