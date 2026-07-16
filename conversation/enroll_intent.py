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
    # Parity with the retired introductions duplicate (F9, review 7-07): its
    # pattern list also handled "I go by X" and bare "name's X" (no my/the).
    (re.compile(rf"\bI\s+go\s+by\s+{_NAME_SPAN}\b", re.IGNORECASE), False),
    (re.compile(rf"\bname'?s\s+{_NAME_SPAN}\b", re.IGNORECASE), False),
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
    # temporal fillers the "call me X" frame drags in once it doubles as a
    # passive self-intro trigger ("call me later/back tomorrow" — 7-06)
    "later", "soon", "tomorrow", "tonight", "today", "anytime", "whenever",
    "sometime",
    # connectives/verbs: stop the multi-token span from fusing identities
    # ("I'm Dan and Sarah is here" must parse 'dan', not 'dan_and_sarah' —
    # code review C7) and from canonicalizing evasive replies ("not telling",
    # "never mind" — code review C5).
    "and", "or", "but", "with", "plus", "also", "is", "are", "was", "were",
    "the", "a", "an", "not", "never", "telling", "mind", "nothing", "nobody",
    "none", "guys", "everyone", "everybody",
    # prepositions/predicates in name position — "my name is hard to
    # pronounce" / "on the whiteboard" / "not important" hijacked the
    # correction FSM as claims 'hard_to_pronounce' etc. (review 7-06)
    "to", "on", "in", "of", "at", "for", "from", "by", "about", "behind",
    "under", "over", "off", "out", "up", "down", "into", "onto", "written",
    "hard", "easy", "difficult", "weird", "funny", "strange", "unusual",
    "common", "important", "irrelevant", "secret", "private", "long",
    "short", "simple", "complicated", "spelled", "spelt", "pronounced",
    "supposed", "gonna", "going", "trying", "actually", "really", "kind",
    "sort", "like", "still", "already", "only", "even",
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

@dataclass
class EnrollIntent:
    matched: bool
    name: Optional[str] = None
    scope: str = "face"                 # "both" | "face" | "voice"
    keyword_present: bool = False


def _extract_name(text: str) -> Optional[str]:
    """Return a canonical (lowercase, underscore-joined) name from ``text`` or
    None, via the local patterns (they parse "... as X" and multi-word
    names). This module IS the canonical conversational extractor now (F9,
    review 7-07): introductions.py delegates here instead of keeping its
    2026-06-06 duplicate, so the old fallback import of that duplicate is
    gone with it."""
    for pat, strong in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = _clean_name_tokens(m.group(1), explicit=strong)
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


def extract_spelled_name(text: str) -> Optional[str]:
    """Join a spelled-out letter run into a name: "T-U-S-H-A-R", "o t i s",
    "B, O, B" -> "tushar"/"otis"/"bob". The spelled form is the attendee's
    most deliberate signal (Dan 2026-07-15 — "Tushar" was mis-heard and the
    spelling was IGNORED), so callers should prefer it over a word-form name
    in the same utterance ("Odis, O-T-I-S" -> "otis"). Requires a run of >=3
    single-letter tokens — ordinary sentences don't produce those, and 2-runs
    collide with "a"/"I"."""
    if not text:
        return None
    toks = re.sub(r"[-,.]", " ", text.lower()).split()
    best: list = []
    run: list = []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
        else:
            if len(run) > len(best):
                best = run
            run = []
    if len(run) > len(best):
        best = run
    if len(best) >= 3:
        return "".join(best)
    return None


def extract_reply_name(text: str) -> Optional[str]:
    """Extract a canonical name from a reply to the ask-name latch.

    Unlike _extract_name (full enroll utterances), this handles bare replies:
    "Mary Jane." / "It's Bob" / "My name is Mary Jane". A spelled-out letter
    run wins over everything (it corrects the very mishear the word form just
    made); then the local multi-word-capable patterns (the shared extractor
    captures a single \\w+ and would truncate "My name is Mary Jane" to
    'mary' — code review C6), rejects evasive replies (C5), and only then
    falls back to treating a short 1-3-token utterance as the name itself.
    """
    if not text or _EVASIVE_RE.search(text):
        return None
    spelled = extract_spelled_name(text)
    if spelled:
        return spelled
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


def latch_speaker_ok(latch_key: Optional[str], speaker_name: str) -> bool:
    """Cross-category latch speaker gate (Dan 2026-07-16, option C).

    A latch armed by a KNOWN speaker is answerable only by that exact
    speaker. A latch armed by an unknown_N is answerable by ANY unknown_*
    — the same human drifts across unknown clusters on short utterances
    (live-proven 7-16: unknown_10 -> unknown_11 mid-dialog) — but never by
    a known speaker, whose words near the mic are coaching or cross-talk,
    not the visitor's reply. Live failure this gates out: a dan-keyed
    correction latch consumed unknown_11's "My name is Frank!" as a
    confirm reply, and unknown_11's own self-intro never armed."""
    if not latch_key:
        return True  # legacy latch without a key — no gate to apply
    if latch_key.startswith("unknown_"):
        return bool(speaker_name) and speaker_name.startswith("unknown_")
    return speaker_name == latch_key


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
    that the extractor (and STT) parse most reliably; the escalation asks for
    a SPELLING (Dan 2026-07-15 — a spelled run beats an STT mishear, see
    extract_spelled_name)."""
    if attempt <= 1:
        return "I didn't catch a name. Say 'my name is', then the name."
    return ("I still didn't catch it. Say your name, then spell it out "
            "letter by letter.")


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


# --- Identity-correction protest (Dan 2026-07-06) ---------------------------
# A misidentified user pushes back on the name Timmy used: "No, my name is
# not Walter, my name is Flynn", "stop calling me Walter", or a bare
# "My name is Flynn" that CONTRADICTS the current attribution. Deterministic
# regex like detect_enroll_intent above; the caller routes a match into the
# SAME never-silent confirm FSM, and commit_identity's mismatch/lookalike
# guards do the biometric gating at commit time — a stranger cannot talk
# their way into an enrolled identity. Lexicon kept TIGHT; expect live
# phrasing tuning (feedback: green suites miss what people actually say).

@dataclass
class CorrectionIntent:
    matched: bool
    name: Optional[str] = None    # claimed replacement name (canonical), if any
    denied: Optional[str] = None  # the rejected name, when the phrase named it


# LAZY name span for denial frames. The greedy _NAME_SPAN would swallow the
# following claim in unpunctuated STT ("my name is not walter my name is
# flynn" -> denied 'walter_my') and the strip below would then delete the
# claim too. Lazy keeps the denied capture minimal (first token) — cosmetic
# loss on multi-word denied names, but the claim survives.
_DENY_SPAN = rf"({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}?)"

# Denial frames that NAME the rejected identity. "I'm not X" is vocative-
# loose ("I'm not patting myself on the back", live 7-06) so it only counts
# when X matches the attributed speaker; the "my name is not X" / "stop
# calling me X" frames are unambiguous identity statements and count as-is.
_DENY_NAMED_STRONG_RES = [
    # name(?:s|'s)? absorbs the STT contraction/flattening ("my name's not
    # walter", "my names not walter") — review 7-06: the contraction miss
    # dropped the whole protest, even punctuated.
    re.compile(rf"\bmy\s+name(?:s|'s)?\s+(?:is\s+not|isn'?t|ain'?t"
               rf"|was\s+not|wasn'?t|not)\s+{_DENY_SPAN}\b", re.IGNORECASE),
    re.compile(rf"\b(?:stop|quit|don'?t)\s+call(?:ing)?\s+me\s+{_DENY_SPAN}\b",
               re.IGNORECASE),
]


def _deny_weak_re(spk: str) -> re.Pattern:
    """Compile "I'm not <attributed name>" for the CURRENT speaker. The frame
    only ever counts when the denied name equals the attribution, so matching
    the attribution directly sidesteps span-capture pitfalls entirely: the
    greedy _NAME_SPAN swallowed the claim in unpunctuated STT ("i'm not
    walter i'm flynn" -> capture 'walter i'm flynn' != spk -> protest
    silently dropped, review 7-06), and a lazy span would under-capture
    multi-word attributions ("dan the barbarian" -> 'dan' != spk)."""
    toks = r"\s+".join(re.escape(t) for t in spk.split("_") if t)
    return re.compile(rf"\bI(?:'m|\s+am)\s+not\s+{toks}\b", re.IGNORECASE)
# Frameless denials. No bare "wrong name" — too loose outside this frame.
_DENY_BARE_RE = re.compile(
    r"\bthat(?:'s|\s+is)\s+not\s+my\s+name\b|"
    r"\byou(?:'ve|\s+have)?\s+got\s+my\s+name\s+wrong\b", re.IGNORECASE)


def _claim_name(text: str, allow_weak: bool) -> Optional[str]:
    """Claimed-name extraction for correction turns. Reuses _NAME_PATTERNS
    but (unlike extract_reply_name) has NO bare-token fallback — a correction
    rides inside a full sentence, so only framed names count. Weak (vocative-
    capable) frames like "I'm Y" are honored only after a NAMED denial
    ("Stop calling me Walter, I'm Flynn") — a bare "that's not my name"
    doesn't qualify (live 7-06: "I'm supposed to..." became the claim)."""
    for pat, strong in _NAME_PATTERNS:
        if not strong and not allow_weak:
            continue
        # ALL matches, not just the first: in "no my names not walter my
        # name is flynn" the first "my name(s)..." hit cleans to None and
        # the claim lives at the SECOND match of the same pattern (7-06).
        for m in pat.finditer(text):
            cand = _clean_name_tokens(m.group(1), explicit=strong)
            if cand:
                return cand
    return None


def detect_identity_correction(
        text: str, speaker_name: Optional[str] = None,
        speaker_enrolled: bool = False) -> CorrectionIntent:
    """Detect a misidentification protest.

    Args:
        text: ASR transcript of the user turn.
        speaker_name: current voiceprint attribution (canonical), or None.
        speaker_enrolled: True when speaker_name is an enrolled identity —
            gates the bare-claim path (unknown speakers are introductions'
            turf; enroll keywords are detect_enroll_intent's, which the
            caller must run FIRST).

    Matches:
      * denial + claim  ("no, my name is not Walter, my name is Flynn")
        -> matched, name=claim — any speaker state.
      * bare claim      ("My name is Flynn") -> matched ONLY when it
        contradicts an enrolled attribution (Dan 7-06: fire on
        contradiction, not on every self-introduction).
      * denial only     ("that's not my name") -> matched, name=None —
        caller routes to the ask-name latch.
    """
    if not text:
        return CorrectionIntent(matched=False)
    spk = (speaker_name or "").strip().lower()

    denied: Optional[str] = None
    stripped = text
    for pat in _DENY_NAMED_STRONG_RES:
        m = pat.search(stripped)
        if m:
            cand = _clean_name_tokens(m.group(1))
            if cand:
                denied = cand
                stripped = stripped[:m.start()] + " , " + stripped[m.end():]
                break
    if denied is None and spk:
        m = _deny_weak_re(spk).search(stripped)
        if m:
            denied = spk
            stripped = stripped[:m.start()] + " , " + stripped[m.end():]
    bare_denial = bool(_DENY_BARE_RE.search(stripped))
    has_denial = denied is not None or bare_denial

    # Weak "I'm Y" claims unlock only on a NAMED denial ("Stop calling me
    # Walter, I'm Flynn"). A bare denial must NOT unlock them: "So I'm
    # supposed to just say to you, that's not my name" (live 7-06) had the
    # weak frame swallow "supposed to" as the claim. Bare-denial turns route
    # to the ask-name latch instead — never-silent, one extra turn at worst.
    claim = _claim_name(stripped, allow_weak=denied is not None)
    if claim and spk and claim == spk:
        # Claiming the CURRENT attribution corrects nothing ("My name is
        # Dan" while attributed dan) — stay out of the LLM's way.
        return CorrectionIntent(matched=False)

    if claim:
        if has_denial:
            return CorrectionIntent(matched=True, name=claim, denied=denied)
        if speaker_enrolled and spk and not spk.startswith("unknown_"):
            return CorrectionIntent(matched=True, name=claim, denied=spk)
        return CorrectionIntent(matched=False)
    if has_denial:
        return CorrectionIntent(matched=True, denied=denied or (spk or None))
    return CorrectionIntent(matched=False)


# --- Passive self-introduction (LED-mic anchor, 2026-07-06) -----------------
# An UNKNOWN speaker volunteering their name ("my name is Flynn") feeds the
# introductions confirm flow without Timmy having to ask first. Deliberately
# NOT folded into detect_identity_correction: its bare-claim branch is
# contractually scoped to CONTRADICTING an enrolled attribution (Dan 7-06 —
# "fire on contradiction, not on every self-introduction"); unknown-speaker
# self-intro is the Introductions FSM's turf. Framed names ONLY — no bare
# tokens and no weak "I'm X" frame ("I'm tired" is the false-positive
# vector). The caller gates on speaker_name.startswith("unknown_") and runs
# detect_enroll_intent / detect_identity_correction FIRST (keywords and
# denials win).
_SELF_INTRO_PATTERNS = [
    # "my name is X" / "my name's X" (strong: soft filler may lead)
    (re.compile(rf"\bmy\s+name(?:\s+is|'?s)\s+{_NAME_SPAN}\b", re.IGNORECASE), True),
    (re.compile(rf"\bcall\s+me\s+{_NAME_SPAN}\b", re.IGNORECASE), False),
    (re.compile(rf"\bI\s+go\s+by\s+{_NAME_SPAN}\b", re.IGNORECASE), True),
]


def detect_self_intro(text: str) -> Optional[str]:
    """Canonical name from an unsolicited self-introduction, or None.

    Negated frames fail safe through _clean_name_tokens ("my name is not
    Walter" -> first token 'not' is a non-name -> None), so a denial never
    reads as an intro even when the correction detector didn't run."""
    if not text or _EVASIVE_RE.search(text):
        return None
    for pat, strong in _SELF_INTRO_PATTERNS:
        for m in pat.finditer(text):
            cand = _clean_name_tokens(m.group(1), explicit=strong)
            if cand:
                return cand
    return None
