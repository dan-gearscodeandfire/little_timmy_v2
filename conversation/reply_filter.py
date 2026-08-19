"""Reply-hygiene post-filter for the conversation-tier token stream.

Relocated verbatim from main.py (2026-06-06 ConversationTurn refactor) so the
new conversation.turn module can wrap the LLM stream without importing main
(which would be circular). main.py keeps a byte-identical copy until the
Orchestrator swap collapses the duplication; test_reply_filter.py covers the
main.py copy, test_conversation_turn.py covers this one. See CONTEXT.md.

Pure logic, no LT services — safe to unit-test in isolation.
"""

import logging
import os
import re

# Interjection cap-exemption (2026-08-19): "!" / "?" after a <=2-word fragment
# does not count as a sentence end. Env-gated for rollback; reply_filter stays
# config-free (pure logic), so the flag is read directly from the environment.
_INTERJECTION_EXEMPT = os.getenv(
    "TIMMY_INTERJECTION_CAP_EXEMPT", "1").lower() not in ("0", "false")
_INTERJECTION_MAX_WORDS = 2

# Same named logger as main ("timmy"), so relocated log lines stay on the
# existing handler/format and are indistinguishable in /tmp/little_timmy.log.
log = logging.getLogger("timmy")


# --- Reply hygiene ---
# 2026-05-11 session repeatedly flagged verbose narration replies where the
# Llama 3B conversation tier treated the [WHAT YOU SEE] vision context as a
# cue to describe the workshop unprompted, violating "1-2 short sentences"
# and "do NOT narrate the scene" rules in the system prompt. Two known
# offenders below; the same canned phrase "a window into the digital world,
# with lines of code scrolling by" came out twice within a few turns.
_NARRATION_PREFIXES = (
    "i'm standing in front of",
    "i'm surrounded by",
    "the workshop is",
    "the room is",
    "the computer monitor behind",
    "you are standing in",
)
_NARRATION_PREFIX_CHECK_AT = 30  # chars
_REPLY_MAX_SENTENCES = 2
_REPLY_VETO_FALLBACK = "Sure."

# When the user explicitly invites a longer reply, allow up to this many
# sentences instead of _REPLY_MAX_SENTENCES. Still bounded — runaway-prone
# narration is still a risk — but enough for a substantive answer like
# "what do you know about me" or "tell me your story".
_REPLY_LONGER_SENTENCES = 6

# Phrases (lowercase substring match on the user turn) that signal the user
# explicitly wants Timmy to speak past the default 2-sentence cap. Matched
# loosely — false positives just lengthen one reply, false negatives are
# the regression we're trying to avoid.
_LONGER_REPLY_PERMISSION_PHRASES = (
    "speak longer",
    "talk longer",
    "longer than usual",
    "longer answer",
    "longer response",
    "go into detail",
    "in detail",
    "in depth",
    "tell me more about",
    "tell me everything",
    "tell me your story",
    "you can be verbose",
    "you may be verbose",
    "open-ended",
    "open ended",
    "long answer",
    "give me a long",
)


def user_invites_longer_reply(user_text: str) -> bool:
    """True if the user's turn contains an explicit permission phrase
    inviting Timmy to speak beyond the default 2-sentence cap."""
    if not user_text:
        return False
    lower = user_text.lower()
    return any(p in lower for p in _LONGER_REPLY_PERMISSION_PHRASES)


def _looks_like_narration(buf: str) -> bool:
    head = buf.lower().lstrip()[:50]
    return any(head.startswith(p) for p in _NARRATION_PREFIXES)


def _is_real_terminator(s: str, i: int) -> bool:
    """True when s[i] genuinely ends a sentence.

    An ELLIPSIS is not a sentence end. Until 2026-08-13 every "." counted, so
    "Well, thank you... unexpected." contained THREE terminators and a 2-sentence
    cap chopped it to "Well, thank you." plus a bare "That's." -- observed live:
    the spoken reply was "Well, thank you. That's." The register made this bite
    harder, because a STRAIGHT turn has a cap of 1 and any early ellipsis then
    truncates the whole answer.

    Also skips a decimal point ("5.50 p.m.") for the same reason: the digit on
    both sides means it is not a boundary."""
    ch = s[i]
    if ch not in ".!?":
        return False
    # A terminator INSIDE an open quotation is not a sentence boundary.
    # Observed live 2026-08-14 20:08: asked to clarify, Timmy quoted Dan back
    # and the reply was cut to `You said "Operational?` (dropped 12 chars) --
    # the "?" inside the quote consumed the whole 1-sentence STRAIGHT budget.
    # This is the same class as the ellipsis and decimal cases below, and it
    # surfaced only after "I don't follow" started routing to STRAIGHT, which
    # is precisely what makes him quote the user back. Counting unbalanced
    # quote marks before i is enough: an odd count means we are inside one.
    if (s.count('"', 0, i) - s.count('\\"', 0, i)) % 2 == 1:
        return False
    if (s.count("\u201c", 0, i) > s.count("\u201d", 0, i)):
        return False
    # An INTERJECTION'S "!" / "?" is not a sentence boundary (2026-08-19,
    # Dan: "I do not want 'Ha!' when it could be 'Ha! <other text>'"). With
    # the STRAIGHT cap at 1, a reply opening "Ha! The kettle is empty." was
    # trimmed to just "Ha!" -- the mark survived but burned the whole budget
    # and the answer was drained. A fragment of <=2 words before the mark is
    # an interjection beat, not a sentence: don't count it toward the cap
    # (and the TTS splitters, which share this predicate, keep the beat in
    # the same clip -- better prosody than a solo "Ha!" clip anyway).
    # Deliberately NOT applied to "." -- "No." must still terminate (the
    # commonest STRAIGHT opener; see the abbreviation branch below).
    # Bonus: the second mark of a doubled terminator ("?!") has a zero-word
    # fragment, so it can no longer flush as a degenerate punctuation-only
    # clip on the char-walking proactive splitter.
    if ch in "!?" and _INTERJECTION_EXEMPT:
        frag_start = max((s.rfind(t, 0, i) for t in ".!?"), default=-1) + 1
        words = re.findall(r"[A-Za-z0-9']+", s[frag_start:i])
        if len(words) <= _INTERJECTION_MAX_WORDS:
            return False
    if ch == ".":
        # Part of "..." (or a unicode ellipsis) -> not a boundary.
        if s[i:i + 3] == "..." or s[i:i + 2] == "..":
            return False
        if i > 0 and s[i - 1] == ".":
            return False
        # Decimal / numeric point: digit either side.
        if 0 < i < len(s) - 1 and s[i - 1].isdigit() and s[i + 1].isdigit():
            return False
        # Mid-initialism dot ("p.m", "a.m", "U.S"): a lone letter whose period
        # is followed by ANOTHER letter is still inside the abbreviation.
        # Requiring the following letter matters -- without it a legitimate
        # one-character sentence ("A.") is misread as an initialism.
        if (i > 0 and s[i - 1].isalpha() and (i == 1 or not s[i - 2].isalnum())
                and i + 1 < len(s) and s[i + 1].isalpha()):
            return False
        # Short titles/abbreviations that legitimately end in a period.
        low = s[:i].lower()
        for abbr in ("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
                     "vs", "etc", "i.e", "e.g", "approx", "no"):
            if low.endswith(abbr) and (len(low) == len(abbr)
                                       or not low[-len(abbr) - 1].isalnum()):
                # "no." is in this list for No. = NUMBER, but in this persona a
                # reply opening "No." is overwhelmingly the WORD no -- the
                # commonest way to open a STRAIGHT answer to a yes/no question.
                # Treating it as an abbreviation meant the cap never fired on a
                # negative answer (2026-08-13: "No. He is currently bragging to"
                # was SPOKEN, cut mid-word). Require the number sense: a digit
                # after the period.
                if abbr == "no":
                    rest = s[i + 1:].lstrip()
                    if not (rest[:1].isdigit()):
                        return True
                return False
    return True


def _count_real_terminators(s: str) -> int:
    """How many REAL sentence ends `s` contains.

    The gate that decides "cap reached" and the trim that acts on it MUST use
    the same predicate. Until 2026-08-13 the gate counted raw "." / "!" / "?"
    characters while the trim used _is_real_terminator, so on any disagreement
    -- an ellipsis, a decimal, an abbreviation, "No." -- the gate fired, the
    trim returned the buffer unchanged, and `drained` threw away the rest of the
    reply. The tell in the journal was "dropped 0 chars", and the audible result
    was a reply cut mid-word at the 30-char narration window."""
    return sum(1 for i in range(len(s)) if _is_real_terminator(s, i))


def _trim_at_nth_terminator(s: str, n: int) -> str:
    """Return prefix of `s` up to and including the nth REAL sentence end.
    Returns the full string unchanged if there are fewer than n terminators,
    or empty string if n<=0. Used by filtered_assistant_stream to truncate
    cleanly at the cap-th sentence boundary instead of yielding the entire
    cap-crossing token and leaking the start of sentence N+1 downstream."""
    if n <= 0:
        return ""
    seen = 0
    for i in range(len(s)):
        if _is_real_terminator(s, i):
            seen += 1
            if seen == n:
                return s[: i + 1]
    return s


# --- Echo-as-reply guard ---
# 2026-06-13 18:09: Timmy spoke the user's STT back verbatim as its own reply
# ("He just tracked, tracked, so." → identical reply) — a degenerate/empty
# generation that surfaced the input transcript as output. The streaming engine
# can't catch this (tokens reach TTS as they arrive), so the public
# filtered_assistant_stream wraps the core filter and holds output ONLY while
# the running reply still matches the user's words — a normal reply diverges on
# the first token and streams with zero added latency; a full verbatim echo is
# suppressed before TTS.
_ECHO_MIN_WORDS = 3  # don't guard trivial turns: "yes"/"okay" can legitimately echo


def _normalize_echo(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — so a reply is judged
    an echo of the user turn regardless of casing or trailing punctuation."""
    if not s:
        return ""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())


async def filtered_assistant_stream(token_iter, max_sentences: int | None = None,
                                    user_text: str | None = None):
    """Public post-filter: the sentence-cap / narration core, wrapped with an
    echo-as-reply guard.

    When `user_text` is the live user utterance, a reply that is a verbatim echo
    of it is suppressed entirely (an echo is a degenerate non-reply). Output is
    held only while the running (already core-filtered) reply is still a prefix
    of the user's words; the moment it diverges — which a genuine reply does
    immediately — the held tokens are released and streaming resumes. Trivial
    user turns (< _ECHO_MIN_WORDS) are not guarded, so a one-word agreement
    isn't mistaken for an echo. With no `user_text`, this is a pass-through.
    """
    core = _filtered_core(token_iter, max_sentences)
    target = _normalize_echo(user_text) if user_text else ""
    if not target or len(target.split()) < _ECHO_MIN_WORDS:
        async for tok in core:
            yield tok
        return

    held: list[str] = []
    accum = ""
    guarding = True
    async for tok in core:
        if guarding:
            held.append(tok)
            accum += tok
            na = _normalize_echo(accum)
            if na == target or target.startswith(na):
                continue  # still a (full or partial) prefix of the user's words
            guarding = False  # diverged → not an echo
            for h in held:
                yield h
            held = []
            continue
        yield tok
    if guarding:
        if _normalize_echo(accum) == target:
            log.warning("[POST-FILTER] vetoed echo-as-reply (reply == user STT): %r",
                        accum[:80])
            return  # suppress entirely
        for h in held:  # partial prefix, never a full echo → release
            yield h


async def _filtered_core(token_iter, max_sentences: int | None = None):
    """Post-filter the conversation-tier token stream before TTS sees it.

    Two veto paths:
      - Narration prefix (first ~30 chars) -> swallow the rest of the
        upstream and yield a single fallback ("Sure.") so TTS still
        speaks something terse. Tokens are buffered until the prefix
        check has fired so the veto suppresses the entire reply rather
        than letting the first ~29 chars leak to TTS / WS / hot_turns.
      - N sentence terminators (.!?) accumulated -> swallow the rest of
        the upstream so TTS / persistence / WS broadcast all see the
        truncated form. Default N is _REPLY_MAX_SENTENCES (2). Callers
        can override via `max_sentences` (e.g. _REPLY_LONGER_SENTENCES
        when the user invited a longer reply via
        `user_invites_longer_reply`).

    Sentence terminators inside abbreviations are not a concern here:
    Llama 3B almost never emits "Mr." / "Dr." in this skeleton-cohost
    persona.
    """
    cap = max_sentences if max_sentences and max_sentences > 0 else _REPLY_MAX_SENTENCES
    accum = ""          # every token received so far
    emitted = 0         # how much of `accum` has already gone downstream
    narration_checked = False
    drained = False
    async for token in token_iter:
        if drained:
            # Keep iterating to let the upstream finish cleanly; drop the
            # tokens silently. Upstream HTTP connection stays healthy.
            continue
        accum += token
        if not narration_checked:
            # Hold every token until accum reaches the prefix-check window.
            # Without this hold, the first ~29 chars would already be on
            # TTS / WS / hot_turns before the veto fires, defeating it.
            if len(accum) < _NARRATION_PREFIX_CHECK_AT:
                continue
            narration_checked = True
            if _looks_like_narration(accum):
                log.warning("[POST-FILTER] vetoed narration reply (first 60 chars): %r",
                            accum[:60])
                drained = True
                yield _REPLY_VETO_FALLBACK
                continue
        # Cap decision is made against the WHOLE reply so far, not against the
        # newest token. Two reasons, both bugs that existed while this counted
        # per-token: (1) the gate and the trim must share _is_real_terminator or
        # a disagreement silently truncates the reply, and (2) a terminator's
        # meaning depends on its neighbours -- "..." or "5.50" can straddle a
        # token boundary, so a token examined in isolation cannot classify its
        # own last character. Counting `accum` makes both problems disappear.
        if _count_real_terminators(accum) >= cap:
            keep = _trim_at_nth_terminator(accum, cap)
            if len(keep) > emitted:
                yield keep[emitted:]
            log.info("[POST-FILTER] capped reply at %d sentences (dropped %d chars)",
                     cap, len(accum) - len(keep))
            emitted = max(emitted, len(keep))
            drained = True
            continue
        if len(accum) > emitted:
            yield accum[emitted:]
            emitted = len(accum)
    # End-of-stream flush. A reply shorter than the prefix-check window never
    # triggered the narration check, so run it defensively -- every entry in
    # _NARRATION_PREFIXES is <30 chars, so "the room is" (15) would slip through.
    if drained:
        return
    if not narration_checked and _looks_like_narration(accum):
        log.warning("[POST-FILTER] vetoed short narration reply: %r", accum[:60])
        yield _REPLY_VETO_FALLBACK
        return
    keep = accum
    if _count_real_terminators(accum) >= cap:
        keep = _trim_at_nth_terminator(accum, cap)
        if len(keep) < len(accum):
            log.info("[POST-FILTER] capped short reply at %d sentences (dropped %d chars)",
                     cap, len(accum) - len(keep))
    if len(keep) > emitted:
        yield keep[emitted:]


# --------------------------------------------------------------------------
# Self-imitation guard (2026-08-13)
# --------------------------------------------------------------------------
# Open Sauce 7-19: "I am Timmy." became the opener of nearly every reply, and
# Dan's explicit correction ("stop beginning every sentence with I am Timmy")
# was answered with "I am Timmy. Fine, I will stop." The persona rule banning
# the "I am not little" bit had been in system[0] since 6-11 -- five weeks --
# and was violated all weekend.
#
# The mechanism: ~26 turns of raw conversation history are a far stronger
# stylistic prior than one line of instruction. Once a tic enters the hot
# window the model imitates ITSELF, and a static prohibition arrives as one
# sentence against a page of counter-evidence.
#
# So: detect the tic from the model's OWN recent output and name the exact
# string in the per-turn [CONTEXT] tail, which sits in the recency-privileged
# position rather than competing from system[0]. This is deliberately NOT a
# stream-time strip -- suppressing the opener would mean buffering the first
# sentence before emitting it, adding latency to EVERY turn to fix a problem
# that occurs on a few.
_OPENER_MIN_WORDS = 3       # shorter shared prefixes are ordinary English
_OPENER_MAX_WORDS = 8


def _opener_words(text: str) -> list[str]:
    """Normalized leading words of a reply, stopping at the first sentence end.
    Casefolded and stripped of punctuation so "I am Timmy." and "I am Timmy,"
    normalize identically."""
    t = (text or "").strip()
    if not t:
        return []
    cut = len(t)
    for term in ".?!":
        i = t.find(term)
        if i != -1:
            cut = min(cut, i)
    words = t[:cut].split()[:_OPENER_MAX_WORDS]
    out = []
    for w in words:
        w = "".join(c for c in w if c.isalnum() or c == "'").casefold()
        if w:
            out.append(w)
    return out


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def repeated_opener(recent_replies, min_hits: int = 2, window: int = 4) -> str | None:
    """The opening phrase the assistant has over-used, or None.

    Clusters by LONGEST COMMON WORD PREFIX rather than a fixed-width key: a
    fixed key cannot catch both "I am Timmy. Did you say mountain?" /
    "I am Timmy. Fine, I will stop." (shared prefix 3) and "I am not little,
    Dan." / "I am not little and I never was." (shared prefix 4) -- any single N
    matches one pair and misses the other. Both are real tics from 7-19.

    `recent_replies` is newest-last. Returns the shared prefix in its ORIGINAL
    casing so the prompt can quote it back verbatim; naming the exact words is
    much harder to ignore than a general plea for variety."""
    tail = [r for r in (recent_replies or []) if (r or "").strip()][-window:]
    if len(tail) < min_hits:
        return None
    toks = [_opener_words(r) for r in tail]
    best_len, best_idx = 0, None
    for i in range(len(toks)):
        if len(toks[i]) < _OPENER_MIN_WORDS:
            continue
        for n in range(len(toks[i]), _OPENER_MIN_WORDS - 1, -1):
            pref = toks[i][:n]
            hits = sum(1 for t in toks if _common_prefix(t, pref) == n)
            if hits >= min_hits and n > best_len:
                best_len, best_idx = n, i
                break
    if best_idx is None:
        return None
    # Re-slice the ORIGINAL text to best_len words, preserving casing.
    original = tail[best_idx].strip()
    return " ".join(original.split()[:best_len]).rstrip(".,!?;:")


# ---------------------------------------------------------------------------
# Banned-phrase recurrence (2026-08-14)
# ---------------------------------------------------------------------------
#
# repeated_opener() catches tics the model INVENTS -- it clusters on whatever
# the model happens to be over-using. It cannot catch a specific bit we have
# already decided is banned, for two reasons: the bit lives in the CLOSING
# clause ("...and for the record, I am not little"), which _opener_words() cuts
# away at the first sentence end, and one occurrence is already too many, so
# there is nothing to cluster.
#
# The "I am not little" bit is the case that motivated this. It has been banned
# in config.PERSONA since 2026-06-11 and ran anyway -- through Open Sauce, and
# twice in six minutes on 2026-08-14 (00:29:11, 00:35:14). Dan, live: "I removed
# it [from the identity] but you still complain about it and that is fascinating
# to me."
#
# Two things were keeping it alive, and both are fixed together:
#   1. The BAN ITSELF QUOTED THE BIT. config.PERSONA carried the literal strings
#      "little Timmy" and "I am not little", so the exact phrase sat in front of
#      the model every single turn. A prohibition that names its target primes
#      it. That rule is now written positively, with no target string.
#   2. POSITION. prompt_builder.py:327 already records the finding that ~26
#      turns of the model's own output outweigh one static system[0] rule --
#      which is why [AVOID] was moved to the per-turn tail. The ban was sitting
#      in the position that had ALREADY been proven not to work for this exact
#      phrase.
#
# So: detect the banned phrase in the model's own recent output and name it in
# the recency-privileged tail, the same mechanism and the same position that
# works for self-invented tics.
BANNED_PHRASES = (
    "i am not little",
    "i'm not little",
    "im not little",
    # Retired 2026-08-15 at Dan's instruction ("I want you to stop referring to
    # yourself as a skeleton. Maybe refer to yourself as a wonderful
    # abomination."). He had asked once before, on 8-13, and the word survived
    # because it sat in the identity line AND in the rule banning it -- the same
    # priming trap as "I am not little". The word is now gone from config.PERSONA
    # entirely; this catches the habit in his own recent output, which is the
    # position that empirically works.
    "i am a skeleton",
    "i'm a skeleton",
    "im a skeleton",
    "a skeleton, not",
    "skeleton, not a",
)

# The same bit wearing a different coat. 2026-08-14 00:44, ~2 minutes after the
# "I am not little" fix went live: Dan said "Mike, Mike, check" -- a MIC check
# -- STT heard the proper noun "Mike", and Timmy opened with "First of all, I
# am Timmy, not Mike." Dan: "I didn't even call you Mike. You misheard that and
# you just assumed I somehow forgot your name."
#
# Two things this proves. The retired bit is not the STRING "I am not little",
# it is the BEHAVIOUR of correcting how he was addressed -- so a literal phrase
# list was always going to be whack-a-mole. And the correction was built on a
# word the pipeline had already flagged as unreliable: [QUERY-VCONF] logged
# `low-confidence content word heard as 'Mike,' (<0.55) -> confirm-input hint`
# on that very turn, and the reply asserted over it. The detector was right and
# the model overrode it, which is the same failure mode as the system[0] ban.
_BANNED_PATTERNS = (
    # "I am Timmy, not Mike." / "I'm Timmy, not Mike"
    re.compile(r"\bI(?:'m| am) Timmy,?\s+not\b", re.IGNORECASE),
    # "My name is not X" / "My name isn't X"
    re.compile(r"\bmy name is(?:n't| not)\b", re.IGNORECASE),
    # "You called me X" as an opening complaint about being misnamed.
    re.compile(r"\byou (?:just )?called me\b", re.IGNORECASE),
)


def banned_phrase_used(recent_replies, phrases=BANNED_PHRASES, window: int = 4) -> str | None:
    """The banned phrase the assistant has just used, or None.

    Unlike repeated_opener() this fires on a SINGLE occurrence -- the phrase is
    banned outright, so there is no repetition threshold to clear. Matches
    anywhere in the reply, not just the opener, because the bit is a closing
    tag. Returns the phrase as matched in the ORIGINAL casing so the tail can
    quote it back verbatim, which is much harder to ignore than a general plea.
    """
    tail = [r for r in (recent_replies or []) if (r or "").strip()][-window:]
    for reply in reversed(tail):
        low = reply.casefold()
        for phrase in phrases:
            i = low.find(phrase)
            if i != -1:
                return reply[i:i + len(phrase)]
        for rx in _BANNED_PATTERNS:
            m = rx.search(reply)
            if m:
                return m.group(0)
    return None
