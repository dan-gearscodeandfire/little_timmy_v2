"""Reply-hygiene post-filter for the conversation-tier token stream.

Relocated verbatim from main.py (2026-06-06 ConversationTurn refactor) so the
new conversation.turn module can wrap the LLM stream without importing main
(which would be circular). main.py keeps a byte-identical copy until the
Orchestrator swap collapses the duplication; test_reply_filter.py covers the
main.py copy, test_conversation_turn.py covers this one. See CONTEXT.md.

Pure logic, no LT services — safe to unit-test in isolation.
"""

import logging
import re

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


def _trim_at_nth_terminator(s: str, n: int) -> str:
    """Return prefix of `s` up to and including the nth `.!?` occurrence.
    Returns the full string unchanged if there are fewer than n terminators,
    or empty string if n<=0. Used by filtered_assistant_stream to truncate
    cleanly at the cap-th sentence boundary instead of yielding the entire
    cap-crossing token and leaking the start of sentence N+1 downstream."""
    if n <= 0:
        return ""
    seen = 0
    for i, ch in enumerate(s):
        if ch in ".!?":
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
    accum = ""
    buffered = ""
    sentence_count = 0
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
            buffered += token
            if len(accum) < _NARRATION_PREFIX_CHECK_AT:
                continue
            narration_checked = True
            if _looks_like_narration(accum):
                log.warning("[POST-FILTER] vetoed narration reply (first 60 chars): %r",
                            accum[:60])
                drained = True
                yield _REPLY_VETO_FALLBACK
                continue
            # Safe prefix — flush the buffer in one chunk and resume
            # streaming. Sentence counting catches up on the buffered text.
            # 2026-05-15: if the buffer already crosses the cap, trim it at
            # the cap-th terminator so we don't leak the start of sentence
            # N+1 into TTS / hot_turns. Previously the entire buffered
            # prefix was yielded and the leaked partial got picked up by
            # the end-of-stream flush in _stream_to_tts, producing audible
            # mid-sentence cutoffs (e.g. "Fine. Dexter and Preston. I'll").
            buf_terminators = sum(1 for ch in buffered if ch in ".!?")
            if sentence_count + buf_terminators >= cap:
                trimmed = _trim_at_nth_terminator(buffered, cap - sentence_count)
                yield trimmed
                log.info("[POST-FILTER] capped reply at %d sentences (trimmed in narration-flush; dropped %d chars)",
                         cap, len(buffered) - len(trimmed))
                sentence_count = cap
                buffered = ""
                drained = True
                continue
            sentence_count += buf_terminators
            yield buffered
            buffered = ""
            continue
        # Token branch: same trim-at-cap discipline as the narration flush.
        tok_terminators = sum(1 for ch in token if ch in ".!?")
        if sentence_count + tok_terminators >= cap:
            trimmed = _trim_at_nth_terminator(token, cap - sentence_count)
            yield trimmed
            log.info("[POST-FILTER] capped reply at %d sentences (trimmed mid-token; dropped %d chars)",
                     cap, len(token) - len(trimmed))
            sentence_count = cap
            drained = True
            continue
        sentence_count += tok_terminators
        yield token
    # End-of-stream flush: a reply shorter than the prefix-check window
    # never triggered the narration check. Every entry in _NARRATION_PREFIXES
    # is <30 chars, so a reply that is exactly "the room is" (15 chars) and
    # then stops would otherwise slip through. Run the check defensively.
    # Also apply the sentence cap here — short replies that fit entirely
    # inside the buffer bypass the per-token cap check otherwise.
    if buffered and not drained:
        if _looks_like_narration(accum):
            log.warning("[POST-FILTER] vetoed short narration reply: %r", accum[:60])
            yield _REPLY_VETO_FALLBACK
        else:
            # Trim to the cap-th terminator if the buffer has ≥cap terminators
            # AND any trailing content after the cap-th. Catches both
            # "A. B. C." (3 terminators, cap=2) and "A. B. C" (2 terminators
            # + partial third sentence) — both previously fell through as
            # incomplete-trailing-sentence cutoffs.
            trimmed = _trim_at_nth_terminator(buffered, cap)
            if 0 < len(trimmed) < len(buffered):
                log.info("[POST-FILTER] capped short reply at %d sentences (dropped %d chars)",
                         cap, len(buffered) - len(trimmed))
                buffered = trimmed
            yield buffered
