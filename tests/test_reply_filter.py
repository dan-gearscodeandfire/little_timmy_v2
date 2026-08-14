"""Unit tests for filtered_assistant_stream — the post-filter that wraps
the Llama 3B conversation-tier stream before TTS / WS / hot_turns see it.

Pure-logic tests, no LT services. Run:
    .venv/bin/pytest tests/test_reply_filter.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation.reply_filter import (
    banned_phrase_used,
    filtered_assistant_stream,
    user_invites_longer_reply,
    _REPLY_VETO_FALLBACK,
)


async def _agen(tokens):
    for t in tokens:
        yield t


async def _collect(token_iter):
    out = []
    async for t in token_iter:
        out.append(t)
    return out


@pytest.mark.asyncio
async def test_clean_short_reply_passes_through():
    tokens = ["Sure.", " ", "Got it."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert "".join(out) == "Sure. Got it."


@pytest.mark.asyncio
async def test_narration_prefix_single_token_blocks_full_reply():
    # First veto path: a single >30-char narration token triggers the check
    # inside the buffering branch.
    tokens = ["I'm standing in front of a wall of monitors. And one of them..."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert out == [_REPLY_VETO_FALLBACK]
    joined = "".join(out)
    assert "standing" not in joined
    assert "monitors" not in joined


@pytest.mark.asyncio
async def test_narration_prefix_split_across_many_tokens_blocks_all():
    # Regression: the pre-fix loop yielded each token immediately, so the
    # first ~29 chars of narration leaked to TTS / WS / hot_turns before the
    # veto fired. The buffered version must hold every token until the
    # check has resolved.
    tokens = ["I", "'m ", "standing ", "in ", "front ", "of ", "a ", "wall ",
              "of ", "monitors."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert out == [_REPLY_VETO_FALLBACK]
    joined = "".join(out)
    assert "standing" not in joined


@pytest.mark.asyncio
async def test_short_narration_under_check_window_vetoed_on_eos():
    # "the room is" is 11 chars — well under the 30-char check threshold.
    # End-of-stream flush must still run the narration check, otherwise a
    # reply that is exactly the prefix would slip through unvetoed.
    tokens = ["the room is"]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert out == [_REPLY_VETO_FALLBACK]


@pytest.mark.asyncio
async def test_short_safe_reply_under_window_flushes_cleanly_on_eos():
    # Safe reply, under window, must flush on EOS — not get swallowed by the
    # buffering branch.
    tokens = ["Hello", " ", "there."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert "".join(out) == "Hello there."


@pytest.mark.asyncio
async def test_two_sentence_cap_drops_third_and_later():
    tokens = ["First sentence. ", "Second sentence. ", "Third sentence. ",
              "Fourth."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    joined = "".join(out)
    assert "First sentence" in joined
    assert "Second sentence" in joined
    assert "Third sentence" not in joined
    assert "Fourth" not in joined


@pytest.mark.asyncio
async def test_two_sentence_cap_keeps_second_terminator():
    # The second sentence's terminator must be yielded — drain triggers
    # after the yield, not before.
    tokens = ["A.", " ", "B."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert "".join(out) == "A. B."


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing():
    tokens = []
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert out == []


@pytest.mark.asyncio
async def test_single_short_token_passes():
    tokens = ["Yeah."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert "".join(out) == "Yeah."


@pytest.mark.asyncio
async def test_check_window_exactly_30_chars_safe_flushes():
    text = "x" * 30
    tokens = [text]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert "".join(out) == text


@pytest.mark.asyncio
async def test_mid_sentence_narration_phrase_does_not_match():
    # Narration check is startswith on the lowercased lstripped first 50
    # chars. "the room is" appearing mid-reply must not trigger the veto.
    tokens = ["Yeah, in fact the room is bigger than I thought."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    joined = "".join(out)
    assert "Yeah" in joined
    assert "the room is" in joined


@pytest.mark.asyncio
async def test_narration_with_leading_whitespace_still_blocked():
    # _looks_like_narration lstrips before matching, so leading spaces
    # should not save a narration reply.
    tokens = ["   ", "the workshop is dim and full of screens..."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert out == [_REPLY_VETO_FALLBACK]


@pytest.mark.asyncio
async def test_post_flush_two_sentence_cap_in_buffered_text():
    # If the prefix-window flush itself already contains two terminators,
    # drain must fire and the rest of the stream must be dropped.
    tokens = ["Hi there friend. Done already. ", "And more.", " And more."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    joined = "".join(out)
    assert "Hi there friend" in joined
    assert "Done already" in joined
    assert "And more" not in joined


# ---------------------------------------------------------------------------
# max_sentences override (Supervisor M5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_sentences_override_lets_more_sentences_through():
    tokens = ["First. ", "Second. ", "Third. ", "Fourth. ", "Fifth.", " Sixth."]
    out = await _collect(filtered_assistant_stream(_agen(tokens), max_sentences=5))
    joined = "".join(out)
    assert "First" in joined and "Second" in joined and "Third" in joined
    assert "Fourth" in joined and "Fifth" in joined
    # Sixth would push past the 5-sentence cap; should be dropped.
    assert "Sixth" not in joined


@pytest.mark.asyncio
async def test_max_sentences_none_uses_default_two():
    tokens = ["One. ", "Two. ", "Three."]
    out = await _collect(filtered_assistant_stream(_agen(tokens), max_sentences=None))
    joined = "".join(out)
    assert "One" in joined and "Two" in joined
    assert "Three" not in joined


@pytest.mark.asyncio
async def test_max_sentences_invalid_falls_back_to_default():
    """Zero / negative caps fall back to the default 2."""
    tokens = ["A. ", "B. ", "C."]
    out = await _collect(filtered_assistant_stream(_agen(tokens), max_sentences=0))
    joined = "".join(out)
    assert "A" in joined and "B" in joined
    assert "C" not in joined


# ---------------------------------------------------------------------------
# user_invites_longer_reply detector
# ---------------------------------------------------------------------------


def test_user_invites_longer_reply_positive_cases():
    assert user_invites_longer_reply("you can speak longer than usual")
    assert user_invites_longer_reply("Tell me more about your life")
    assert user_invites_longer_reply("Go into detail please")
    assert user_invites_longer_reply("This is open-ended, no rush")
    assert user_invites_longer_reply("Give me a long answer")
    assert user_invites_longer_reply("Tell me your story")


def test_user_invites_longer_reply_negative_cases():
    assert not user_invites_longer_reply("How are you")
    assert not user_invites_longer_reply("what time is it")
    assert not user_invites_longer_reply("")
    assert not user_invites_longer_reply("describe the room")  # narration test, not length permission


# ---------------------------------------------------------------------------
# echo-as-reply guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_echo_reply_verbatim_is_suppressed():
    # The 2026-06-13 18:09 defect: reply == the user's STT, verbatim.
    user = "He just tracked, tracked, so."
    tokens = ["He just ", "tracked, ", "tracked, so."]
    out = await _collect(filtered_assistant_stream(_agen(tokens), user_text=user))
    assert out == []  # suppressed entirely — nothing reaches TTS


@pytest.mark.asyncio
async def test_echo_guard_ignores_casing_and_punctuation():
    user = "what is the torque spec on this bolt"
    tokens = ["What is the torque spec on this bolt?"]
    out = await _collect(filtered_assistant_stream(_agen(tokens), user_text=user))
    assert out == []


@pytest.mark.asyncio
async def test_genuine_reply_passes_through_with_user_text():
    # A real reply diverges immediately → released, no suppression, no loss.
    user = "He just tracked, tracked, so."
    tokens = ["Yeah, ", "I locked onto ", "you for a second."]
    out = await _collect(filtered_assistant_stream(_agen(tokens), user_text=user))
    assert "".join(out) == "Yeah, I locked onto you for a second."


@pytest.mark.asyncio
async def test_reply_that_starts_like_user_then_continues_is_kept():
    # Echo PREFIX but then continues — a real (if odd) reply, not a pure echo.
    user = "are you a robot"
    tokens = ["Are you a robot? ", "No, I'm a skeleton."]
    out = await _collect(filtered_assistant_stream(_agen(tokens), user_text=user))
    assert "skeleton" in "".join(out)


@pytest.mark.asyncio
async def test_trivial_user_turn_not_guarded():
    # A one/two-word turn under _ECHO_MIN_WORDS: "Yes." echoing "yes" is fine.
    user = "yes"
    tokens = ["Yes."]
    out = await _collect(filtered_assistant_stream(_agen(tokens), user_text=user))
    assert "".join(out) == "Yes."


@pytest.mark.asyncio
async def test_no_user_text_is_pure_passthrough():
    tokens = ["Sure. ", "Got it."]
    out = await _collect(filtered_assistant_stream(_agen(tokens)))
    assert "".join(out) == "Sure. Got it."


# ---------------------------------------------------------------------------
# sentence-boundary detection (2026-08-13)
# ---------------------------------------------------------------------------

def test_ellipsis_is_not_a_sentence_boundary():
    """Live acoustic 2026-08-13: the model wrote "Well, thank you. That's...
    unexpected." and every "." counted, so a 2-sentence cap chopped it to
    "Well, thank you. That's." and Timmy SPOKE that. The register made it bite
    harder -- a STRAIGHT turn caps at 1, so any early ellipsis truncates the
    whole answer."""
    from conversation.reply_filter import _trim_at_nth_terminator as trim
    assert trim("Well, thank you. That's... unexpected. And rare.", 2) == \
        "Well, thank you. That's... unexpected."
    assert trim("Well... Paris is the capital. Obviously.", 1) == \
        "Well... Paris is the capital."


def test_decimal_point_is_not_a_sentence_boundary():
    from conversation.reply_filter import _trim_at_nth_terminator as trim
    assert trim("It cost 5.50 today. Then more.", 1) == "It cost 5.50 today."


def test_title_abbreviation_is_not_a_sentence_boundary():
    from conversation.reply_filter import _trim_at_nth_terminator as trim
    assert trim("Ask Dr. Smith about it. He knows.", 1) == "Ask Dr. Smith about it."


def test_single_letter_sentence_still_terminates():
    """The abbreviation guard must not swallow a legitimate one-character
    sentence -- "p.m" is protected because a LETTER follows the period."""
    from conversation.reply_filter import _trim_at_nth_terminator as trim
    assert trim("A. B. C.", 2) == "A. B."


# ---------------------------------------------------------------------------
# Gate/trim predicate parity (2026-08-13)
# ---------------------------------------------------------------------------
# The cap GATE counted raw "." / "!" / "?" characters while the TRIM counted
# only _is_real_terminator. On any disagreement the gate fired, the trim
# returned the buffer unchanged, and `drained` discarded the rest of the reply,
# so the SPOKEN output ended mid-word at the 30-char narration window. The
# journal tell was "dropped 0 chars". Found by listening, not by this suite.

async def _stream(text, chunk=7):
    for i in range(0, len(text), chunk):
        yield text[i:i + chunk]


async def _run(text, cap, chunk=7):
    out = ""
    async for tok in filtered_assistant_stream(_stream(text, chunk), max_sentences=cap):
        out += tok
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("text,cap,want", [
    # "No." is the word no, not No.=number -- the commonest opener of a
    # STRAIGHT answer. The cap must FIRE on it, cleanly.
    ("No. He is currently bragging to anyone who will listen about it.", 1, "No."),
    ("No. And I would not tell you if he had, obviously.", 1, "No."),
    ("Yes. He is currently bragging to anyone who will listen.", 1, "Yes."),
    # An ellipsis is not a sentence end, so this is TWO sentences, not four.
    ("Well, thank you. That's... genuinely unexpected coming from you.", 2,
     "Well, thank you. That's... genuinely unexpected coming from you."),
    # A decimal point is not a sentence end.
    ("It costs 5.50 dollars, which is robbery for a bag of screws.", 1,
     "It costs 5.50 dollars, which is robbery for a bag of screws."),
    # Real multi-sentence replies still get capped.
    ("First one. Second one. Third one.", 2, "First one. Second one."),
    ("First one. Second one. Third one.", 1, "First one."),
])
async def test_cap_never_truncates_mid_word(text, cap, want):
    assert await _run(text, cap) == want


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk", [1, 3, 7, 13, 400])
async def test_cap_is_independent_of_token_boundaries(chunk):
    # A terminator's meaning depends on its neighbours, so a token examined in
    # isolation cannot classify its own last character -- "..." and "5.50" can
    # straddle a chunk boundary. The result must not depend on how the stream
    # happens to be split.
    text = "Well, thank you. That's... genuinely unexpected. And 5.50 is robbery."
    assert await _run(text, 2, chunk=chunk) == "Well, thank you. That's... genuinely unexpected."


@pytest.mark.asyncio
async def test_no_as_number_still_reads_as_an_abbreviation():
    # The reason "no" was in the abbreviation list in the first place.
    assert await _run("Bay No. 4 is where he keeps it. Second sentence here.", 1) == \
        "Bay No. 4 is where he keeps it."


# ---------------------------------------------------------------------------
# banned_phrase_used -- the retired "I am not little" bit
# ---------------------------------------------------------------------------


class TestBannedPhrase:
    """Live evidence, 2026-08-14: the bit ran twice in six minutes (00:29:11,
    00:35:14) while config.PERSONA had banned it since 6-11. Dan: "I removed it
    but you still complain about it and that is fascinating to me." Two causes,
    both now fixed -- the ban quoted the phrase, and it sat in system[0]."""

    def test_fires_on_a_single_use(self):
        # Not a repetition tic: once is already too many, so there is no
        # threshold to clear the way repeated_opener() needs one.
        assert banned_phrase_used(
            ["And for the record, I am not little."]) == "I am not little"

    def test_matches_in_the_closing_clause(self):
        # This is why repeated_opener() could never see it: _opener_words()
        # cuts at the first sentence end, and the bit is a trailing tag.
        assert banned_phrase_used(
            ["I do not speak to the therapist, and for the record, "
             "I am not little."]) == "I am not little"

    def test_returns_original_casing_for_quoting_back(self):
        assert banned_phrase_used(["I AM NOT LITTLE."]) == "I AM NOT LITTLE"

    def test_catches_contractions(self):
        assert banned_phrase_used(["I'm not little, Dan."]) == "I'm not little"

    def test_silent_on_clean_replies(self):
        assert banned_phrase_used(
            ["I heard you the first time.", "It is 10:04 PM, Dan."]) is None

    def test_ignores_the_word_little_used_normally(self):
        # The children rule and ordinary speech both use "little" constantly.
        assert banned_phrase_used(
            ["There is a little girl in the doorway.",
             "Give me a little more detail."]) is None

    def test_only_looks_at_the_recent_window(self):
        old = ["I am not little."] + ["clean reply"] * 5
        assert banned_phrase_used(old) is None

    def test_handles_empty_and_none(self):
        assert banned_phrase_used([]) is None
        assert banned_phrase_used(None) is None
        assert banned_phrase_used([None, ""]) is None
