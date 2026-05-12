"""Unit tests for filtered_assistant_stream — the post-filter that wraps
the Llama 3B conversation-tier stream before TTS / WS / hot_turns see it.

Pure-logic tests, no LT services. Run:
    .venv/bin/pytest tests/test_reply_filter.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (
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
