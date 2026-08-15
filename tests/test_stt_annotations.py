"""Whisper non-speech annotations must never reach the turn as spoken words.

Live defect, 2026-08-15 01:06. Dan sang in the shop; whisper transcribed
`*singing* Sorry, I'm singing...` and nothing stripped the stage direction, so
it was consumed as if he had said the word "singing" in three places at once:

  1. the LLM prompt, as part of his utterance
  2. `prop_search`, as a retrieval query term
  3. `[QUERY-VCONF] low-confidence content word heard as '*singing*' (<0.55)`
     -> which arms the confirm-input path, so Timmy can ask Dan to clarify a
     "word" no human ever spoke.

(3) is why `words` is asserted on as hard as `text`: VCONF scores the per-word
probability list, so stripping the text alone leaves the bug fully armed.

An earlier version of this finding claimed whisper marks singing with `♪`. That
was incomplete -- this box emitted `*singing*`. The annotation vocabulary is
OPEN, hence the mixed cases below.
"""

import pytest

from stt.client import _strip_annotations


@pytest.mark.parametrize("raw,want_text,want_non_speech", [
    ("*singing* Sorry, I'm singing. It's hard to me.",
     "Sorry, I'm singing. It's hard to me.", True),
    ("♪ Kiss from a rose ♪ and then he stops", "and then he stops", True),
    ("[Music] Damn, my god bitches", "Damn, my god bitches", True),
    ("(upbeat music) hello there", "hello there", True),
    ("[BLANK_AUDIO]", "[BLANK_AUDIO]", True),
    # Ordinary speech must pass through byte-identical.
    ("Timmy, do you know the lyrics?", "Timmy, do you know the lyrics?", False),
    ("", "", False),
])
def test_strip_annotations(raw, want_text, want_non_speech):
    text, _words, non_speech = _strip_annotations(raw, [])
    assert text == want_text
    assert non_speech is want_non_speech


def test_whole_utterance_annotation_is_flagged_not_erased():
    """A turn that is ENTIRELY singing keeps its text and raises the flag.

    Erasing it would make a sung turn indistinguishable from silence, and the
    caller needs the turn in order to GATE it (no speaker attribution, no
    unknown_N mint, no stranger greeting, never solicit a name).
    """
    text, _words, non_speech = _strip_annotations("♪ What's that ♪", [])
    assert text == "♪ What's that ♪"
    assert non_speech is True


def test_annotation_removed_from_word_list():
    """The half that actually disarms the confirm-input bug."""
    words = [("*singing*", 0.31), ("Sorry,", 0.98), ("I'm", 0.97)]
    _text, clean, non_speech = _strip_annotations("*singing* Sorry, I'm", words)
    assert non_speech is True
    assert all(w != "*singing*" for w, _ in clean)
    assert [w for w, _ in clean] == ["Sorry,", "I'm"]


def test_asterisk_arithmetic_is_not_an_annotation():
    """Regression guard: `*` is also a multiplication sign and a censor mark.

    The annotation pattern requires a CLOSING asterisk with no whitespace-only
    content, so ordinary speech containing `*` survives untouched.
    """
    raw = "I said 5 * 3 is fifteen"
    text, _words, non_speech = _strip_annotations(raw, [])
    assert text == raw
    assert non_speech is False


@pytest.mark.parametrize("text,want", [
    ("♪ What's that ♪", True),
    ("*singing*", True),
    ("[Music]", True),
    ("♪ ♪", True),
    # A turn with real words is NOT annotation-only, even when it also sings.
    ("*singing* Sorry, I'm singing.", False),
    ("Timmy, do you know the lyrics?", False),
    ("", False),
])
def test_annotation_only(text, want):
    """Drives the SING-GATE in main.process_speech: annotation-only turns are
    dropped before reply and before the unknown_N observation is kept."""
    from stt.client import annotation_only
    assert annotation_only(text) is want
