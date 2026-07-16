"""Hermetic tests for the whisper hallucination filter — especially the
reply-window relaxation (allow_short_replies, Dan 2026-07-15) and its
repetition backstop (review 7-15). Pure function, no network. Run:

    .venv/bin/pytest tests/test_stt_hallucination.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt.client import _is_likely_hallucination


# --- standing filter (outside a reply window) — behavior unchanged ----------

@pytest.mark.parametrize("text", ["yes", "okay", "hmm", "oh right"])
def test_standing_filter_eats_short_fillers(text):
    assert _is_likely_hallucination(text, 0.1) is True


def test_standing_filter_eats_repetition():
    assert _is_likely_hallucination("yes, yes, yes", 0.1) is True


def test_standing_filter_passes_real_sentences():
    assert _is_likely_hallucination("my name is tushar", 0.1) is False


def test_high_no_speech_prob_always_filters():
    # The acoustic gate applies in AND out of the reply window.
    assert _is_likely_hallucination("yes", 0.9) is True
    assert _is_likely_hallucination("yes", 0.9, allow_short_replies=True) is True


# --- reply window (allow_short_replies) --------------------------------------

def test_reply_window_admits_expected_one_worders():
    # The whole point of the window: "Yes." IS the expected answer.
    for text in ("yes", "no", "sure", "yep"):
        assert _is_likely_hallucination(text, 0.1,
                                        allow_short_replies=True) is False


def test_reply_window_admits_live_seen_double():
    # "Yes, yes," was produced by a REAL yes on the rig (7-15 acoustics) —
    # a 2x double must keep passing or the confirm dialog stalls again.
    assert _is_likely_hallucination("Yes, yes,", 0.1,
                                    allow_short_replies=True) is False


def test_reply_window_still_filters_triple_repetition():
    # Review 7-15: a >=3x single-word loop is classic whisper-on-noise, and
    # the reply window is exactly where a false "yes" would commit a name.
    assert _is_likely_hallucination("yes yes yes", 0.5,
                                    allow_short_replies=True) is True
    assert _is_likely_hallucination("no, no, no, no", 0.5,
                                    allow_short_replies=True) is True
