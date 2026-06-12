"""Tests for the Slice A manual [SITUATION] regime knob in build_ephemeral_block.

Context (2026-06-12, party-prep): a human-set operating regime
(SOLO/GUEST/SMALL_GROUP/PARTY/EXPO) injects an NL [SITUATION] line at the head
of the ephemeral block, framing the WHO-IS-PRESENT / WHO-IS-SPEAKING lines
below it. At a party the prior flips from SOLO's "ambiguous is probably Dan" to
"assume strangers" — reinforcing the WHO-IS-SPEAKING fix. Ships dark: empty/None
emits nothing, so nothing changes until Dan sets a regime.

Run:
    .venv/bin/pytest tests/test_situation_regime.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from llm.prompt_builder import build_ephemeral_block, _SITUATION_TEXT


def _block(situation_regime):
    return build_ephemeral_block(
        memories=[],
        facts=[],
        speaker_name="dan",
        situation_regime=situation_regime,
    )


@pytest.mark.parametrize("regime", sorted(_SITUATION_TEXT.keys()))
def test_each_regime_emits_situation_line(regime):
    block = _block(regime)
    assert "[SITUATION]" in block
    # The regime's own NL text must be present verbatim.
    assert _SITUATION_TEXT[regime] in block


def test_party_assumes_strangers_never_dan():
    block = _block("PARTY")
    assert "[SITUATION]" in block
    assert "never default an unknown to" in block
    assert "stranger" in block.lower()


def test_solo_biases_toward_dan():
    block = _block("SOLO")
    assert "[SITUATION]" in block
    assert "alone with Dan" in block


def test_empty_regime_emits_nothing():
    assert "[SITUATION]" not in _block("")


def test_none_regime_emits_nothing():
    assert "[SITUATION]" not in _block(None)


def test_unknown_regime_fails_safe_to_nothing():
    # The API boundary rejects unknowns, but the builder must also fail safe.
    assert "[SITUATION]" not in _block("CHAOS")


def test_regime_is_case_insensitive():
    assert "[SITUATION]" in _block("party")
    assert _SITUATION_TEXT["PARTY"] in _block("party")


def test_situation_line_precedes_who_is_speaking():
    # The regime must frame (come before) the speaker directive so the prior is
    # established before the model reads who is talking.
    block = _block("PARTY")
    # speaker_name="dan" -> a [WHO IS SPEAKING] line is emitted.
    assert "[WHO IS SPEAKING]" in block
    assert block.index("[SITUATION]") < block.index("[WHO IS SPEAKING]")
