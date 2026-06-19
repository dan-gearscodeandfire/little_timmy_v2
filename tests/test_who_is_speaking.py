"""Regression tests for the per-turn [WHO IS SPEAKING] addressee steering in
build_ephemeral_block.

Context (2026-06-11): the voice matcher correctly tags turns ([Sky], [Dan],
[Unknown_15]) but Timmy's *replies* kept calling non-Dan / unrecognized speakers
"Dan" — the Dan-anchored persona steamrolls the "[Name]:" history prefix, the
only speaker signal the model otherwise gets. Party-critical: every OpenSauce
guest would be misnamed Dan. Fix: emit an explicit current-speaker directive
every turn, and for strangers forbid the Dan default outright.

These tests pin that directive for each speaker class.

Run:
    .venv/bin/pytest tests/test_who_is_speaking.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.prompt_builder import build_ephemeral_block


def _block(speaker_name, *, fusion_source=None, face_hint_name=None):
    return build_ephemeral_block(
        memories=[],
        facts=[],
        speaker_name=speaker_name,
        fusion_source=fusion_source,
        face_hint_name=face_hint_name,
    )


def test_known_non_dan_speaker_is_named_and_dan_default_forbidden():
    block = _block("sky")
    assert "[WHO IS SPEAKING]" in block
    assert "speaking with Sky" in block
    # The crux: it must explicitly steer away from Dan.
    assert "NOT Dan" in block


def test_unknown_speaker_is_never_called_dan():
    block = _block("unknown_15")
    assert "[WHO IS SPEAKING]" in block
    assert "do NOT assume this is Dan" in block.replace("Do NOT", "do NOT")
    assert '"Dan"' in block
    # Must not leak the temp id as a name to address.
    assert "Unknown_15" not in block


def test_dan_is_addressed_normally_without_negation():
    block = _block("dan")
    assert "[WHO IS SPEAKING]" in block
    assert "speaking with Dan" in block
    # No "NOT Dan" negation when Dan really is speaking.
    assert "NOT Dan" not in block


def test_face_hint_keeps_working_hypothesis_framing():
    block = _block("unknown_3", fusion_source="face_hint", face_hint_name="erin")
    assert "[WHO IS SPEAKING]" in block
    assert "working hypothesis" in block
    assert "Erin" in block
    # Even the face-hint branch should not silently default to Dan.
    assert "Dan" in block  # the "Do NOT default to calling them Dan" guard


def test_timmy_speaker_emits_no_directive():
    # Timmy's own narration loopback should not produce a speaker directive.
    block = _block("timmy")
    assert "[WHO IS SPEAKING]" not in block


def test_none_speaker_emits_no_directive():
    block = _block(None)
    assert "[WHO IS SPEAKING]" not in block
