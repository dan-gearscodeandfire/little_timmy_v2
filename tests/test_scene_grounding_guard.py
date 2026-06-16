"""Unit tests for the scene-grounding guard (2026-06-16).

In a mean/sarcastic mood, with [WHO IS PRESENT] listing ONLY Dan and the
face/vision/presence pipelines all reporting Dan alone, the persona announced
"...the guest who just walked in, who clearly has better taste in partners than
you do." — inventing a non-existent occupant for insult color (supervisor flag,
2026-06-16 01:01). One turn later it denied saying it; the denial is the mean
persona doing a bit (the reply WAS in hot history), but the INVENTION is the
real, fixable defect.

The guard is a tail-of-[CONTEXT] negative constraint: it forbids announcing
arrivals/occupants not in [WHO IS PRESENT]/[WHAT YOU SEE], WITHOUT asserting who
is present (sensors under-observe — a real unsensed person must never be denied).

Run:
    .venv/bin/pytest tests/test_scene_grounding_guard.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from llm.prompt_builder import build_ephemeral_block


def test_guard_fires_by_default():
    blk = build_ephemeral_block(memories=[], facts=[], speaker_name="Dan")
    assert "[SCENE GROUNDING]" in blk
    assert "never invent guests" in blk
    # Negative-constraint phrasing — bans invention, doesn't assert presence.
    assert "walked in" in blk


def test_guard_present_with_dan_only_room():
    """The exact failing scenario: only Dan present, guard still fires."""
    presence = {"present": [{"name": "dan", "last_seen_face_age_s": 1.0}]}
    blk = build_ephemeral_block(
        memories=[], facts=[], speaker_name="Dan", presence_state=presence
    )
    assert "[WHO IS PRESENT]" in blk
    assert "[SCENE GROUNDING]" in blk


def test_guard_sits_before_who_is_speaking():
    """Recency: the grounding rule must come AFTER presence but BEFORE the
    deliberately-last WHO IS SPEAKING addressee directive."""
    presence = {"present": [{"name": "dan", "last_seen_face_age_s": 1.0}]}
    blk = build_ephemeral_block(
        memories=[], facts=[], speaker_name="Dan", presence_state=presence
    )
    i_present = blk.index("[WHO IS PRESENT]")
    i_ground = blk.index("[SCENE GROUNDING]")
    i_speak = blk.index("[WHO IS SPEAKING]")
    assert i_present < i_ground < i_speak


def test_guard_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "SCENE_GROUNDING_GUARD", False)
    blk = build_ephemeral_block(memories=[], facts=[], speaker_name="Dan")
    assert "[SCENE GROUNDING]" not in blk


def test_guard_does_not_assert_who_is_present():
    """Must NOT claim nobody/somebody is present (under-observation safety):
    it only forbids inventing arrivals, never denies unsensed people."""
    blk = build_ephemeral_block(memories=[], facts=[], speaker_name="Dan")
    lowered = blk.lower()
    assert "no one is here" not in lowered
    assert "nobody is" not in lowered
    assert "you are alone" not in lowered
