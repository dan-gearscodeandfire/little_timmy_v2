"""Hermetic tests for the shared F1 anchored-crop bind-eligibility rule
(main._may_bind_anchored, review 7-15 — was duplicated at the co-sample feed
and the enroll live-grab; now one predicate) and the fail-CLOSED contract of
the enrolled-face check it leans on. Run:

    .venv/bin/pytest tests/test_bind_anchored_guard.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from presence import face_identifier


def _stub_identifier(monkeypatch, has_face: bool):
    """get_shared_identifier -> a store whose path_for(name).exists() is fixed."""
    store = SimpleNamespace(
        path_for=lambda name: SimpleNamespace(exists=lambda: has_face))
    monkeypatch.setattr(face_identifier, "get_shared_identifier",
                        lambda: SimpleNamespace(_store=store))


# --- recognized anchored face: must match the voice --------------------------

def test_recognized_face_binds_only_on_match():
    assert main._may_bind_anchored("bob", "bob") is True
    assert main._may_bind_anchored("bob", "alice") is False


# --- unrecognized anchored face ----------------------------------------------

def test_unrecognized_binds_for_fresh_visitor():
    assert main._may_bind_anchored(None, "unknown_3") is True


def test_unrecognized_binds_for_known_but_faceless(monkeypatch):
    # Voice-only-promotion bootstrap (Tushar catch-22, 7-15): known voice,
    # no face prototype yet -> the mic-holder's face may bind.
    _stub_identifier(monkeypatch, has_face=False)
    assert main._may_bind_anchored(None, "tushar") is True


def test_unrecognized_skips_known_with_enrolled_face(monkeypatch):
    # F1: known speaker already HAS a face -> an unrecognized mic-holder face
    # means they're off-mic; never bind someone else's face to them.
    _stub_identifier(monkeypatch, has_face=True)
    assert main._may_bind_anchored(None, "dan") is False


# --- review 7-15: the check FAILS CLOSED ---------------------------------------

def test_enrolled_face_check_fails_closed(monkeypatch):
    """An error in the face-store lookup must read as 'has a face' (skip the
    bind), NOT 'face-less' (allow it) — failing open would let an off-mic
    bystander's face bind to a known identity, the exact F1 misID."""
    def _boom():
        raise RuntimeError("store unavailable")
    monkeypatch.setattr(face_identifier, "get_shared_identifier", _boom)
    assert main._speaker_has_enrolled_face("dan") is True
    assert main._may_bind_anchored(None, "dan") is False
