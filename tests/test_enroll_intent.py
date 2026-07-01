"""Tests for dual-modality enroll-intent detection (scope + canonical name)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation.enroll_intent import detect_enroll_intent  # noqa: E402


def test_face_scope_with_explicit_name():
    r = detect_enroll_intent("learn my face, my name is Dan")
    assert r.matched and r.scope == "face"
    assert r.name == "dan"          # canonical lowercase
    assert r.keyword_present


def test_voice_scope():
    r = detect_enroll_intent("please remember my voice, I'm Erin")
    assert r.matched and r.scope == "voice"
    assert r.name == "erin"


def test_both_scope_enroll_me():
    r = detect_enroll_intent("enroll me as Tomasz")
    assert r.matched and r.scope == "both"
    assert r.name == "tomasz"


def test_both_scope_remember_me_speaker_fallback():
    r = detect_enroll_intent("remember me", speaker_name="dan")
    assert r.matched and r.scope == "both"
    assert r.name == "dan"
    assert r.used_speaker_fallback


def test_face_and_voice_both_keywords_is_both():
    r = detect_enroll_intent("learn my face and my voice, my name is Nate")
    assert r.scope == "both"
    assert r.name == "nate"


def test_keyword_without_name_unknown_speaker_asks():
    # Phrase present, no name, voice unknown -> not "matched" but keyword_present
    # so the FSM knows to ask.
    r = detect_enroll_intent("enroll me", speaker_name="unknown_3")
    assert r.matched is False
    assert r.keyword_present is True
    assert r.scope == "both"


def test_no_enroll_phrase():
    r = detect_enroll_intent("what's the weather like")
    assert r.matched is False
    assert r.keyword_present is False


def test_enroll_as_pattern_face():
    r = detect_enroll_intent("enroll my face as Keith")
    assert r.matched and r.scope == "face" and r.name == "keith"


def test_non_name_rejected():
    # "save my face" with no name + unknown speaker -> keyword only.
    r = detect_enroll_intent("save my face", speaker_name="unknown_1")
    assert r.matched is False
    assert r.keyword_present and r.scope == "face"
