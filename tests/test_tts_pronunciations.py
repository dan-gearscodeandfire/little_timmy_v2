"""Hermetic test for the TTS pronunciation-override substitution.

Booth fix (live 2026-07-16): Piper/espeak said "Erin" as "Karen". A whole-word,
case-insensitive respell map (config.TTS_PRONUNCIATIONS) rewrites the token
before synthesis. The load-bearing property is WHOLE-WORD matching: "erin" is a
substring of "gathering"/"engineering" and must NOT be touched inside them.

No Piper, no audio -- pure string logic.

Run: .venv/bin/pytest tests/test_tts_pronunciations.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.engine import _apply_pronunciations


def test_whole_word_name_is_respelled():
    assert _apply_pronunciations("Hi Erin") == "Hi Airin"


def test_case_insensitive_match():
    assert _apply_pronunciations("hi erin, welcome") == "hi Airin, welcome"


def test_substring_inside_other_words_untouched():
    # The exact trap from the 2026-07-16 handoff: "erin" hides in these.
    assert _apply_pronunciations("gathering") == "gathering"
    assert _apply_pronunciations("engineering the booth") == "engineering the booth"


def test_word_at_sentence_boundaries():
    assert _apply_pronunciations("Erin.") == "Airin."
    assert _apply_pronunciations("(Erin)") == "(Airin)"


def test_no_overrides_is_identity(monkeypatch):
    import config
    monkeypatch.setattr(config, "TTS_PRONUNCIATIONS", {})
    assert _apply_pronunciations("Hi Erin") == "Hi Erin"


def test_bad_config_fails_open(monkeypatch):
    import config
    monkeypatch.setattr(config, "TTS_PRONUNCIATIONS", None)
    assert _apply_pronunciations("Hi Erin") == "Hi Erin"
