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


def test_both_scope_remember_me_known_speaker_asks():
    # Speaker-name fallback REMOVED 2026-07-02 (live test E): a keyword with
    # no usable name must always route to the ask-name latch, even for a
    # voice-known speaker — the fallback silently committed garbled names
    # ("enroll me as here") under the current speaker's identity.
    r = detect_enroll_intent("remember me", speaker_name="dan")
    assert not r.matched
    assert r.keyword_present and r.scope == "both"


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


# ---- 2026-07-02 additions (live adversarial test findings E/F/G) ----

def test_multiword_name_captured_and_canonicalized():
    # Test F: "Mary Jane" used to truncate to "mary".
    r = detect_enroll_intent("remember my face as Mary Jane")
    assert r.matched and r.scope == "face"
    assert r.name == "mary_jane"


def test_multiword_name_trailing_filler_trimmed():
    r = detect_enroll_intent("enroll me as Billy Bob please")
    assert r.matched and r.name == "billy_bob"


def test_reject_word_no_speaker_fallback():
    # Test E: "here" is a non-name; a KNOWN speaker must get the ask, not a
    # silent commit under their own identity.
    r = detect_enroll_intent("enroll me as here", speaker_name="dan")
    assert not r.matched
    assert r.keyword_present and r.scope == "both"


def test_affirmation_negation_parsing():
    from conversation.enroll_intent import is_affirmation, is_negation
    assert is_affirmation("yes")
    assert is_affirmation("yeah, that's right")
    assert is_negation("no, that's not right")
    assert not is_affirmation("no, that's not right")  # negation wins ties
    assert not is_affirmation("what time is it")
    assert not is_negation("yes exactly")


# ---- 2026-07-02 code review C1/C5/C6/C7 additions ----

def test_confirm_verdict_negations_with_yes_words():
    # C1: hedged/negative replies that CONTAIN yes-words must never read as
    # yes — a false yes commits wrong biometrics, a false no merely re-asks.
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("I'm not sure") == "no"
    assert confirm_verdict("not exactly") == "no"
    assert confirm_verdict("that is not correct") == "no"
    assert confirm_verdict("no, that's not right") == "no"
    assert confirm_verdict("that isn't right") == "no"


def test_confirm_verdict_asides_are_unclear():
    # C1: yes-words buried in an unrelated aside must not commit.
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("sure is loud in here") == "unclear"
    assert confirm_verdict("turn right up there") == "unclear"
    assert confirm_verdict("what's the weather") == "unclear"
    assert confirm_verdict("hmm okay whatever fine then") == "unclear"


def test_confirm_verdict_real_confirmations():
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("Yes, that's right.") == "yes"
    assert confirm_verdict("yeah") == "yes"
    assert confirm_verdict("Sure.") == "yes"
    assert confirm_verdict("exactly right") == "yes"
    assert confirm_verdict("that's correct") == "yes"
    assert confirm_verdict("you got it") == "yes"
    assert confirm_verdict("spot on") == "yes"


def test_extract_reply_name_multiword_leadin():
    # C6: the shared extractor truncated "My name is Mary Jane" to 'mary';
    # the local multi-word patterns must run first.
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("My name is Mary Jane") == "mary_jane"
    assert extract_reply_name("Mary Jane.") == "mary_jane"
    assert extract_reply_name("call me Billy Bob") == "billy_bob"
    assert extract_reply_name("It's Bob") == "bob"
    assert extract_reply_name("Bob") == "bob"


def test_extract_reply_name_rejects_evasives():
    # C5: evasive replies must never canonicalize into names.
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("not telling") is None
    assert extract_reply_name("never mind") is None
    # Caught live 2026-07-02 (acoustic run 2, T4): "Forget about it." was
    # canonicalized to 'forget_about_it' and spoken back for confirm.
    assert extract_reply_name("Forget about it.") is None
    assert extract_reply_name("forget it") is None
    assert extract_reply_name("drop it") is None
    assert extract_reply_name("nothing") is None
    assert extract_reply_name("no name for you") is None
    assert extract_reply_name("I'd rather not say") is None
    assert extract_reply_name("") is None


def test_conjunction_stops_name_span():
    # C7: the 3-token span must not fuse two people into one identity.
    r = detect_enroll_intent("Enroll me, I'm Dan and Sarah is here too")
    assert r.matched and r.name == "dan"


def test_apostrophe_name_canonicalized():
    # O'Brien previously failed NAME_RE downstream; now canonicalizes.
    r = detect_enroll_intent("enroll me as O'Brien")
    assert r.matched and r.name == "obrien"
