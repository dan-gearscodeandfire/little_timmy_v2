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


# ---- 2026-07-05 re-ask fix (never-silent latches, Dan's spec) ----

def test_confirm_verdict_paraphrased_yes():
    # Live-proven failure 2026-07-02: "You did get that right" read as
    # unclear -> silent latch drop -> the LLM confabulated the commit. STT
    # filters one-word turns, so paraphrases are the DEFAULT confirm reply.
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("You did get that right") == "yes"
    assert confirm_verdict("you got that right") == "yes"
    assert confirm_verdict("you got it right") == "yes"
    assert confirm_verdict("you heard me right") == "yes"
    assert confirm_verdict("that's my name") == "yes"
    assert confirm_verdict("that is my name") == "yes"
    assert confirm_verdict("nailed it") == "yes"
    assert confirm_verdict("yes that is correct") == "yes"   # level-3 script


def test_confirm_verdict_paraphrased_yes_negated_stays_no():
    # _NO_RE runs first: widened yes-anchors must not flip negated forms.
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("you did not get that right") == "no"
    assert confirm_verdict("you didn't get it right") == "no"
    assert confirm_verdict("that is not my name") == "no"
    assert confirm_verdict("no that is wrong") == "no"       # level-3 script


def test_confirm_verdict_name_babble_stays_unclear():
    # Live repro 2: "Joe, Joey, Jojo, Shabbadu" — neither yes nor no; the
    # caller must RE-ASK (escalation), never commit or silently drop.
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("Joe, Joey, Jojo, Shabbadu") == "unclear"
    assert confirm_verdict("you got me confused with someone") == "unclear"


def test_enroll_cancel_detection():
    from conversation.enroll_intent import is_enroll_cancel
    assert is_enroll_cancel("never mind")
    assert is_enroll_cancel("forget about it")
    assert is_enroll_cancel("cancel that")
    assert is_enroll_cancel("stop asking")
    assert is_enroll_cancel("no thanks")
    assert not is_enroll_cancel("yes that is right")
    assert not is_enroll_cancel("my name is Dan")
    assert not is_enroll_cancel("")


def test_reask_scripts_escalate():
    from conversation.enroll_intent import confirm_reask_line, name_reask_line
    first = confirm_reask_line("Joe", 1)
    later = confirm_reask_line("Joe", 2)
    assert first == "Yes or no — is the name Joe?"
    assert "YES THAT IS CORRECT" in later and "Joe" in later
    assert confirm_reask_line("Joe", 7) == later    # blunt level is terminal
    assert "my name is" in name_reask_line(1)
    assert "MY NAME IS" in name_reask_line(2)


def test_full_name_inference_with_connective():
    # Dan 2026-07-05: infer the FULL stated name, not the first token.
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("My name is Dan the Barbarian") == \
        "dan_the_barbarian"
    r = detect_enroll_intent("enroll me as Dan the Barbarian")
    assert r.matched and r.name == "dan_the_barbarian"
    # Trailing connective is trimmed, leading connective still breaks.
    assert extract_reply_name("I'm Dan the") == "dan"
    assert extract_reply_name("the door") is None


def test_conjunction_still_stops_span():
    # C7 must survive the connective change: "and" is a hard break.
    r = detect_enroll_intent("Enroll me, I'm Dan and Sarah is here too")
    assert r.matched and r.name == "dan"


# ---- 2026-07-05 code-review fixes (false-yes / false-cancel / junk names) ----

def test_confirm_verdict_questions_are_not_confirms():
    # Review 7-05: optional-tail anchors read questions as yes — the
    # wrong-biometrics commit C1 exists to prevent. Declarative forms only.
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("where did you get that name") == "unclear"
    assert confirm_verdict("can you get it right this time") == "unclear"
    assert confirm_verdict("do you have it saved") == "unclear"
    assert confirm_verdict("did you get that right") == "unclear"
    assert confirm_verdict("you got me confused with someone") == "unclear"


def test_confirm_verdict_positive_idioms_not_no():
    # "no worries/no problem" carry a surface negation cue but are positive.
    from conversation.enroll_intent import confirm_verdict
    assert confirm_verdict("no worries, you got it right") == "yes"
    assert confirm_verdict("no problem, that's correct") == "yes"
    # ...but a real negation elsewhere still wins.
    assert confirm_verdict("no worries, but that is wrong") == "no"


def test_affirmative_with_cancel_fragment_commits():
    # "yes, leave it as it is" is a confirm; "leave it" must not abort it.
    from conversation.enroll_intent import confirm_verdict, is_enroll_cancel
    assert confirm_verdict("yeah just leave it as Dan") == "yes"
    assert not is_enroll_cancel("yeah just leave it as Dan")
    assert not is_enroll_cancel("yes, leave it as it is")


def test_bare_verb_replies_not_names():
    # "leave it" -> 'leave' and "call me buddy" -> 'call'/'buddy' regressions.
    # (NOT covered: "this is weird" -> 'weird' via the pre-existing
    # this-is pattern — indistinguishable from "this is Bob" without
    # semantics; the confirm turn is the guard for that class.)
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("leave it") is None
    assert extract_reply_name("call me buddy") is None
    assert extract_reply_name("I'm buddy") is None      # vocative frame stays weak


def test_incidental_it_is_clause_not_a_name():
    # "it is X" is a bare-reply lead-in only, never a full-utterance pattern.
    r = detect_enroll_intent("remember my face when it is dark in here")
    assert not r.matched
    assert r.keyword_present and r.scope == "face"


def test_name_turn_miss_shapes_2026_07_02():
    # The four live miss shapes get sane parses instead of garbage.
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("It is Bob") == "bob"            # was 'it'
    assert extract_reply_name("My name's Mary Jane") == "mary_jane"  # was 'mary'
    assert extract_reply_name("hang on") is None               # was 'hang_on'
    assert extract_reply_name("Buddy") is None                 # bare filler
    # ...but a real Buddy can enroll via the coached explicit phrasing.
    assert extract_reply_name("My name is Buddy") == "buddy"


# ---- 2026-07-06 identity-correction protest (misID pushback) ----

def test_correction_denial_plus_claim():
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "No, my name is not Walter, my name is Flynn.", "walter", True)
    assert r.matched and r.name == "flynn" and r.denied == "walter"


def test_correction_denial_plus_claim_unpunctuated_stt():
    # STT drops commas; the lazy denial span must not swallow the claim.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "no my name is not walter my name is flynn", "walter", True)
    assert r.matched and r.name == "flynn" and r.denied == "walter"


def test_correction_stop_calling_me_with_weak_claim():
    # "I'm Y" is honored as the claim only AFTER an explicit denial.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "Stop calling me Walter, I'm Flynn.", "walter", True)
    assert r.matched and r.name == "flynn" and r.denied == "walter"


def test_correction_bare_claim_contradicting_enrolled():
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction("My name is Flynn.", "dan", True)
    assert r.matched and r.name == "flynn" and r.denied == "dan"


def test_correction_bare_claim_matching_attribution_inert():
    from conversation.enroll_intent import detect_identity_correction
    assert not detect_identity_correction("My name is Dan.", "dan", True).matched


def test_correction_bare_claim_unknown_speaker_inert():
    # Unknown speakers are introductions' turf, not a correction.
    from conversation.enroll_intent import detect_identity_correction
    assert not detect_identity_correction(
        "My name is Flynn.", "unknown_3", False).matched


def test_correction_denial_only_routes_to_name_ask():
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction("That's not my name.", "dan", True)
    assert r.matched and r.name is None and r.denied == "dan"
    r2 = detect_identity_correction("You've got my name wrong.", "dan", True)
    assert r2.matched and r2.name is None


def test_correction_im_not_gated_to_attribution():
    from conversation.enroll_intent import detect_identity_correction
    # Live 7-06: idioms must stay inert.
    assert not detect_identity_correction(
        "I'm not patting myself on the back", "dan", True).matched
    assert not detect_identity_correction("I'm not sure", "dan", True).matched
    assert not detect_identity_correction(
        "I am not hungry today", "dan", True).matched
    # ...but denying the ATTRIBUTED name counts.
    r = detect_identity_correction("I'm not Walter.", "walter", True)
    assert r.matched and r.name is None and r.denied == "walter"


def test_correction_weak_claim_without_denial_inert():
    # A casual "I'm Flynn" (no denial) must not hijack the turn.
    from conversation.enroll_intent import detect_identity_correction
    assert not detect_identity_correction("I'm Flynn", "dan", True).matched


def test_correction_full_name_claim():
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "No, my name is not Dan, my name is Dan the Barbarian.", "dan", True)
    assert r.matched and r.name == "dan_the_barbarian" and r.denied == "dan"


def test_correction_plain_talk_inert():
    from conversation.enroll_intent import detect_identity_correction
    for utt in ("Today I played Legend of Zelda.",
                "Can you get it right this time?",
                "Never mind.",
                "What's the weather like?"):
        assert not detect_identity_correction(utt, "dan", True).matched, utt


def test_correction_bare_denial_does_not_unlock_weak_claim():
    # Live 7-06 false positive: "So I'm supposed to just say to you, that's
    # not my name." — the bare denial unlocked the weak "I'm Y" frame, which
    # swallowed "supposed to" as the claim ("My mistake. Supposed To — did I
    # get that right?"). Bare denials route to the name-ask latch instead.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "So I'm supposed to just say to you, that's not my name.",
        "dan", True)
    assert r.matched and r.name is None and r.denied == "dan"


def test_correction_bare_denial_plus_strong_claim_still_extracts():
    # A STRONG frame after a bare denial is still an in-turn claim. The
    # bare denial names nobody, so denied stays None at the detector level
    # (the caller falls back to the current attribution).
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "That's not my name, my name is Flynn.", "dan", True)
    assert r.matched and r.name == "flynn" and r.denied is None


def test_correction_bare_denial_weak_claim_goes_to_name_ask():
    # Cost of the gate: "That's not my name, I'm Flynn" takes the name-ask
    # path (one extra never-silent turn) instead of extracting flynn.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "That's not my name, I'm Flynn.", "dan", True)
    assert r.matched and r.name is None and r.denied == "dan"
