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
    # Escalation coaches a spelled-out name (Dan 2026-07-15): a letter run
    # beats an STT mishear — see extract_spelled_name.
    assert "spell" in name_reask_line(2)


def test_extract_spelled_name():
    # Dan 2026-07-15: "Tushar" was mis-heard and the spelling was ignored.
    from conversation.enroll_intent import (extract_spelled_name,
                                            extract_reply_name)
    assert extract_spelled_name("T-U-S-H-A-R") == "tushar"
    assert extract_spelled_name("o t i s") == "otis"
    assert extract_spelled_name("B, O, B") == "bob"
    assert extract_spelled_name("My name is Otis, O-T-I-S.") == "otis"
    # <3 letters or no run: not a spelling
    assert extract_spelled_name("B-O") is None
    assert extract_spelled_name("I am a person") is None
    assert extract_spelled_name("") is None
    # The spelled form WINS over the (possibly mis-heard) word form.
    assert extract_reply_name("Odis, O-T-I-S") == "otis"
    assert extract_reply_name("My name is Odis. O-T-I-S.") == "otis"


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


# ---- 2026-07-06 code-review fixes (post-live-test) ----

def test_correction_weak_denial_unpunctuated_stt():
    # Review 7-06: greedy span swallowed the claim — "i'm not walter i'm
    # flynn" (STT drops commas) captured 'walter i'm flynn' != spk and the
    # flagship protest was silently dropped. Weak frame now matches the
    # attribution directly.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction("i'm not walter i'm flynn", "walter", True)
    assert r.matched and r.name == "flynn" and r.denied == "walter"
    # Punctuated variant keeps working.
    r2 = detect_identity_correction("I'm not Walter, I'm Flynn", "walter", True)
    assert r2.matched and r2.name == "flynn" and r2.denied == "walter"


def test_correction_weak_denial_multiword_attribution():
    # Direct-attribution matching handles multi-word canonical names a
    # captured span (greedy OR lazy) could not.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "I'm not Dan the Barbarian, I'm Flynn", "dan_the_barbarian", True)
    assert r.matched and r.name == "flynn" and r.denied == "dan_the_barbarian"


def test_correction_names_not_contraction():
    # Review 7-06: "name's not" was missing from the strong denial lexicon
    # and the first-match-only claim walk never reached "my name is flynn".
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction(
        "My name's not Walter, my name is Flynn", "walter", True)
    assert r.matched and r.name == "flynn" and r.denied == "walter"
    r2 = detect_identity_correction(
        "no my names not walter my name is flynn", "walter", True)
    assert r2.matched and r2.name == "flynn" and r2.denied == "walter"


def test_correction_non_name_predicates_inert():
    # Review 7-06: predicates in name position hijacked the FSM as claims
    # ('hard_to_pronounce', 'on_the_whiteboard_behind', denied='important').
    from conversation.enroll_intent import detect_identity_correction
    for utt in ("my name is hard to pronounce",
                "my name is on the whiteboard behind you",
                "my name is not important, just turn the lights on",
                "my name is kind of long",
                "my name is spelled with two n's"):
        r = detect_identity_correction(utt, "dan", True)
        assert not r.matched, (utt, r)


def test_correction_real_claims_survive_stoplist():
    # The stoplist must trim trailing filler without killing the name.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction("My name is Flynn actually", "dan", True)
    assert r.matched and r.name == "flynn"


# --- auto-suffixed forks (expo duplicate names, 2026-07-16) ------------------

def test_correction_claiming_display_base_is_noop():
    # mike_2's owner is SPOKEN to as "Mike"; claiming it corrects nothing.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction("My name is Mike", "mike_2", True,
                                   speaker_display_base="mike")
    assert not r.matched


def test_correction_display_base_still_fires_on_contradiction():
    # ...but a DIFFERENT claim from the fork's owner is a real protest.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction("My name is Flynn", "mike_2", True,
                                   speaker_display_base="mike")
    assert r.matched and r.name == "flynn" and r.denied == "mike_2"


def test_correction_weak_denial_matches_display_base():
    # "I'm not Mike" while attributed mike_2 — the spoken form is the display
    # base; the canonical "mike 2" can never occur in speech.
    from conversation.enroll_intent import detect_identity_correction
    r = detect_identity_correction("I'm not Mike, I'm Flynn", "mike_2", True,
                                   speaker_display_base="mike")
    assert r.matched and r.name == "flynn" and r.denied == "mike_2"


# --- cross-category latch speaker gate (option C, Dan 2026-07-16) ----------

from conversation.enroll_intent import latch_speaker_ok  # noqa: E402


def test_latch_gate_known_armed_accepts_only_owner():
    assert latch_speaker_ok("dan", "dan")
    assert not latch_speaker_ok("dan", "erin")


def test_latch_gate_known_armed_rejects_unknown():
    # THE live failure 7-16 01:33: dan-keyed correction latch must NOT
    # consume unknown_11's turn.
    assert not latch_speaker_ok("dan", "unknown_11")


def test_latch_gate_unknown_armed_tolerates_cluster_drift():
    # Same human drifted unknown_10 -> unknown_11 mid-dialog (7-16); the
    # dialog only completed because unknown-armed accepts any unknown_*.
    assert latch_speaker_ok("unknown_10", "unknown_10")
    assert latch_speaker_ok("unknown_10", "unknown_11")


def test_latch_gate_unknown_armed_rejects_known():
    # Option C's second direction: Dan's coaching words near the mic must
    # not resolve a visitor's confirm.
    assert not latch_speaker_ok("unknown_10", "dan")


def test_latch_gate_legacy_latch_without_key_is_ungated():
    assert latch_speaker_ok(None, "dan")
    assert latch_speaker_ok("", "unknown_3")


def test_latch_gate_empty_speaker_rejected_by_keyed_latch():
    # An unattributed turn is never graded as anyone's answer.
    assert not latch_speaker_ok("dan", "")
    assert not latch_speaker_ok("unknown_10", "")


def test_confused_interrogative_replies_not_names():
    # Live 7-16 19:12: "Wha-wha-what?" -> Timmy: "Did you say Wha_Wha_What?"
    # Booth visitors answer the confirm ask with confusion constantly; a
    # confused reply must count as an unanswered attempt (rig f0b re-ask),
    # never become the candidate.
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("Wha-wha-what?") is None
    assert extract_reply_name("wha wha what") is None      # no '?' -- collapse+stoplist
    assert extract_reply_name("What?") is None
    assert extract_reply_name("What") is None
    assert extract_reply_name("Who?") is None
    assert extract_reply_name("Huh?") is None
    assert extract_reply_name("Pardon?") is None
    assert extract_reply_name("Say again?") is None
    assert extract_reply_name("Say that again") is None
    assert extract_reply_name("wait what") is None
    assert extract_reply_name("What's that?") is None
    assert extract_reply_name("Come again?") is None


def test_question_mark_gates_bare_reply_only():
    # A '?'-terminated bare reply is a question, not a name-tell; an
    # explicit frame already proved name-position intent so a trailing
    # STT '?' must not reject it.
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("Bob?") is None
    assert extract_reply_name("My name is Bob?") == "bob"
    assert extract_reply_name("T-U-S-H-A-R?") == "tushar"  # spelled run still wins


def test_stutter_collapse_rescues_stuttered_names():
    # The collapse must reject STRUCTURAL junk while rescuing a nervous
    # real name -- bias against repetition, not unfamiliar strings.
    from conversation.enroll_intent import extract_reply_name
    assert extract_reply_name("Th-th-Thomas") == "thomas"
    assert extract_reply_name("B-b-bob.") == "bob"
    assert extract_reply_name("no no no") is None
    # A short real first name that prefixes the surname must NOT collapse.
    assert extract_reply_name("Dan Daniels") == "dan_daniels"
    assert extract_reply_name("Tushar") == "tushar"        # weird != junk
