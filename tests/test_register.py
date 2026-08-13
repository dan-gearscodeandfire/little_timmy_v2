"""Tests for the per-turn conversational register (2026-08-13).

Pure functions -- no DB, no LLM, no audio. The register exists to remove the
[answer][jab] shape that the fixed 2-sentence cap imposed on every reply; these
pin the classification boundaries and, critically, the sentence budget.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation.register import (
    classify, register_line, SENTENCE_CAP, STRAIGHT, WARM, BANTER,
)


def test_factual_question_is_straight_and_gets_one_sentence():
    """The load-bearing assertion. A one-sentence budget is what actually
    removes the jab beat -- instruction alone demonstrably did not."""
    for q in ["Timmy, what are my cats' names?",
              "who directed Interstellar?",
              "How many bones are in the human skeleton?",
              "where is my microphone?",
              "do you remember the bucket of screws?"]:
        assert classify(q) == STRAIGHT, q
    assert SENTENCE_CAP[STRAIGHT] == 1


def test_opinion_prompt_stays_banter_even_though_it_is_a_question():
    """Wit is the point when there is no right answer -- these must NOT be
    flattened to one dry sentence."""
    for q in ["what do you think of Frankenstein?",
              "What is your favorite color?",
              "tell me a poem about dogs",
              "would you rather be taller or louder?"]:
        assert classify(q) == BANTER, q


def test_repetition_complaint_is_a_correction():
    """Live acoustic 2026-08-13: "You said that already." classified BANTER, so
    the reply got a second sentence and spent it on "I have said nothing. You
    are misinterpreting my silence." -- wrong AND a jab, aimed at someone who
    was correcting him."""
    for q in ["You said that already.",
              "you keep saying that",
              "you're repeating yourself"]:
        assert classify(q) == STRAIGHT, q


def test_correction_is_straight():
    """A jab on top of a correction produced the two worst-received replies in
    the whole Open Sauce audit."""
    for q in ["No, you got it wrong.",
              "Timmy, you have my name wrong",
              "that's not my name",
              "stop saying that"]:
        assert classify(q) == STRAIGHT, q


def test_child_in_frame_is_warm_and_outranks_a_factual_question():
    r = classify("what is your favorite color?",
                 vision_description="a little girl is standing at the workbench")
    assert r == WARM
    # Protective register must win even over a correction.
    assert classify("no, that's wrong",
                    vision_description="a child watching") == WARM


def test_stranger_first_turn_is_warm_then_relaxes():
    assert classify("hello there", speaker_is_unknown=True, turns_with_speaker=0) == WARM
    assert classify("that is interesting", speaker_is_unknown=True,
                    turns_with_speaker=5) == BANTER


def test_known_speaker_banter_is_unaffected():
    assert classify("that is interesting.") == BANTER
    assert classify("you're creeping people out again") == BANTER


def test_register_line_present_for_each_and_none_for_unknown():
    for r in (STRAIGHT, WARM, BANTER):
        line = register_line(r)
        assert line and line.startswith("[REGISTER]")
    assert register_line(None) is None
    assert register_line("NOPE") is None


def test_every_register_has_a_cap():
    for r in (STRAIGHT, WARM, BANTER):
        assert SENTENCE_CAP[r] >= 1


# ---------------------------------------------------------------------------
# introductions latch (2026-08-13) — see conversation/introductions.py
# ---------------------------------------------------------------------------

def test_sentence_fragment_is_not_a_name():
    """Live 2026-08-13: STT dropped "What do you" from "What do you think of
    Dan?", leaving "Think of Dan." The bare fallback accepted any <=3-token
    reply, so "Think" was confirmed back as a name, latched the confirm dialog,
    and ate the following turn."""
    from conversation.enroll_intent import extract_reply_name as name
    assert name("Think of Dan.") is None
    assert name("Can you help me with something?") is None
    assert name("What do you think of Dan?") is None


def test_real_name_tells_still_parse():
    """The guard rejects CLAUSES, not unfamiliar strings — real names can be
    weird, and name particles must survive."""
    from conversation.enroll_intent import extract_reply_name as name
    assert name("Bob") == "bob"
    assert name("Mary Jane") == "mary_jane"
    assert name("My name is Tushar") == "tushar"
    assert name("It is Bob") == "bob"
    assert name("Ann de Vries") == "ann_de_vries"
    assert name("Th-th-Thomas") == "thomas"
    # Explicit frame proves name-position intent, so a particle-as-name works.
    assert name("my name is Van") == "van"


def test_question_during_confirm_is_not_an_unanswered_confirm():
    """A pending name-confirm must not swallow a real question. Live: the
    visitor asked "Can you help me with something?" and got a byte-identical
    re-ask of "Did you say Think?" instead of an answer."""
    from conversation.introductions import _looks_like_question as isq
    assert isq("Can you help me with something?")
    assert isq("What is the capital of France?")
    assert isq("Tell me a poem")
    # Answers and mumbles must still count as unanswered, keeping the
    # never-silent re-ask behaviour intact.
    assert not isq("yes")
    assert not isq("no")
    assert not isq("Bob")
    assert not isq("uh hang on")
