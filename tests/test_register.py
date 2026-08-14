"""Tests for the per-turn conversational register (2026-08-13).

Pure functions -- no DB, no LLM, no audio. The register exists to remove the
[answer][jab] shape that the fixed 2-sentence cap imposed on every reply; these
pin the classification boundaries and, critically, the sentence budget.
"""
import pytest
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


# ---------------------------------------------------------------------------
# Widened STRAIGHT (2026-08-13, Dan: "jabs are not necessary in every response")
# ---------------------------------------------------------------------------
# _FACTUAL_RE is ^-anchored on fronted wh-/aux- forms. Ordinary speech asks
# factual questions in shapes that anchor never sees, and BANTER is the
# fallthrough -- so those turns got the 2-sentence jab budget. Found by styling
# the acoustic test as a conversation instead of a question battery.

@pytest.mark.parametrize("text", [
    "Somebody walked off with one of your microphones at that party, right?",
    "You lost one of your microphones at that party, didn't you?",
    "That was Sierra who asked about the screws, wasn't it?",
    "So the Knicks won game four, right?",
    "Dan built you in the shop, correct?",
    "You have not forgotten the Dinobots argument, have you?",
    "It was a bucket of screws, yeah?",
])
def test_tag_questions_are_straight(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    "Tell me about the party.",
    "Remind me what Sierra ordered at Taco Bell.",
    "Catch me up on the Dinobots argument.",
    "Walk me through what happened at Open Sauce.",
    "Refresh my memory on the microphone.",
])
def test_imperative_recall_asks_are_straight(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    # The opinion veto must still beat all three answer-shapes -- same frame,
    # opposite request. "tell me a joke" is the collision that matters.
    "Tell me a joke.",
    "Tell me about your favorite album.",
    "Rate my outfit, would you?",
    "Would you rather be a toaster or a lamp?",
])
def test_opinion_veto_still_wins_over_the_widened_shapes(text):
    assert classify(text) == BANTER


@pytest.mark.parametrize("text", [
    # Ordinary conversation keeps its second sentence -- this is the half of
    # the persona the widening must NOT eat.
    "Hey Timmy. Been a while.",
    "It is freezing in this shop.",
    "Anyway, I brought coffee but I drank the whole thing on the way over.",
    "You are less annoying than the last time I was in here.",
    "Alright, I am out. Tell Dan I stopped by.",
    "Honestly I think Transformers peaked with the 1986 movie.",
])
def test_social_turns_stay_banter(text):
    assert classify(text) == BANTER


@pytest.mark.parametrize("text", [
    # A FRONTED question must not be parsed as <clause> + <tag>. "how are you?"
    # ends in "are you?" and was swallowed by the first draft of the tag rule,
    # collapsing a social greeting to a one-sentence answer. Caught by
    # tests/test_conversation_turn.py before it shipped.
    "how are you?",
    "How are you?",
    # "Are you serious?" was pinned here too, until Dan's 8-13 correction moved
    # it: see test_exasperation_is_a_correction_not_banter below. The tag-rule
    # property this case was really guarding (a fronted question must not parse
    # as <clause>+<tag>) is still covered by the remaining four.
    "Can you hear me?",
    "Is that so?",
    "Was it you?",
])
def test_fronted_questions_are_not_tag_questions(text):
    assert classify(text) == BANTER


@pytest.mark.parametrize("text", [
    # Dan's correction, 2026-08-13, on the first draft of this classifier, which
    # filed these under a rhetorical-question veto next to "tell me a joke":
    # "'Are you kidding me right now?' is not banter. It's a strong 'what's
    # wrong with you' indicator, in re: LT's codebase."
    #
    # That is the whole point -- from Dan to Timmy these are COMPLAINTS: Timmy
    # has just malfunctioned and Dan is naming it. BANTER grants the 2-sentence
    # budget, and the second sentence is a jab, which is the single worst reply
    # to a bug report. STRAIGHT's one-sentence budget removes the beat it lives
    # in. Live precedent, 22:33: "Why the hell did you just call me Nathan?"
    # classified BANTER and drew "I didn't call you Nathan, Dan. You're just
    # projecting your own confusion onto me." -- both sentences false.
    "Are you serious?",
    "Are you kidding me right now?",
    "What's wrong with you?",
    "Do you even hear yourself?",
    "What are you doing?",
    "You've got to be kidding.",
])
def test_exasperation_is_a_correction_not_banter(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    # Challenging what Timmy JUST said -- verifiable one turn back in his own
    # hot history, so a jab here denies the record rather than landing an edge.
    "Why the hell did you just call me Nathan?",
    "Why did you say that?",
    "What did you just call me?",
    "Did you just say Nathan?",
    "You just said the opposite.",
])
def test_own_turn_challenge_is_straight(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    # Substantive polar (yes/no) questions. _FACTUAL_RE is a wh-word list, so
    # every one of these fell to BANTER and bought the jab sentence. Live cost
    # 22:40: "Is Aliens Earth in your training data?" -> "Yes, Dan." -> "I don't
    # know it, Dan. Stop fishing for compliments about my knowledge base."
    "Is Alien Earth in your training data?",
    "Did you already move the servos?",
    "Have you ever met Erin?",
    "Can you actually see the workbench right now?",
])
def test_substantive_polar_questions_are_straight(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    # Decorum / content-safety corrections and clarification requests. Both
    # drew refusals on 8-13: "maybe you should be the one screening the chat"
    # and "I don't have the patience to repeat myself for the half-brained".
    "That was a little racy, dude.",
    "It sounded slightly sexualized and this needs to be an all ages thing.",
    "Keep it clean, there are kids watching.",
    "I said I don't completely follow.",
    "Say that again.",
])
def test_decorum_and_clarification_are_straight(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    # Imperative recall in the shapes people actually use. _RECALL_ASK_RE only
    # had "tell me about", so "tell me something about X" fell to BANTER.
    "Tell me something about the Voyager probe.",
    "Tell me more about that party.",
    "What do you know about the Voyager probe?",
])
def test_recall_asks_are_straight(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    # The question is the LAST sentence of a multi-sentence turn -- which is how
    # real STT delivers speech. All the question patterns are ^-anchored, so
    # before 2026-08-14 these graded BANTER and bought a jab/invention sentence.
    # Live cost 00:44: this exact turn produced "And yes, it reminds me of the
    # time she tried to fix the sink and flooded the kitchen" -- a story that
    # exists in no store. Dan: "Boo! Hallucination."
    "Timmy, Timmy, you legitimately just saw me trying to drill a hole in a "
    "pottery pot. Does that remind you of any stories with my wife, Erin?",
    "I was out in the shop all day. What time is it?",
    "So I finished the bracket. Do you remember what I called it?",
    "Long preamble that means nothing. Tell me something about the party.",
])
def test_question_in_the_last_sentence_is_straight(text):
    assert classify(text) == STRAIGHT


@pytest.mark.parametrize("text", [
    # De-anchoring must not drag ordinary multi-sentence chat into STRAIGHT.
    "I finished the bracket today. It came out clean.",
    "That was a long day. I am going to bed.",
    "Tell me a joke. Make it a good one.",
])
def test_de_anchoring_leaves_ordinary_talk_in_banter(text):
    assert classify(text) == BANTER
