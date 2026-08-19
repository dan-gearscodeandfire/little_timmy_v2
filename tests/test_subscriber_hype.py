"""Subscriber-hype detector: announcement phrasings trigger, everything else
stays silent. Hermetic — pure regex, no network, no DB.

Per feedback_live_test_classifiers_with_real_phrasing: the positive set is
seeded with Dan's five canonical phrasings VERBATIM, plus natural variants;
the negative set leads with his named false-positive ("we don't have new
subscribers").
"""

import pytest

from conversation import subscriber_hype


POSITIVES = [
    # Dan's five canonical triggers, verbatim
    "we have a new subscriber",
    "look, new subscriber",
    "another new subscriber",
    "somebody just subscribed",
    "hey hey new subscriber",
    # Natural variants (STT punctuation/casing, mid-sentence, plural, filler)
    "We have a new subscriber!",
    "Dude, we have a new subscriber.",
    "We've got a new subscriber!",
    "We just got a new subscriber.",
    "We got ourselves a new subscriber!",
    "We picked up another new subscriber.",
    "Look at that, a new subscriber!",
    "Looky here, new subscriber!",
    "Hey, hey, a new subscriber!",
    "Someone just subscribed.",
    "Somebody subscribed!",
    "New subscriber alert!",
    "We have a brand new subscriber!",
    "Holy crap, we have a new subscriber.",
]

NEGATIVES = [
    # Dan's named false-positive class
    "we don't have new subscribers",
    "We don't have a new subscriber.",
    "We do not have any new subscribers.",
    # Bare keyword — explicitly must NOT fire
    "new subscriber",
    "The new subscriber count is flat.",
    "I was reading about subscriber churn.",
    # Questions
    "Do we have a new subscriber?",
    "Did somebody just subscribe?",
    "Have we got a new subscriber",
    # Hypotheticals / wishes / negation variants
    "If we got a new subscriber I'd celebrate.",
    "I wish we had a new subscriber.",
    "I hope somebody just subscribed.",
    "We never get a new subscriber.",
    "We haven't got a new subscriber yet.",
    "Nobody subscribed today.",
    "We need a new subscriber.",
    # Adjacent vocabulary that must stay quiet
    "I unsubscribed from that channel.",
    "My subscription renewed.",
    "",
]


@pytest.mark.parametrize("text", POSITIVES)
def test_positive(text):
    assert subscriber_hype.detect(text), f"should trigger: {text!r}"


@pytest.mark.parametrize("text", NEGATIVES)
def test_negative(text):
    assert not subscriber_hype.detect(text), f"must NOT trigger: {text!r}"


def test_pick_line_no_immediate_repeat():
    lines = subscriber_hype._load_lines()
    assert len(lines) >= 2, "lines file missing or too short"
    prev = subscriber_hype.pick_line()
    for _ in range(50):
        cur = subscriber_hype.pick_line()
        assert cur != prev
        assert cur in lines
        prev = cur


def test_pick_line_covers_all_lines_eventually():
    lines = set(subscriber_hype._load_lines())
    seen = {subscriber_hype.pick_line() for _ in range(400)}
    assert seen == lines
