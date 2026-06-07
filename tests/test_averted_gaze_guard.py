"""Unit tests for the averted-gaze guard (C6, 2026-06-07).

A prompted visual question about the user ("what's on my shoulder?") used to be
answered confidently from the cached scene even when Timmy's head had looked
away -- so the frame didn't contain the subject and the answer was a confident
hallucination (supervisor flag, 2026-06-06 23:44).

Two pieces are pinned here:

1. Detection. The real failing utterance was NOT caught by is_visual_question
   at all (it took the background-awareness vision branch), yet
   is_self_referential_visual_question DOES catch it. main.py ORs the two, so
   self-referential questions are treated as visual questions AND drive the
   guard. Scene questions ("what do you see") stay non-self-referential so the
   guard never fires on them (consistent with the narration-veto rule that a
   direct "describe the room" is correct).

2. Prompt shaping. With vision_subject_absent=True, build_ephemeral_block must
   emit an honest "not looking at you" deflection and SUPPRESS the (wrong)
   cached scene text. With it False, the normal descriptive instruction stands.

Run:
    .venv/bin/pytest tests/test_averted_gaze_guard.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.visual_question import (
    is_visual_question,
    is_self_referential_visual_question,
)
from llm.prompt_builder import build_ephemeral_block


def _visual_q(text: str) -> bool:
    """Mirror main.py: a self-referential question counts as a visual one."""
    return is_visual_question(text) or is_self_referential_visual_question(text)


# (utterance, expected combined visual_q, expected self_referential)
CASES = [
    # The exact supervisor-flagged utterance. is_visual_question misses it; the
    # self-ref detector is what saves it. This is the regression that matters.
    ("I am holding something on my left shoulder, a small character. "
     "Can you tell what it is?", True, True),
    ("what is on my shoulder", True, True),
    ("what am I wearing", True, True),
    ("how do I look", True, True),
    ("can you see me", True, True),
    # Live C6 test (2026-06-07): the phrasing Dan actually used. "the X I'm
    # holding" has no "my"/"am i" and the what..i span is >20 chars, so it
    # slipped the self-ref detector and Timmy confabulated "it's black" off a
    # stale frame. The "i'm <presenting-verb>" pattern is what now catches it.
    ("what color is the thermos I'm holding here", True, True),
    ("what color is this thing I'm holding", True, True),
    ("what's this I’m showing you", True, True),  # curly apostrophe (STT)
    # Scene questions: visual, but NOT self-referential -> guard must not fire.
    ("what do you see", True, False),
    ("describe the room", True, False),
    ("who is here", True, False),
    ("what color is that", True, False),
    # Not a visual question at all.
    ("what time is it", False, False),
]


def test_detection_table():
    for text, exp_visual, exp_selfref in CASES:
        assert _visual_q(text) is exp_visual, f"visual_q wrong for {text!r}"
        assert is_self_referential_visual_question(text) is exp_selfref, (
            f"self_ref wrong for {text!r}")


def test_real_c6_utterance_drives_guard():
    """The flagged utterance must be both a visual question and self-referential
    so the averted-gaze guard can fire on it."""
    u = ("I am holding something on my left shoulder, a small character. "
         "Can you tell what it is?")
    assert _visual_q(u)
    assert is_self_referential_visual_question(u)


def test_deflection_suppresses_wrong_scene():
    blk = build_ephemeral_block(
        memories=[], facts=[], speaker_name="Dan",
        vision_description="A desk with a monitor and a Darth Vader figure.",
        visual_question=True, vision_subject_absent=True,
    )
    assert "NOT currently looking" in blk          # honest deflection injected
    assert "Darth Vader" not in blk                # wrong cached scene suppressed
    assert "Be specific and descriptive" not in blk


def test_in_view_still_descriptive():
    blk = build_ephemeral_block(
        memories=[], facts=[], speaker_name="Dan",
        vision_description="Dan wearing a blue shirt.",
        visual_question=True, vision_subject_absent=False,
    )
    assert "Be specific and descriptive" in blk
    assert "blue shirt" in blk
    assert "NOT currently looking" not in blk


def test_non_visual_background_branch_intact():
    blk = build_ephemeral_block(
        memories=[], facts=[], speaker_name="Dan",
        vision_description="A desk and a window.",
        visual_question=False, vision_subject_absent=False,
    )
    assert "background awareness" in blk
    assert "NOT currently looking" not in blk
