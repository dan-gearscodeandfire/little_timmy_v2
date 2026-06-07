"""Regression: a direct visual question must be grounded in the fresh raw frame.

Live finding 2026-06-07: Dan held up a teal thermos and asked "what am I
holding?". Two bugs stacked:

  1. The turn answered from a cached frame that predated the gesture (the
     background speech-onset capture raced the turn and lost), so the brain saw
     "empty hands". Fixed by block-on-fresh (main.py awaits trigger_capture when
     the cached frame is older than VISION_VISUAL_Q_MAX_AGE_S).

  2. Even with a fresh frame, get_description() relevance-filtered the scene to
     None (a teal bottle in a cluttered workshop scores low novelty), so the
     prompt carried no [WHAT YOU SEE] block and the brain confabulated ("it's a
     wrench"). Fixed by get_raw_description(), which bypasses the relevance gate
     for explicitly-asked visual questions.

This pins bug #2: the relevance filter must NOT suppress object detail when the
user directly asks. (Block-on-fresh / scene_age gating is covered by the live
test + the scene_age() unit below.)
"""
import time

from vision.context import VisionContext
from vision.analyzer import SceneRecord
from vision.relevance import RelevanceResult


def _ctx_with_low_novelty_bottle():
    """A fresh scene holding a teal bottle that the relevance filter would skip."""
    ctx = VisionContext()
    ctx._enabled = True
    ctx._current = SceneRecord(
        people=["man with glasses"],
        actions=["holding up a bottle"],
        objects=["teal bottle", "workbench", "tools"],
        scene_state="cluttered workshop",
    )
    ctx._last_update = time.monotonic()
    # Relevance says "not worth volunteering" -- the exact condition that made
    # get_description() return None and starved the prompt.
    ctx._last_relevance = RelevanceResult(should_inject=False, filtered_summary="")
    return ctx


def test_relevance_filter_suppresses_unsolicited():
    """Baseline: get_description() returns None for the low-novelty scene."""
    ctx = _ctx_with_low_novelty_bottle()
    assert ctx.get_description() is None


def test_raw_description_bypasses_relevance_for_visual_question():
    """get_raw_description() must surface the object detail regardless of novelty."""
    ctx = _ctx_with_low_novelty_bottle()
    raw = ctx.get_raw_description()
    assert raw is not None
    assert "teal bottle" in raw
    assert "holding up a bottle" in raw


def test_scene_age_tracks_last_update():
    """scene_age() drives the block-on-fresh gate; None before any frame."""
    ctx = VisionContext()
    assert ctx.scene_age() is None  # disabled / no frame
    ctx._enabled = True
    ctx._current = SceneRecord(objects=["thermos"])
    ctx._last_update = time.monotonic() - 5.0
    age = ctx.scene_age()
    assert age is not None and 4.5 < age < 6.0


def test_raw_description_honors_staleness_cutoff():
    """A frame older than the 60s hard cutoff yields no description at all."""
    ctx = _ctx_with_low_novelty_bottle()
    ctx._last_update = time.monotonic() - 120.0
    assert ctx.get_raw_description() is None
