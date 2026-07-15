"""Regression: the averted-gaze background recapture must not contend or waste.

Live finding 2026-07-15 (supervisor gripe 01:07, "two visual LLM calls in a
row"): the guard's _delayed_recapture fired 0.6s after deflecting -- mid-turn,
because FrameCapture.trigger() bypasses the poll pause. The VLM ran
concurrently with the reply's conversation-tier generation and both halved
(turn e2e 6792ms vs p50 2636ms; VLM 3.8s vs 2.0-2.4s). The frame it grabbed
was also still empty (0.6s < look-at pan time) and its product was never used
(the next visual question's block-on-fresh re-captured anyway).

Fix: _delayed_recapture now (a) waits for the turn to release the poll pause
(cap VISION_RECAPTURE_MAX_WAIT_S, then skip), and (b) gates on a cheap /faces
detection check (detection-not-ID, proximity-gate idiom), failing open when
the Pi is unreachable. This pins the two VisionContext helpers the gate reads.
"""
import asyncio

from vision.context import VisionContext


def test_is_polling_paused_tracks_pause_counting():
    """Composes with overlapping pause windows, same as FrameCapture."""
    ctx = VisionContext()
    assert ctx.is_polling_paused is False
    ctx.pause_polling()
    ctx.pause_polling()
    assert ctx.is_polling_paused is True
    ctx.resume_polling()
    assert ctx.is_polling_paused is True  # second holder still active
    ctx.resume_polling()
    assert ctx.is_polling_paused is False


class _FakeFaceRemote:
    def __init__(self, full):
        self._full = full

    async def fetch_full(self):
        return self._full


def _visible(ctx):
    return asyncio.run(ctx.face_currently_visible())


def test_face_visible_none_when_remote_not_ready():
    """Remote face state never came up -> None (caller fails open)."""
    ctx = VisionContext()
    ctx._face_remote_ready = False
    assert _visible(ctx) is None


def test_face_visible_none_when_pi_unreachable():
    """fetch_full() connection failure -> None (caller fails open)."""
    ctx = VisionContext()
    ctx._face_remote_ready = True
    ctx._face_remote = _FakeFaceRemote(None)
    assert _visible(ctx) is None


def test_face_visible_false_on_empty_faces():
    """Reachable but no (fresh) face -> False: the recapture must skip."""
    ctx = VisionContext()
    ctx._face_remote_ready = True
    ctx._face_remote = _FakeFaceRemote(
        {"faces": [], "image_size": (640, 480), "age_s": 0.2})
    assert _visible(ctx) is False


def test_face_visible_true_on_fresh_face():
    ctx = VisionContext()
    ctx._face_remote_ready = True
    ctx._face_remote = _FakeFaceRemote(
        {"faces": [{"name": "dan", "bbox": [10, 10, 80, 120]}],
         "image_size": (640, 480), "age_s": 0.2})
    assert _visible(ctx) is True
