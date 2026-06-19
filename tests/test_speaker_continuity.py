"""Hermetic red->green test for the short-audio speaker-continuity false-accept.

Party-prep, 2026-06-12. Live test 2026-06-11 (Dan + YouTube voice clips)
characterized the scariest party vector: a guest who speaks a SHORT (<5s)
utterance within the continuity window, whose nearest enrolled identity happens
to be whoever just spoke, gets stamped as that person via the short-audio
continuity fallback in speaker/identifier.py.

Observed live: a stranger clip sat at 0.52 cosine distance from Dan on ~3s of
audio — inside the OLD 0.55 cap — while Dan self-matches at 0.13-0.21. The only
thing that saved it on the night was an expired 60s timer. At a party where Dan
talks constantly, that timer is always open.

The gate is now the pure function `continuity_allowed`, so this reproduces the
bug and proves the fix deterministically — no audio, no embeddings, no VAD.

    RED  (old params: cap 0.55, window 60s)  -> the 0.52/3s stranger is stamped Dan
    GREEN(WeSpeaker default cap 0.60, 15s)    -> a ~0.70 stranger abstains (False)
    PARTY regime                              -> disabled outright regardless of cap

    (The RED scenario's 0.52 distance is a Resemblyzer-era live observation, kept
    as a pure-logic fixture with an explicit 0.55 cap. The continuity_allowed()
    function is encoder-agnostic; the module CONSTANTS were recalibrated to
    WeSpeaker space 2026-06-17, where a real stranger sits at ~0.70 — see the
    GREEN test for the WeSpeaker-scale stranger.)

Run:
    .venv/bin/pytest tests/test_speaker_continuity.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speaker.identifier import (
    continuity_allowed,
    SHORT_AUDIO_DIST_CAP,
    CONTINUITY_WINDOW_SEC,
    CONTINUITY_MARGIN,
    SHORT_AUDIO_SAMPLES,
)

# The exact live scenario: a 3s (48k-sample) stranger clip whose nearest known
# identity is Dan at 0.52, 5s after Dan last spoke, normal (non-party) regime.
LIVE_STRANGER = dict(
    audio_len=49152,          # ~3.07s @ 16k — short
    best_known_name="dan",    # nearest enrolled identity
    last_known_name="dan",    # Dan was the last confident speaker
    best_known_dist=0.52,     # observed 0.5206 live
    elapsed_s=5.0,            # well inside any window
    regime="",
)

# Old (pre-fix) params, to demonstrate the bug existed.
OLD_CAP = 0.55
OLD_WINDOW = 60.0


def test_RED_old_params_false_accept_stranger_as_dan():
    """With the old 0.55 cap the live stranger WOULD have been stamped Dan."""
    assert continuity_allowed(**LIVE_STRANGER, dist_cap=OLD_CAP, window_s=OLD_WINDOW) is True


def test_GREEN_new_default_cap_abstains():
    """With the shipped WeSpeaker default cap (0.60) a real stranger abstains.

    On the WeSpeaker scale an unenrolled voice sits at ~0.70+ from the nearest
    enrolled print (calibrated 2026-06-17: genuine on-mic Dan 0.295-0.405,
    impostors >= 0.704), so the 0.60 cap excludes it while genuine Dan continues.
    """
    assert SHORT_AUDIO_DIST_CAP == 0.60
    ws_stranger = {**LIVE_STRANGER, "best_known_dist": 0.70}   # WeSpeaker-scale stranger
    assert continuity_allowed(**ws_stranger) is False
    assert continuity_allowed(**ws_stranger, dist_cap=SHORT_AUDIO_DIST_CAP) is False


def test_party_regime_disables_continuity_even_with_loose_cap():
    """PARTY/EXPO disable continuity outright — belt-and-suspenders vs the cap."""
    for regime in ("PARTY", "EXPO", "party", " party "):
        params = {**LIVE_STRANGER, "regime": regime}
        assert continuity_allowed(**params, dist_cap=OLD_CAP, window_s=OLD_WINDOW) is False


def test_window_tightened_guest_after_15s_abstains():
    """A guest 20s after Dan is no longer 'Dan continuing' under the 15s window."""
    params = {**LIVE_STRANGER, "best_known_dist": 0.30, "elapsed_s": 20.0}
    # Old 60s window would have allowed it; new 15s default rejects.
    assert continuity_allowed(**params, window_s=OLD_WINDOW) is True
    assert continuity_allowed(**params) is False
    assert CONTINUITY_WINDOW_SEC == 15.0


def test_dan_real_short_utterance_still_accepted():
    """The fix must not false-REJECT Dan: his own short utterances sit well
    under 0.40 (live: 0.13-0.21), so genuine continuity still applies."""
    dan_short = dict(
        audio_len=40000, best_known_name="dan", last_known_name="dan",
        best_known_dist=0.21, elapsed_s=4.0, regime="",
    )
    assert continuity_allowed(**dan_short) is True


# --- margin-aware continuity (anti-latch), added 2026-06-15 ------------------

def test_latch_rejected_when_runnerup_is_close():
    """The live Archer->sky latch: a kid whose nearest known IS the last speaker
    (sky, within the cap) but who sits nearly as close to ANOTHER known identity
    must NOT be continued — the small margin means we can't be sure it's sky
    continuing rather than a different similar voice."""
    params = dict(
        audio_len=40000, best_known_name="sky", last_known_name="sky",
        best_known_dist=0.33, elapsed_s=4.0, regime="",
    )
    # Without runner-up info (default inf) the old gate would have latched:
    assert continuity_allowed(**params) is True
    # Ambiguous: another known (e.g. erin) at 0.36 -> margin 0.03 < 0.10 -> abstain.
    assert continuity_allowed(**params, second_best_known_dist=0.36) is False


def test_continuity_held_when_last_speaker_is_decisively_nearest():
    """Genuine 1-on-1 continuity is preserved: the last speaker is clearly the
    nearest and no other known identity is close, so the margin is wide."""
    params = dict(
        audio_len=40000, best_known_name="dan", last_known_name="dan",
        best_known_dist=0.21, elapsed_s=4.0, regime="",
    )
    assert continuity_allowed(**params, second_best_known_dist=0.55) is True


def test_continuity_margin_boundary():
    params = dict(
        audio_len=40000, best_known_name="dan", last_known_name="dan",
        best_known_dist=0.20, elapsed_s=4.0, regime="",
    )
    assert CONTINUITY_MARGIN == 0.12   # WeSpeaker-rescaled 2026-06-17 (was 0.10)
    # Comfortably above the margin -> allowed; clearly below -> rejected.
    # (Avoid testing the exact 0.12 edge: float cancellation makes a 0.32 runner-up
    # evaluate to 0.1199… — real distances never land on the precise boundary.)
    assert continuity_allowed(**params, second_best_known_dist=0.35) is True
    assert continuity_allowed(**params, second_best_known_dist=0.27) is False


def test_long_audio_never_uses_continuity():
    """Long audio is not 'short' — continuity must not apply (confident path
    or unknown handles it instead)."""
    params = {**LIVE_STRANGER, "audio_len": SHORT_AUDIO_SAMPLES + 1, "best_known_dist": 0.30}
    assert continuity_allowed(**params) is False


def test_nearest_identity_must_match_last_speaker():
    """Continuity only fires when the stranger's nearest identity IS the last
    speaker — a stranger nearest 'robin' after Dan spoke must abstain."""
    params = {**LIVE_STRANGER, "best_known_name": "robin", "best_known_dist": 0.30}
    assert continuity_allowed(**params) is False


def test_none_names_abstain():
    assert continuity_allowed(
        audio_len=40000, best_known_name=None, last_known_name="dan",
        best_known_dist=0.2, elapsed_s=1.0,
    ) is False
    assert continuity_allowed(
        audio_len=40000, best_known_name="dan", last_known_name=None,
        best_known_dist=0.2, elapsed_s=1.0,
    ) is False


def test_timmy_never_continued():
    """Timmy's own loopback must never be stamped via continuity."""
    params = {**LIVE_STRANGER, "best_known_name": "timmy", "last_known_name": "timmy",
              "best_known_dist": 0.2}
    assert continuity_allowed(**params) is False
