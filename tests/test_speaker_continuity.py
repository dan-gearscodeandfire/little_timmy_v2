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
    GREEN(new params: cap 0.40, window 15s)  -> abstains (returns False)
    PARTY regime                              -> disabled outright regardless of cap

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
    """With the shipped defaults (cap 0.40) the same stranger abstains."""
    assert continuity_allowed(**LIVE_STRANGER) is False
    # And explicitly with the new constant, to pin the value.
    assert SHORT_AUDIO_DIST_CAP == 0.40
    assert continuity_allowed(**LIVE_STRANGER, dist_cap=SHORT_AUDIO_DIST_CAP) is False


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


def test_long_audio_never_uses_continuity():
    """Long audio is not 'short' — continuity must not apply (confident path
    or unknown handles it instead)."""
    params = {**LIVE_STRANGER, "audio_len": SHORT_AUDIO_SAMPLES + 1, "best_known_dist": 0.30}
    assert continuity_allowed(**params) is False


def test_nearest_identity_must_match_last_speaker():
    """Continuity only fires when the stranger's nearest identity IS the last
    speaker — a stranger nearest 'devon' after Dan spoke must abstain."""
    params = {**LIVE_STRANGER, "best_known_name": "devon", "best_known_dist": 0.30}
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
