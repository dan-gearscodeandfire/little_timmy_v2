"""Tests for memory.decay — recency/usage re-rank for episodic semantic recall
(plan Session 5). Pure functions, fully hermetic (no DB, no embeddings).

Run: .venv/bin/pytest tests/test_decay.py -v
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import config
from memory.decay import recency_weight, access_boost, decay_multiplier, decayed_score

TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 6, 20, 12, 0, tzinfo=TZ)
DAY = 86400.0


def test_zero_age_is_full_weight():
    assert recency_weight(0.0, halflife_seconds=30 * DAY) == 1.0


def test_one_halflife_is_half():
    # floor=0.0 is the pre-2026-08-13 unbounded curve — still the shape under test.
    assert recency_weight(30 * DAY, halflife_seconds=30 * DAY, floor=0.0) == pytest.approx(0.5)


def test_two_halflives_is_quarter():
    assert recency_weight(60 * DAY, halflife_seconds=30 * DAY, floor=0.0) == pytest.approx(0.25)


# --- range bounding (2026-08-13) ---------------------------------------------
# Decay must MODIFY relevance, never out-swing it. See memory/decay.py for the
# measurement: unfloored decay spanned 4.2x against a 3x relevance range and
# evicted the correct answer from the injected top-5 on 12 of 14 real utterances.

def test_floor_compresses_range_without_changing_shape():
    hl = 30 * DAY
    for age in (0, 7 * DAY, 30 * DAY, 60 * DAY, 365 * DAY):
        raw = recency_weight(age, halflife_seconds=hl, floor=0.0)
        got = recency_weight(age, halflife_seconds=hl, floor=0.85)
        assert got == pytest.approx(0.85 + 0.15 * raw)


def test_floor_bounds_how_far_decay_can_reorder():
    # With floor f the multiplier lives in [f, 1.0], so decay can only overturn
    # a base-score gap smaller than 1/f. At 0.85 that is 17.6% — comfortably
    # inside the observed relevance spread, so real gaps survive.
    hl = 30 * DAY
    oldest = recency_weight(10 * 365 * DAY, halflife_seconds=hl, floor=0.85)
    newest = recency_weight(0.0, halflife_seconds=hl, floor=0.85)
    assert oldest >= 0.85
    assert newest / oldest < 1.18


def test_floor_zero_is_the_ab_control():
    hl = 30 * DAY
    for age in (0, 30 * DAY, 90 * DAY):
        assert recency_weight(age, halflife_seconds=hl, floor=0.0) == pytest.approx(0.5 ** (age / hl))


def test_a_stale_but_far_more_relevant_claim_now_survives():
    # The live regression, as a unit test. prop 1243 (the near-verbatim answer,
    # 60d old) vs prop 9 (a weak match, 3d old) — real measured base scores.
    hl = 30 * DAY
    stale_relevant = 0.0763 * recency_weight(60 * DAY, halflife_seconds=hl)
    fresh_weak     = 0.0613 * recency_weight(3 * DAY, halflife_seconds=hl)
    assert stale_relevant > fresh_weak, "the correct answer must outrank a fresher weaker one"


def test_access_boost_is_capped():
    # Self-reinforcing: the injected top-K is touched every turn. log1p saturates
    # but never stops growing, so it needs a hard ceiling.
    assert access_boost(10_000) <= config.EPISODE_ACCESS_BOOST_MAX
    assert access_boost(29) <= config.EPISODE_ACCESS_BOOST_MAX


def test_negative_age_clamps_to_one():
    # Future span_end / clock skew must not blow the weight up past 1.0.
    assert recency_weight(-5 * DAY, halflife_seconds=30 * DAY) == 1.0


def test_nonpositive_halflife_disables_decay():
    assert recency_weight(99 * DAY, halflife_seconds=0) == 1.0
    assert recency_weight(99 * DAY, halflife_seconds=-1) == 1.0


def test_recency_is_monotonic_decreasing():
    hl = 30 * DAY
    ages = [0, 1 * DAY, 7 * DAY, 30 * DAY, 90 * DAY, 365 * DAY]
    for floor in (0.0, 0.85):
        weights = [recency_weight(a, halflife_seconds=hl, floor=floor) for a in ages]
        assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1)), \
            f"fresh must still outrank stale at equal similarity (floor={floor})"


def test_access_boost_zero_is_neutral():
    assert access_boost(0) == 1.0
    assert access_boost(-3) == 1.0


def test_access_boost_grows_but_saturates():
    # Growth + concavity are asserted BELOW the cap; above it the curve is flat
    # by design (see test_access_boost_is_capped).
    b1, b3, b6 = access_boost(1), access_boost(3), access_boost(6)
    assert 1.0 < b1 < b3 < b6 <= config.EPISODE_ACCESS_BOOST_MAX
    # Concave: the marginal (per-unit) lift diminishes — the 1->2 step is
    # bigger than the 5->6 step.
    assert (access_boost(2) - access_boost(1)) > (access_boost(6) - access_boost(5))


def test_access_boost_respects_config_zero(monkeypatch):
    monkeypatch.setattr(config, "EPISODE_ACCESS_BOOST", 0.0)
    assert access_boost(50) == 1.0


def test_decay_multiplier_combines_recency_and_usage():
    hl = 30 * DAY
    span_end = NOW - timedelta(seconds=30 * DAY)  # one half-life old
    # One half-life at the default floor: 0.85 + 0.15*0.5 = 0.925, plus a small
    # (capped) usage lift for access_count>0.
    m_unused = decay_multiplier(span_end, NOW, access_count=0, halflife_seconds=hl)
    m_used = decay_multiplier(span_end, NOW, access_count=20, halflife_seconds=hl)
    expected = config.RECENCY_WEIGHT_FLOOR + (1 - config.RECENCY_WEIGHT_FLOOR) * 0.5
    assert m_unused == pytest.approx(expected)
    assert m_used > m_unused


def test_fresh_beats_stale_at_equal_similarity():
    hl = 30 * DAY
    fresh = decayed_score(0.8, NOW - timedelta(days=1), NOW, halflife_seconds=hl)
    stale = decayed_score(0.8, NOW - timedelta(days=120), NOW, halflife_seconds=hl)
    assert fresh > stale


def test_strong_stale_can_still_beat_weak_fresh():
    hl = 30 * DAY
    strong_old = decayed_score(0.95, NOW - timedelta(days=20), NOW, halflife_seconds=hl)
    weak_new = decayed_score(0.20, NOW - timedelta(days=0), NOW, halflife_seconds=hl)
    assert strong_old > weak_new
