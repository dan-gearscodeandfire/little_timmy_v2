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
    assert recency_weight(30 * DAY, halflife_seconds=30 * DAY) == pytest.approx(0.5)


def test_two_halflives_is_quarter():
    assert recency_weight(60 * DAY, halflife_seconds=30 * DAY) == pytest.approx(0.25)


def test_negative_age_clamps_to_one():
    # Future span_end / clock skew must not blow the weight up past 1.0.
    assert recency_weight(-5 * DAY, halflife_seconds=30 * DAY) == 1.0


def test_nonpositive_halflife_disables_decay():
    assert recency_weight(99 * DAY, halflife_seconds=0) == 1.0
    assert recency_weight(99 * DAY, halflife_seconds=-1) == 1.0


def test_recency_is_monotonic_decreasing():
    hl = 30 * DAY
    ages = [0, 1 * DAY, 7 * DAY, 30 * DAY, 90 * DAY, 365 * DAY]
    weights = [recency_weight(a, halflife_seconds=hl) for a in ages]
    assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))


def test_access_boost_zero_is_neutral():
    assert access_boost(0) == 1.0
    assert access_boost(-3) == 1.0


def test_access_boost_grows_but_saturates():
    b1, b10, b100 = access_boost(1), access_boost(10), access_boost(100)
    assert 1.0 < b1 < b10 < b100
    # Concave: the marginal (per-unit) lift diminishes — the 1->2 step is
    # bigger than the 10->11 step.
    assert (access_boost(2) - access_boost(1)) > (access_boost(11) - access_boost(10))


def test_access_boost_respects_config_zero(monkeypatch):
    monkeypatch.setattr(config, "EPISODE_ACCESS_BOOST", 0.0)
    assert access_boost(50) == 1.0


def test_decay_multiplier_combines_recency_and_usage():
    hl = 30 * DAY
    span_end = NOW - timedelta(seconds=30 * DAY)  # one half-life old
    # recency 0.5, plus a small usage lift for access_count>0.
    m_unused = decay_multiplier(span_end, NOW, access_count=0, halflife_seconds=hl)
    m_used = decay_multiplier(span_end, NOW, access_count=20, halflife_seconds=hl)
    assert m_unused == pytest.approx(0.5)
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
