"""Tests for episodic temporal recall (plan Session 2).

Two layers:
  - resolve_date_range(): pure phrase -> [start, end) resolution, fixed `now`.
  - query_episodes_by_range(): span-overlap query over seeded episodes (hits
    local Postgres, NOT the :8083 LLM slot; self-cleans).

Run:
    .venv/bin/pytest tests/test_temporal_resolver.py -v
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from memory.temporal import resolve_date_range

# Fixed reference instant: Wednesday 2026-06-17 14:30 local (-04:00 EDT).
TZ = timezone(timedelta(hours=-4))
NOW = datetime(2026, 6, 17, 14, 30, 0, tzinfo=TZ)


def _d(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=TZ)


# --------------------------------------------------------------------------
# resolve_date_range — exact days
# --------------------------------------------------------------------------

def test_today():
    assert resolve_date_range("what did we talk about today", NOW) == (
        _d(2026, 6, 17), _d(2026, 6, 18))


def test_yesterday():
    assert resolve_date_range("yesterday", NOW) == (_d(2026, 6, 16), _d(2026, 6, 17))


def test_day_before_yesterday():
    assert resolve_date_range("the day before yesterday", NOW) == (
        _d(2026, 6, 15), _d(2026, 6, 16))


def test_earlier_today_runs_to_now():
    assert resolve_date_range("earlier today", NOW) == (_d(2026, 6, 17), NOW)


# --------------------------------------------------------------------------
# day parts
# --------------------------------------------------------------------------

def test_this_morning():
    assert resolve_date_range("this morning", NOW) == (
        _d(2026, 6, 17, 5), _d(2026, 6, 17, 12))


def test_this_afternoon():
    assert resolve_date_range("this afternoon", NOW) == (
        _d(2026, 6, 17, 12), _d(2026, 6, 17, 17))


def test_yesterday_evening():
    assert resolve_date_range("yesterday evening", NOW) == (
        _d(2026, 6, 16, 17), _d(2026, 6, 17, 0))


def test_last_night():
    assert resolve_date_range("last night", NOW) == (
        _d(2026, 6, 16, 18), _d(2026, 6, 17, 6))


# --------------------------------------------------------------------------
# weekdays — NOW is a Wednesday (weekday()==2)
# --------------------------------------------------------------------------

def test_last_saturday():
    # Most recent past Saturday before Wed 6-17 is 6-13.
    assert resolve_date_range("what about last Saturday", NOW) == (
        _d(2026, 6, 13), _d(2026, 6, 14))


def test_bare_monday_is_this_week():
    # Monday of the current week = 6-15.
    assert resolve_date_range("Monday", NOW) == (_d(2026, 6, 15), _d(2026, 6, 16))


def test_last_weekday_same_as_today_goes_back_a_week():
    # "last Wednesday" on a Wednesday => previous Wednesday 6-10.
    assert resolve_date_range("last Wednesday", NOW) == (
        _d(2026, 6, 10), _d(2026, 6, 11))


def test_bare_weekday_today_is_today():
    # bare "Wednesday" on a Wednesday => today.
    assert resolve_date_range("Wednesday", NOW) == (_d(2026, 6, 17), _d(2026, 6, 18))


# --------------------------------------------------------------------------
# weeks / weekends / months
# --------------------------------------------------------------------------

def test_last_week():
    # ISO week: previous Mon..Mon = 6-8 .. 6-15.
    assert resolve_date_range("last week", NOW) == (_d(2026, 6, 8), _d(2026, 6, 15))


def test_this_week():
    assert resolve_date_range("this week", NOW) == (_d(2026, 6, 15), _d(2026, 6, 22))


def test_last_weekend():
    # Sat..Mon preceding this Monday (6-15) => 6-13 .. 6-15.
    assert resolve_date_range("last weekend", NOW) == (_d(2026, 6, 13), _d(2026, 6, 15))


def test_last_month():
    assert resolve_date_range("last month", NOW) == (_d(2026, 5, 1), _d(2026, 6, 1))


def test_this_month():
    assert resolve_date_range("this month", NOW) == (_d(2026, 6, 1), _d(2026, 7, 1))


# --------------------------------------------------------------------------
# named months (S3 follow-up: "back in March", "in April 2025", "last December")
# --------------------------------------------------------------------------

def test_back_in_march_is_this_year():
    # NOW = June 2026; March already passed this year.
    assert resolve_date_range("what did we talk about back in March", NOW) == (
        _d(2026, 3, 1), _d(2026, 4, 1))


def test_in_april_full_word():
    assert resolve_date_range("in April", NOW) == (_d(2026, 4, 1), _d(2026, 5, 1))


def test_in_december_resolves_to_last_year():
    # December is after June -> most recent PAST December is last year.
    assert resolve_date_range("anything from in December", NOW) == (
        _d(2025, 12, 1), _d(2026, 1, 1))


def test_december_rollover_end_is_january_next_year():
    start, end = resolve_date_range("in December 2025", NOW)
    assert start == _d(2025, 12, 1)
    assert end == _d(2026, 1, 1)


def test_month_with_explicit_year():
    assert resolve_date_range("back in April 2025", NOW) == (
        _d(2025, 4, 1), _d(2025, 5, 1))


def test_month_year_no_cue_word():
    assert resolve_date_range("March 2024", NOW) == (_d(2024, 3, 1), _d(2024, 4, 1))


def test_month_of_year_with_of():
    assert resolve_date_range("sometime in March of 2023", NOW) == (
        _d(2023, 3, 1), _d(2023, 4, 1))


def test_abbreviated_month():
    assert resolve_date_range("in Feb", NOW) == (_d(2026, 2, 1), _d(2026, 3, 1))


def test_last_march():
    # "last March" -> most recent past March (this year, since it's passed).
    assert resolve_date_range("last March", NOW) == (_d(2026, 3, 1), _d(2026, 4, 1))


def test_bare_ambiguous_month_without_cue_does_not_match():
    # "may"/"march" as common words must NOT resolve without a temporal cue.
    assert resolve_date_range("you may have told me", NOW) is None
    assert resolve_date_range("we should march forward", NOW) is None


def test_in_june_is_current_month():
    assert resolve_date_range("in June", NOW) == (_d(2026, 6, 1), _d(2026, 7, 1))


# --------------------------------------------------------------------------
# N ago + fuzzy windows
# --------------------------------------------------------------------------

def test_three_days_ago():
    assert resolve_date_range("3 days ago", NOW) == (_d(2026, 6, 14), _d(2026, 6, 15))


def test_two_weeks_ago_is_that_iso_week():
    # 2 ISO weeks before this Monday (6-15) => week of 6-1 .. 6-8.
    assert resolve_date_range("two weeks ago", NOW) == (_d(2026, 6, 1), _d(2026, 6, 8))


def test_a_couple_weeks_ago_is_a_window():
    start, end = resolve_date_range("a couple weeks ago", NOW)
    # span=2 -> [today-21, today-7]
    assert start == _d(2026, 5, 27)
    assert end == _d(2026, 6, 10)
    assert start < end


def test_a_few_days_ago_is_a_small_window():
    start, end = resolve_date_range("a few days ago", NOW)
    # span=3 -> [today-4, today-2]
    assert start == _d(2026, 6, 13)
    assert end == _d(2026, 6, 15)


def test_recently():
    assert resolve_date_range("recently", NOW) == (_d(2026, 6, 10), NOW)


def test_last_few_days():
    assert resolve_date_range("the last few days", NOW) == (_d(2026, 6, 13), NOW)


# --------------------------------------------------------------------------
# non-matches
# --------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "", "   ", "tell me a joke", "what's the weather",
    "how do I make pasta", "turn on the lights",
])
def test_unrecognized_returns_none(phrase):
    assert resolve_date_range(phrase, NOW) is None


def test_all_ranges_are_ordered_and_tzaware():
    phrases = [
        "today", "yesterday", "this morning", "last saturday", "last week",
        "last weekend", "last month", "3 days ago", "two weeks ago",
        "a couple weeks ago", "recently",
    ]
    for ph in phrases:
        r = resolve_date_range(ph, NOW)
        assert r is not None, ph
        start, end = r
        assert start < end, ph
        assert start.tzinfo is not None and end.tzinfo is not None, ph


# --------------------------------------------------------------------------
# query_episodes_by_range — DB overlap (seeds + cleans up)
# --------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_overlap_query_seeded():
    from memory.manager import store_episode, query_episodes_by_range
    from db.connection import get_pool

    async def go():
        import time
        base = time.time() - 30 * 86400  # 30 days ago, clear of live data
        # Three episodes on three consecutive days.
        ids = []
        for i, marker in enumerate(("DAY_A", "DAY_B", "DAY_C")):
            s = base + i * 86400
            e = s + 1800  # 30-min span
            ids.append(await store_episode(
                s, e, f"__OVERLAP_TEST__ {marker}", token_count=5,
                source={"trigger": "unit_test"}))
        try:
            # Window covering only DAY_B (+/- a few hours around its day).
            day_b_start = datetime.fromtimestamp(base + 1 * 86400, tz=timezone.utc) - timedelta(hours=2)
            day_b_end = datetime.fromtimestamp(base + 1 * 86400, tz=timezone.utc) + timedelta(hours=6)
            rows = await query_episodes_by_range(day_b_start, day_b_end, limit=20)
            markers = [r["text"] for r in rows if "__OVERLAP_TEST__" in r["text"]]
            assert len(markers) == 1, markers
            assert "DAY_B" in markers[0]

            # Wide window covering all three -> ordered oldest-first.
            wide_start = datetime.fromtimestamp(base - 86400, tz=timezone.utc)
            wide_end = datetime.fromtimestamp(base + 4 * 86400, tz=timezone.utc)
            rows = await query_episodes_by_range(wide_start, wide_end, limit=20)
            ours = [r for r in rows if "__OVERLAP_TEST__" in r["text"]]
            assert len(ours) == 3
            assert ours[0]["span_start"] <= ours[1]["span_start"] <= ours[2]["span_start"]
            assert "DAY_A" in ours[0]["text"] and "DAY_C" in ours[2]["text"]

            # Window entirely before -> no rows.
            far_start = datetime.fromtimestamp(base - 10 * 86400, tz=timezone.utc)
            far_end = datetime.fromtimestamp(base - 9 * 86400, tz=timezone.utc)
            rows = await query_episodes_by_range(far_start, far_end, limit=20)
            assert not [r for r in rows if "__OVERLAP_TEST__" in r["text"]]
        finally:
            pool = await get_pool()
            await pool.execute("DELETE FROM episodes WHERE id = ANY($1::int[])", ids)

    _run(go())
