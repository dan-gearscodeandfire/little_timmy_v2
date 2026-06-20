"""Deterministic natural-language date-range resolution for episodic recall.

`resolve_date_range(phrase, now)` maps an English time expression ("yesterday",
"last Saturday", "a couple weeks ago") to a half-open `[start, end)` datetime
window, or None if nothing recognizable is found. This is the *deterministic
core* of episodic recall (plan Session 2): it sidesteps the recency-blind vector
ranker entirely — episodes are then fetched by span overlap, ordered by time.

Pure module: no DB, no LLM, no I/O. `now` is injected so it is fully unit
testable. Everything works in `now`'s timezone — the caller passes a tz-aware
local `now` (e.g. `datetime.now().astimezone()`); the returned bounds carry the
same tzinfo, and Postgres compares them against TIMESTAMPTZ as instants.

Convention: ranges are HALF-OPEN `[start, end)`. A day is `[00:00, next 00:00)`.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

__all__ = ["resolve_date_range"]

# Monday=0 .. Sunday=6 (matches datetime.weekday()).
_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

# Small number-word map for "X days/weeks ago" and "a couple/few".
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2, "few": 3, "several": 4,
}

# Day-part windows (local hours), half-open.
_DAYPARTS = {
    "morning": (5, 12),
    "afternoon": (12, 17),
    "evening": (17, 24),
    "tonight": (17, 24),
}


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _add_hours(day0: datetime, h: int) -> datetime:
    # h may be 24 (end-of-day) — represent as next-day 00:00.
    return day0 + timedelta(hours=h)


def _week_start(dt: datetime) -> datetime:
    """Monday 00:00 of dt's ISO week."""
    return _day_start(dt) - timedelta(days=dt.weekday())


def _word_to_int(tok: str) -> int | None:
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok)


def resolve_date_range(phrase: str, now: datetime):
    """Resolve an English time phrase to a half-open `[start, end)` window.

    Returns `(start, end)` (both tz-aware, same tzinfo as `now`) or None.
    `start < end` always. Unrecognized / non-temporal input returns None.
    """
    if not phrase:
        return None
    p = phrase.lower().strip()
    # Normalize punctuation/whitespace; keep word boundaries.
    p = re.sub(r"[^\w\s]", " ", p)
    p = re.sub(r"\s+", " ", p).strip()
    if not p:
        return None

    today0 = _day_start(now)

    # Specific phrases first — broader matches ("today", "yesterday") below
    # would otherwise shadow "earlier today" / "day before yesterday" /
    # "yesterday evening".

    # --- "earlier today" / "so far today" ---------------------------------
    if re.search(r"\bearlier today\b", p) or re.search(r"\bso far today\b", p):
        return (today0, now)

    # --- "day before yesterday" -------------------------------------------
    if re.search(r"\bday before yesterday\b", p):
        return (today0 - timedelta(days=2), today0 - timedelta(days=1))

    # --- day parts: "this morning", "yesterday afternoon", "last night" ---
    if re.search(r"\blast night\b", p):
        # Previous evening into the early hours: yesterday 18:00 -> today 06:00.
        return (today0 - timedelta(days=1) + timedelta(hours=18),
                today0 + timedelta(hours=6))
    m = re.search(r"\b(this|yesterday|last)?\s*(morning|afternoon|evening|tonight)\b", p)
    if m:
        qualifier, part = m.group(1), m.group(2)
        lo, hi = _DAYPARTS[part]
        base = today0
        if qualifier == "yesterday":
            base = today0 - timedelta(days=1)
        # "tonight"/"this evening" with no qualifier -> today.
        return (_add_hours(base, lo), _add_hours(base, hi))

    # --- exact single days -------------------------------------------------
    if re.search(r"\btoday\b", p):
        return (today0, today0 + timedelta(days=1))
    if re.search(r"\byesterday\b", p):
        return (today0 - timedelta(days=1), today0)

    # --- weeks -------------------------------------------------------------
    this_monday = _week_start(now)
    if re.search(r"\blast week\b", p):
        return (this_monday - timedelta(days=7), this_monday)
    if re.search(r"\bthis week\b", p):
        return (this_monday, this_monday + timedelta(days=7))
    if re.search(r"\blast weekend\b", p):
        # Sat 00:00 -> Mon 00:00 of the most recent completed weekend.
        return (this_monday - timedelta(days=2), this_monday)
    if re.search(r"\bthis weekend\b", p):
        return (this_monday + timedelta(days=5), this_monday + timedelta(days=7))

    # --- months ------------------------------------------------------------
    if re.search(r"\blast month\b", p):
        this_month0 = today0.replace(day=1)
        prev_month_end = this_month0  # exclusive
        prev_month_start = (this_month0 - timedelta(days=1)).replace(day=1)
        return (prev_month_start, prev_month_end)
    if re.search(r"\bthis month\b", p):
        this_month0 = today0.replace(day=1)
        # First day of next month.
        next_month0 = (this_month0 + timedelta(days=32)).replace(day=1)
        return (this_month0, next_month0)

    # --- "last <weekday>" and bare "<weekday>" -----------------------------
    m = re.search(r"\blast (" + "|".join(_WEEKDAYS) + r")\b", p)
    if m:
        target = _WEEKDAYS[m.group(1)]
        days_back = (now.weekday() - target) % 7
        if days_back == 0:
            days_back = 7  # "last Saturday" on a Saturday => the prior one
        d0 = today0 - timedelta(days=days_back)
        return (d0, d0 + timedelta(days=1))
    m = re.search(r"\b(" + "|".join(_WEEKDAYS) + r")\b", p)
    if m:
        target = _WEEKDAYS[m.group(1)]
        days_back = (now.weekday() - target) % 7  # 0 => today
        d0 = today0 - timedelta(days=days_back)
        return (d0, d0 + timedelta(days=1))

    # --- fuzzy multi-day windows (BEFORE numeric: "couple"/"few" are fuzzy) -
    # "a couple/few days ago" -> a small window (the exact day is unknown).
    m = re.search(r"\b(couple|few|several) (?:of )?days ago\b", p)
    if m:
        span = _NUM_WORDS[m.group(1)]
        # Center on `span` days back, widen by ~1 day each side.
        return (today0 - timedelta(days=span + 1), today0 - timedelta(days=max(1, span - 1)))
    m = re.search(r"\b(couple|few|several) (?:of )?weeks ago\b", p)
    if m:
        span = _NUM_WORDS[m.group(1)]
        return (today0 - timedelta(days=7 * span + 7), today0 - timedelta(days=7 * max(1, span - 1)))

    # --- "N days ago" / "N weeks ago" (digit or word) ----------------------
    m = re.search(r"\b(\w+) days? ago\b", p)
    if m:
        n = _word_to_int(m.group(1))
        if n is not None:
            d0 = today0 - timedelta(days=n)
            return (d0, d0 + timedelta(days=1))
    m = re.search(r"\b(\w+) weeks? ago\b", p)
    if m:
        n = _word_to_int(m.group(1))
        if n is not None:
            wk = this_monday - timedelta(days=7 * n)
            return (wk, wk + timedelta(days=7))

    # "recently" / "lately" / "the last few days" / "past week"
    if re.search(r"\b(the )?(last|past) few days\b", p):
        return (today0 - timedelta(days=4), now)
    if re.search(r"\b(the )?(last|past) week\b", p):
        return (now - timedelta(days=7), now)
    if re.search(r"\b(recently|lately)\b", p):
        return (today0 - timedelta(days=7), now)

    return None
