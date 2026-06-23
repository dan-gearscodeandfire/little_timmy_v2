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

# Named months + common abbreviations -> month number (1..12).
_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
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


def _month_window(year: int, month: int, now: datetime):
    """Half-open `[first of month 00:00, first of next month 00:00)`, carrying
    `now`'s tzinfo. Handles the December -> next-January rollover."""
    start = now.replace(year=year, month=month, day=1,
                        hour=0, minute=0, second=0, microsecond=0)
    end = (start.replace(year=year + 1, month=1) if month == 12
           else start.replace(month=month + 1))
    return (start, end)


def _resolve_month_year(month: int, now: datetime) -> int:
    """Year of the most recent occurrence of `month` at/before the current
    month. For recall we only ever look backward, so a month later in the
    calendar than `now`'s resolves to last year (e.g. "in December" in June)."""
    return now.year if month <= now.month else now.year - 1


# Absolute calendar dates & explicit from–to ranges --------------------------
# Reached BEFORE the named-month branch so "June 13" resolves to a single day
# rather than being mistaken for the whole month. By the time these run the
# caller has already stripped punctuation to spaces (see resolve_date_range),
# so ISO "2025-04-01" arrives as "2025 04 01" and "June 10-13" as "june 10 13".
_MONTH_ALT = "|".join(_MONTHS)
_ORD = r"(?:st|nd|rd|th)?"  # optional ordinal suffix on a day number


def _mk_day(year: int, month: int, day: int, now: datetime):
    """Day-start datetime in `now`'s tz, or None if the date is invalid
    (e.g. Feb 30, day 0). Lets callers reject impossible day numbers cheaply."""
    try:
        return now.replace(year=year, month=month, day=day,
                           hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        return None


def _infer_md_year(month: int, day: int, now: datetime):
    """Most-recent past-or-today occurrence of month/day (recall looks back):
    this year unless that date is still in the future, then last year."""
    cand = _mk_day(now.year, month, day, now)
    if cand is None:
        return None
    if cand > now:
        cand = _mk_day(now.year - 1, month, day, now)
    return cand


def _find_full_dates(p: str, now: datetime):
    """Every explicit calendar date in `p`, as (start_pos, day_start) pairs,
    left-to-right and non-overlapping. Recognizes ISO `yyyy mm dd`,
    `month day [year]`, and `day [of] month [year]`. A bare year (no day) is
    NOT a full date — that stays the named-month branch's job."""
    cands = []  # (start, end, day_start)
    # ISO: 2025 04 01  (dashes already normalized to spaces upstream)
    for m in re.finditer(r"\b(\d{4})\s+(\d{1,2})\s+(\d{1,2})\b", p):
        d = _mk_day(int(m.group(1)), int(m.group(2)), int(m.group(3)), now)
        if d is not None:
            cands.append((m.start(), m.end(), d))
    # month day [year]: "June 13", "June 13th", "June 13 2025"
    for m in re.finditer(r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})" + _ORD +
                         r"(?:\s+(?:of\s+)?(\d{4}))?\b", p):
        mon, day = _MONTHS[m.group(1)], int(m.group(2))
        d = (_mk_day(int(m.group(3)), mon, day, now) if m.group(3)
             else _infer_md_year(mon, day, now))
        if d is not None:
            cands.append((m.start(), m.end(), d))
    # day [of] month [year]: "13 June", "13th of June 2025"
    for m in re.finditer(r"\b(\d{1,2})" + _ORD + r"\s+(?:of\s+)?(" + _MONTH_ALT +
                         r")\b(?:\s+(\d{4}))?", p):
        mon, day = _MONTHS[m.group(2)], int(m.group(1))
        d = (_mk_day(int(m.group(3)), mon, day, now) if m.group(3)
             else _infer_md_year(mon, day, now))
        if d is not None:
            cands.append((m.start(), m.end(), d))
    # Resolve overlaps: earliest start wins, longest match breaks ties.
    cands.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    chosen, last_end = [], -1
    for s, e, d in cands:
        if s >= last_end:
            chosen.append((s, d))
            last_end = e
    return chosen


def _resolve_absolute(p: str, now: datetime):
    """Resolve explicit calendar dates / from–to ranges to a half-open
    `[start, end)` window (ranges are INCLUSIVE of both endpoint days), or
    None if `p` carries no absolute date. Order matters: two full dates form a
    range; a single month + two day numbers ("June 10 13", "June 10 to 13") is
    a same-month range; one full date is a single day; a bare ordinal
    ("the 13th") is the most recent month carrying that day."""
    full = _find_full_dates(p, now)
    # Two explicit dates -> span from the earliest day to the latest day inclusive.
    if len(full) >= 2:
        starts = [d for _, d in full]
        return (min(starts), max(starts) + timedelta(days=1))
    # Same-month numeric range with ONE month word: "June 10 13" (from
    # "June 10-13"), "June 10 to 13", "March 3 through 9".
    m = re.search(r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})" + _ORD +
                  r"\s+(?:to|through|thru|until|till|and)?\s*(\d{1,2})" + _ORD + r"\b", p)
    if m:
        mon, d1, d2 = _MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
        a = _infer_md_year(mon, min(d1, d2), now)
        b = _infer_md_year(mon, max(d1, d2), now)
        if a is not None and b is not None:
            return (a, b + timedelta(days=1))
    # Single explicit date -> that day.
    if len(full) == 1:
        d = full[0][1]
        return (d, d + timedelta(days=1))
    # Day-of-month only: "the 13th", "on the 13th", bare ordinal "13th".
    # Requires "the" or an ordinal suffix so a plain "13" (e.g. "13 days ago")
    # is NOT mistaken for a date.
    m = re.search(r"\b(?:on\s+)?the\s+(\d{1,2})" + _ORD + r"\b", p)
    if not m:
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", p)
    if m:
        day = int(m.group(1))
        year, month = now.year, now.month
        cand = _mk_day(year, month, day, now)
        if cand is None or cand > now:  # not yet this month -> walk back one month
            month -= 1
            if month == 0:
                month, year = 12, year - 1
            cand = _mk_day(year, month, day, now)
        if cand is not None:
            return (cand, cand + timedelta(days=1))
    return None


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

    # --- absolute dates & explicit ranges: "June 13", "between June 10 and
    # June 13", "2025-04-01", "the 13th". Before day-parts so "June 13
    # afternoon" lands on the date, not today; before named-month so "June 13"
    # is a day, not the whole month. Returns None for bare months/years
    # ("June 2025") so those fall through to the named-month branch.
    abs_window = _resolve_absolute(p, now)
    if abs_window is not None:
        return abs_window

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

    # --- named months: "in March", "back in April 2025", "last December" ---
    # Cue-guarded to avoid false matches on month names that double as common
    # words ("may", "march" the verb, "august" the adjective): a bare month
    # only resolves when led by a temporal cue OR followed by an explicit year.
    # No year given -> the most recent PAST occurrence (recall looks backward).
    _month_alt = "|".join(_MONTHS)
    m = re.search(
        r"\b(?:back in|in|during|around|last|this past|sometime in|month of)\s+"
        r"(" + _month_alt + r")\b(?:\s+(?:of\s+)?(\d{4}))?", p)
    if not m:
        # "<month> <year>" with no leading cue ("March 2026", "Apr 2025").
        m = re.search(r"\b(" + _month_alt + r")\s+(?:of\s+)?(\d{4})\b", p)
    if m:
        month = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else _resolve_month_year(month, now)
        return _month_window(year, month, now)

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
