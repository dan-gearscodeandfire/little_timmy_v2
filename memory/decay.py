"""Recency decay for episodic semantic ranking (plan Session 5).

Pure functions, no I/O — `now`/timestamps are injected so they're fully unit
testable. Exponential half-life decay multiplies a similarity/fusion score so a
fresh episode outranks a stale one of EQUAL similarity. This is the fix the plan
requires before episode embeddings are restored: it closes the documented
no-recency hole in the vector ranker (see feedback_lt_semantic_retrieval_no_recency,
where created_at is display-only and access_count is written-but-never-read) for
the episodes path specifically.

`access_count` — written on every recall, previously unused — is folded back in
as a mild, saturating usage boost (the "free signal" the plan calls out).

2026-08-13: both terms are now RANGE-BOUNDED (RECENCY_WEIGHT_FLOOR,
EPISODE_ACCESS_BOOST_MAX) so neither can out-swing the relevance score it
multiplies. See recency_weight's docstring for the measurement that forced it.
"""

from __future__ import annotations

import math
from datetime import datetime

import config


def recency_weight(age_seconds: float, halflife_seconds: float | None = None,
                   floor: float | None = None) -> float:
    """Multiplicative recency weight in [floor, 1.0]: 1.0 at age 0, decaying
    exponentially toward `floor` (NOT toward 0). A negative age (future
    timestamp / clock skew) clamps to 1.0; a non-positive half-life disables
    decay (returns 1.0) so the feature is a clean no-op when turned off.

    WHY THE FLOOR (2026-08-13). Measured on the live 1,239-proposition corpus:
    the corpus spans 2.07 half-lives, so an unfloored weight ranges 1.00 -> 0.24
    (4.2x), while fused RRF relevance across the same candidate pool spans only
    ~0.025 -> 0.076 (3x). Recency therefore had MORE dynamic range than
    relevance and became the PRIMARY sort key, with similarity demoted to a
    tiebreaker -- backwards. Live cost: the near-verbatim answer to "somebody
    walked off with one of your microphones at that party" scored rank 1 of all
    candidates on relevance and fell to rank 7 (uninjected) after x0.25, and
    Timmy answered "I don't recall any party." Across 14 replayed utterances,
    decay evicted a top-5-by-relevance claim on 12 of them.

    Compressing into [floor, 1.0] keeps the curve's SHAPE -- still monotonic,
    still smooth, a fresh claim still outranks a stale one of equal similarity
    -- while bounding how far recency can move a result. At the default floor
    the whole 2-month corpus spans 15%, comfortably inside the relevance range,
    so decay breaks near-ties instead of reordering across real relevance gaps.

    This is deliberately NOT "lengthen the half-life": that trades one blunt
    prior for another and still lets an arbitrarily old item be swamped once the
    corpus outgrows the new half-life. `floor=0.0` reproduces the pre-2026-08-13
    behaviour exactly, which is the A/B control (TIMMY_RECENCY_WEIGHT_FLOOR=0)."""
    if halflife_seconds is None:
        halflife_seconds = config.EPISODE_DECAY_HALFLIFE_S
    if halflife_seconds <= 0:
        return 1.0
    if floor is None:
        floor = config.RECENCY_WEIGHT_FLOOR
    floor = min(max(0.0, float(floor)), 1.0)
    age = max(0.0, age_seconds)
    return floor + (1.0 - floor) * (0.5 ** (age / halflife_seconds))


def access_boost(access_count: int) -> float:
    """Saturating usage lift from the free access_count signal: 0 -> 1.0,
    growing log-wise so a frequently-recalled episode gets a small, BOUNDED
    boost that never swamps similarity or recency. Scaled by
    config.EPISODE_ACCESS_BOOST (0 disables)."""
    if access_count <= 0 or config.EPISODE_ACCESS_BOOST <= 0:
        return 1.0
    # Hard cap (2026-08-13). The boost is SELF-REINFORCING: the injected top-K
    # is touch_*()-ed every turn, so whatever won last turn is a little more
    # likely to win the next. Measured, the runaway was a contentless meta-claim
    # ("Timmy recently recalled a conversation from OpenSauce", access_count 29,
    # highest in the corpus) appearing in the injected top-5 on 5 of 9 live
    # turns. log1p saturates but never stops growing; the cap does.
    raw = 1.0 + config.EPISODE_ACCESS_BOOST * math.log1p(access_count)
    return min(raw, config.EPISODE_ACCESS_BOOST_MAX)


def decay_multiplier(span_end: datetime, now: datetime,
                     access_count: int = 0,
                     halflife_seconds: float | None = None) -> float:
    """The combined re-rank multiplier for an episode: recency × usage. Multiply
    a base similarity/fusion score by this. `span_end` is the episode's event
    end (when it actually happened), `now` the query instant — both tz-aware."""
    age = (now - span_end).total_seconds()
    return recency_weight(age, halflife_seconds) * access_boost(access_count)


def decayed_score(similarity: float, span_end: datetime, now: datetime,
                  access_count: int = 0,
                  halflife_seconds: float | None = None) -> float:
    """Convenience: similarity × decay_multiplier. Matches the plan's
    `score = similarity × halflife_decay(now − span_end)` (plus the usage lift)."""
    return similarity * decay_multiplier(span_end, now, access_count, halflife_seconds)
