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
"""

from __future__ import annotations

import math
from datetime import datetime

import config


def recency_weight(age_seconds: float, halflife_seconds: float | None = None) -> float:
    """Multiplicative recency weight in (0, 1]: 1.0 at age 0, 0.5 at one
    half-life, decaying exponentially after. A negative age (future timestamp /
    clock skew) clamps to 1.0; a non-positive half-life disables decay (returns
    1.0) so the feature is a clean no-op when turned off."""
    if halflife_seconds is None:
        halflife_seconds = config.EPISODE_DECAY_HALFLIFE_S
    if halflife_seconds <= 0:
        return 1.0
    age = max(0.0, age_seconds)
    return 0.5 ** (age / halflife_seconds)


def access_boost(access_count: int) -> float:
    """Saturating usage lift from the free access_count signal: 0 -> 1.0,
    growing log-wise so a frequently-recalled episode gets a small, BOUNDED
    boost that never swamps similarity or recency. Scaled by
    config.EPISODE_ACCESS_BOOST (0 disables)."""
    if access_count <= 0 or config.EPISODE_ACCESS_BOOST <= 0:
        return 1.0
    return 1.0 + config.EPISODE_ACCESS_BOOST * math.log1p(access_count)


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
