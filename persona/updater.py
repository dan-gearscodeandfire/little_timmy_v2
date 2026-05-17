"""Async mood updater. Runs after each turn finalizes; never on the hot path.

Y signal: VADER compound score on Dan's user_text, INVERTED. Happy Dan → push
Timmy mean. Grumpy Dan → push Timmy nice.

X signal: built from two embedding-derived components on Dan's recent turns:
  continuity = cosine(latest_turn, EMA centroid of prior turns)  [0..1, ~]
  novelty    = 1 - cosine(latest_turn, immediately previous turn) [0..1, ~]

Combined X signal:
  high continuity + high novelty → +project-progression
  high continuity + low novelty  → -rambling
  low continuity                  → near zero (topic shift, neutral)

Uses LT's existing nomic-embed-text Ollama embedder (memory.manager.embed)
so no new model is loaded.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from typing import Optional

import numpy as np

from memory.manager import embed
from persona import state as mood_state

log = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception as e:  # pragma: no cover
    log.warning("VADER unavailable (%s); Y axis will receive 0 signal", e)
    _vader = None


# Rolling window of the most recent Dan-turn embeddings. Module-level so it
# survives across turns within a single LT process. Lost on restart, which
# is fine — the X-axis state is persisted; only the working memory of recent
# embeddings is ephemeral, and a few turns of warmup recovers signal.
_RECENT_EMBEDDINGS: deque[np.ndarray] = deque(maxlen=6)


_CONTINUITY_HIGH = 0.55   # cos sim above this counts as "same topic"
                          # tuned against nomic-embed-text: sustained project
                          # talk lands ~0.56-0.64 vs centroid; off-topic ~0.30-0.42
_NOVELTY_LOW = 0.10       # cos distance below this counts as "repeating"
_NOVELTY_HIGH = 0.40      # cos distance above this counts as "new angle"
                          # consecutive on-topic-but-new-detail pairs at 0.46-0.54

# 2026-05-16 mood overhaul ---------------------------------------------------
# Stage 2 (symmetric Y): warmth markers in Dan's text. Inverted-VADER alone is
# one-directional in practice (upbeat-but-not-affectionate Dan keeps pushing
# mean), so we add a positive bias when Dan is warm TO TIMMY specifically:
# compliments, affection, thanks, "you're (great|funny|smart…)".
_WARMTH_PATTERNS = re.compile(
    r"\b("
    r"thanks?(?:\s+you)?|thank\s+you|"
    r"love\s+you|i\s+love|"
    r"good\s+(?:job|boy|one|work|point|call|catch)|"
    r"well\s+done|nice\s+(?:one|work|going|job)|"
    r"appreciate|amazing|brilliant|hilarious|adorable|cute|sweet|"
    r"you(?:'re|\s+are)\s+(?:great|right|funny|smart|clever|good|nice|hilarious|fantastic|the\s+best)|"
    r"that(?:'s|\s+is|\s+was)\s+(?:funny|great|hilarious|clever|smart|good|nice|true|fair|accurate)|"
    r"sorry|my\s+bad|i\s+was\s+wrong|fair\s+enough"
    r")\b",
    re.IGNORECASE,
)

# Stage 4 (symmetric X): idle-bored counter. _x_signal_from_embeddings returns
# 0 on topic shifts; a long stretch of 0s with no genuine engagement signal
# means Dan is hopping topics — which is the conversational equivalent of
# being bored at Timmy. After this many consecutive zeros, return a soft
# negative bias so the X axis can eventually decay toward bored.
_X_IDLE_THRESHOLD = 4      # consecutive 0-classified turns before bored kicks in
_X_IDLE_BORED_BIAS = -0.5  # value to return once threshold is crossed
_x_idle_count = 0          # module-level counter, ephemeral like the embeddings deque


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _y_signal_base(user_text: str) -> float:
    """Return the inverted-VADER component in [-1, +1]."""
    if not _vader or not user_text:
        return 0.0
    s = _vader.polarity_scores(user_text)
    # Compound is in [-1, +1]. Invert for Timmy (happy Dan → mean Timmy).
    return -float(s.get("compound", 0.0))


def _warmth_bonus(user_text: str) -> float:
    """Detect compliments / affection / explicit warmth toward Timmy. Returns a
    positive bonus (push toward nice) in [0, +0.6]. Conservative — at most one
    cluster of patterns per turn fires, with diminishing return on repeats."""
    if not user_text:
        return 0.0
    n = len(_WARMTH_PATTERNS.findall(user_text))
    if n == 0:
        return 0.0
    # +0.4 for the first match, +0.1 per additional, capped at +0.6.
    return min(0.6, 0.4 + 0.1 * (n - 1))


def _reciprocity_bonus(user_text: str, prev_y: int) -> float:
    """If Timmy was mean last turn (prev_y == -1) and Dan didn't bite back
    (his VADER compound is neutral or positive), reward with a small positive
    Y nudge — closes the loop the one-shot inverted-VADER signal misses.
    Symmetric on the other end: if Timmy was nice and Dan turned hostile,
    nudge negative to mirror the slap."""
    if not _vader or not user_text:
        return 0.0
    compound = float(_vader.polarity_scores(user_text).get("compound", 0.0))
    if prev_y == -1 and compound >= -0.1:
        return +0.3  # Dan stayed patient through Timmy's hostility — soften.
    if prev_y == +1 and compound <= -0.3:
        return -0.3  # Dan bit Timmy back during a nice stretch — harden.
    return 0.0


def _y_signal(user_text: str, prev_y: int = 0) -> float:
    """Composed Y signal in [-1, +1]: inverted VADER + warmth bonus +
    reciprocity bonus. Final value is clamped; sign matters more than
    magnitude for the ratchet's _classify_sign check."""
    raw = _y_signal_base(user_text) + _warmth_bonus(user_text) + _reciprocity_bonus(user_text, prev_y)
    if raw > 1.0:  return 1.0
    if raw < -1.0: return -1.0
    return raw


def _x_signal_from_embeddings(latest: np.ndarray) -> float:
    """Derive the X signal from the rolling embedding buffer.

    Returns a value in [-1, +1]. 0 when there isn't enough context yet, but
    sustained zeros (topic-shift hopping) eventually flip to a bored bias.
    """
    global _x_idle_count
    if len(_RECENT_EMBEDDINGS) == 0:
        return 0.0

    prev = _RECENT_EMBEDDINGS[-1]
    novelty = 1.0 - _cos(latest, prev)

    # Continuity vs. centroid of all prior turns (excluding the latest).
    if len(_RECENT_EMBEDDINGS) >= 2:
        centroid = np.mean(np.stack(list(_RECENT_EMBEDDINGS)), axis=0)
        continuity = _cos(latest, centroid)
    else:
        continuity = _cos(latest, prev)

    def _raw():
        if continuity < _CONTINUITY_HIGH:
            return 0.0
        if novelty >= _NOVELTY_HIGH:
            return +1.0  # progression
        if novelty <= _NOVELTY_LOW:
            return -1.0  # repetition
        return 0.0

    raw = _raw()
    # Stage 4: idle-bored fallback. The genuine -1 (repetition) case is rare —
    # in practice Dan rarely repeats himself. So short-term topic hopping
    # leaves the axis pinned at +1 forever once we're there. After a few
    # consecutive zero-signal turns, return a soft negative bias so the
    # ratchet can decay toward bored without requiring perfect repetition.
    if raw == 0.0:
        _x_idle_count += 1
        if _x_idle_count >= _X_IDLE_THRESHOLD:
            return _X_IDLE_BORED_BIAS
        return 0.0
    _x_idle_count = 0
    return raw


async def update_async(user_text: str, prev_assistant_text: str = "") -> None:
    """Fire-and-forget mood update. Safe to call from the orchestrator."""
    try:
        # Read prev_y BEFORE we call update() so the reciprocity bonus sees
        # the axis value that drove Timmy's most recent reply.
        prev_y = mood_state.get().y
        y = _y_signal(user_text, prev_y=prev_y)

        x = 0.0
        if user_text.strip():
            try:
                emb = await embed(user_text)
                x = _x_signal_from_embeddings(emb)
                _RECENT_EMBEDDINGS.append(emb)
            except Exception as e:
                log.warning("mood updater: embed failed (%s); skipping X signal", e)

        result = mood_state.update(x_signal=x, y_signal=y)
        if result["moved_x"] or result["moved_y"]:
            log.info(
                "[MOOD] step x=%+d y=%+d -> (%+d,%+d) | xs=%.2f ys=%.2f",
                result["moved_x"], result["moved_y"], result["x"], result["y"],
                result["x_signal"], result["y_signal"],
            )
        else:
            log.debug(
                "[MOOD] hold (%+d,%+d) | xs=%.2f ys=%.2f",
                result["x"], result["y"], result["x_signal"], result["y_signal"],
            )
    except Exception as e:
        log.warning("mood updater error: %s", e)


def schedule(user_text: str, prev_assistant_text: str = "") -> None:
    """Fire-and-forget: schedule update_async as a background task."""
    asyncio.create_task(update_async(user_text, prev_assistant_text))
