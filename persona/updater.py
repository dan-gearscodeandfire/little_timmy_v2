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


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _y_signal(user_text: str) -> float:
    """Return the inverted-VADER signal in [-1, +1]."""
    if not _vader or not user_text:
        return 0.0
    s = _vader.polarity_scores(user_text)
    # Compound is in [-1, +1]. Invert for Timmy (happy Dan → mean Timmy).
    return -float(s.get("compound", 0.0))


def _x_signal_from_embeddings(latest: np.ndarray) -> float:
    """Derive the X signal from the rolling embedding buffer.

    Returns a value in [-1, +1]. 0 when there isn't enough context yet.
    """
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

    # Topic shift → small magnitude, near zero.
    if continuity < _CONTINUITY_HIGH:
        return 0.0

    # Same topic — direction depends on whether Dan is moving the topic forward.
    if novelty >= _NOVELTY_HIGH:
        return +1.0  # progression: push toward interested
    if novelty <= _NOVELTY_LOW:
        return -1.0  # repetition: push toward bored
    return 0.0


async def update_async(user_text: str, prev_assistant_text: str = "") -> None:
    """Fire-and-forget mood update. Safe to call from the orchestrator."""
    try:
        y = _y_signal(user_text)

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
