"""Two-axis mood state persisted to disk.

X = engagement: -1 BORED, 0 NEUTRAL, +1 SLIGHTLY_INTERESTED
Y = warmth:    -1 MEAN,  0 NEUTRAL, +1 BEGRUDGINGLY_NICE

Transitions step at most ±1 per axis per turn, require 2 consecutive
same-direction signals to fire (persistence), and use hysteresis so coming
back toward 0 needs a stronger opposing signal than going further out.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

STATE_PATH = Path.home() / "little_timmy" / "data" / "mood_state.json"

# Thresholds. A signal in (MOVE_THRESH, +inf) counts as "+", in (-inf, -MOVE_THRESH) as "-",
# else neutral (no contribution to persistence).
MOVE_THRESH = 0.20
# Hysteresis: when at +1, signal needs to be < -RETURN_THRESH (wider) to start
# moving back to 0. Same on the negative side.
RETURN_THRESH = 0.40

PERSISTENCE = 2  # consecutive same-sign signals required to step

_AXES = ("x", "y")
_VALID = (-1, 0, 1)


@dataclass
class MoodState:
    x: int = 0
    y: int = 0
    x_signals: deque = field(default_factory=lambda: deque(maxlen=PERSISTENCE))
    y_signals: deque = field(default_factory=lambda: deque(maxlen=PERSISTENCE))
    last_update_ts: float = 0.0
    last_x_signal: float = 0.0
    last_y_signal: float = 0.0

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "x_signals": list(self.x_signals),
            "y_signals": list(self.y_signals),
            "last_update_ts": self.last_update_ts,
            "last_x_signal": self.last_x_signal,
            "last_y_signal": self.last_y_signal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MoodState":
        s = cls(
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            last_update_ts=float(d.get("last_update_ts", 0.0)),
            last_x_signal=float(d.get("last_x_signal", 0.0)),
            last_y_signal=float(d.get("last_y_signal", 0.0)),
        )
        for v in d.get("x_signals", [])[-PERSISTENCE:]:
            s.x_signals.append(float(v))
        for v in d.get("y_signals", [])[-PERSISTENCE:]:
            s.y_signals.append(float(v))
        # Clamp in case of stale persisted values outside the valid range.
        if s.x not in _VALID:
            s.x = max(-1, min(1, s.x))
        if s.y not in _VALID:
            s.y = max(-1, min(1, s.y))
        return s


_lock = threading.Lock()
_state: MoodState | None = None


def _load() -> MoodState:
    if not STATE_PATH.exists():
        return MoodState()
    try:
        return MoodState.from_dict(json.loads(STATE_PATH.read_text()))
    except Exception as e:
        log.warning("mood_state load failed (%s); resetting to neutral", e)
        return MoodState()


def _save(s: MoodState) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(s.to_dict(), indent=2))
    except Exception as e:
        log.warning("mood_state save failed: %s", e)


def get() -> MoodState:
    """Return the current state. Initializes from disk on first call."""
    global _state
    with _lock:
        if _state is None:
            _state = _load()
        return _state


def _classify_sign(signal: float, current: int) -> int:
    """+1 / -1 / 0 with hysteresis depending on current position on the axis."""
    if current == 0:
        if signal > MOVE_THRESH:
            return 1
        if signal < -MOVE_THRESH:
            return -1
        return 0
    if current > 0:
        # at +1: easier to push further (no +2 in design, so stay), harder to come back
        if signal < -RETURN_THRESH:
            return -1
        if signal > MOVE_THRESH:
            return 1
        return 0
    # current < 0
    if signal > RETURN_THRESH:
        return 1
    if signal < -MOVE_THRESH:
        return -1
    return 0


def _step(buf: deque, current: int) -> int:
    """Return the new axis value given the most recent signal classifications.

    Steps at most ±1 and only when the last PERSISTENCE entries all agree
    in the same non-zero direction.
    """
    if len(buf) < PERSISTENCE:
        return current
    signs = list(buf)
    if all(s > 0 for s in signs):
        return min(1, current + 1)
    if all(s < 0 for s in signs):
        return max(-1, current - 1)
    return current


# Per-axis instrumentation log (Bundle C / supervisor_issues.md 2026-05-14 00:42).
# Zero behavior change -- just captures every update() input + decision so we
# can see why each axis's ratchet stays stuck. Tail with:
#   tail -F ~/little_timmy/data/mood_debug.jsonl | jq .
DEBUG_LOG_PATH = Path.home() / "little_timmy" / "data" / "mood_debug.jsonl"


def _debug_log(record: dict) -> None:
    """Append one JSON line to the mood-debug log. Best effort; never raises."""
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as e:
        log.debug("mood_debug log append failed: %s", e)


def update(x_signal: float, y_signal: float) -> dict:
    """Feed one turn's signals; transition state if persistence+threshold met.

    Returns a small dict describing what happened (for logging / inspection).
    """
    with _lock:
        s = _load() if _state is None else _state
        prev_x, prev_y = s.x, s.y
        sx = _classify_sign(x_signal, s.x)
        sy = _classify_sign(y_signal, s.y)
        s.x_signals.append(sx)
        s.y_signals.append(sy)
        x_buf_after = list(s.x_signals)
        y_buf_after = list(s.y_signals)
        s.x = _step(s.x_signals, s.x)
        s.y = _step(s.y_signals, s.y)
        s.last_update_ts = time.time()
        s.last_x_signal = float(x_signal)
        s.last_y_signal = float(y_signal)
        # Reset the buffer after a successful transition so the next step
        # requires another PERSISTENCE-length agreement, not just one more.
        if s.x != prev_x:
            s.x_signals.clear()
        if s.y != prev_y:
            s.y_signals.clear()
        _save(s)
        globals()["_state"] = s
        moved_x = s.x - prev_x
        moved_y = s.y - prev_y
        # Per-axis ratchet instrumentation. Includes the hysteresis-aware
        # classified signs and the persistence buffer contents so we can see
        # whether the ratchet is stuck because (a) raw signals never cross
        # the return threshold, (b) classification flips happen but the
        # persistence buffer keeps getting mixed signs, or (c) something else.
        _debug_log({
            "ts": s.last_update_ts,
            "raw_x": float(x_signal),
            "raw_y": float(y_signal),
            "classified_sx": sx,
            "classified_sy": sy,
            "x_buf_after": x_buf_after,
            "y_buf_after": y_buf_after,
            "prev_x": prev_x,
            "prev_y": prev_y,
            "new_x": s.x,
            "new_y": s.y,
            "moved_x": moved_x,
            "moved_y": moved_y,
            "move_thresh": MOVE_THRESH,
            "return_thresh": RETURN_THRESH,
            "persistence": PERSISTENCE,
        })
        return {
            "x": s.x,
            "y": s.y,
            "moved_x": moved_x,
            "moved_y": moved_y,
            "x_signal": x_signal,
            "y_signal": y_signal,
        }


def reset() -> None:
    """Reset to neutral. Test/admin helper."""
    global _state
    with _lock:
        _state = MoodState()
        _save(_state)
