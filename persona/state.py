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
# Bundle C phase 2 (2026-05-14): RETURN_THRESH was 0.40 -- one-way ratchet
# that prevented return-to-center because Y-axis raw signal rarely sustains
# >0.40 for two turns in a row. Now matches MOVE_THRESH so both directions
# are equally easy to cross. The "stickiness" the original asymmetry was
# trying to provide is preserved by PERSISTENCE (still needs 2 same-sign
# signals to step) and by the new DECAY_TURNS gate below.
RETURN_THRESH = 0.20

PERSISTENCE = 2  # consecutive same-sign signals required to step

# Bundle C phase 2: decay-to-center for an axis stuck at an extreme. After
# this many consecutive turns of neutral-classified signal (raw signal
# inside +/-MOVE_THRESH) at +/-1, force one step toward 0 even without an
# opposing-direction signal. Mitigates the case where the X-axis signal
# generator (which is ternary {-1,0,+1}) just doesn't produce a -1 for a
# long stretch -- the axis would otherwise stay at +1 forever.
# 2026-05-16: tightened 8 -> 3. Mood was pinned at (+1,-1) for hours because
# same-direction signals kept interleaving the zero stretches, so 8-in-a-row
# never landed. 3-in-a-row is reachable in a typical conversation pause and
# pairs with the symmetric Y warmth/reciprocity bonuses + the X idle-bored
# signal added in updater.py so the axes breathe.
DECAY_TURNS = 3

_AXES = ("x", "y")
_VALID = (-1, 0, 1)


@dataclass
class MoodState:
    # Product default (2026-06-07): boot pinned to Interested (x=+1) +
    # Begrudgingly-Nice (y=+1) under manual override (see `override` below), so
    # a fresh install / reset starts in the cell Dan drives from the dashboard
    # rather than dead-neutral. Auto drift only takes over once override is
    # released from the LT-OS Mood panel.
    x: int = 1
    y: int = 1
    x_signals: deque = field(default_factory=lambda: deque(maxlen=PERSISTENCE))
    y_signals: deque = field(default_factory=lambda: deque(maxlen=PERSISTENCE))
    last_update_ts: float = 0.0
    last_x_signal: float = 0.0
    last_y_signal: float = 0.0
    # Bundle C phase 2: per-axis idle-counters used by the decay-to-center
    # mechanism. Increment when at extreme + classified signal is 0; reset
    # on any actual step OR on any non-zero classified signal (a real signal
    # in any direction means the axis isn't being silently abandoned).
    x_idle_turns: int = 0
    y_idle_turns: int = 0
    # Manual override: when True, the axes are pinned to (x, y) by an operator
    # (LT-OS dashboard) and update() will NOT transition them. Signals are
    # still recorded for instrumentation so we can see what the auto system
    # *would* have done. Clearing the override resumes automatic drift from
    # wherever the axes were pinned. Defaults ON (2026-06-07) so Timmy boots
    # under manual control at the nice+interested cell above.
    override: bool = True

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "x_signals": list(self.x_signals),
            "y_signals": list(self.y_signals),
            "last_update_ts": self.last_update_ts,
            "last_x_signal": self.last_x_signal,
            "last_y_signal": self.last_y_signal,
            "x_idle_turns": self.x_idle_turns,
            "y_idle_turns": self.y_idle_turns,
            "override": self.override,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MoodState":
        s = cls(
            x=int(d.get("x", 1)),
            y=int(d.get("y", 1)),
            last_update_ts=float(d.get("last_update_ts", 0.0)),
            last_x_signal=float(d.get("last_x_signal", 0.0)),
            last_y_signal=float(d.get("last_y_signal", 0.0)),
            x_idle_turns=int(d.get("x_idle_turns", 0)),
            y_idle_turns=int(d.get("y_idle_turns", 0)),
            override=bool(d.get("override", True)),
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

        # Manual override: axes are pinned by an operator. Record what the
        # auto system saw (last signals + a debug line tagged override) but do
        # NOT transition x/y. This keeps the instrumentation honest about what
        # would have happened while the human is driving.
        if s.override:
            sx_obs = _classify_sign(x_signal, s.x)
            sy_obs = _classify_sign(y_signal, s.y)
            s.last_update_ts = time.time()
            s.last_x_signal = float(x_signal)
            s.last_y_signal = float(y_signal)
            _save(s)
            globals()["_state"] = s
            _debug_log({
                "ts": s.last_update_ts,
                "raw_x": float(x_signal),
                "raw_y": float(y_signal),
                "classified_sx": sx_obs,
                "classified_sy": sy_obs,
                "new_x": s.x,
                "new_y": s.y,
                "moved_x": 0,
                "moved_y": 0,
                "override": True,
            })
            return {"x": s.x, "y": s.y, "moved_x": 0, "moved_y": 0,
                    "x_signal": x_signal, "y_signal": y_signal, "override": True}

        prev_x, prev_y = s.x, s.y
        sx = _classify_sign(x_signal, s.x)
        sy = _classify_sign(y_signal, s.y)
        s.x_signals.append(sx)
        s.y_signals.append(sy)
        x_buf_after = list(s.x_signals)
        y_buf_after = list(s.y_signals)
        s.x = _step(s.x_signals, s.x)
        s.y = _step(s.y_signals, s.y)

        # Bundle C phase 2: decay-to-center. Track consecutive turns where
        # the axis is at an extreme AND the classified signal contributes
        # nothing (sx/sy = 0). If we hit DECAY_TURNS in a row, force one
        # step toward 0 even with no opposing-direction signal. Resets on
        # any actual step (handled below) OR on any non-zero classified
        # signal (the axis is still receiving real input).
        decay_x = False
        decay_y = False
        if s.x != prev_x:
            s.x_idle_turns = 0
        elif s.x != 0 and sx == 0:
            s.x_idle_turns += 1
            if s.x_idle_turns >= DECAY_TURNS:
                s.x = s.x - 1 if s.x > 0 else s.x + 1  # one step toward 0
                s.x_idle_turns = 0
                s.x_signals.clear()
                decay_x = True
        else:
            # axis didnt move but signal is non-zero -- real input, no decay
            s.x_idle_turns = 0
        if s.y != prev_y:
            s.y_idle_turns = 0
        elif s.y != 0 and sy == 0:
            s.y_idle_turns += 1
            if s.y_idle_turns >= DECAY_TURNS:
                s.y = s.y - 1 if s.y > 0 else s.y + 1
                s.y_idle_turns = 0
                s.y_signals.clear()
                decay_y = True
        else:
            s.y_idle_turns = 0

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
            "x_idle_turns": s.x_idle_turns,
            "y_idle_turns": s.y_idle_turns,
            "decay_x": decay_x,
            "decay_y": decay_y,
            "move_thresh": MOVE_THRESH,
            "return_thresh": RETURN_THRESH,
            "persistence": PERSISTENCE,
            "decay_turns": DECAY_TURNS,
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


def set_override(x: int, y: int) -> dict:
    """Pin the mood to (x, y) and freeze automatic transitions.

    x, y are clamped to {-1, 0, +1}. Clears the persistence buffers so that
    if the override is later released, the auto system starts fresh from the
    pinned cell rather than acting on stale pre-override signals.
    """
    global _state
    with _lock:
        s = _load() if _state is None else _state
        s.x = max(-1, min(1, int(x)))
        s.y = max(-1, min(1, int(y)))
        s.override = True
        s.x_signals.clear()
        s.y_signals.clear()
        s.x_idle_turns = 0
        s.y_idle_turns = 0
        s.last_update_ts = time.time()
        _save(s)
        _state = s
        log.info("mood override SET to (x=%d, y=%d)", s.x, s.y)
        return {"x": s.x, "y": s.y, "override": True}


def clear_override() -> dict:
    """Release the manual override; automatic drift resumes from current cell."""
    global _state
    with _lock:
        s = _load() if _state is None else _state
        s.override = False
        s.x_signals.clear()
        s.y_signals.clear()
        s.x_idle_turns = 0
        s.y_idle_turns = 0
        _save(s)
        _state = s
        log.info("mood override CLEARED; resuming auto drift from (x=%d, y=%d)", s.x, s.y)
        return {"x": s.x, "y": s.y, "override": False}
