"""Atomic JSONL append for feedback events."""

import json
import os
import time
from pathlib import Path

PERSONA_TUNING_DIR = Path(os.path.expanduser("~/little_timmy/persona_tuning"))

INBOX_PATH = Path(os.path.expanduser("~/little_timmy/feedback_inbox.jsonl"))


def append_event(entry: dict) -> str:
    """Append one feedback event to the JSONL inbox. Returns the event id.

    Uses fsync + rename-style atomicity is overkill for an append-only log;
    a single write() with O_APPEND is atomic on POSIX for buffers under
    PIPE_BUF and our entries are well under 4 KiB. We open in 'a' which
    sets O_APPEND and the kernel serializes concurrent appends.
    """
    INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "id" not in entry:
        entry["id"] = f"{int(entry.get('ts', time.time()) * 1000)}"
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(INBOX_PATH, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    return entry["id"]


def read_events(since_ts: float | None = None, limit: int = 500) -> list[dict]:
    """Read events, optionally filtered to ts > since_ts. Returns newest last."""
    if not INBOX_PATH.exists():
        return []
    out: list[dict] = []
    with open(INBOX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_ts is not None and entry.get("ts", 0) <= since_ts:
                continue
            out.append(entry)
    return out[-limit:]


def write_persona_tuning_negative(entry: dict) -> Path:
    """Mirror of _check_compliment's positive-example shape but for the
    LoRA negative bin: {timestamp, penultimate_user, system_prompt,
    flag_reason, response, source}. Filename: example_neg_<ts>.json so
    sort order interleaves with positive examples (example_<ts>.json).
    """
    PERSONA_TUNING_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(entry.get("timestamp", time.time()))
    path = PERSONA_TUNING_DIR / f"example_neg_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return path
