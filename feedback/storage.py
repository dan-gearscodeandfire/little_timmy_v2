"""Atomic JSONL append for feedback events."""

import json
import os
import time
from pathlib import Path

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
