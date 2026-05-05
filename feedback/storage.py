"""Atomic JSONL append for feedback events."""

import json
import os
import time
from pathlib import Path

INBOX_PATH = Path(os.path.expanduser("~/little_timmy/feedback_inbox.jsonl"))
PERSONA_TUNING_DIR = Path(os.path.expanduser("~/little_timmy/persona_tuning"))
FLAGGED_PATH = PERSONA_TUNING_DIR / "flagged.jsonl"


def append_event(entry: dict) -> str:
    """Append one feedback event to the JSONL inbox. Returns the event id."""
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
    """LoRA negative example. Mirror of _check_compliment positive shape."""
    PERSONA_TUNING_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(entry.get("timestamp", time.time()))
    path = PERSONA_TUNING_DIR / f"example_neg_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return path


def write_persona_tuning_positive(entry: dict) -> Path:
    """LoRA positive example. Same shape as _check_compliment writes,
    used by the UI thumbs-up path so verbal-praise and click-praise files
    interleave by timestamp."""
    PERSONA_TUNING_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(entry.get("timestamp", time.time()))
    path = PERSONA_TUNING_DIR / f"example_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return path


def append_flagged(kind: str, entry: dict) -> None:
    """Append one line to the consolidated good+bad running log at
    ~/little_timmy/persona_tuning/flagged.jsonl. Independent of Obsidian;
    meant for grep/jq/tail consumption.

    `kind`: "good" | "bad".
    `entry`: dict with at minimum (ts, source, speaker, user_prompt,
              response, comment, system_prompt). Extra keys preserved.
    """
    PERSONA_TUNING_DIR.mkdir(parents=True, exist_ok=True)
    ts = float(entry.get("ts", time.time()))
    line = {
        "ts": ts,
        "iso_ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        "kind": kind,
    }
    line.update(entry)
    with open(FLAGGED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
