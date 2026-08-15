#!/usr/bin/env python3
"""Backfill DELIVERY x SUBSTANCE labels onto historical feedback rows.

Approved by Dan 2026-08-15 00:50. Rationale in feedback/axes.py -- a
single-axis thumbs-up cannot distinguish "funny AND right" from "funny BUT
wrong", and the second cell is the one that turns an amusing defect into a
permanent trait.

WHAT IT GRADES
  feedback_inbox.jsonl              288 rows, all critique
  persona_tuning/flagged.jsonl      400 rows: 270 bad + 130 good
The 231 bad/verbal_meta_feedback rows in flagged.jsonl mirror inbox rows one
for one, so events are keyed by rounded `ts` and graded ONCE. The 130 `good`
rows are the point of the exercise: they are the only place a praised-but-wrong
response can be sitting today.

WHAT IT WRITES
  feedback_axes.jsonl -- a SIDECAR, keyed by event ts. The source files are
  never rewritten. That keeps the operation reversible (delete the sidecar) and
  means a crash halfway through cannot corrupt 688 rows of real history, which
  is the failure mode that actually matters here. Re-running skips keys already
  present, so it resumes.

WHY THE THROTTLE
  Grading calls go to :8084. `generate_memory`'s `_wait_for_conversation_idle`
  gate is IN-PROCESS state, so it does nothing from a separate script -- the
  gate would report idle no matter how hard Dan was talking. Two 35B servers on
  one GPU halve each other cross-process (decode 0.50x, prefill 0.71x --
  `feedback_two_35b_servers_halve_cross_process`), so an ungated batch here
  would slow live replies exactly the way the mail-triage run did at 20:17.
  Hence: poll LT's own conversation endpoint and pause while a turn is recent.

  python3 ops/backfill_feedback_axes.py --dry-run          # counts, no calls
  python3 ops/backfill_feedback_axes.py --limit 20         # small live batch
  python3 ops/backfill_feedback_axes.py                    # full run
  python3 ops/backfill_feedback_axes.py --ignore-live      # no backoff (idle box only)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/little_timmy"))

import httpx

from feedback import axes as _axes

ROOT = Path(os.path.expanduser("~/little_timmy"))
INBOX = ROOT / "feedback_inbox.jsonl"
FLAGGED = ROOT / "persona_tuning" / "flagged.jsonl"
SIDECAR = ROOT / "feedback_axes.jsonl"

CONVO_URL = "http://localhost:8893/api/conversation"
IDLE_SECONDS = 45.0      # how quiet the conversation must be before we grade
POLL_SECONDS = 15.0      # how often to re-check while waiting
PACE_SECONDS = 1.0       # breather between gradings even when idle


def _key(ts) -> str:
    """Stable event key. Rounded to the millisecond because the same event is
    written to both files with float timestamps that round-trip through JSON."""
    try:
        return f"{float(ts):.3f}"
    except (TypeError, ValueError):
        return str(ts)


def load_rows() -> dict:
    """Collect unique gradable events from both files, newest first.

    Field names differ between the two files for the same concept, so they are
    normalised here rather than in the grader:
        inbox    prev_user / prev_assistant / feedback_text
        flagged  user_prompt / response     / comment
    """
    events: dict[str, dict] = {}

    def add(ts, prev_user, prev_assistant, reaction, kind, src):
        if not prev_assistant:
            return                      # nothing to grade
        k = _key(ts)
        if k in events:
            events[k].setdefault("srcs", []).append(src)
            return
        events[k] = {
            "key": k, "ts": ts, "kind": kind, "srcs": [src],
            "prev_user": prev_user or "",
            "prev_assistant": prev_assistant or "",
            "reaction": reaction or "",
        }

    if INBOX.exists():
        for line in INBOX.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            add(d.get("ts"), d.get("prev_user"), d.get("prev_assistant"),
                d.get("feedback_text"), "bad", "inbox")

    if FLAGGED.exists():
        for line in FLAGGED.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            add(d.get("ts"), d.get("user_prompt"), d.get("response"),
                d.get("comment"), d.get("kind") or "bad", "flagged")

    return events


def load_done() -> set:
    if not SIDECAR.exists():
        return set()
    done = set()
    for line in SIDECAR.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["key"])
        except Exception:
            continue
    return done


def append_label(rec: dict) -> None:
    with open(SIDECAR, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def seconds_since_last_turn(client: httpx.AsyncClient) -> float:
    """Age of LT's most recent conversation turn. Returns +inf when LT is
    unreachable or has no history -- an unreachable LT is not one we can stall,
    so grading may proceed."""
    try:
        r = await client.get(CONVO_URL, timeout=5.0)
        r.raise_for_status()
        hot = r.json().get("hot") or []
        if not hot:
            return float("inf")
        return time.time() - float(hot[-1].get("timestamp", 0))
    except Exception:
        return float("inf")


async def wait_until_quiet(client: httpx.AsyncClient, ignore_live: bool) -> None:
    if ignore_live:
        return
    announced = False
    while True:
        age = await seconds_since_last_turn(client)
        if age >= IDLE_SECONDS:
            if announced:
                print(f"    ...conversation idle {age:.0f}s, resuming", flush=True)
            return
        if not announced:
            print(f"    conversation active ({age:.0f}s since last turn) -- "
                  f"pausing so we do not slow Timmy", flush=True)
            announced = True
        await asyncio.sleep(POLL_SECONDS)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ignore-live", action="store_true",
                    help="skip the conversation-activity backoff")
    args = ap.parse_args()

    events = load_rows()
    done = load_done()
    todo = [e for k, e in events.items() if k not in done]
    todo.sort(key=lambda e: float(e["ts"]), reverse=True)

    good = sum(1 for e in events.values() if e["kind"] == "good")
    print(f"gradable events : {len(events)}  (good={good} bad={len(events)-good})")
    print(f"already labelled: {len(done)}")
    print(f"to grade        : {len(todo)}" + (f" (limited to {args.limit})" if args.limit else ""))
    if args.dry_run:
        for e in todo[:5]:
            print(f"  - {e['kind']:4} {e['key']} {e['prev_assistant'][:60]!r}")
        return 0

    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do")
        return 0

    counts: dict[str, int] = {}
    async with httpx.AsyncClient() as client:
        for i, e in enumerate(todo, 1):
            await wait_until_quiet(client, args.ignore_live)
            graded = await _axes.label_axes(
                e["prev_user"], e["prev_assistant"], e["reaction"])
            rec = {
                "key": e["key"], "ts": e["ts"], "kind": e["kind"],
                "srcs": sorted(set(e["srcs"])),
                "delivery": graded["delivery"],
                "substance": graded["substance"],
                "axes_source": graded["axes_source"],
                "graded_at": time.time(),
            }
            append_label(rec)
            cell = f"{graded['delivery']}/{graded['substance']}"
            counts[cell] = counts.get(cell, 0) + 1
            print(f"[{i}/{len(todo)}] {e['kind']:4} {cell:15} "
                  f"{e['prev_assistant'][:55]!r}", flush=True)
            await asyncio.sleep(PACE_SECONDS)

    print("\n--- delivery/substance distribution ---")
    for cell, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cell:16} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
