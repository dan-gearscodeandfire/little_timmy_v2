#!/usr/bin/env python3
"""Per-turn prefill benchmark for the conversation slot (2026-08-17).

Measures the only latency lever the prompt layout actually exposes: how many
tokens the brain must re-process on each turn, and how long that takes.

WHY A SCRIPT AND NOT A ONE-OFF CURL
  The number is easy to measure wrong. Sending the SAME payload twice makes
  llama.cpp re-prefill the WHOLE prompt (it must evaluate at least one token,
  and this build drops the slot rather than reusing 3499 of 3500) — so a naive
  A/B "run it again and compare" reports a 3.9s cache miss and looks like the
  cache is broken. Production never does that: every turn appends the previous
  exchange and carries a fresh clock. This replays that real shape — N turns,
  each one strictly extending the last — which is the only way the cache-reuse
  boundary lands where it does in production.

WHAT IT READS
  Seeds from a live payload (GET :8893/api/last_payload) so history depth,
  persona size and block shape are whatever Timmy is actually running. Reports
  llama.cpp's own accounting via `timings_per_token`:
     cache_n  = tokens reused from KV        (free)
     prompt_n = tokens re-prefilled          (what you pay, every turn)
     prompt_ms= wall time for that re-prefill

USE FOR BEFORE/AFTER
  Take a baseline, change ONE thing (--cache-reuse on the server, a prompt-block
  edit, top-k), re-run, compare medians. Same seed payload both times or the
  comparison is meaningless.

USAGE
  .venv/bin/python ops/prefill_bench.py                 # 6 turns, live payload
  .venv/bin/python ops/prefill_bench.py --turns 10 --label "cache-reuse-256"
  .venv/bin/python ops/prefill_bench.py --save base.json
  .venv/bin/python ops/prefill_bench.py --compare base.json

Read-only against the brain: max_tokens=1, nothing persisted, no LT state
touched. It DOES occupy the conversation slot briefly — don't run it mid-
conversation.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
import urllib.request

BRAIN = os.getenv("TIMMY_CONVERSATION_URL", "http://127.0.0.1:8083")
TIMMY = os.getenv("TIMMY_BASE_URL", "http://127.0.0.1:8893")

# Varied so no two turns are byte-identical (see module docstring); short so the
# utterance itself is never the dominant term in what gets re-prefilled.
UTTERANCES = [
    "So what do you make of that?",
    "Alright, and then what happened?",
    "That is not what I asked you.",
    "Say more about the second part.",
    "Do you still think that was the right call?",
    "Fine. Move on.",
    "What about the other one?",
    "Give me the short version.",
    "And you are sure about that?",
    "Let's try something else.",
]
REPLIES = [
    "If you say so.",
    "It was exactly what I said it was, Dan.",
    "Obviously.",
    "You already know the answer to that.",
    "Sure. Whatever helps.",
]


def _post(path: str, obj: dict, timeout: float = 300.0) -> dict:
    r = urllib.request.urlopen(urllib.request.Request(
        BRAIN + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}), timeout=timeout)
    return json.load(r)


def _seed(path: str | None = None) -> dict:
    """A saved seed beats the live one for A/B work: the live payload changes
    between runs (history grows, blocks differ), and comparing two runs on two
    different prompts measures nothing. --seed pins it."""
    if path:
        p = json.load(open(path))
        if not p.get("messages"):
            sys.exit(f"{path} has no .messages — not a payload capture")
        return p
    try:
        with urllib.request.urlopen(TIMMY + "/api/last_payload", timeout=8) as r:
            p = json.load(r)
        if p.get("messages"):
            return p
    except Exception as e:                          # noqa: BLE001
        print(f"could not read a live payload from {TIMMY}: {e}")
    sys.exit("no seed payload — pass --seed FILE, or let little-timmy take a turn first")


def run(turns: int, seed_path: str | None = None) -> list[dict]:
    seed = _seed(seed_path)
    msgs = list(seed["messages"])
    block = seed.get("ephemeral_block") or ""
    prev_user = "[Dan]: " + (seed.get("user_text") or "hello")
    # Drop the seed's wrapped tail; we rebuild a fresh one each turn exactly as
    # prompt_builder does (context block + utterance inside the wrap).
    base = msgs[:-1]
    out = []
    for i in range(turns):
        # Clock moves every turn in production; mirror that so the block is
        # never byte-identical to the previous one.
        ctx = block.replace("PM", f"PM").replace(
            block.split("\n")[0], f"Current time: turn {i:02d}", 1) if block else ""
        tail = ("[CONTEXT]\n" + ctx + "\n[/CONTEXT]\n[UTTERANCE]\n"
                + UTTERANCES[i % len(UTTERANCES)] + "\n[/UTTERANCE]")
        msgs = base + [{"role": "user", "content": tail}]
        t0 = time.time()
        j = _post("/v1/chat/completions", {
            "messages": msgs, "max_tokens": 1, "temperature": 0.7, "stream": False,
            "timings_per_token": True,
            "chat_template_kwargs": {"enable_thinking": False}})
        wall = (time.time() - t0) * 1000
        tm = j.get("timings", {})
        total = (j.get("usage") or {}).get("prompt_tokens", 0)
        rec = {"turn": i, "total": total, "cache_n": tm.get("cache_n", 0),
               "prompt_n": tm.get("prompt_n", 0),
               "prompt_ms": round(tm.get("prompt_ms", 0.0), 1),
               "wall_ms": round(wall, 1)}
        out.append(rec)
        print(f"  turn {i:>2}  total={rec['total']:>6}  cached={rec['cache_n']:>6}  "
              f"re-prefill={rec['prompt_n']:>5} tok  {rec['prompt_ms']:>7.1f} ms")
        # Grow the conversation the way a real turn does: the wrapped tail
        # becomes a raw user turn plus Timmy's reply.
        base = base + [{"role": "user", "content": prev_user},
                       {"role": "assistant", "content": REPLIES[i % len(REPLIES)]}]
        prev_user = "[Dan]: " + UTTERANCES[i % len(UTTERANCES)]
    return out


def summarize(rows: list[dict], label: str) -> dict:
    # Turn 0 lands on whatever the slot happened to hold, so it is a cold/foreign
    # measurement, not a per-turn cost. Every steady-state figure excludes it.
    warm = rows[1:] or rows
    s = {"label": label, "turns": len(rows),
         "median_reprefill_tok": st.median(r["prompt_n"] for r in warm),
         "median_prefill_ms": st.median(r["prompt_ms"] for r in warm),
         "median_total_tok": st.median(r["total"] for r in warm),
         "max_reprefill_tok": max(r["prompt_n"] for r in warm),
         "misses": sum(1 for r in warm if r["cache_n"] == 0)}
    print(f"\n  [{label}]  warm turns={len(warm)}")
    print(f"    median re-prefill : {s['median_reprefill_tok']:.0f} tok  "
          f"({s['median_prefill_ms']:.0f} ms)")
    print(f"    median prompt     : {s['median_total_tok']:.0f} tok")
    print(f"    worst re-prefill  : {s['max_reprefill_tok']} tok")
    print(f"    full cache misses : {s['misses']}/{len(warm)}"
          + ("   <- a co-tenant is sharing the slot" if s["misses"] else ""))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seed", help="payload JSON capture; pins the prompt across runs")
    ap.add_argument("--label", default="run")
    ap.add_argument("--save", help="write summary+rows to JSON")
    ap.add_argument("--compare", help="baseline JSON to diff against")
    args = ap.parse_args()

    print(f"benchmarking {BRAIN} — {args.turns} consecutive turns\n")
    rows = run(args.turns, args.seed)
    s = summarize(rows, args.label)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"summary": s, "rows": rows}, f, indent=2)
        print(f"\n  saved -> {args.save}")

    if args.compare:
        base = json.load(open(args.compare))["summary"]
        dt = s["median_reprefill_tok"] - base["median_reprefill_tok"]
        dm = s["median_prefill_ms"] - base["median_prefill_ms"]
        print(f"\n  vs [{base['label']}]:")
        print(f"    re-prefill {base['median_reprefill_tok']:.0f} -> "
              f"{s['median_reprefill_tok']:.0f} tok   ({dt:+.0f})")
        print(f"    prefill    {base['median_prefill_ms']:.0f} -> "
              f"{s['median_prefill_ms']:.0f} ms    ({dm:+.0f} ms/turn)")
        if base["median_prefill_ms"]:
            print(f"    change     {100 * dm / base['median_prefill_ms']:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
