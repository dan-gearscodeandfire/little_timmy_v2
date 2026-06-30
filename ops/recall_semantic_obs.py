#!/usr/bin/env python3
"""Summarize recall_semantic retrieval quality for Phase-2 tuning (2026-06-30).

Parses the `[recall_semantic-obs]` journal lines emitted by
conversation/tool_router._resolve_semantic_block (one per retrieved episode,
with pre-decay base + post-decay scores, age, snippet) and reports score/age
distributions split by rank, so the distance floor (when does a hit stop being
relevant?) and the 30d half-life (EPISODE_DECAY_HALFLIFE_S) can be tuned against
real traffic instead of guessed.

USAGE (reads stdin):
  sudo journalctl -u little-timmy.service --since "7 days ago" -o cat \
    | .venv/bin/python ops/recall_semantic_obs.py
"""
import re
import sys
import statistics as st

LINE = re.compile(
    r"\[recall_semantic-obs\] q=(?P<q>.*?) r(?P<rank>\d+) ep(?P<id>\d+) "
    r"base=(?P<base>[-\d.]+) decayed=(?P<dec>[-\d.]+) age=(?P<age>[-\d.]+)h \| (?P<snip>.*)"
)


def pctl(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main() -> None:
    rows = []
    for line in sys.stdin:
        m = LINE.search(line)
        if m:
            rows.append(m.groupdict())
    if not rows:
        print("no [recall_semantic-obs] lines found on stdin.")
        return

    queries = {r["q"] for r in rows}
    top = [r for r in rows if r["rank"] == "1"]
    base_all = [float(r["base"]) for r in rows]
    dec_all = [float(r["dec"]) for r in rows]
    base_top = [float(r["base"]) for r in top]
    age_top = [float(r["age"]) for r in top if float(r["age"]) >= 0]

    print(f"recall_semantic hits: {len(queries)} queries, {len(rows)} episodes returned "
          f"({len(rows)/max(1,len(queries)):.1f} avg/hit)\n")

    def dist(name, xs):
        if not xs:
            print(f"  {name}: (none)")
            return
        print(f"  {name}: min={min(xs):.4f} p10={pctl(xs,10):.4f} "
              f"med={st.median(xs):.4f} p90={pctl(xs,90):.4f} max={max(xs):.4f}")

    print("BASE score (pre-decay fusion):")
    dist("all returned", base_all)
    dist("top hit (r1) ", base_top)
    print("\nDECAYED score (post recency):")
    dist("all returned", dec_all)
    print("\nTOP-HIT age (hours):")
    dist("r1 age", age_top)

    # Floor-sweep: how many hits' TOP episode would survive a given base floor.
    print("\nTop-hit survival vs a hypothetical base-score floor:")
    for floor in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        kept = sum(1 for b in base_top if b >= floor)
        print(f"  floor {floor:.2f}: {kept}/{len(base_top)} top hits kept "
              f"({100*kept/max(1,len(base_top)):.0f}%)")

    print("\nWeakest 5 top hits (candidate false-positives to eyeball):")
    for r in sorted(top, key=lambda r: float(r["base"]))[:5]:
        print(f"  base={float(r['base']):.4f} dec={float(r['dec']):.4f} "
              f"age={float(r['age']):.1f}h q={r['q']} | {r['snip']}")


if __name__ == "__main__":
    main()
