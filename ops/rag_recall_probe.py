#!/usr/bin/env python3
"""Offline RAG recall probe for the always-on memory channel (2026-08-17).

Answers one question with numbers instead of vibes: when Timmy spends ~275
tokens per turn on RECALLED FROM PAST CONVERSATIONS, does the answer actually
show up in there, and at what rank?

WHY THIS EXISTS
  Retrieval quality and reply quality are different failures with the same
  symptom ("Timmy didn't remember"). This probe isolates the READ PATH: no
  brain call, no TTS, no audio, no writes. If recall@k is fine here and the
  live answer was still wrong, the bug is downstream (persona, register,
  stranger-path refusal, introductions doorway) — go look there, not at the
  ranker. ops/rag_acoustic_probe.py covers the end-to-end half.

GROUND TRUTH IS THE CORPUS ITSELF
  Probes are generated FROM `propositions`: sample a row, ask the memory-tier
  LLM to write the question that row answers, then check whether that exact
  row comes back in top-k. The answer key is a primary key, not a fuzzy string
  match, and it regenerates itself as the corpus grows — no hand-maintained
  fixture to rot. Question generation runs on LLM_MEMORY_URL (:8084), never on
  the conversation slot (:8083), so building a probe set cannot evict Timmy's
  KV cache mid-conversation.

CONTROLS ARE THE POINT
  Half the value is the control probes — questions whose answers are nowhere in
  the corpus. Retrieval always returns its top-k, so a control still comes back
  with 5 confident-looking lines. Comparing the score distributions of hits vs
  controls is how you find out whether ANY threshold could gate relevance.
  As of 2026-08-17 the answer is no: controls outscore genuine rank-1 hits.

USAGE
  # build a probe set (LLM calls, ~2s each; cached to disk)
  .venv/bin/python ops/rag_recall_probe.py build --n 30

  # score the read path against it (fast, no LLM)
  .venv/bin/python ops/rag_recall_probe.py run
  .venv/bin/python ops/rag_recall_probe.py run --top-k 3   # what would K=3 cost?

Exit 0 = recall at or above --min-recall. Exit 1 = below (CI-usable).
"""
import argparse
import asyncio
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULT_SET = os.path.join(REPO, "data", "rag_probes.json")

# Controls: plausible-sounding, corpus-absent. Deliberately shaped like real
# questions (possessive + concrete noun) so they exercise the same FTS/trigram
# channels a genuine question would -- a control that looks nothing like a
# question would take an easier path and flatter the scores.
CONTROLS = [
    "What is the name of Dan's boat?",
    "Which airport did Dan fly out of?",
    "What did Dan eat for breakfast on Tuesday?",
    "How many years did Dan spend in the navy?",
    "What colour is Dan's motorcycle?",
    "Which hospital was Dan born in?",
]

QUESTION_PROMPT = """You are writing a test question for a memory-retrieval system.

Below is a single fact recorded from a past conversation. Write the ONE natural
spoken question a person would ask that this fact answers.

RULES
- Output ONLY the question. No preamble, no quotes, no explanation.
- Use the names as they appear. Do not invent details.
- Ask it the way someone would say it out loud, in one sentence.
- Do NOT quote the fact back verbatim -- ask about it.

FACT: {text}

QUESTION:"""


def _clean_question(raw: str) -> str | None:
    """First non-empty line, stripped of quoting/labels. None if unusable."""
    if not raw:
        return None
    for line in raw.strip().splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if line.lower().startswith("question:"):
            line = line.split(":", 1)[1].strip()
        # A model that ignored the format and explained itself gives a long
        # line with no question mark -- reject rather than probe with prose.
        if line and "?" in line and len(line) < 200:
            return line
    return None


async def _sample_propositions(n: int) -> list[dict]:
    """Age-stratified sample. Recency decay is part of the ranker, so a probe
    set drawn only from this week's rows would measure the easy case and miss
    the failure the decay was tuned to avoid."""
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, text, span_end FROM propositions "
            "WHERE embedding IS NOT NULL AND length(text) >= 40")
    rows = [dict(r) for r in rows]
    if not rows:
        return []
    rows.sort(key=lambda r: r["span_end"])
    # Even spread across the age range, deterministic under a fixed seed.
    buckets, per = 5, max(1, n // 5)
    size = max(1, len(rows) // buckets)
    picked, rng = [], random.Random(1729)
    for b in range(buckets):
        chunk = rows[b * size: (b + 1) * size] if b < buckets - 1 else rows[b * size:]
        if chunk:
            picked += rng.sample(chunk, min(per, len(chunk)))
    return picked[:n]


async def cmd_build(args):
    import config
    from llm.client import generate_memory
    print(f"[build] sampling {args.n} propositions (age-stratified)…")
    props = await _sample_propositions(args.n)
    if not props:
        print("[build] no propositions with embeddings — nothing to probe.")
        return 1
    print(f"[build] generating questions on {config.LLM_MEMORY_URL} "
          f"(NOT the conversation slot)…")
    probes, skipped = [], 0
    for i, p in enumerate(props, 1):
        try:
            raw = await generate_memory(
                QUESTION_PROMPT.format(text=p["text"]),
                thinking=False, temperature=0.3)
        except Exception as e:                     # noqa: BLE001 - probe build is best-effort
            print(f"  [{i}/{len(props)}] LLM error, skipping: {e}")
            skipped += 1
            continue
        q = _clean_question(raw)
        if not q:
            skipped += 1
            print(f"  [{i}/{len(props)}] unusable question, skipping: {raw[:60]!r}")
            continue
        probes.append({"question": q, "prop_id": p["id"], "answer": p["text"],
                       "span_end": p["span_end"].isoformat()})
        print(f"  [{i}/{len(props)}] {q}")
    out = {"generated_from": "propositions", "n": len(probes),
           "skipped": skipped, "controls": CONTROLS, "probes": probes}
    os.makedirs(os.path.dirname(args.set), exist_ok=True)
    with open(args.set, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[build] wrote {len(probes)} probes (+{len(CONTROLS)} controls) "
          f"-> {args.set}   ({skipped} skipped)")
    return 0


async def _retrieve(question: str, top_k: int):
    """Exactly what a live turn runs for the always-on channel."""
    from conversation.turn import _retrieve_episodes_as_memories
    return await _retrieve_episodes_as_memories(question, top_k, [])


def _pctl(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


async def cmd_run(args):
    import config
    if not os.path.exists(args.set):
        print(f"[run] no probe set at {args.set} — run `build` first.")
        return 1
    data = json.load(open(args.set))
    probes, controls = data["probes"], data.get("controls", CONTROLS)
    top_k = args.top_k or config.RETRIEVAL_TOP_K
    print(f"[run] {len(probes)} probes + {len(controls)} controls | top_k={top_k} | "
          f"episodic_always_on={config.EPISODIC_ALWAYS_ON_RETRIEVAL}\n")

    ranks, hit_scores, rows = [], [], []
    for p in probes:
        mems = await _retrieve(p["question"], top_k)
        texts = [(m.content or "").strip() for m in mems]
        target = p["answer"].strip()
        rank = next((i + 1 for i, t in enumerate(texts) if t == target), None)
        top = max([getattr(m, "score", 0) or 0 for m in mems], default=0.0)
        ranks.append(rank)
        if rank:
            hit_scores.append(top)
        rows.append((p["question"], rank, top, texts[0][:60] if texts else ""))

    ctl_scores, ctl_filled = [], 0
    for q in controls:
        mems = await _retrieve(q, top_k)
        ctl_filled += bool(mems)
        ctl_scores.append(max([getattr(m, "score", 0) or 0 for m in mems], default=0.0))

    if args.verbose:
        print(f"  {'rank':<6}{'score':<9}probe")
        print("  " + "-" * 86)
        for q, rank, top, _ in rows:
            print(f"  {str(rank or 'MISS'):<6}{top:<9.4f}{q[:70]}")
        print()

    n = len(ranks)
    got = [r for r in ranks if r]
    at1 = sum(1 for r in got if r <= 1)
    at3 = sum(1 for r in got if r <= 3)
    print(f"  recall@1 : {at1}/{n}  ({100 * at1 / n:.0f}%)")
    print(f"  recall@3 : {at3}/{n}  ({100 * at3 / n:.0f}%)")
    print(f"  recall@{top_k} : {len(got)}/{n}  ({100 * len(got) / n:.0f}%)")
    hist = {}
    for r in got:
        hist[r] = hist.get(r, 0) + 1
    print(f"  rank histogram : "
          + ", ".join(f"r{k}={v}" for k, v in sorted(hist.items())) or "  (none)")
    tail = sum(v for k, v in hist.items() if k > 3)
    print(f"  answers found ONLY at rank 4-5 : {tail}"
          + ("   <- these are what K=3 would cost you" if tail else
             "   <- K=3 would lose nothing on this set"))

    print(f"\n  controls filled a block : {ctl_filled}/{len(controls)}"
          "   (retrieval never abstains)")
    if hit_scores and ctl_scores:
        worst_hit, best_ctl = min(hit_scores), max(ctl_scores)
        print(f"  true-hit score   p10={_pctl(hit_scores, 10):.4f} "
              f"med={_pctl(hit_scores, 50):.4f} min={worst_hit:.4f}")
        print(f"  control score    med={_pctl(ctl_scores, 50):.4f} max={best_ctl:.4f}")
        if best_ctl >= worst_hit:
            print(f"  VERDICT: score CANNOT gate relevance — the best control "
                  f"({best_ctl:.4f}) outscores the worst true hit ({worst_hit:.4f}).\n"
                  f"           Do not build a floor on it. Use rank-1 margin, or "
                  f"check only the top hit.")
        else:
            print(f"  VERDICT: a floor between {best_ctl:.4f} and {worst_hit:.4f} "
                  f"separates hits from controls ON THIS SET — re-run before trusting it.")

    recall = len(got) / n if n else 0.0
    if recall < args.min_recall:
        print(f"\n[FAIL] recall@{top_k} {recall:.2f} < --min-recall {args.min_recall}")
        return 1
    print(f"\n[OK] recall@{top_k} {recall:.2f} >= {args.min_recall}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate a probe set from the live corpus")
    b.add_argument("--n", type=int, default=30)
    b.add_argument("--set", default=DEFAULT_SET)
    r = sub.add_parser("run", help="score the read path against a probe set")
    r.add_argument("--set", default=DEFAULT_SET)
    r.add_argument("--top-k", type=int, default=None, help="override RETRIEVAL_TOP_K")
    r.add_argument("--min-recall", type=float, default=0.8)
    r.add_argument("-v", "--verbose", action="store_true", help="per-probe rows")
    args = ap.parse_args()
    fn = cmd_build if args.cmd == "build" else cmd_run
    sys.exit(asyncio.run(fn(args)))


if __name__ == "__main__":
    main()
