#!/usr/bin/env python3
"""End-to-end RAG probe: retrieved vs. actually USED (2026-08-17).

ops/rag_recall_probe.py proves the answer was put in front of the brain.
This proves what the brain did with it — the two are not the same, and the
gap between them is where the interesting failures live.

WHAT THIS CATCHES THAT THE OFFLINE PROBE STRUCTURALLY CANNOT
  Measured on the 2026-08-17 run, all four in one six-turn session:
    - the introductions doorway eating a turn (a greeting, not an answer)
    - the stranger path REFUSING to answer from memory it had already retrieved
      ("I don't know who you are... ask Dan yourself" — for a question whose
      answer sits at rank 1)
    - both retrieval channels firing at once (always-on propositions AND the
      recall_semantic tool), with items duplicated verbatim across the two
    - real llm_ft, including the cache-miss spikes a new speaker triggers

  Every one of those turns paid the full retrieval payload and spent none of it.
  A read-path probe scores them as perfect.

HOW IT SCORES
  For each probe it runs BOTH layers and reports them side by side:
    retrieved? -> did the target proposition come back in top-k (exact row match)
    used?      -> did the reply carry the answer's distinctive content words
  The diagnostic column is the combination. retrieved+used = working.
  retrieved+NOT used = tokens bought and thrown away: look downstream at
  persona/register/stranger-path, NOT at the ranker. NOT retrieved+used = the
  model answered from parametric knowledge or history, not from memory.

SAFETY
  Synthesized speech runs the full production path and writes REAL rows, so
  this ALWAYS brackets the run with ops/synthtest_guard.py: fresh snapshot
  before, cleanup + integrity verify after, even if the driver dies. A stale
  snapshot cannot restore — this takes a new one every time, never reuses.

USAGE
  .venv/bin/python ops/rag_acoustic_probe.py --n 4 --controls 2
  .venv/bin/python ops/rag_acoustic_probe.py --n 4 --voice en_US-amy-medium

NOTE ON --voice: en_US-kristin-medium is the couple's-therapist voice, and it
comes back as `unknown_N` — but NOT because a print went stale. As of 2026-08-18
there is **no `couples_therapist_wespeaker.npy` at all**; the `speakers` row
(id 6, 2026-06-28) is a name record with no voiceprint behind it. Only two live
voiceprints exist on this box: `dan` and `devon`. So `known_best_dist=0.916` was
measured against those two, and every run of this harness exercises the STRANGER
path. That is useful on purpose (it is the guest case), but do not read the
results as the known-speaker case.
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

GUARD = os.path.join(REPO, "ops", "synthtest_guard.py")
DRIVER = os.path.join(REPO, "ops", "acoustic_convo_driver.py")
PROBE_SET = os.path.join(REPO, "data", "rag_probes.json")

_STOP = {
    "about", "after", "again", "their", "there", "these", "those", "which",
    "while", "would", "could", "should", "being", "because", "before", "where",
    "what", "when", "that", "this", "with", "from", "have", "into", "than",
    "then", "they", "them", "were", "your", "just", "like", "over", "also",
    "does", "said", "says", "timmy", "dans",
}


def _content_words(text: str) -> set[str]:
    """Distinctive tokens for overlap scoring. 'timmy' is a stop word here —
    it appears in nearly every proposition, so counting it would score any
    reply mentioning himself as a hit."""
    return {w for w in re.findall(r"[a-z0-9']{4,}", (text or "").lower())
            if w not in _STOP}


def _payload_tokens(answer: str, question: str) -> set[str]:
    """The tokens ONLY memory could have supplied: in the answer, absent from
    the question.

    This is the whole trick. Scoring on raw overlap between reply and answer
    fails badly here because the persona caps replies at 1-2 short sentences:
    "Devon named Ennard." shares almost nothing with the 18-word proposition it
    came from, yet it is a perfect answer. Worse, a reply that merely parrots
    the question ("your latency dropped") would score as a hit on the shared
    words. Subtracting the question isolates the added information — Ennard,
    2.69 — which is exactly what retrieval was supposed to contribute.

    Numbers count even when short (2.69, 6.28); words need 4+ chars.
    """
    q = _content_words(question) | {w.lower() for w in re.findall(r"[\w.']+", question)}
    nums = set(re.findall(r"\d+(?:\.\d+)?", answer or ""))
    words = _content_words(answer)
    return (nums | words) - q


def _used(reply: str, answer: str, question: str = "") -> tuple[bool, float]:
    """Did the reply carry information that only the retrieved answer had?

    Returns (used, fraction_of_payload_tokens_present). One payload token is
    enough — a correct terse answer often supplies exactly one ("Ennard").
    Falls back to whole-answer overlap when the question already contains
    everything distinctive (nothing left to subtract).
    """
    payload = _payload_tokens(answer, question)
    r = _content_words(reply) | {n for n in re.findall(r"\d+(?:\.\d+)?", reply or "")}
    if not payload:
        a = _content_words(answer)
        frac = len(a & r) / len(a) if a else 0.0
        return frac >= 0.34, frac
    hits = payload & r
    return bool(hits), len(hits) / len(payload)


def _run_guard(action: str, baseline: str) -> int:
    r = subprocess.run([sys.executable, GUARD, action, baseline],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    return r.returncode


async def _retrieval_for(questions: list[str], top_k: int) -> dict:
    """Same read path a live turn runs, so the two columns are comparable."""
    from conversation.turn import _retrieve_episodes_as_memories
    out = {}
    for q in questions:
        mems = await _retrieve_episodes_as_memories(q, top_k, [])
        out[q] = [(m.content or "").strip() for m in mems]
    return out


def _report(turns, probes, retrieved, guard_rc):
    """Score + print one run. Shared by the live path and --rescore."""
    print(f"\nretrieved vs used\n")
    hdr = f"  {'ret':<5}{'used':<6}{'ovl':<7}{'route':<17}probe / reply"
    print(hdr)
    print("  " + "-" * (len(hdr) + 34))
    paid_unspent = 0
    for i, t in enumerate(turns):
        q = t.get("said", "")
        reply = t.get("reply") or ""
        is_control = i >= len(probes)
        answer = "" if is_control else probes[i]["answer"]
        got = retrieved.get(q, [])
        was_ret = (not is_control) and answer.strip() in got
        used, frac = (False, 0.0) if is_control else _used(reply, answer, q)
        if was_ret and not used:
            paid_unspent += 1
        if is_control:
            # A control is right when he declines. Cheap check, but the failure
            # it guards against (confident invention) is loud and obvious.
            declined = bool(re.search(r"\bdon'?t (know|remember)|no idea|not sure",
                                      reply, re.I))
            mark = "OK" if declined else "INVENTED?"
            print(f"  {'—':<5}{mark:<6}{'—':<7}{str(t.get('route')):<17}"
                  f"{q[:56]}")
        else:
            print(f"  {'YES' if was_ret else 'no':<5}{'YES' if used else 'no':<6}"
                  f"{frac:<7.2f}{str(t.get('route')):<17}{q[:56]}")
        # Surface the transcript whenever it drifted from what was spoken. Both
        # "retrieved but not used" turns in the 2026-08-17 run were STT mangles
        # ("shroud" -> "shrub"), not ranker or persona failures — without this
        # column they read as RAG misses and send you debugging the wrong layer.
        heard = (t.get("heard") or "").strip()
        if heard and _content_words(heard) != _content_words(q):
            print(f"  {'':<35}   heard: {heard[:88]!r}")
        print(f"  {'':<35}-> {reply[:96]!r}")

    scored = [t for i, t in enumerate(turns) if i < len(probes)]
    both = sum(1 for i, t in enumerate(scored)
               if probes[i]["answer"].strip() in retrieved.get(t.get("said", ""), [])
               and _used(t.get("reply") or "", probes[i]["answer"], t.get("said", ""))[0])
    print(f"\n  turns scored          : {len(scored)}")
    print(f"  retrieved AND used    : "
          f"{both}")
    print(f"  retrieved, NOT used   : {paid_unspent}"
          + ("   <- tokens bought and thrown away"
             if paid_unspent else ""))
    if paid_unspent:
        print("     check the `heard:` lines FIRST — a mangled transcript is an STT\n"
              "     failure, not a RAG one. Only turns that were heard correctly and\n"
              "     still ignored the memory point downstream (persona, register,\n"
              "     stranger path, introductions doorway).")
    if guard_rc != 0:
        print(f"\n  WARNING: guard cleanup returned {guard_rc} — verify the store by hand.")
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=PROBE_SET)
    ap.add_argument("--n", type=int, default=4, help="answerable probes to speak")
    ap.add_argument("--controls", type=int, default=2, help="corpus-absent probes")
    ap.add_argument("--voice", default="en_US-kristin-medium")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--reply-window", type=float, default=18.0)
    ap.add_argument("--rescore", metavar="RESULTS_JSON",
                    help="re-score a previous run's results; no audio, no writes")
    ap.add_argument("--keep-baseline", action="store_true",
                    help="do not delete the snapshot file (for manual verify)")
    args = ap.parse_args()

    if not os.path.exists(args.set):
        print(f"no probe set at {args.set} — run `rag_recall_probe.py build` first.")
        return 1
    data = json.load(open(args.set))
    probes = data["probes"][:args.n]
    controls = data.get("controls", [])[:args.controls]
    if not probes:
        print("probe set is empty.")
        return 1

    import config
    top_k = args.top_k or config.RETRIEVAL_TOP_K
    questions = [p["question"] for p in probes] + list(controls)

    print("[1/5] offline retrieval for the same questions (read path only)…")
    retrieved = asyncio.run(_retrieval_for(questions, top_k))

    # --rescore: the metric is the part most likely to need tuning (a terse
    # persona breaks naive overlap scoring), and re-speaking six turns to try a
    # threshold is both slow and noisy. Replay a saved run instead: no audio,
    # no guard, no writes.
    if args.rescore:
        if not os.path.exists(args.rescore):
            print(f"no results file at {args.rescore}")
            return 1
        turns = json.load(open(args.rescore)).get("results", [])
        return _report(turns, probes, retrieved, guard_rc=0)

    baseline = tempfile.mktemp(prefix="lt_ragprobe_", suffix=".json")
    print(f"\n[2/5] snapshot -> {baseline}")
    if _run_guard("snapshot", baseline) != 0:
        print("snapshot failed — refusing to speak into a live memory store.")
        return 1

    scenario = [{"say": q, "expect": (p["answer"][:70] if p else "CONTROL — should admit not knowing")}
                for q, p in zip(questions, list(probes) + [None] * len(controls))]
    scen_path = tempfile.mktemp(prefix="lt_ragscen_", suffix=".json")
    with open(scen_path, "w") as f:
        json.dump(scenario, f)

    results_path = tempfile.mktemp(prefix="lt_ragres_", suffix=".json")
    print(f"\n[3/5] speaking {len(scenario)} turns as {args.voice}…\n")
    rc = 0
    try:
        rc = subprocess.run(
            [sys.executable, DRIVER, "--scenario", scen_path, "--voice", args.voice,
             "--reply-window", str(args.reply_window), "--out", results_path]
        ).returncode
    finally:
        # Cleanup runs even if the driver crashed or was interrupted; leaving
        # synthetic rows in the store is the one outcome that is not allowed.
        print(f"\n[4/5] cleanup + verify")
        guard_rc = _run_guard("cleanup", baseline)
        if not args.keep_baseline and os.path.exists(baseline):
            os.unlink(baseline)

    if rc != 0 or not os.path.exists(results_path):
        print(f"\ndriver exited {rc} — no results to score.")
        return 1

    turns = json.load(open(results_path)).get("results", [])
    print(f"\n[5/5] retrieved vs used")
    return _report(turns, probes, retrieved, guard_rc)


if __name__ == "__main__":
    sys.exit(main())
