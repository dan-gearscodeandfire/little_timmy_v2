"""Controlled microbenchmark: does running the coref resolver (:8093) CONCURRENTLY
with the tool-call classifier (:8092) inflate resolve latency on this single
Strix Halo GPU?

Settles the speculative-coref question without the live-mic confounds (cold
start, brain/vision background load, varying utterance length) that muddied the
2026-06-22 A/B. Both servers are warmed first; then for each fixed query we
measure :8093 resolve latency ISOLATED vs CONCURRENT-with-a-:8092-classify,
interleaved per rep so any drifting background load averages across both modes.

Dan's hypothesis: separate llama.cpp processes (independent contexts/command
streams) interleave fine, unlike in-server `-np` slots -> CONCURRENT ~= ISOLATED.
Contention hypothesis: the Vulkan driver serializes cross-process compute ->
CONCURRENT >> ISOLATED.

Run (from ~/little_timmy, idle line only):
    .venv/bin/python -m ops.coref_contention_bench [reps]
"""

import asyncio
import sys
import time
import statistics

from llm import client
from conversation import tool_router

# Fixed deictic follow-ups + a short 2-turn context, similar rewrite lengths so
# decode-token count doesn't differ between modes (the resolver is decode-bound).
QUERIES = [
    ("What did she do in the backyard?",
     "user: I used to have a dog named Molly.\nassistant: Nice."),
    ("Why do you think that about her name?",
     "user: My sister had a parakeet named Mrs. Lopez.\nassistant: Got it."),
    ("Who am I talking about?",
     "user: I had a dog named Fred who sang at sirens.\nassistant: Fred."),
    ("What does he do for a living?",
     "user: My friend Voss just moved to town.\nassistant: Okay."),
]


async def _timed_resolve(q, ctx):
    t = time.perf_counter()
    r = await client.resolve_query(q, ctx)
    return (time.perf_counter() - t) * 1000.0, r


async def _timed_classify(q):
    t = time.perf_counter()
    await tool_router.classify_intent(q)
    return (time.perf_counter() - t) * 1000.0


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    p90 = xs[min(n - 1, int(round(n * 0.9)) - 1)] if n else 0
    return (f"n={n:<3} median={statistics.median(xs):6.0f}ms  mean={statistics.mean(xs):6.0f}ms  "
            f"p90={p90:6.0f}ms  min={xs[0]:5.0f}  max={xs[-1]:5.0f}")


async def main(reps):
    print(f"warming both servers ({len(QUERIES)} queries x2)...")
    for q, ctx in QUERIES:
        await client.resolve_query(q, ctx)
        await tool_router.classify_intent(q)

    iso, conc, conc_class = [], [], []
    for rep in range(reps):
        for q, ctx in QUERIES:
            # ISOLATED: resolve alone (nothing else fired from here).
            ms, _ = await _timed_resolve(q, ctx)
            iso.append(ms)
            # CONCURRENT: resolve || classify, both launched together. We record
            # the resolve's own wall-clock -- does it stretch while :8092 runs?
            (rms, _), cms = await asyncio.gather(_timed_resolve(q, ctx), _timed_classify(q))
            conc.append(rms)
            conc_class.append(cms)
    print(f"\nreps={reps}  (samples = reps x {len(QUERIES)} queries)\n")
    print("RESOLVE  isolated          :", _stats(iso))
    print("RESOLVE  concurrent w/8092 :", _stats(conc))
    print("classifier (while resolving):", _stats(conc_class))
    im, cm = statistics.median(iso), statistics.median(conc)
    print(f"\n=> concurrent/isolated resolve median ratio = {cm/im:.2f}x")
    print("   ~1.0x  -> separate servers parallelize (Dan's theory); speculative coref is safe")
    print("   >>1.0x -> cross-process GPU contention; serial wins (keep it OFF)")


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    asyncio.run(main(reps))
