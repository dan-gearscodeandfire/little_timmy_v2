#!/usr/bin/env python3
"""Tier-1 route classifier probe (live :8092, NOT hermetic).

Calls conversation.tool_router.classify_intent on a labelled battery N times each
and reports the route distribution -- the right tool for validating a change to
prompts/classify_route.txt (the small Qwen3-4B classifier has sampling variance,
so a single sample lies; N samples reveal real reliability). Isolates the route
DECISION from STT/acoustic noise (unlike ops/acoustic_convo_driver.py).

Hits the classifier server (config.LLM_CLASSIFIER_URL, :8092) only -- NOT the
:8083/:8084 brain -- so it's safe to run during a live conversation. Synthetic
phrases only (no real PII -- see feedback_no_pii_in_test_prompts).

Run: .venv/bin/python ops/route_classification_probe.py [N]
2026-06-22: added when broadening store_fact to capture value-supplying
declaratives ("my dog is named Max") not just imperatives ("remember ...").
"""
import asyncio
import sys
from collections import Counter

sys.path.insert(0, ".")
import conversation.tool_router as tr

# (label, want_route, phrases)
BATTERY = [
    ("STORE", "store_fact", [
        "my dog is named Max", "my favorite color is teal", "I work at Anthropic",
        "my anniversary is June 3rd", "my sister lives in Portland",
        "I drive a Subaru", "my cat is named Onyx", "my landlord is named Greg",
        "remember I have an iguana named Nacho", "don't forget my gate code is 4417",
    ]),
    ("NONE", "none", [
        "what is my dog's name", "I have a hamster", "you are absolutely hilarious",
        "what should we talk about", "do you remember my dog's name",
        "tell me a joke", "remember my face",
    ]),
    ("RECALL", "recall_temporal", [
        "what did we talk about yesterday", "what did I say last week",
        "remind me what I told you this morning",
    ]),
]


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"route probe: each phrase x{n}\n")
    total = ok = 0
    for label, want, phrases in BATTERY:
        print(f"--- {label} (want {want}) ---")
        for p in phrases:
            routes = [await tr.classify_intent(p) for _ in range(n)]
            hits = sum(1 for r in routes if r == want)
            total += n
            ok += hits
            flag = "OK " if hits == n else ("~~ " if hits else "XX ")
            print(f"  {flag}{hits}/{n}  {p!r:52} {dict(Counter(routes))}")
    print(f"\noverall: {ok}/{total} ({100*ok/total:.0f}%) matched expected route")


if __name__ == "__main__":
    asyncio.run(main())
