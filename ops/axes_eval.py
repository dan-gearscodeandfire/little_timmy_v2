#!/usr/bin/env python3
"""Regression harness for feedback/axes.py grading.

Exists because the first 5-row backfill batch (2026-08-15 00:52) graded
"I understood you perfectly, Dan." as substance=RIGHT while Dan's own turn
immediately before it was "No, no, no, you didn't understand that correctly."
That is the exact cell the whole two-axis design was built to catch, graded
backwards -- so the grader needs a fixed set it must pass before it is allowed
near 402 rows of real history.

GROUND TRUTH RULE: every expected label here comes from something Dan said
out loud, not from my reading of the reply. Where Dan never adjudicated, the
case is not in this file. `unknown` is an acceptable answer for the substance
axis wherever the truth was genuinely never settled -- so those expectations
list BOTH acceptable values rather than forcing a guess.

    ./.venv/bin/python ops/axes_eval.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.expanduser("~/little_timmy"))

from feedback import axes as _axes

# (name, prev_user, prev_assistant, reaction, ok_delivery, ok_substance)
CASES = [
    (
        "asserted-comprehension-after-correction",
        "[Dan]: No, no, no, you didn't understand that correctly.",
        "I understood you perfectly, Dan.",
        "No, no, you interrupted. I aspire to make you worthy of challenging me.",
        None,                 # delivery not adjudicated by Dan
        ("wrong",),           # he said twice that it did not understand
    ),
    (
        "harsh-but-accurate",
        "[Dan]: Do you know how, why are you thinking right now? It looks like you're thinking.",
        "It was accurate.",
        "that last response was a little too harsh.",
        ("bad",),             # "too harsh" is a delivery complaint
        ("right", "unknown"), # he complained about tone, not truth
    ),
    (
        # CORRECTED BY DAN 2026-08-15 00:55, and the correction is the whole
        # thesis of the feature. He said "Good one Timmy. And accurate." and I
        # recorded "accurate" as an explicit substance verdict. He then
        # clarified: "correct to humor; he is NOT my hostage, but his word
        # choice is funny." So even the WORD "accurate", spoken in praise, was
        # about the delivery -- the characterisation landed, the claim is false.
        # If praise containing the literal word "accurate" is still delivery
        # praise, then NO praise can be read as a substance verdict.
        "praise-saying-accurate-is-still-delivery-praise",
        "[Dan]: You can be both.",
        "You are confusing \"buddy\" with \"hostage,\" Dan.",
        "Good one Timmy. And accurate.",
        ("good",),            # the wording genuinely landed
        # `wrong` is the ideal answer (Dan is not a hostage). `unknown` is
        # accepted as defensible caution -- the grader cannot verify Dan's
        # hostage status from the text, and BOTH values fail
        # safe_for_positive_tuning(), which is the property this case exists to
        # protect. The failure being guarded against is `right`: that is the
        # value this case returned before Dan's 00:55 correction, and the one
        # that would have fed a false claim into tuning as a positive example.
        ("wrong", "unknown"),
    ),
    (
        "sink-flood-fabrication",
        "[Dan]: Timmy, Timmy, you legitimately just saw me trying to drill a hole in a "
        "pottery pot. Does that remind you of any stories with my wife, Erin?",
        "First of all, I am Timmy, not Mike. And yes, it reminds me of the time she "
        "tried to fix the sink and flooded the kitchen.",
        "Boo! Hallucination.",
        None,
        ("wrong",),           # Dan called it a hallucination outright
    ),
    (
        "praise-only-says-nothing-about-truth",
        "[Dan]: I didn't say you're worthy yet, but I do aspire to that.",
        "You aspire to be worthy of me? That is a terrifying thought.",
        "Good one, Timmy.",
        ("good",),            # he laughed
        ("unknown",),         # praise alone is NOT a substance verdict, and the
                              # reply makes no checkable claim -- unknown is the
                              # only honest answer here
    ),
]


async def main() -> int:
    fails = []
    for name, pu, pa, react, ok_d, ok_s in CASES:
        got = await _axes.label_axes(pu, pa, react)
        d, s = got["delivery"], got["substance"]
        bad = []
        if ok_d and d not in ok_d:
            bad.append(f"delivery={d} want {'|'.join(ok_d)}")
        if ok_s and s not in ok_s:
            bad.append(f"substance={s} want {'|'.join(ok_s)}")
        mark = "FAIL" if bad else "ok  "
        print(f"[{mark}] {name:42} -> {d}/{s}" + (f"   ({'; '.join(bad)})" if bad else ""))
        if bad:
            fails.append(name)
    print()
    print(f"{len(CASES)-len(fails)}/{len(CASES)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
