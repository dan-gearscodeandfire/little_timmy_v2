#!/usr/bin/env python3
"""T1 — coalesced-buffer correction race (the hole in the fc47a96 gate).

fc47a96 added recency-gated source precedence: an 'extraction' write may
overwrite a 'tool'-written (explicit) fact only if its source turn is newer
than the tool write's learned_at. The threading passed the COALESCED buffer's
*max* turn timestamp. That leaves a hole: when the extractor coalesces a buffer
that STRADDLES a correction — old "Sparky" turns PLUS one newer unrelated turn —
max(ts) > the tool write, so the stale "Sparky" the LLM extracts from the old
turns is allowed to clobber the explicit "Rusty" correction. Confident-wrong
recall follows: FALSE, the worst outcome under TRUE > AMBIGUITY > FALSE.

This test simulates exactly what memory/extraction.py does to store_fact: it
passes the buffer-derived gate timestamp. It runs BOTH semantics so the result
is unambiguous:
  - GATE_TS='max'  (fc47a96 behavior)      -> straddle case is expected to FAIL
  - GATE_TS='min'  (proposed strict fix)   -> straddle case is expected to PASS

Isolated to subject '__test_coal__'; self-cleaning. No LLM, no mic.
Run: .venv/bin/python ops/test_coalesced_correction_race.py [max|min]
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_pool
from memory.facts import store_fact

SUBJ = "__test_coal__"
GATE = sys.argv[1] if len(sys.argv) > 1 else "min"  # 'min' = shipped behavior; pass 'max' to demo the hole


def gate_ts(buffer_ts: list[float]) -> float:
    return max(buffer_ts) if GATE == "max" else min(buffer_ts)


async def _val(pool):
    return await pool.fetchval(
        "SELECT value FROM facts WHERE subject=$1 AND predicate='name' "
        "AND superseded_by IS NULL", SUBJ)


async def _learned(pool):
    la = await pool.fetchval(
        "SELECT learned_at FROM facts WHERE subject=$1 AND predicate='name' "
        "AND superseded_by IS NULL", SUBJ)
    return la.timestamp()


async def main():
    pool = await get_pool()
    await pool.execute("DELETE FROM facts WHERE subject=$1", SUBJ)
    fails = []

    def check(name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        print(f"  {tag}  {name}{(' -- ' + detail) if detail else ''}")
        if not cond:
            fails.append(name)

    try:
        # t=1000 two early "Sparky" mentions; the user then explicitly corrects
        # to "Rusty" via the tool route (stored 'now'); afterwards an unrelated
        # newer turn (t=tool+10) lands in the SAME coalesced extraction buffer.
        await store_fact(SUBJ, "name", "Sparky", source="extraction", turn_ts=1000.0)
        await store_fact(SUBJ, "name", "Rusty", source="tool")
        tool_epoch = await _learned(pool)
        check("setup: tool correction holds", await _val(pool) == "Rusty")

        # The coalesced buffer straddles the correction: oldest turn 1000 (pre),
        # newest turn tool+10 (post). The LLM extracts "Sparky" from the old
        # turns. store_fact is called with the buffer-derived gate ts.
        straddle_buffer = [1000.0, 1001.0, tool_epoch + 10.0]
        await store_fact(SUBJ, "name", "Sparky", source="extraction",
                         turn_ts=gate_ts(straddle_buffer))
        held = await _val(pool) == "Rusty"
        check(f"[{GATE}] straddle buffer does NOT clobber 'Rusty' correction",
              held, f"value={await _val(pool)!r} (want Rusty)")

        # Control: a CLEAN buffer fully post-dating the correction (a genuine
        # router-missed new value) SHOULD still be allowed under both semantics.
        clean_buffer = [tool_epoch + 20.0, tool_epoch + 25.0]
        await store_fact(SUBJ, "name", "Zoe", source="extraction",
                         turn_ts=gate_ts(clean_buffer))
        check(f"[{GATE}] clean post-correction buffer MAY update (router-miss)",
              await _val(pool) == "Zoe", f"value={await _val(pool)!r} (want Zoe)")

    finally:
        await pool.execute("DELETE FROM facts WHERE subject=$1", SUBJ)
        try:
            await pool.close()
        except Exception:
            pass

    print()
    print(f"GATE={GATE}: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    asyncio.run(main())
