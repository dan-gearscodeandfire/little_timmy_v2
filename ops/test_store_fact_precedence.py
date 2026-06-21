#!/usr/bin/env python3
"""Integration test for the recency-gated source-precedence guard in
memory.facts.store_fact (added 2026-06-21).

Reproduces the live race found under acoustic multi-turn load: the async
extractor coalesces a debounce buffer and can flush a STALE earlier mention
AFTER an explicit store_fact correction, overwriting the newer value on the
shared (subject,predicate) upsert key (robot Rusty -> Sparky, recall then
served stale Sparky). See lt-store-fact-correction-clobbered-by-extractor-
race-2026-06-21.

Rule under test:
  - 'tool' writes always apply (explicit user intent / current correction).
  - an 'extraction' write may overwrite a 'tool'-written fact ONLY if its
    source turn (turn_ts) is newer than the tool write's learned_at.
  - extraction-over-extraction and tool-over-* are unaffected.

Isolated: operates only on subject '__test_race__' so it never collides with
real facts; deletes its rows on the way out. Quick; no LLM, no mic.
Run: .venv/bin/python ops/test_store_fact_precedence.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_pool
from memory.facts import store_fact

SUBJ = "__test_race__"


async def _value(pool, predicate):
    return await pool.fetchval(
        "SELECT value FROM facts WHERE subject=$1 AND predicate=$2 "
        "AND superseded_by IS NULL", SUBJ, predicate,
    )


async def _learned_epoch(pool, predicate):
    la = await pool.fetchval(
        "SELECT learned_at FROM facts WHERE subject=$1 AND predicate=$2 "
        "AND superseded_by IS NULL", SUBJ, predicate,
    )
    return la.timestamp()


async def _cleanup(pool):
    await pool.execute("DELETE FROM facts WHERE subject=$1", SUBJ)


async def main():
    pool = await get_pool()
    await _cleanup(pool)  # start clean even if a prior run aborted
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
        if not cond:
            fails.append(name)

    try:
        # --- THE RACE (the bug) -------------------------------------------
        # 1. extractor records an early mention.
        await store_fact(SUBJ, "robot", "Sparky", source="extraction", turn_ts=1000.0)
        check("1 extraction insert", await _value(pool, "robot") == "Sparky")

        # 2. user explicitly corrects via the tool route (the gold value).
        await store_fact(SUBJ, "robot", "Rusty", source="tool")
        check("2 tool correction wins", await _value(pool, "robot") == "Rusty")
        tool_epoch = await _learned_epoch(pool, "robot")

        # 3. the coalesced extractor flushes the STALE earlier mention
        #    (turn predates the tool write) -> MUST be skipped.
        await store_fact(SUBJ, "robot", "Sparky", source="extraction",
                         turn_ts=tool_epoch - 5.0)
        check("3 stale extraction does NOT clobber correction",
              await _value(pool, "robot") == "Rusty",
              f"value={await _value(pool, 'robot')!r} (want Rusty)")

        # 3b. turn_ts=None (unknown) is treated as not-newer -> also skipped.
        await store_fact(SUBJ, "robot", "Falcon", source="extraction", turn_ts=None)
        check("3b extraction with no turn_ts does NOT clobber",
              await _value(pool, "robot") == "Rusty")

        # --- THE INVERSE (must NOT over-correct) --------------------------
        # 4. router MISSES a genuinely newer correction; only the extractor
        #    catches it, and its turn is newer than the tool write -> ALLOW.
        await store_fact(SUBJ, "robot", "Bolt", source="extraction",
                         turn_ts=tool_epoch + 5.0)
        check("4 newer extraction MAY overwrite a tool fact (router-miss case)",
              await _value(pool, "robot") == "Bolt",
              f"value={await _value(pool, 'robot')!r} (want Bolt)")

        # --- UNAFFECTED PATHS ---------------------------------------------
        # 5. extraction-over-extraction is normal last-writer-wins.
        await store_fact(SUBJ, "color", "teal", source="extraction", turn_ts=1000.0)
        await store_fact(SUBJ, "color", "green", source="extraction", turn_ts=1.0)
        check("5 extraction-over-extraction unaffected (last writer wins)",
              await _value(pool, "color") == "green")

        # 6. tool always applies, even over a tool fact.
        await store_fact(SUBJ, "drone", "Falcon", source="tool")
        await store_fact(SUBJ, "drone", "Hawk", source="tool")
        check("6 tool-over-tool unaffected", await _value(pool, "drone") == "Hawk")

    finally:
        await _cleanup(pool)
        # leave the pool/loop tidy
        try:
            await pool.close()
        except Exception:
            pass

    print()
    if fails:
        print(f"FAILED ({len(fails)}): {', '.join(fails)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    asyncio.run(main())
