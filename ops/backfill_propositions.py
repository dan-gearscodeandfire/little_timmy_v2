#!/usr/bin/env python3
"""Split existing episodes into propositions (memory/propositions.py).

Resumable and idempotent: skips episodes that already have rows, and the
per-episode (episode_id, content_hash) unique index makes a re-run a no-op.

GPU safety: every generation goes through llm.client.generate_memory, which
blocks on BOTH the conversation-idle and vision-idle gates -- so this cannot
steal the :8084 slot from a live turn. It will simply run slower while someone
is talking to Timmy. --sleep adds extra spacing on top of that.

  python -m ops.backfill_propositions --dry-run --limit 3
  python -m ops.backfill_propositions --limit 20
  python -m ops.backfill_propositions            # everything remaining
"""

import argparse
import asyncio
import logging
import sys
import time

sys.path.insert(0, __file__.rsplit("/ops/", 1)[0])

import config  # noqa: E402
from db.connection import get_pool, close_pool  # noqa: E402
from memory.propositions import (  # noqa: E402
    generate_propositions, store_propositions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_propositions")


async def pending(pool, limit, redo):
    q = """SELECT e.id, e.text, e.span_end
           FROM episodes e
           WHERE e.text IS NOT NULL AND length(e.text) > 0
             {clause}
           ORDER BY e.span_end DESC
           {lim}"""
    clause = "" if redo else "AND NOT EXISTS (SELECT 1 FROM propositions p WHERE p.episode_id = e.id)"
    lim = f"LIMIT {int(limit)}" if limit else ""
    return await pool.fetch(q.format(clause=clause, lim=lim))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max episodes this run (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="generate + print, write nothing")
    ap.add_argument("--redo", action="store_true", help="include episodes that already have rows")
    ap.add_argument("--sleep", type=float, default=0.0, help="extra seconds between episodes")
    args = ap.parse_args()

    pool = await get_pool()
    rows = await pending(pool, args.limit, args.redo)
    total_eps = await pool.fetchval("SELECT count(*) FROM episodes")
    have = await pool.fetchval("SELECT count(DISTINCT episode_id) FROM propositions")
    log.info("corpus: %d episodes, %d already split; %d selected this run (max %d props each)",
             total_eps, have, len(rows), config.PROPOSITION_MAX_PER_EPISODE)

    t0 = time.time()
    written = skipped = failed = 0
    for n, r in enumerate(rows, 1):
        try:
            props = await generate_propositions(r["text"])
        except Exception:
            log.exception("ep%s generation raised", r["id"])
            failed += 1
            continue
        if not props:
            log.warning("ep%s -> 0 propositions (kept episode-tier only)", r["id"])
            skipped += 1
            continue
        if args.dry_run:
            print(f"\n--- ep{r['id']} ({r['span_end'].date()}) "
                  f"{len(r['text'])} chars -> {len(props)} propositions")
            for p in props:
                print(f"    • {p}")
        else:
            w = await store_propositions(r["id"], r["span_end"], props)
            written += w
        if n % 10 == 0 or n == len(rows):
            log.info("progress %d/%d  (%.1fs elapsed, %d rows written)",
                     n, len(rows), time.time() - t0, written)
        if args.sleep:
            await asyncio.sleep(args.sleep)

    log.info("DONE in %.1fs: %d propositions written, %d episodes yielded nothing, %d failed",
             time.time() - t0, written, skipped, failed)
    if not args.dry_run:
        stats = await pool.fetchrow(
            """SELECT count(*) n, count(DISTINCT episode_id) eps,
                      round(avg(length(text))) avg_chars,
                      count(embedding) embedded FROM propositions""")
        log.info("table now: %d propositions across %d episodes, avg %s chars, %d embedded",
                 stats["n"], stats["eps"], stats["avg_chars"], stats["embedded"])
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
