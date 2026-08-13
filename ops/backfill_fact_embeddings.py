#!/usr/bin/env python3
"""Embed existing facts so relevance ranking can see them (2026-08-13).

Idempotent: only touches rows with a NULL embedding. Embeds the same
"subject predicate value" string store_fact writes, so backfilled and
newly-written rows share one representation.

  python -m ops.backfill_fact_embeddings [--limit N] [--all]
"""
import argparse, asyncio, logging, sys, time
sys.path.insert(0, __file__.rsplit("/ops/", 1)[0])
from db.connection import get_pool, close_pool
from memory.manager import embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_fact_embeddings")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="include superseded rows")
    args = ap.parse_args()
    pool = await get_pool()
    q = ("SELECT id, subject, predicate, value FROM facts WHERE embedding IS NULL"
         + ("" if args.all else " AND superseded_by IS NULL")
         + " ORDER BY learned_at DESC" + (f" LIMIT {int(args.limit)}" if args.limit else ""))
    rows = await pool.fetch(q)
    log.info("%d fact(s) to embed", len(rows))
    t0, n = time.time(), 0
    for r in rows:
        try:
            e = await embed(f"{r['subject']} {r['predicate']} {r['value']}")
        except Exception:
            log.exception("embed failed for fact %s", r["id"]); continue
        await pool.execute("UPDATE facts SET embedding = $2 WHERE id = $1", r["id"], e)
        n += 1
        if n % 50 == 0:
            log.info("  %d/%d (%.1fs)", n, len(rows), time.time() - t0)
    tot = await pool.fetchrow(
        "SELECT count(*) t, count(embedding) e FROM facts WHERE superseded_by IS NULL")
    log.info("DONE %d embedded in %.1fs; active facts %d/%d now embedded",
             n, time.time() - t0, tot["e"], tot["t"])
    await close_pool()

if __name__ == "__main__":
    asyncio.run(main())
