"""One-shot: classify existing facts for sensitivity (PII gating).

Run from ~/little_timmy with PYTHONPATH=. and the venv python. Idempotent --
re-running just recomputes. Prints a category breakdown.
"""
import asyncio
from db.connection import get_pool
from memory.pii import classify_sensitivity


async def main():
    pool = await get_pool()
    rows = await pool.fetch("SELECT id, subject, predicate, value FROM facts")
    updates = []
    for r in rows:
        sensitive, category = classify_sensitivity(r["subject"], r["predicate"], r["value"])
        updates.append((r["id"], sensitive, category))
    async with pool.acquire() as conn:
        async with conn.transaction():
            for fid, sensitive, category in updates:
                await conn.execute(
                    "UPDATE facts SET sensitive=$1, pii_category=$2 WHERE id=$3",
                    sensitive, category, fid)
    total = len(updates)
    sens = sum(1 for _, s, _ in updates if s)
    print(f"Classified {total} facts: {sens} sensitive, {total - sens} clear.")
    breakdown = await pool.fetch(
        "SELECT pii_category, count(*) FROM facts WHERE sensitive GROUP BY pii_category ORDER BY 2 DESC")
    for b in breakdown:
        print(f"  {b['pii_category']:18s} {b['count']}")


if __name__ == "__main__":
    asyncio.run(main())
