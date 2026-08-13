#!/usr/bin/env python3
"""Report (and optionally clean) junk in the facts table. 2026-08-13.

Reports four classes found by the Open Sauce audit:
  ephemeral   predicate is true only at the moment it was said ("time")
  lowconf     below config.FACT_MIN_WRITE_CONFIDENCE
  unsensitive PII classifier NOW flags it but the stored row says otherwise
  nearduped   two active facts on one subject whose embeddings are very close

--apply cleans ONLY the first three. Near-duplicates are REPORTED, NEVER
auto-superseded: measured on the live corpus, `favorite_show` and
`favorite_song` sit at cosine 0.125 (genuinely distinct) while the real
duplicate pair -- injury_history "table saw in May 2000" vs cut_off_finger
"table saw in May of 2020" -- is FURTHER apart than that. No threshold
separates "same fact restated" from "adjacent but different fact", so merging
by distance would destroy good facts before catching bad ones. Dan adjudicates.

  python -m ops.fact_hygiene              # report only
  python -m ops.fact_hygiene --apply      # supersede ephemeral + low-conf, fix flags
"""
import argparse, asyncio, sys
sys.path.insert(0, __file__.rsplit("/ops/", 1)[0])
import config
from db.connection import get_pool, close_pool
from memory.facts import _EPHEMERAL_PRED_RE
from memory.pii import classify_sensitivity


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--near-dist", type=float, default=0.13)
    args = ap.parse_args()
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT id, subject, predicate, value, confidence, sensitive, pii_category
           FROM facts WHERE superseded_by IS NULL ORDER BY subject, predicate""")

    ephemeral, lowconf, misflag = [], [], []
    for r in rows:
        if _EPHEMERAL_PRED_RE.search(r["predicate"]):
            ephemeral.append(r)
        elif r["confidence"] is not None and r["confidence"] < config.FACT_MIN_WRITE_CONFIDENCE:
            lowconf.append(r)
        want, cat = classify_sensitivity(r["subject"], r["predicate"], r["value"])
        if want and not r["sensitive"]:
            misflag.append((r, cat))

    near = await pool.fetch(
        """SELECT a.id ai, a.predicate ap, a.value av, b.id bi, b.predicate bp,
                  b.value bv, (a.embedding <=> b.embedding) d, a.subject
           FROM facts a JOIN facts b ON a.id < b.id AND a.subject = b.subject
           WHERE a.superseded_by IS NULL AND b.superseded_by IS NULL
             AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
             AND (a.embedding <=> b.embedding) < $1
           ORDER BY d""", args.near_dist)

    def show(title, items, fmt):
        print(f"\n=== {title} ({len(items)}) ===")
        for i in items:
            print("  " + fmt(i))

    show("EPHEMERAL predicate", ephemeral,
         lambda r: f"[{r['id']}] {r['subject']}.{r['predicate']} = {r['value'][:50]!r} (conf {r['confidence']:.2f})")
    show(f"LOW CONFIDENCE (< {config.FACT_MIN_WRITE_CONFIDENCE})", lowconf,
         lambda r: f"[{r['id']}] {r['subject']}.{r['predicate']} = {r['value'][:50]!r} (conf {r['confidence']:.2f})")
    show("SENSITIVE but stored unflagged", misflag,
         lambda t: f"[{t[0]['id']}] {t[0]['subject']}.{t[0]['predicate']} = {t[0]['value'][:44]!r} -> {t[1]}")
    show("NEAR-DUPLICATE (review by hand, never auto-merged)", near,
         lambda r: f"d={r['d']:.4f} {r['subject']}: [{r['ai']}] {r['ap']}={r['av'][:28]!r}  <->  [{r['bi']}] {r['bp']}={r['bv'][:28]!r}")

    if not args.apply:
        print("\n(report only -- pass --apply to supersede ephemeral + low-conf and fix flags)")
    else:
        drop = [r["id"] for r in ephemeral] + [r["id"] for r in lowconf]
        if drop:
            # Self-supersede = retire without deleting provenance.
            await pool.execute(
                "UPDATE facts SET superseded_by = id WHERE id = ANY($1::int[])", drop)
            print(f"\nretired {len(drop)} fact(s)")
        for r, cat in misflag:
            await pool.execute(
                "UPDATE facts SET sensitive = TRUE, pii_category = $2 WHERE id = $1",
                r["id"], cat)
        if misflag:
            print(f"flagged {len(misflag)} fact(s) sensitive")
    await close_pool()

if __name__ == "__main__":
    asyncio.run(main())
