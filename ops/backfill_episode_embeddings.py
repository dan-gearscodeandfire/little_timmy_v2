"""One-shot: backfill episodes.embedding for rows written while EMBED_EPISODES
was off (plan Session 5). Also (re)computes content_hash for any legacy rows
that predate the dedup floor, so the UNIQUE index covers them.

Run from ~/little_timmy with PYTHONPATH=. and the venv python. Idempotent:
only touches rows with NULL embedding / NULL content_hash. Embeds via the same
Ollama nomic-embed-text path as live writes.

    PYTHONPATH=. .venv/bin/python ops/backfill_episode_embeddings.py
"""
import asyncio

from db.connection import get_pool
from memory.manager import embed, _episode_content_hash


async def main():
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, text, content_hash, (embedding IS NULL) AS no_emb "
        "FROM episodes WHERE embedding IS NULL OR content_hash IS NULL "
        "ORDER BY id")
    if not rows:
        print("Nothing to backfill — all episodes embedded + hashed.")
        return
    print(f"Backfilling {len(rows)} episode(s)…")
    embedded = hashed = 0
    for r in rows:
        new_hash = r["content_hash"] or _episode_content_hash(r["text"])
        if r["no_emb"]:
            emb = await embed(r["text"])
            await pool.execute(
                "UPDATE episodes SET embedding = $1, content_hash = $2 WHERE id = $3",
                emb, new_hash, r["id"])
            embedded += 1
        else:
            await pool.execute(
                "UPDATE episodes SET content_hash = $1 WHERE id = $2",
                new_hash, r["id"])
        if not r["content_hash"]:
            hashed += 1
        print(f"  id={r['id']}: {'embedded+' if r['no_emb'] else ''}hashed")
    print(f"Done. {embedded} embedded, {hashed} hashed.")


if __name__ == "__main__":
    asyncio.run(main())
