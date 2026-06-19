"""One-shot: compress useful conversation_summary rows (re-embed) and delete gist.

Run from ~/little_timmy with the venv python. Backup already taken to
backups/conversation_summary_backup_*.tsv. Transactional: all-or-nothing.
"""
import asyncio
from db.connection import get_pool
from memory.manager import embed

# id -> compressed content (UPDATE + re-embed)
COMPRESS = {
    46:  "Dan worked with Timmy on a dance routine for one of his daughters.",
    97:  "Dan calls building Timmy's physical body \"Operation Pinocchio\"; the project tends to stall. One of Dan's daughters was present and thinks highly of him.",
    133: "At Dan's birthday party (2026-06-13), Timmy met guests Brian, Pierre, and Charlie, correcting them on his name and demanding proper introductions.",
    134: "Party guests included Charlie, Colin, and a young family member. Several guests mis-called Timmy \"Jimmy\"; he corrected them.",
    135: "A daughter of Dan's and \"Mama Bean\" attended the party. Timmy's stated favorite color is \"shiny\"/silver (like a fresh screwdriver).",
    136: "Guest Charlotte was introduced at the party. Dan complained about a missing microphone and asked the \"couples therapist\" to review the chat logs.",
    137: "Party guests Colin and Elizabeth attended; a \"couples therapist\" was monitoring the interaction.",
    138: "Guest Andy attended the party and asked about Dan (his creator); Timmy refused to gossip.",
    140: "Dan's 50th birthday party (2026-06-13) was a success, held during a Knicks playoff game.",
    142: "Guest Peter was enrolled (face and voice) on 2026-06-14 at Dan's request.",
    144: "The Knicks lost the NBA finals (~2026-06-14).",
    148: "Voss (real name V.A.S.) is Dan's \"nerdy friend\"; present around the OpenSauce conference.",
    152: "A lavalier mic fell during the party and was recovered; Dan praised Timmy's party performance.",
    154: "Timmy's brain (Qwen 3.6) runs with an approximately 128k-token context window.",
    155: "Erin asked Timmy for help drilling a hole in a concrete pot (2026-06-15); she learned to use a masonry bit instead of a wood bit.",
    158: "Dan reminded Timmy of his 50th birthday party and a recent basketball game; confirmed Dan lost a lavalier microphone.",
}

DELETE = [57, 70, 90, 91, 92, 93, 95, 96, 98, 99, 100, 101, 103, 104, 105, 106,
          107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
          122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 139, 141, 143,
          145, 146, 147, 149, 150, 151, 153, 156, 157]


async def main():
    pool = await get_pool()
    # Guard: only operate on rows that exist and are conversation_summary.
    existing = {r["id"] for r in await pool.fetch(
        "SELECT id FROM memories WHERE type='conversation_summary'")}
    missing = (set(COMPRESS) | set(DELETE)) - existing
    if missing:
        raise SystemExit(f"ABORT: ids not found / not conversation_summary: {sorted(missing)}")
    overlap = set(COMPRESS) & set(DELETE)
    if overlap:
        raise SystemExit(f"ABORT: ids in both lists: {sorted(overlap)}")

    # Re-embed compressed rows first (network calls, outside txn).
    embeds = {}
    for mid, content in COMPRESS.items():
        embeds[mid] = await embed(content)
    print(f"Embedded {len(embeds)} compressed rows.")

    async with pool.acquire() as conn:
        async with conn.transaction():
            for mid, content in COMPRESS.items():
                await conn.execute(
                    "UPDATE memories SET content=$1, embedding=$2, "
                    "metadata = metadata || '{\"compressed\": true}'::jsonb "
                    "WHERE id=$3",
                    content, embeds[mid], mid,
                )
            deleted = await conn.execute(
                "DELETE FROM memories WHERE id = ANY($1::int[])", DELETE)
    print(f"Compressed (updated+re-embedded): {len(COMPRESS)} rows.")
    print(f"Deleted: {deleted}")

    remaining = await pool.fetch(
        "SELECT count(*) c, count(embedding) e FROM memories WHERE type='conversation_summary'")
    print(f"Remaining conversation_summary: {remaining[0]['c']} rows, {remaining[0]['e']} embedded.")


if __name__ == "__main__":
    asyncio.run(main())
