"""Reconcile the postgres ``speakers`` table with the voiceprint id-map.

The voiceprint id-map (``models/speaker/_id_map.json``, owned by
``speaker.identifier.SpeakerIdentifier``) is the single source of truth for
``name -> speaker_id``. ``facts.speaker_id`` and ``memories.speaker_id`` are
foreign keys into ``speakers(id)``, so every enrolled speaker MUST have a
matching ``speakers`` row or those inserts raise ``facts_speaker_id_fkey`` /
``memories_speaker_id_fkey``.

Nothing in the enrollment path used to insert that row, so a freshly enrolled
voiceprint (e.g. Devon, 2026-06-24) would FK-fail the first fact extracted for
it. This module closes that gap. Call ``sync_speakers_from_id_map`` at startup
(belt — runs on every restart, which is also when a new voiceprint goes live)
and ``ensure_rows_for_enrolled`` right after every enrollment commit
(suspenders — the row appears immediately, no restart needed).

Both are idempotent and non-destructive: existing rows are never renamed or
re-id'd.
"""

import logging

import asyncpg

from db.connection import close_pool, get_pool
from speaker.identifier import SpeakerIdentifier

log = logging.getLogger(__name__)


async def sync_speakers_from_id_map(conn: asyncpg.Connection | None = None) -> int:
    """Upsert a ``speakers`` row for every name in the voiceprint id-map.

    Idempotent and non-destructive: ``ON CONFLICT (id) DO NOTHING`` never
    renames or re-id's an existing row. After upserting, advances
    ``speakers_id_seq`` to >= max(id) so a future ``SERIAL`` insert can't
    collide with an explicit id we wrote. Returns the number of rows inserted.

    Pass an existing ``conn`` to run on it (e.g. from the live pool at startup);
    otherwise a pooled connection is acquired. The caller owns pool lifecycle.
    """
    mapping = SpeakerIdentifier().enrolled_speaker_ids()  # name -> id, no encoder load
    if not mapping:
        return 0

    async def _run(c: asyncpg.Connection) -> int:
        inserted = 0
        for name, sid in sorted(mapping.items(), key=lambda kv: kv[1]):
            try:
                tag = await c.execute(
                    "INSERT INTO speakers (id, name) VALUES ($1, $2) "
                    "ON CONFLICT (id) DO NOTHING",
                    sid, name,
                )
            except asyncpg.UniqueViolationError:
                # The name already exists under a DIFFERENT id than the id-map
                # assigns -> a genuine id-map/DB desync. Don't crash the caller
                # (startup); surface it loudly for manual reconcile instead.
                log.error("speakers sync: name %r already present under a different "
                          "id than id-map's %d -- manual reconcile needed", name, sid)
                continue
            if tag.endswith(" 1"):
                inserted += 1
                log.info("speakers sync: inserted id=%d name=%s", sid, name)
        # Keep the SERIAL sequence ahead of every explicit id we wrote so a
        # later plain INSERT (without an id) can't reuse one.
        await c.execute(
            "SELECT setval('speakers_id_seq', "
            "GREATEST((SELECT COALESCE(MAX(id), 1) FROM speakers), 1))"
        )
        return inserted

    if conn is not None:
        inserted = await _run(conn)
    else:
        pool = await get_pool()
        async with pool.acquire() as c:
            inserted = await _run(c)

    log.info("speakers sync complete: %d row(s) inserted, %d speaker(s) tracked",
             inserted, len(mapping))
    return inserted


def ensure_rows_for_enrolled() -> int:
    """Sync wrapper for the standalone (synchronous) enrollment scripts.

    Refreshes the id-map from disk via ``load_voiceprints`` (so a name written
    by ``persist_voiceprint`` alone gets an id allocated), reconciles the
    ``speakers`` table, and closes the throwaway pool so the script exits
    cleanly. The encoder load inside ``load_voiceprints`` is cheap when already
    warm (every enroll path has just computed an embedding). Returns rows
    inserted. Never raises out -- enrollment already succeeded on disk, so a DB
    hiccup here is reported, not fatal.
    """
    import asyncio

    SpeakerIdentifier().load_voiceprints()  # allocate/persist id-map entry for new files

    async def _go() -> int:
        try:
            return await sync_speakers_from_id_map()
        finally:
            await close_pool()

    return asyncio.run(_go())
