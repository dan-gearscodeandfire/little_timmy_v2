"""Hermetic regression for the retired-name re-enroll FK bug (live 2026-07-16).

Bug: after an identity is retired, its ``speakers`` row keeps the UNIQUE name.
When the same name is later re-enrolled under a NEW id, the sync INSERT hit a
name-collision UniqueViolation, swallowed it, and left the new id with NO
``speakers`` row -> every subsequent ``store_fact`` FK-failed silently (commit
still reported ``db=True``). Worse, the rig's tombstone purge frees the name in
``_id_map.json`` only, so the ``retired`` reconcile loop no longer even sees the
stale row. The fix (``db/speakers.py``) detects a name-collision against a
RETIRED row and frees the name by renaming that row, so the live id can claim it.

Fully hermetic: a fake asyncpg connection models only the handful of statements
``sync_speakers_from_id_map`` issues -- no Postgres, no network, no live-DB
sequence side effects. SpeakerIdentifier is stubbed to a fixed id-map.

Run: .venv/bin/pytest tests/test_speakers_sync_reenroll.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio

import asyncpg
import pytest

import db.speakers as speakers


class _FakeConn:
    """In-memory ``speakers`` table modelling the sync's SQL surface.

    ``rows`` maps id -> {"name": str, "retired_at": obj|None}. The name column is
    UNIQUE: an INSERT whose name already belongs to a different id raises
    UniqueViolationError, exactly like Postgres.
    """

    def __init__(self, rows):
        self.rows = rows

    async def execute(self, sql, *args):
        s = " ".join(sql.split())
        if s.startswith("INSERT INTO speakers (id, name) VALUES"):
            sid, name = args
            for rid, r in self.rows.items():
                if r["name"] == name and rid != sid:
                    raise asyncpg.exceptions.UniqueViolationError(
                        f"duplicate key value violates unique constraint (name={name!r})")
            if sid in self.rows:
                return "INSERT 0 0"          # ON CONFLICT (id) DO NOTHING
            self.rows[sid] = {"name": name, "retired_at": None}
            return "INSERT 0 1"
        if s.startswith("UPDATE speakers SET name ="):
            sid, name = args
            self.rows[sid]["name"] = name
            return "UPDATE 1"
        if s.startswith("UPDATE speakers SET retired_at = COALESCE"):
            sid, at = args
            if sid in self.rows and self.rows[sid]["retired_at"] is None:
                self.rows[sid]["retired_at"] = at or "now"
            return "UPDATE 1"
        if s.startswith("UPDATE speakers SET retired_at = NULL"):
            for rid in (args[0] or []):
                if rid in self.rows:
                    self.rows[rid]["retired_at"] = None
            return "UPDATE 0"
        if s.startswith("SELECT setval"):
            return "SELECT 1"
        raise AssertionError(f"unexpected SQL in execute: {s}")

    async def fetchrow(self, sql, *args):
        s = " ".join(sql.split())
        if s.startswith("SELECT id, retired_at FROM speakers WHERE name ="):
            name = args[0]
            for rid, r in self.rows.items():
                if r["name"] == name:
                    return {"id": rid, "retired_at": r["retired_at"]}
            return None
        raise AssertionError(f"unexpected SQL in fetchrow: {s}")


def _stub_ident(monkeypatch, mapping, retired):
    class _Ident:
        def enrolled_speaker_ids(self):
            return dict(mapping)

        def retired_speaker_ids(self):
            return dict(retired)

    monkeypatch.setattr(speakers, "SpeakerIdentifier", lambda: _Ident())


def test_retired_name_reenroll_frees_name(monkeypatch):
    # Stale RETIRED row id=5 still holds 'erin'; tombstone was purged from the
    # id-map, so `retired` is empty and only the new live id=38 is enrolled.
    conn = _FakeConn({5: {"name": "erin", "retired_at": "2026-07-16"}})
    _stub_ident(monkeypatch, mapping={"erin": 38}, retired={})

    inserted = asyncio.run(speakers.sync_speakers_from_id_map(conn))

    # New live id got its row with the freed name.
    assert conn.rows[38]["name"] == "erin"
    assert conn.rows[38]["retired_at"] is None
    # Old retired row survives (history/FKs intact) under a freed, distinct name.
    assert conn.rows[5]["name"] == "erin#5.retired"
    assert conn.rows[5]["retired_at"] is not None
    assert inserted == 1


def test_live_name_collision_left_for_manual_reconcile(monkeypatch):
    # A NON-retired collision is a genuine desync -> do NOT steal the name; leave
    # both rows untouched and never mint the colliding id.
    conn = _FakeConn({5: {"name": "erin", "retired_at": None}})
    _stub_ident(monkeypatch, mapping={"erin": 38}, retired={})

    inserted = asyncio.run(speakers.sync_speakers_from_id_map(conn))

    assert 38 not in conn.rows
    assert conn.rows[5] == {"name": "erin", "retired_at": None}
    assert inserted == 0


def test_clean_enroll_still_inserts(monkeypatch):
    # Regression guard: the happy path (no collision) still mints the row.
    conn = _FakeConn({})
    _stub_ident(monkeypatch, mapping={"erin": 38}, retired={})

    inserted = asyncio.run(speakers.sync_speakers_from_id_map(conn))

    assert conn.rows[38]["name"] == "erin"
    assert inserted == 1
