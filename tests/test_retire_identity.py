"""Hermetic tests for persona retirement — the mirror of the identity commit.

Pins the 4-store resurrection class (2026-07-02 "john"): deletion used to be
representable only as absence, and every healer (startup DB sync, stray-file
load, ledger reload) read absence as damage to repair. These tests assert the
tombstone makes "deleted" a positive, propagated, irreversible-by-accident
state — and that revive is the one explicit way back.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.identity_commit import (  # noqa: E402
    commit_identity_stores,
    retire_identity_stores,
    revive_identity_stores,
)
from presence.prototype_base import (  # noqa: E402
    NAME_RE,
    RESERVED_NAMES,
    IdMap,
    PrototypeFileStore,
    RetiredNameError,
)


def _unit(seed, dim=8):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def stores(tmp_path):
    vdir = tmp_path / "speaker"
    fdir = tmp_path / "face"
    vdir.mkdir()
    fdir.mkdir()
    id_map = IdMap(vdir / "_id_map.json",
                   reserved_ids={"dan": 1, "timmy": 2}, first_free_id=3)
    voice = PrototypeFileStore(vdir, "_wespeaker",
                               reserved_names=RESERVED_NAMES, name_re=NAME_RE)
    face = PrototypeFileStore(fdir, "_edgeface",
                              reserved_names=RESERVED_NAMES, name_re=NAME_RE)
    return id_map, voice, face


@pytest.fixture()
def enrolled(stores):
    """sarah enrolled with both modalities, ready to retire."""
    id_map, voice, face = stores
    res = commit_identity_stores(
        "sarah", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(i) for i in range(3)],
        face_embeddings=[_unit(100 + i) for i in range(3)])
    assert res.status == "ok"
    return stores, res.speaker_id


# ── retire ───────────────────────────────────────────────────────────────────

def test_retire_archives_files_and_tombstones(enrolled, tmp_path):
    (id_map, voice, face), sid = enrolled
    trash = tmp_path / "trash"
    res = retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                                 face_store=face, trash_root=trash, at=1000.0)
    assert res.status == "ok"
    assert res.speaker_id == sid
    # Files GONE from the stores, PRESENT in the archive (never destroyed).
    assert not voice.path_for("sarah").exists()
    assert not face.path_for("sarah").exists()
    archive = trash / "sarah.1000"
    assert (archive / "sarah_wespeaker.npy").exists()
    assert (archive / "sarah_edgeface.npy").exists()
    # Tombstoned: out of the active map, in the retired section, same id.
    assert "sarah" not in id_map.enrolled_ids()
    assert id_map.is_retired("sarah")
    assert id_map.retired()["sarah"]["id"] == sid


def test_retire_is_idempotent(enrolled, tmp_path):
    (id_map, voice, face), sid = enrolled
    trash = tmp_path / "trash"
    retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=trash)
    res2 = retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                                  face_store=face, trash_root=trash)
    assert res2.status == "ok"
    assert res2.speaker_id == sid
    assert "already_retired" in res2.warnings


def test_retire_reserved_refused(stores, tmp_path):
    id_map, voice, face = stores
    res = retire_identity_stores("dan", id_map=id_map, voice_store=voice,
                                 face_store=face, trash_root=tmp_path / "trash")
    assert res.status == "reserved"
    with pytest.raises(ValueError):
        id_map.retire("dan")


def test_retire_unknown_not_found(stores, tmp_path):
    id_map, voice, face = stores
    res = retire_identity_stores("nobody", id_map=id_map, voice_store=voice,
                                 face_store=face, trash_root=tmp_path / "trash")
    assert res.status == "not_found"


def test_retire_exact_suffix_never_sweeps_prefix_names(stores, tmp_path):
    """`dan_the_barbarian`'s files must survive retiring a hypothetical `dan_the`."""
    id_map, voice, face = stores
    for nm in ("dan_the", "dan_the_barbarian"):
        commit_identity_stores(nm, id_map=id_map, voice_store=voice,
                               voice_embeddings=[_unit(hash(nm) % 100)])
    retire_identity_stores("dan_the", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=tmp_path / "trash")
    assert not voice.path_for("dan_the").exists()
    assert voice.path_for("dan_the_barbarian").exists()


def test_retire_sweeps_legacy_voice_files(enrolled, tmp_path):
    (id_map, voice, face), _ = enrolled
    legacy = voice.dir / "sarah_resemblyzer.npy"
    np.save(legacy, np.zeros((1, 4), dtype=np.float32))
    res = retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                                 face_store=face,
                                 trash_root=tmp_path / "trash", at=1000.0)
    assert res.status == "ok"
    assert not legacy.exists()
    assert (tmp_path / "trash" / "sarah.1000" / "sarah_resemblyzer.npy").exists()


# ── resurrection regressions (the 2026-07-02 class) ─────────────────────────

def test_allocate_refuses_retired_name(enrolled, tmp_path):
    (id_map, voice, face), _ = enrolled
    retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=tmp_path / "trash")
    with pytest.raises(RetiredNameError):
        id_map.allocate("sarah")


def test_commit_refuses_retired_name(enrolled, tmp_path):
    (id_map, voice, face), _ = enrolled
    retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=tmp_path / "trash")
    res = commit_identity_stores(
        "sarah", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(7)])
    assert res.status == "retired_name"
    assert not voice.path_for("sarah").exists()


def test_retired_id_never_reused(enrolled, tmp_path):
    """A new enroll after a retire must NOT recycle the tombstoned id —
    facts/memories FK history stays unambiguous forever (S1)."""
    (id_map, voice, face), sid = enrolled
    retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=tmp_path / "trash")
    res = commit_identity_stores(
        "bob", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(50)])
    assert res.speaker_id != sid


def test_tombstone_survives_read_write_roundtrip(enrolled, tmp_path):
    """load_voiceprints-style callers read() the whole map and write() it
    back; the _retired section must round-trip or a startup would erase
    every tombstone."""
    (id_map, voice, face), sid = enrolled
    retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=tmp_path / "trash")
    m = id_map.read()
    m["newperson"] = 99
    id_map.write(m)
    assert id_map.is_retired("sarah")
    assert id_map.retired()["sarah"]["id"] == sid


def test_enrolled_ids_excludes_retired(enrolled, tmp_path):
    (id_map, voice, face), _ = enrolled
    retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=tmp_path / "trash")
    ids = id_map.enrolled_ids()
    assert "sarah" not in ids
    assert ids["dan"] == 1  # reserved untouched


# ── revive ───────────────────────────────────────────────────────────────────

def test_revive_restores_tombstone_same_id(enrolled, tmp_path, monkeypatch):
    (id_map, voice, face), sid = enrolled
    trash = tmp_path / "trash"
    retire_identity_stores("sarah", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=trash, at=1000.0)

    # revive_identity_stores restores to the PRODUCTION dirs; point them at
    # the fixture stores for hermeticity.
    import presence.identity_commit as ic
    monkeypatch.setattr(ic, "VOICEPRINT_DIR", voice.dir)
    import presence.face_identifier as fi
    monkeypatch.setattr(fi, "FACE_DIR", face.dir)

    res = revive_identity_stores("sarah", id_map=id_map, trash_root=trash)
    assert res.status == "ok"
    assert res.speaker_id == sid
    assert not id_map.is_retired("sarah")
    assert id_map.enrolled_ids()["sarah"] == sid
    assert voice.path_for("sarah").exists()
    assert face.path_for("sarah").exists()
    # Re-enrolling (augment) works again after revive.
    res2 = commit_identity_stores(
        "sarah", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(0)], require_match_for_known=True)
    assert res2.status == "ok"
    assert res2.speaker_id == sid


def test_revive_without_tombstone_not_found(stores, tmp_path):
    id_map, voice, face = stores
    res = revive_identity_stores("nobody", id_map=id_map,
                                 trash_root=tmp_path / "trash")
    assert res.status == "not_found"


# ── room ledger forget ───────────────────────────────────────────────────────

def test_ledger_forget_survives_reload(tmp_path):
    """forget() must beat the flush-wins race: after forgetting, a reload of
    the save file must NOT re-mint the record (the disk-edit failure mode)."""
    from presence.ledger import RoomLedger
    save = tmp_path / "ledger.json"
    led = RoomLedger(save_path=str(save))
    led.update_from_voice("sarah", ts=__import__("time").time())
    assert any(p["name"] == "sarah" for p in led.current_state()["present"])

    assert led.forget("sarah") is True
    assert not any(p["name"] == "sarah" for p in led.current_state()["present"])
    # Reload from disk — the forget was flushed, nothing re-mints.
    led2 = RoomLedger(save_path=str(save))
    assert not any(p["name"] == "sarah" for p in led2.current_state()["present"])
    # Forgetting a missing record is a clean no-op.
    assert led.forget("sarah") is False
