"""Tests for the name-preserving identity audit (Phase C)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.migrate_reenroll import audit  # noqa: E402
from presence.prototype_base import (  # noqa: E402
    NAME_RE,
    RESERVED_NAMES,
    IdMap,
    PrototypeFileStore,
)


def _stores(tmp_path):
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


def _vec():
    v = np.random.default_rng(0).standard_normal(8).astype(np.float32)
    return (v / np.linalg.norm(v))[None, :]


def test_audit_reports_modalities_and_incomplete(tmp_path):
    id_map, voice, face = _stores(tmp_path)
    id_map.allocate("alice")
    voice.persist("alice", _vec())
    face.persist("alice", _vec())
    id_map.allocate("bob")
    voice.persist("bob", _vec())          # voice only -> needs enroll
    rep = audit(id_map, voice, face)
    by = {r.name: r for r in rep["identities"]}
    assert by["alice"].complete and by["alice"].modalities == "voice+face"
    assert not by["bob"].complete and by["bob"].has_voice and not by["bob"].has_face
    # Same speaker_id came from the shared map.
    assert by["alice"].speaker_id == id_map.id_for("alice")


def test_audit_detects_orphan_prototype(tmp_path):
    id_map, voice, face = _stores(tmp_path)
    # A face file with no id-map entry -> orphan.
    face.persist("ghost", _vec())
    rep = audit(id_map, voice, face)
    assert "ghost" in rep["orphan_face"]
    ghost = next(r for r in rep["identities"] if r.name == "ghost")
    assert ghost.speaker_id is None


def test_audit_includes_reserved(tmp_path):
    id_map, voice, face = _stores(tmp_path)
    rep = audit(id_map, voice, face)
    names = {r.name for r in rep["identities"]}
    assert {"dan", "timmy"} <= names   # reserved always present via enrolled_ids
