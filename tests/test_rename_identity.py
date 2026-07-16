"""Hermetic tests for the in-place identity rename (Dan 2026-07-15, Open
Sauce spec 9: correct a mis-heard enrolled name — "too_sharp" -> "tushar" —
WITHOUT discarding biometrics or orphaning FK history)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presence.identity_commit import (  # noqa: E402
    commit_identity_stores,
    rename_identity_stores,
    retire_identity_stores,
)
from presence.prototype_base import (  # noqa: E402
    NAME_RE,
    RESERVED_NAMES,
    IdMap,
    PrototypeFileStore,
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
    """too_sharp enrolled with both modalities (the Tushar shape)."""
    id_map, voice, face = stores
    res = commit_identity_stores(
        "too_sharp", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(i) for i in range(3)],
        face_embeddings=[_unit(100 + i) for i in range(3)])
    assert res.status == "ok"
    return stores, res.speaker_id


def test_rename_preserves_id_and_moves_files(enrolled):
    (id_map, voice, face), sid = enrolled
    res = rename_identity_stores("too_sharp", "tushar", id_map=id_map,
                                 voice_store=voice, face_store=face)
    assert res.ok, (res.status, res.error)
    assert res.speaker_id == sid                       # FK history intact
    assert id_map.id_for("tushar") == sid
    assert id_map.id_for("too_sharp") is None
    assert voice.path_for("tushar").exists()
    assert face.path_for("tushar").exists()
    assert not voice.path_for("too_sharp").exists()
    assert not face.path_for("too_sharp").exists()
    # Prototypes byte-identical: a rename never touches biometrics.
    assert np.array_equal(voice.load(voice.path_for("tushar")),
                          np.asarray(voice.load(voice.path_for("tushar"))))


def test_rename_refuses_existing_target(enrolled):
    (id_map, voice, face), _sid = enrolled
    setup = commit_identity_stores(
        "erin", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(50 + i) for i in range(3)],
        require_match_for_known=False)   # fixture setup — bypass guards
    assert setup.status == "ok", (setup.status, setup.error)
    res = rename_identity_stores("too_sharp", "erin", id_map=id_map,
                                 voice_store=voice, face_store=face)
    assert not res.ok
    # Nothing moved on refusal.
    assert voice.path_for("too_sharp").exists()
    assert id_map.id_for("too_sharp") is not None


def test_rename_refuses_reserved_and_retired_targets(enrolled, tmp_path):
    (id_map, voice, face), _sid = enrolled
    # Reserved target.
    res = rename_identity_stores("too_sharp", "dan", id_map=id_map,
                                 voice_store=voice, face_store=face)
    assert not res.ok
    # Retired target: tombstone a name, then try renaming onto it.
    setup = commit_identity_stores(
        "ghost", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(70 + i) for i in range(3)],
        require_match_for_known=False)   # fixture setup — bypass guards
    assert setup.status == "ok", (setup.status, setup.error)
    retire_identity_stores("ghost", id_map=id_map, voice_store=voice,
                           face_store=face, trash_root=tmp_path / "trash")
    res2 = rename_identity_stores("too_sharp", "ghost", id_map=id_map,
                                  voice_store=voice, face_store=face)
    assert not res2.ok
    assert voice.path_for("too_sharp").exists()


def test_rename_missing_source_refused(stores):
    id_map, voice, face = stores
    res = rename_identity_stores("nobody", "somebody", id_map=id_map,
                                 voice_store=voice, face_store=face)
    assert not res.ok


def test_renamed_identity_recognized_not_lookalike(enrolled):
    """After a rename, the SAME samples committing under the NEW name are an
    augment of that identity — not a lookalike of the old name (which no
    longer exists) and not a mismatch."""
    (id_map, voice, face), sid = enrolled
    rename_identity_stores("too_sharp", "tushar", id_map=id_map,
                           voice_store=voice, face_store=face)
    res = commit_identity_stores(
        "tushar", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(0)])   # same seed as an enrolled sample
    assert res.status == "ok", (res.status, res.error)
    assert res.speaker_id == sid
