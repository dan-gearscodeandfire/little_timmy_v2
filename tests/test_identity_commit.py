"""Hermetic tests for the unified dual-modality identity commit core.

Exercises ``commit_identity_stores`` with temp dirs + plain numpy embeddings —
no encoder, no Postgres, no live singletons. Covers the ranked edge cases:
S1 (one shared id across modalities), S2 (partial = valid), S3 (augment vs the
stranger-claims-known-name guard), S8 (lowercase / reserved), S9 (thin warn).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.identity_commit import (  # noqa: E402
    CommitResult,
    commit_identity,
    commit_identity_stores,
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


def _near(base, jitter=0.001, seed=0):
    rng = np.random.default_rng(seed)
    v = base + jitter * rng.standard_normal(base.shape).astype(np.float32)
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


def test_new_identity_commits_both_modalities_one_id(stores):
    id_map, voice, face = stores
    v = [_unit(i) for i in range(3)]
    f = [_unit(100 + i) for i in range(3)]
    res = commit_identity_stores(
        "alice", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=v, face_embeddings=f)
    assert res.status == "ok"
    assert res.created is True
    assert res.voice_committed and res.face_committed
    # S1: face + voice share ONE speaker_id.
    assert res.speaker_id == id_map.id_for("alice")
    assert voice.path_for("alice").exists()
    assert face.path_for("alice").exists()


def test_casing_lowercased_at_boundary(stores):
    id_map, voice, face = stores
    res = commit_identity_stores(
        "Alice", id_map=id_map, voice_store=voice, voice_embeddings=[_unit(1)])
    assert res.name == "alice"
    assert voice.path_for("alice").exists()


def test_reserved_or_malformed_name_refused(stores):
    id_map, voice, face = stores
    for bad in ("timmy", "unknown", "9lives", ""):
        res = commit_identity_stores(
            bad, id_map=id_map, voice_store=voice, voice_embeddings=[_unit(1)])
        assert res.status == "invalid_name"
        assert res.speaker_id is None


def test_nothing_to_commit(stores):
    id_map, voice, face = stores
    res = commit_identity_stores("alice", id_map=id_map, voice_store=voice)
    assert res.status == "nothing_to_commit"


def test_augment_existing_keeps_id_and_merges(stores):
    id_map, voice, face = stores
    v1 = [_unit(i) for i in range(3)]
    r1 = commit_identity_stores(
        "bob", id_map=id_map, voice_store=voice, voice_embeddings=v1)
    sid = r1.speaker_id
    n1 = voice.load(voice.path_for("bob")).shape[0]
    # A genuinely-new pose of the same person (far from existing prototypes).
    v2 = [_unit(500 + i) for i in range(2)]
    r2 = commit_identity_stores(
        "bob", id_map=id_map, voice_store=voice, voice_embeddings=v2,
        require_match_for_known=False)
    assert r2.created is False
    assert r2.speaker_id == sid                      # id stable (S1)
    assert voice.load(voice.path_for("bob")).shape[0] > n1  # merged in


def test_stranger_claiming_known_name_is_refused(stores):
    id_map, voice, face = stores
    # Enroll the real bob.
    commit_identity_stores(
        "bob", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(i) for i in range(3)])
    # A different voice claims "bob".
    imposter = [_unit(9000 + i) for i in range(3)]
    res = commit_identity_stores(
        "bob", id_map=id_map, voice_store=voice, voice_embeddings=imposter,
        require_match_for_known=True)
    assert res.status == "mismatch"
    # Original prototypes untouched (no overwrite).
    assert "voice_failed" not in " ".join(res.warnings)


def test_matching_samples_augment_known_name(stores):
    id_map, voice, face = stores
    base = _unit(7)
    commit_identity_stores(
        "carol", id_map=id_map, voice_store=voice,
        voice_embeddings=[base, _near(base, seed=1), _near(base, seed=2)])
    # Same voice returns (near the enrolled prototype) -> passes the guard.
    again = [_near(base, jitter=0.15, seed=k) for k in range(3, 6)]
    res = commit_identity_stores(
        "carol", id_map=id_map, voice_store=voice, voice_embeddings=again,
        require_match_for_known=True)
    assert res.status in ("ok", "partial")
    assert res.created is False


def test_known_by_voice_add_face_unverified_warns_but_commits(stores):
    id_map, voice, face = stores
    # Known by voice only.
    commit_identity_stores(
        "dave", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(i) for i in range(3)])
    # Now add a face with NO voice sample -> nothing to verify against.
    res = commit_identity_stores(
        "dave", id_map=id_map, voice_store=voice, face_store=face,
        face_embeddings=[_unit(200 + i) for i in range(3)],
        require_match_for_known=True)
    assert res.status in ("ok", "partial")
    assert "augment_unverified" in res.warnings
    assert face.path_for("dave").exists()


def test_thin_buffer_warns(stores):
    id_map, voice, face = stores
    res = commit_identity_stores(
        "erin", id_map=id_map, voice_store=voice, voice_embeddings=[_unit(1)],
        min_voice=3)
    assert any(w.startswith("voice_thin") for w in res.warnings)
    assert res.voice_committed  # still commits what it has (S9 advisory)


def test_callbacks_fire_with_persisted_protos(stores):
    id_map, voice, face = stores
    seen = {}
    commit_identity_stores(
        "frank", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(i) for i in range(3)],
        face_embeddings=[_unit(300 + i) for i in range(3)],
        on_voice=lambda n, s, p: seen.setdefault("voice", (n, s, p.shape)),
        on_face=lambda n, s, p: seen.setdefault("face", (n, s, p.shape)))
    assert seen["voice"][0] == "frank"
    assert seen["face"][1] == seen["voice"][1]   # same id passed to both


def test_partial_when_one_store_missing_dir(stores, tmp_path):
    id_map, voice, _ = stores
    # Point face store at a path whose parent is a file -> persist raises.
    broken_parent = tmp_path / "afile"
    broken_parent.write_text("x")
    broken_face = PrototypeFileStore(
        broken_parent / "sub", "_edgeface",
        reserved_names=RESERVED_NAMES, name_re=NAME_RE)
    res = commit_identity_stores(
        "grace", id_map=id_map, voice_store=voice, face_store=broken_face,
        voice_embeddings=[_unit(i) for i in range(3)],
        face_embeddings=[_unit(1) for _ in range(3)])
    # Voice landed, face failed -> partial, id + voice valid (S2: no orphan).
    assert res.status == "partial"
    assert res.voice_committed and not res.face_committed
    assert res.speaker_id == id_map.id_for("grace")


# --- live async wrapper (fake identifiers + real stores, no DB / encoder) ---

class _FakeFaceId:
    def __init__(self, id_map, store):
        self._id_map = id_map
        self._store = store
        self._known = []


class _FakeVoiceId:
    def __init__(self, store):
        self._store = store
        self._known_speakers = []


@pytest.mark.asyncio
async def test_live_wrapper_binds_both_and_refreshes_memory(stores):
    id_map, voice, face = stores
    fface = _FakeFaceId(id_map, face)
    fvoice = _FakeVoiceId(voice)
    res = await commit_identity(
        "heidi",
        voice_embeddings=[_unit(i) for i in range(3)],
        face_embeddings=[_unit(400 + i) for i in range(3)],  # skips the encoder
        speaker_identifier=fvoice,
        face_identifier=fface,
        db_sync=False,
    )
    assert res.status == "ok"
    assert res.speaker_id == id_map.id_for("heidi")
    # In-memory refresh happened on BOTH live identifiers (no restart needed).
    assert [k.name for k in fface._known] == ["heidi"]
    assert [k.name for k in fvoice._known_speakers] == ["heidi"]
    assert fface._known[0].speaker_id == fvoice._known_speakers[0].speaker_id


@pytest.mark.asyncio
async def test_live_wrapper_voice_only_scope(stores):
    id_map, voice, face = stores
    fface = _FakeFaceId(id_map, face)
    fvoice = _FakeVoiceId(voice)
    res = await commit_identity(
        "ivan",
        voice_embeddings=[_unit(i) for i in range(3)],
        speaker_identifier=fvoice,
        face_identifier=fface,
        db_sync=False,
    )
    assert res.voice_committed and not res.face_committed
    assert not face.path_for("ivan").exists()
    assert voice.path_for("ivan").exists()
