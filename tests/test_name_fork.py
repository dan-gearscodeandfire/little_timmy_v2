"""Hermetic tests for duplicate-display-name forking (expo 2026-07-16).

Exercises ``resolve_fork_name`` and the ``fork_on_name_collision`` routing in
``commit_identity_stores``: a biometrically-different second "Mike" silently
forks to ``mike_2`` (id-map ``_meta`` marks the base); the SAME person
re-claiming their display name augments their existing record (sibling
check), never minting a third identity; a retired base name forks to a new
id without resurrecting the tombstone; and NONE of this happens without the
flag (augment paths keep the mismatch/retired refusals).

Fixture style per tests/test_identity_commit.py — temp dirs + plain numpy.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.identity_commit import (  # noqa: E402
    commit_identity_stores,
    resolve_fork_name,
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


def _enroll(stores_t, name, seed_base, flag=False):
    id_map, voice, face = stores_t
    return commit_identity_stores(
        name, id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(seed_base + i) for i in range(3)],
        fork_on_name_collision=flag)


# ── resolve_fork_name collision matrix ───────────────────────────────────────

def test_resolver_skips_active_retired_and_strays(stores):
    id_map, voice, face = stores
    id_map.allocate("mike")                       # active
    id_map.allocate("mike_2")
    id_map.retire("mike_2")                       # tombstone burns _2
    np.save(voice.dir / "mike_3_wespeaker.npy",   # on-disk stray
            np.vstack([_unit(1)]))
    np.save(face.dir / "mike_4_edgeface.npy",     # stray in the OTHER store
            np.vstack([_unit(2)]))
    got = resolve_fork_name("mike", id_map=id_map,
                            voice_store=voice, face_store=face)
    assert got == "mike_5"


def test_resolver_skips_legacy_voice_strays(stores):
    id_map, voice, _ = stores
    id_map.allocate("mike")
    np.save(voice.dir / "mike_2_resemblyzer.npy", np.vstack([_unit(1)]))
    got = resolve_fork_name("mike", id_map=id_map, voice_store=voice)
    assert got == "mike_3"


def test_resolver_respects_name_length_cap(stores):
    id_map, voice, _ = stores
    base = "a" * 32                               # at the NAME_RE cap
    got = resolve_fork_name(base, id_map=id_map, voice_store=voice)
    assert len(got) <= 32 and got.endswith("_2")


# ── fork on enroll (the second Mike) ─────────────────────────────────────────

def test_different_person_same_name_forks_silently(stores):
    id_map, voice, face = stores
    r1 = _enroll(stores, "mike", 0)
    assert r1.status == "ok" and r1.name == "mike"
    # A biometrically-different second Mike, WITH the flag.
    r2 = _enroll(stores, "mike", 9000, flag=True)
    assert r2.status == "ok"
    assert r2.name == "mike_2"
    assert r2.requested_name == "mike"
    assert r2.forked_from == "mike"
    assert r2.created is True
    assert r2.speaker_id not in (None, r1.speaker_id)
    assert id_map.base_name("mike_2") == "mike"   # display marker set
    assert voice.path_for("mike_2").exists()
    # Original mike untouched.
    assert id_map.id_for("mike") == r1.speaker_id


def test_third_mike_takes_next_suffix(stores):
    # Orthogonal basis vectors per person: cosine distance is exactly 1.0
    # across persons, so the routing is deterministic (a random-seed variant
    # of this test flaked when one seed landed inside the voice threshold —
    # which the code correctly AUGMENTED).
    id_map, voice, face = stores
    e = np.eye(8, dtype=np.float32)

    def commit(*rows):
        return commit_identity_stores(
            "mike", id_map=id_map, voice_store=voice, face_store=face,
            voice_embeddings=[e[r] for r in rows],
            fork_on_name_collision=True)

    assert commit(0, 1).name == "mike"
    r2 = commit(2, 3)
    r3 = commit(4, 5)
    assert r2.name == "mike_2"
    assert r3.name == "mike_3" and r3.forked_from == "mike"


def test_no_fork_without_flag(stores):
    _enroll(stores, "mike", 0)
    res = _enroll(stores, "mike", 9000)           # flag OFF (augment paths)
    assert res.status == "mismatch"


def test_same_person_reclaiming_name_augments_not_forks(stores):
    id_map, voice, face = stores
    base = _unit(7)
    commit_identity_stores(
        "mike", id_map=id_map, voice_store=voice,
        voice_embeddings=[base, _near(base, seed=1)],
        fork_on_name_collision=True)
    # Same voice says "enroll me as Mike" again — S3 passes, augments.
    res = commit_identity_stores(
        "mike", id_map=id_map, voice_store=voice,
        voice_embeddings=[_near(base, seed=2)],
        fork_on_name_collision=True)
    assert res.status == "ok"
    assert res.name == "mike" and res.created is False
    assert res.forked_from is None


def test_sibling_reenroll_augments_the_fork(stores):
    id_map, voice, face = stores
    _enroll(stores, "mike", 0)                    # mike #1
    fork_base = _unit(9000)
    commit_identity_stores(                       # mike #2 -> mike_2
        "mike", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[fork_base, _unit(9001), _unit(9002)],
        fork_on_name_collision=True)
    sid2 = id_map.id_for("mike_2")
    # Mike #2 says "enroll me as Mike" again: S3 vs mike mismatches, but the
    # sibling check matches mike_2 -> augment, never mike_3.
    res = commit_identity_stores(
        "mike", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_near(fork_base, seed=3)],
        fork_on_name_collision=True)
    assert res.status == "ok"
    assert res.name == "mike_2"
    assert res.requested_name == "mike"
    assert res.created is False
    assert res.speaker_id == sid2
    assert any(w.startswith("sibling_augment:mike_2") for w in res.warnings)
    assert id_map.id_for("mike_3") is None


def test_free_base_name_matching_sibling_augments(stores):
    id_map, voice, face = stores
    # Only the fork remains active (base renamed away — freed).
    _enroll(stores, "mike", 0)
    fork_base = _unit(9000)
    commit_identity_stores(
        "mike", id_map=id_map, voice_store=voice,
        voice_embeddings=[fork_base, _unit(9001)],
        fork_on_name_collision=True)
    id_map.rename("mike", "michael")              # base name now FREE
    # mike_2's owner re-claims "Mike": sibling match -> augment mike_2, not
    # a NEW person minted under the freed base name.
    res = commit_identity_stores(
        "mike", id_map=id_map, voice_store=voice,
        voice_embeddings=[_near(fork_base, seed=4)],
        fork_on_name_collision=True)
    assert res.name == "mike_2" and res.created is False


# ── retired base names ───────────────────────────────────────────────────────

def test_retired_base_forks_new_id_tombstone_intact(stores):
    id_map, voice, face = stores
    r1 = _enroll(stores, "mike", 0)
    id_map.retire("mike")
    res = _enroll(stores, "mike", 9000, flag=True)
    assert res.status == "ok"
    assert res.name == "mike_2"                   # allocator skips tombstone
    assert res.speaker_id != r1.speaker_id        # NEW id — no resurrection
    assert id_map.is_retired("mike")              # tombstone untouched
    assert id_map.retired()["mike"]["id"] == r1.speaker_id


def test_retired_name_still_refused_without_flag(stores):
    id_map, voice, face = stores
    _enroll(stores, "mike", 0)
    id_map.retire("mike")
    res = _enroll(stores, "mike", 9000)
    assert res.status == "retired_name"
