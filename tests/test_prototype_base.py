"""Unit tests for the modality-agnostic K-prototype machinery."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.prototype_base import (  # noqa: E402
    IdMap,
    PrototypeFileStore,
    build_prototypes,
    merge_prototypes,
    min_cosine_distance,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_min_cosine_distance_picks_nearest_prototype():
    protos = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    assert min_cosine_distance(_unit([1, 0, 0]), protos) == pytest.approx(0.0, abs=1e-6)
    # Nearer to the second prototype than the first.
    d = min_cosine_distance(_unit([0.1, 1, 0]), protos)
    assert d < 0.02


def test_build_prototypes_dedups_near_duplicates():
    a = _unit([1, 0, 0])
    near = _unit([1, 0.01, 0])          # within a loose dedup dist of a
    b = _unit([0, 1, 0])
    protos = build_prototypes([a, near, b], dedup_dist=0.05, max_protos=12)
    assert protos.shape[0] == 2         # near-dup dropped


def test_build_prototypes_caps_at_max():
    embs = [_unit([1, i, i * 2]) for i in range(1, 20)]  # all distinct
    protos = build_prototypes(embs, dedup_dist=0.0, max_protos=5)
    assert protos.shape[0] == 5


def test_build_prototypes_falls_back_to_mean_when_all_dedup():
    a = _unit([1, 0, 0])
    protos = build_prototypes([a, a, a], dedup_dist=0.5, max_protos=12)
    assert protos.shape[0] == 1         # collapsed to the mean


def test_merge_prototypes_appends_only_novel():
    existing = np.vstack([_unit([1, 0, 0])])
    dup = _unit([1, 0.01, 0])
    novel = _unit([0, 0, 1])
    new, added = merge_prototypes(existing, [dup, novel], dedup_dist=0.05, max_protos=12)
    assert added == 1
    assert new.shape[0] == 2


def test_merge_prototypes_keeps_most_recent_over_cap():
    existing = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    novel = [_unit([0, 0, 1]), _unit([1, 1, 1])]
    new, added = merge_prototypes(existing, novel, dedup_dist=0.0, max_protos=2)
    assert added == 2
    assert new.shape[0] == 2            # capped, most-recent kept


def test_prototype_file_store_roundtrip_and_legacy_1d(tmp_path):
    store = PrototypeFileStore(
        tmp_path, "_edgeface",
        reserved_names=frozenset({"timmy"}),
        name_re=re.compile(r"^[a-z][a-z0-9_-]{1,31}$"),
    )
    protos = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    out = store.persist("pat", protos)
    assert out.name == "pat_edgeface.npy"
    loaded = store.load(out)
    assert loaded.shape == (2, 3)
    # Legacy single-vector file coerced to (1, D).
    legacy = tmp_path / "leg_edgeface.npy"
    np.save(legacy, _unit([1, 0, 0]))
    assert store.load(legacy).shape == (1, 3)


def test_prototype_file_store_refuses_bad_names(tmp_path):
    store = PrototypeFileStore(
        tmp_path, "_edgeface",
        reserved_names=frozenset({"timmy"}),
        name_re=re.compile(r"^[a-z][a-z0-9_-]{1,31}$"),
    )
    assert not store.valid_name("timmy")
    assert not store.valid_name("")
    assert not store.valid_name("9bad")
    assert store.valid_name("pat")
    with pytest.raises(ValueError):
        store.persist("timmy", np.vstack([_unit([1, 0, 0])]))


def test_prototype_file_store_backs_up_existing(tmp_path):
    store = PrototypeFileStore(
        tmp_path, "_wespeaker",
        reserved_names=frozenset(),
        name_re=re.compile(r"^[a-z][a-z0-9_-]{1,31}$"),
    )
    store.persist("dan", np.vstack([_unit([1, 0, 0])]))
    store.persist("dan", np.vstack([_unit([0, 1, 0])]))
    baks = list(tmp_path.glob("dan_wespeaker.npy.bak.*"))
    assert len(baks) == 1
    # iter skips the .bak file.
    names = [n for n, _ in store.iter_prototype_files()]
    assert names == ["dan"]


def test_idmap_reserved_and_allocation(tmp_path):
    m = IdMap(tmp_path / "_id_map.json",
              reserved_ids={"dan": 1, "timmy": 2}, first_free_id=3)
    assert m.id_for("dan") == 1
    assert m.id_for("pat") is None
    pat_id = m.allocate("pat")
    assert pat_id == 3
    assert m.allocate("pat") == 3               # idempotent
    quinn_id = m.allocate("quinn")
    assert quinn_id == 4
    enrolled = m.enrolled_ids()
    assert enrolled["dan"] == 1 and enrolled["timmy"] == 2
    assert enrolled["pat"] == 3 and enrolled["quinn"] == 4
    assert "_next_id" not in enrolled


def test_idmap_reserved_never_collide(tmp_path):
    # A file that (wrongly) mapped a name onto a reserved id must not hand that id
    # out again to a new name.
    p = tmp_path / "_id_map.json"
    p.write_text('{"pat": 3, "_next_id": 3}')
    m = IdMap(p, reserved_ids={"dan": 1, "timmy": 2}, first_free_id=3)
    new_id = m.allocate("quinn")
    assert new_id not in (1, 2, 3)
    assert new_id == 4


def test_idmap_allocate_persists(tmp_path):
    p = tmp_path / "_id_map.json"
    m = IdMap(p, reserved_ids={"dan": 1}, first_free_id=3)
    m.allocate("pat")
    # A fresh instance reads the persisted allocation.
    m2 = IdMap(p, reserved_ids={"dan": 1}, first_free_id=3)
    assert m2.id_for("pat") == 3


# ── Auto-suffix meta (_meta section) ─────────────────────────────────────────

def test_idmap_meta_roundtrip_and_base_name(tmp_path):
    p = tmp_path / "_id_map.json"
    m = IdMap(p, reserved_ids={"dan": 1}, first_free_id=3)
    m.allocate("mike")
    m.allocate("mike_2")
    m.mark_auto_suffixed("mike_2", "mike")
    assert m.base_name("mike_2") == "mike"
    assert m.base_name("mike") == "mike"           # no marker -> itself
    assert m.base_name("dan_the_barbarian") == "dan_the_barbarian"
    # A fresh instance reads the persisted marker.
    m2 = IdMap(p, reserved_ids={"dan": 1}, first_free_id=3)
    assert m2.meta()["mike_2"]["base"] == "mike"
    # Later writes through OTHER paths must round-trip the section.
    m2.allocate("quinn")
    assert IdMap(p, reserved_ids={"dan": 1}, first_free_id=3).base_name("mike_2") == "mike"
    # Bookkeeping keys never leak into the roster.
    assert "_meta" not in m2.enrolled_ids()


def test_idmap_meta_malformed_entry_tolerated(tmp_path):
    p = tmp_path / "_id_map.json"
    p.write_text(
        '{"mike": 3, "mike_2": 4, "_next_id": 5,'
        ' "_meta": {"mike_2": {"base": "mike", "at": 1.0},'
        '           "broken": {"nope": true}}}')
    m = IdMap(p, reserved_ids={"dan": 1}, first_free_id=3)
    # One malformed meta entry drops alone — bindings and good meta survive.
    assert m.id_for("mike") == 3 and m.id_for("mike_2") == 4
    meta = m.meta()
    assert meta["mike_2"]["base"] == "mike"
    assert "broken" not in meta


def test_idmap_mark_auto_suffixed_validates(tmp_path):
    m = IdMap(tmp_path / "_id_map.json", reserved_ids={}, first_free_id=3)
    with pytest.raises(ValueError):
        m.mark_auto_suffixed("mike", "mike")       # self-marker
    with pytest.raises(ValueError):
        m.mark_auto_suffixed("", "mike")


def test_idmap_rename_migrates_meta(tmp_path):
    p = tmp_path / "_id_map.json"
    m = IdMap(p, reserved_ids={}, first_free_id=3)
    m.allocate("too_sharp")
    m.allocate("mike_2")
    m.mark_auto_suffixed("mike_2", "mike")
    # Rename to a deliberately chosen name clears the marker.
    sid = m.rename("mike_2", "tushar")
    assert m.id_for("tushar") == sid
    assert "mike_2" not in m.meta()
    assert m.base_name("tushar") == "tushar"
    # Rename INTO a pre-resolved suffixed target sets the marker atomically.
    sid2 = m.rename("too_sharp", "tushar_2", auto_suffix_base="tushar")
    assert m.id_for("tushar_2") == sid2
    assert m.base_name("tushar_2") == "tushar"
    with pytest.raises(ValueError):
        m.rename("tushar", "zed", auto_suffix_base="zed")  # base == new


def test_idmap_retire_keeps_meta(tmp_path):
    p = tmp_path / "_id_map.json"
    m = IdMap(p, reserved_ids={}, first_free_id=3)
    m.allocate("mike_2")
    m.mark_auto_suffixed("mike_2", "mike")
    m.retire("mike_2")
    # Tombstoned forks keep their display marker (identity list rendering).
    assert m.base_name("mike_2") == "mike"
    m.revive("mike_2")
    assert m.base_name("mike_2") == "mike"
