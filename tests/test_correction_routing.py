"""Hermetic tests for classify_correction (Dan 2026-07-16 three-way ruling).

A confirmed "my name is X" protest routes by voice distance:
  rename — protester matches the DENIED record (mis-heard name, the Tushar
           trap) -> relabel in place;
  rebind — protester matches the CLAIMED identity (or a display-sibling) ->
           augment it;
  fork   — matches neither -> fresh enroll.

Temp stores + plain numpy per tests/test_identity_commit.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.identity_commit import (  # noqa: E402
    classify_correction,
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
    vdir.mkdir()
    id_map = IdMap(vdir / "_id_map.json",
                   reserved_ids={"dan": 1, "timmy": 2}, first_free_id=3)
    voice = PrototypeFileStore(vdir, "_wespeaker",
                               reserved_names=RESERVED_NAMES, name_re=NAME_RE)
    return id_map, voice


def _enroll(id_map, voice, name, base_emb):
    commit_identity_stores(
        name, id_map=id_map, voice_store=voice,
        voice_embeddings=[base_emb, _near(base_emb, seed=1)])


def test_rename_when_voice_matches_denied_record(stores):
    """The Tushar case: enrolled under the mishear, protests with the SAME
    voice -> rename in place."""
    id_map, voice = stores
    tushar = _unit(7)
    _enroll(id_map, voice, "too_sharp", tushar)
    route = classify_correction(
        "too_sharp", "tushar", [_near(tushar, seed=2)],
        attributed="too_sharp", id_map=id_map, voice_store=voice)
    assert route.branch == "rename"
    assert route.target == "tushar"
    assert route.denied_canonical == "too_sharp"
    assert route.target_base is None
    assert route.d_denied is not None and route.d_denied < 0.3


def test_rename_spoken_denied_resolves_to_attributed_fork(stores):
    """Attributed mike_2 (auto-suffix); the protest SAYS "Mike" — the denied
    token must resolve onto the attribution, not the other Mike."""
    id_map, voice = stores
    real_mike = _unit(3)
    _enroll(id_map, voice, "mike", real_mike)
    fork_voice = _unit(9000)
    _enroll(id_map, voice, "mike_2", fork_voice)
    id_map.mark_auto_suffixed("mike_2", "mike")
    route = classify_correction(
        "mike", "flynn", [_near(fork_voice, seed=2)],
        attributed="mike_2", id_map=id_map, voice_store=voice)
    assert route.branch == "rename"
    assert route.denied_canonical == "mike_2"
    assert route.target == "flynn"


def test_rename_onto_taken_name_pre_resolves_fork(stores):
    """Rename target already belongs to ANOTHER active person -> the rename
    lands on an auto-suffixed fork of the claimed name."""
    id_map, voice = stores
    walter = _unit(5)
    _enroll(id_map, voice, "walter", walter)      # mis-labelled protester
    _enroll(id_map, voice, "tushar", _unit(8000))  # a DIFFERENT tushar exists
    route = classify_correction(
        "walter", "tushar", [_near(walter, seed=2)],
        attributed="walter", id_map=id_map, voice_store=voice)
    assert route.branch == "rename"
    assert route.target == "tushar_2"
    assert route.target_base == "tushar"


def test_rebind_when_voice_matches_claimed_identity(stores):
    """Enrolled flynn mis-attributed as walter -> voice matches flynn's own
    prototypes -> rebind (augment), current behavior."""
    id_map, voice = stores
    flynn = _unit(11)
    _enroll(id_map, voice, "flynn", flynn)
    _enroll(id_map, voice, "walter", _unit(8500))
    route = classify_correction(
        "walter", "flynn", [_near(flynn, seed=2)],
        attributed="walter", id_map=id_map, voice_store=voice)
    assert route.branch == "rebind"
    assert route.target == "flynn"


def test_rebind_matches_display_sibling_of_claim(stores):
    """Claimed "mike" but the protester's record is the fork mike_2 -> the
    rebind lands on the sibling."""
    id_map, voice = stores
    _enroll(id_map, voice, "mike", _unit(3))
    fork_voice = _unit(9000)
    _enroll(id_map, voice, "mike_2", fork_voice)
    id_map.mark_auto_suffixed("mike_2", "mike")
    _enroll(id_map, voice, "walter", _unit(8500))
    route = classify_correction(
        "walter", "mike", [_near(fork_voice, seed=2)],
        attributed="walter", id_map=id_map, voice_store=voice)
    assert route.branch == "rebind"
    assert route.target == "mike_2"


def test_fork_when_voice_matches_neither(stores):
    id_map, voice = stores
    _enroll(id_map, voice, "walter", _unit(5))
    _enroll(id_map, voice, "flynn", _unit(11))
    route = classify_correction(
        "walter", "flynn", [_unit(7777)],
        attributed="walter", id_map=id_map, voice_store=voice)
    assert route.branch == "fork"
    assert route.target == "flynn"


def test_denied_without_voice_prototypes_falls_through(stores):
    """Face-only denied identity: rename branch unavailable -> (b)/(c)."""
    id_map, voice = stores
    id_map.allocate("ghost")                      # id but no voiceprint
    route = classify_correction(
        "ghost", "flynn", [_unit(7777)],
        attributed="ghost", id_map=id_map, voice_store=voice)
    assert route.branch == "fork"
    assert route.d_denied is None
