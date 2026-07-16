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


# ---- 2026-07-02 additions: lookalike guard (live test G) ----

def test_new_name_matching_existing_face_is_lookalike(stores):
    id_map, voice, face = stores
    # Enroll dan's face.
    base = _unit(42)
    commit_identity_stores(
        "danny", id_map=id_map, face_store=face,
        face_embeddings=[base, _near(base, seed=1)])
    # STT homophone mints a "new" person from the SAME face.
    res = commit_identity_stores(
        "johnny", id_map=id_map, voice_store=voice, face_store=face,
        face_embeddings=[_near(base, seed=2)])
    assert res.status == "lookalike"
    assert res.lookalike_of == "danny"
    # Nothing minted: no id, no prototype file.
    assert id_map.id_for("johnny") is None
    assert not face.path_for("johnny").exists()


def test_new_name_matching_existing_voice_is_lookalike(stores):
    id_map, voice, face = stores
    base = _unit(77)
    commit_identity_stores(
        "carla", id_map=id_map, voice_store=voice,
        voice_embeddings=[base, _near(base, seed=3)])
    res = commit_identity_stores(
        "karla", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_near(base, seed=4)])
    assert res.status == "lookalike"
    assert res.lookalike_of == "carla"
    assert id_map.id_for("karla") is None


def test_explicit_name_tell_forks_past_lookalike(stores):
    """fork_on_lookalike (Dan 2026-07-15, Open Sauce spec 5): a visitor who
    SAID and CONFIRMED "my name is X" gets X even when their samples resemble
    an enrolled Z — the refusal trapped Tushar under a mis-heard name. The
    resemblance is surfaced (lookalike_of + warning) so the caller can speak
    the "you look like Z" disclosure."""
    id_map, voice, face = stores
    base = _unit(88)
    commit_identity_stores(
        "zora", id_map=id_map, face_store=face,
        face_embeddings=[base, _near(base, seed=6)])
    res = commit_identity_stores(
        "zola", id_map=id_map, voice_store=voice, face_store=face,
        face_embeddings=[_near(base, seed=7)],
        fork_on_lookalike=True)
    assert res.status == "ok"
    assert res.face_committed
    assert res.lookalike_of == "zora"          # disclosure input
    assert any(w.startswith("lookalike_fork:zora") for w in res.warnings)
    # A genuinely NEW identity minted alongside the existing one.
    assert id_map.id_for("zola") is not None
    assert id_map.id_for("zola") != id_map.id_for("zora")
    assert face.path_for("zola").exists()
    # S3 mismatch guard is untouched: a stranger claiming the EXISTING name
    # is still refused even with the fork flag.
    stranger = _unit(99)
    res2 = commit_identity_stores(
        "zora", id_map=id_map, face_store=face,
        face_embeddings=[stranger],
        fork_on_lookalike=True)
    assert res2.status == "mismatch"


def test_genuinely_new_person_not_lookalike(stores):
    id_map, voice, face = stores
    commit_identity_stores(
        "dave", id_map=id_map, face_store=face,
        face_embeddings=[_unit(10), _unit(11)])
    # A genuinely different face enrolls fine.
    res = commit_identity_stores(
        "eve", id_map=id_map, voice_store=voice, face_store=face,
        face_embeddings=[_unit(9100)])
    assert res.status == "ok"
    assert res.created


def test_lookalike_guard_off_without_require_match(stores):
    id_map, voice, face = stores
    base = _unit(55)
    commit_identity_stores(
        "frank", id_map=id_map, face_store=face,
        face_embeddings=[base])
    res = commit_identity_stores(
        "francis", id_map=id_map, voice_store=voice, face_store=face,
        face_embeddings=[_near(base, seed=5)],
        require_match_for_known=False)
    assert res.status == "ok"


# ---- 2026-07-02 code review C9: prototype-less known names ----

def test_prototypeless_known_name_still_guarded_as_lookalike(stores):
    """A leaked id-map entry with ZERO prototype files (the live 'erin id 5'
    shape, or any crash-after-allocate leak) must NOT exempt the name from
    the lookalike guard: an enrolled person's samples arriving under that
    name are an identity fork, not an augment."""
    id_map, voice, face = stores
    # Enroll dan's voice properly.
    base = _unit(50)
    commit_identity_stores(
        "danvoice", id_map=id_map, voice_store=voice,
        voice_embeddings=[base, _near(base, seed=5)])
    # 'erin' exists in the id-map but has no prototype files anywhere.
    id_map.allocate("erin")
    assert id_map.id_for("erin") is not None
    assert not voice.path_for("erin").exists()
    assert not face.path_for("erin").exists()
    # dan's voice claiming 'erin' must be refused, not warn-and-commit.
    res = commit_identity_stores(
        "erin", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_near(base, seed=6)])
    assert res.status == "lookalike"
    assert res.lookalike_of == "danvoice"
    assert not voice.path_for("erin").exists()


def test_prototypeless_known_name_novel_samples_commit(stores):
    """A genuinely novel person enrolling under a prototype-less id-map name
    commits fine (same semantics as a fresh name — there is no biometric
    identity to protect)."""
    id_map, voice, face = stores
    commit_identity_stores(
        "danvoice", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(60), _unit(61)])
    id_map.allocate("erin")
    prior_id = id_map.id_for("erin")
    res = commit_identity_stores(
        "erin", id_map=id_map, voice_store=voice, face_store=face,
        voice_embeddings=[_unit(7200), _unit(7201)])
    assert res.status == "ok"
    assert res.voice_committed
    assert id_map.id_for("erin") == prior_id      # id stable, no re-mint
    assert "augment_unverified" in res.warnings   # S3 had nothing to verify


# ---- 2026-07-02 code review R1: per-modality lookalike gating ----

def test_single_modality_known_name_missing_modality_guarded(stores):
    """Code review R1: a voice-only identity receiving FACE samples that
    match another enrolled face must be refused as lookalike, not
    warn-and-commit. The original per-name gate (any prototype file in
    EITHER store) skipped the scan because the voiceprint existed — leaving
    every single-modality identity open in its missing modality."""
    id_map, voice, face = stores
    # devon is known by voice only (the live household shape).
    commit_identity_stores(
        "devon", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(90), _unit(91)])
    # danface's face is enrolled.
    base = _unit(95)
    commit_identity_stores(
        "danface", id_map=id_map, face_store=face,
        face_embeddings=[base, _near(base, seed=9)])
    # A danface-lookalike face claiming devon must be refused.
    res = commit_identity_stores(
        "devon", id_map=id_map, voice_store=voice, face_store=face,
        face_embeddings=[_near(base, seed=10)])
    assert res.status == "lookalike"
    assert res.lookalike_of == "danface"
    assert not face.path_for("devon").exists()


def test_single_modality_known_name_novel_missing_modality_commits(stores):
    """R1 counterpart: a genuinely NOVEL face arriving for a voice-only
    identity still commits (with the augment_unverified warning) — the
    per-modality guard only refuses matches to OTHER identities."""
    id_map, voice, face = stores
    commit_identity_stores(
        "devon", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(90), _unit(91)])
    commit_identity_stores(
        "danface", id_map=id_map, face_store=face,
        face_embeddings=[_unit(95)])
    res = commit_identity_stores(
        "devon", id_map=id_map, voice_store=voice, face_store=face,
        face_embeddings=[_unit(9300), _unit(9301)])
    assert res.status == "ok"
    assert res.face_committed
    assert "augment_unverified" in res.warnings
    assert face.path_for("devon").exists()


def test_augment_with_prototypes_not_lookalike_refused(stores):
    """Regression guard for the C9 fix: a legit augment (name HAS prototypes,
    samples match) must still route through S3 verify, not the lookalike
    guard — even though other identities exist in the store."""
    id_map, voice, face = stores
    base = _unit(80)
    commit_identity_stores(
        "walt", id_map=id_map, voice_store=voice,
        voice_embeddings=[base, _near(base, seed=7)])
    commit_identity_stores(
        "skyler", id_map=id_map, voice_store=voice,
        voice_embeddings=[_unit(8100), _unit(8101)])
    res = commit_identity_stores(
        "walt", id_map=id_map, voice_store=voice,
        voice_embeddings=[_near(base, seed=8)])
    assert res.status == "ok"
    assert not res.created
    assert res.voice_committed
