"""FaceIdentifier enroll/match/reject mechanics (hermetic: encoder patched).

Recognition QUALITY (does it actually tell people apart) is validated separately
against the real galleries in ops/edgeface_calibrate + build_maker_gallery; this
locks the identifier's logic: prototype build/persist, shared-id-map allocation,
min-cosine match, threshold rejection, band assignment, augment.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence import face_identifier as FI  # noqa: E402
from presence.prototype_base import IdMap, PrototypeFileStore  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def ident(tmp_path, monkeypatch):
    """FaceIdentifier on temp dirs, with a deterministic fake encoder mapping a
    crop's mean-color to a fixed embedding direction (distinct 'identities')."""
    face_dir = tmp_path / "face"
    id_map = tmp_path / "_id_map.json"
    # Patch encoder: embedding = L2-normalized 8-dim vector seeded by crop[0,0].
    def fake_embed_one(crop):
        seed = int(crop[0, 0, 0])
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(8)
        # tiny jitter from the whole crop so same-id crops are near but not equal
        base = base + 0.01 * np.random.default_rng(int(crop.sum()) % 7).standard_normal(8)
        return _unit(base)
    monkeypatch.setattr(FI, "extract_embedding", fake_embed_one)
    monkeypatch.setattr(FI, "embed_batch",
                        lambda crops: np.vstack([fake_embed_one(c) for c in crops]))
    monkeypatch.setattr(FI, "SHARED_ID_MAP", id_map)
    fi = FI.FaceIdentifier(face_dir=face_dir)
    fi._id_map = IdMap(id_map, reserved_ids=FI.RESERVED_IDS, first_free_id=FI.FIRST_FREE_ID)
    return fi


def _crops(seed, n):
    """n near-identical crops of 'identity' seed (crop[0,0,0]==seed)."""
    out = []
    for i in range(n):
        c = np.full((112, 112, 3), 0, dtype=np.uint8)
        c[0, 0, 0] = seed
        c[1, 1, 1] = i  # vary the sum slightly
        out.append(c)
    return out


def test_enroll_persists_and_allocates_shared_id(ident, tmp_path):
    n = ident.enroll("pat", _crops(50, 4))
    assert n >= 1
    assert (tmp_path / "face" / "pat_edgeface.npy").exists()
    # Shared id-map allocated pat an id in the shared space.
    assert ident._id_map.id_for("pat") == FI.FIRST_FREE_ID


def test_reserved_id_shared_with_voice(ident):
    ident.enroll("dan", _crops(10, 3))
    # dan is reserved id 1 in the shared map (same as voice).
    assert ident._id_map.id_for("dan") == 1


def test_match_recognizes_enrolled_identity(ident):
    ident.enroll("pat", _crops(50, 5))
    ident.enroll("sam", _crops(200, 5))
    # A fresh 'pat' crop matches pat, not sam.
    pred = ident.identify_crop(_crops(50, 1)[0], (0, 0, 50, 50))
    assert pred is not None and pred.user_id == "pat"
    assert pred.band in ("high", "medium")
    assert pred.sticky is False


def test_rejects_unknown_below_threshold(ident, monkeypatch):
    ident.enroll("pat", _crops(50, 5))
    # Force a far embedding for the probe -> distance >= threshold -> None.
    monkeypatch.setattr(FI, "extract_embedding", lambda crop: _unit([1, 0, 0, 0, 0, 0, 0, 0]))
    monkeypatch.setattr(ident, "_known", ident._known)
    ident._known[0].prototypes = np.vstack([_unit([0, 0, 0, 0, 0, 0, 0, 1])])
    pred = ident.identify_crop(np.zeros((112, 112, 3), np.uint8), (0, 0, 1, 1))
    assert pred is None  # orthogonal -> dist ~1.0 >= threshold


def test_empty_gallery_returns_none(ident):
    assert ident.identify_crop(np.zeros((112, 112, 3), np.uint8), (0, 0, 1, 1)) is None


def test_load_roundtrip(ident, tmp_path, monkeypatch):
    ident.enroll("pat", _crops(50, 4))
    ident.enroll("sam", _crops(200, 4))
    # Fresh identifier loads persisted prototypes + ids.
    fi2 = FI.FaceIdentifier(face_dir=tmp_path / "face")
    fi2._id_map = ident._id_map
    fi2.load()
    assert set(fi2.known_names) == {"pat", "sam"}


def test_band_of_cutoffs():
    assert FI.band_of(FI.FACE_BAND_HIGH - 0.01) == "high"
    assert FI.band_of((FI.FACE_BAND_HIGH + FI.FACE_BAND_MEDIUM) / 2) == "medium"
    assert FI.band_of(FI.FACE_BAND_MEDIUM + 0.01) == "low"
