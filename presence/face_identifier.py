"""Face recognition on okDemerzel (EdgeFace-S), replacing the Pi's SFace.

Mirrors ``speaker/identifier.py`` for the FACE modality over the shared
K-prototype base (``presence.prototype_base``): an identity is a SET of EdgeFace
prototypes; a probe matches on the MINIMUM cosine distance across them. Uses the
SAME ``models/speaker/_id_map.json`` id space as voice, so a person keeps one
``speaker_id`` across both biometrics (Postgres FKs bind either modality).

Face prototypes live at ``models/face/<name>_edgeface.npy``. Recognition emits
the existing ``FaceObservation``/``FacePrediction`` shape (``presence.types``) so
``IdentityFusion.resolve()`` (main.py:851) is a drop-in — the Pi keeps supplying
the ``BehaviorSnapshot``; only the identity source moves here.

Enrollment builds prototypes from aligned 112x112 crops (from the gallery
builder, or live LED-speaker crops in Phase B). Thresholds are the calibrated
EdgeFace constants in ``presence.face_thresholds`` (NOT the WeSpeaker scale).

`sticky` is intentionally always False: there is no Pi identity-stabilizer on
this side, so the auto-enroll streak narrows to high-band only (deliberate; see
plan). `band` is computed here against the EdgeFace-calibrated cutoffs and
carried through so fuse_identity uses honest bands rather than re-deriving.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from presence.face_encoder import embed_batch, extract_embedding
from presence.face_thresholds import (
    FACE_BAND_HIGH,
    FACE_BAND_MEDIUM,
    FACE_DEDUP_DIST,
    KNOWN_FACE_THRESHOLD,
    MAX_FACE_PROTOTYPES,
    MIN_FACE_ENROLL_SAMPLES,
)
from presence.prototype_base import (
    NAME_RE,
    RESERVED_NAMES,
    IdMap,
    PrototypeFileStore,
    build_prototypes,
    merge_prototypes,
    min_cosine_distance,
)
from presence.types import FaceObservation, FacePrediction

log = logging.getLogger(__name__)

MODELS_DIR = Path(os.path.expanduser("~/little_timmy/models"))
FACE_DIR = MODELS_DIR / "face"
# Shared id-map with voice (one speaker_id per person across both biometrics).
SHARED_ID_MAP = MODELS_DIR / "speaker" / "_id_map.json"
RESERVED_IDS = {"dan": 1, "timmy": 2}
FIRST_FREE_ID = 3


def _toggle(key, fallback):
    """Read a runtime toggle live, falling back to a static default (keeps this
    module import-clean/testable). Lazy import avoids an import cycle."""
    try:
        from persistence import runtime_toggles
        v = runtime_toggles.get(key)
        return v if v is not None else fallback
    except Exception:
        return fallback


def accept_threshold() -> float:
    """Live accept cutoff (runtime toggle `face_threshold`, default the calibrated
    KNOWN_FACE_THRESHOLD). Tunable on the day at OpenSauce with no restart."""
    return float(_toggle("face_threshold", KNOWN_FACE_THRESHOLD))


_shared = None


def get_shared_identifier():
    """Process-wide FaceIdentifier, loaded once (shared by shadow + authority
    paths so the gallery loads a single time)."""
    global _shared
    if _shared is None:
        fi = FaceIdentifier()
        fi.load()
        _shared = fi
    return _shared


def band_of(dist: float, high: float = FACE_BAND_HIGH,
            medium: float = FACE_BAND_MEDIUM) -> str:
    """Confidence band for a match distance. Cutoffs default to the calibrated
    constants; callers pass live-scaled cutoffs so the bands track the tunable
    accept threshold (medium == accept, high == 0.8 * accept)."""
    if dist < high:
        return "high"
    if dist < medium:
        return "medium"
    return "low"


@dataclass
class KnownFace:
    speaker_id: int
    name: str
    prototypes: np.ndarray  # (K, 512) L2-normalized rows

    def distance(self, emb: np.ndarray) -> float:
        return min_cosine_distance(emb, self.prototypes)


class FaceIdentifier:
    def __init__(self, face_dir: Path = FACE_DIR):
        self._store = PrototypeFileStore(
            face_dir, "_edgeface", reserved_names=RESERVED_NAMES, name_re=NAME_RE)
        self._id_map = IdMap(SHARED_ID_MAP, reserved_ids=RESERVED_IDS,
                             first_free_id=FIRST_FREE_ID)
        self._known: list[KnownFace] = []

    # ---------- load ----------
    def load(self) -> None:
        """Load every ``<name>_edgeface.npy`` into memory, id from the shared map."""
        self._known = []
        ids = self._id_map.enrolled_ids()
        for name, path in self._store.iter_prototype_files():
            try:
                protos = self._store.load(path)
            except Exception as e:
                log.warning("Failed to load face prototypes %s: %s", path.name, e)
                continue
            sid = ids.get(name) or self._id_map.allocate(name)
            self._known.append(KnownFace(speaker_id=sid, name=name, prototypes=protos))
        log.info("Face prototypes loaded: %d (%s)", len(self._known),
                 ", ".join(k.name for k in self._known) or "none")

    @property
    def known_names(self) -> list:
        return [k.name for k in self._known]

    # ---------- enroll ----------
    def enroll(self, name: str, aligned_crops: list, *, augment: bool = True) -> int:
        """Enroll/augment ``name`` from aligned 112x112 RGB crops. Builds EdgeFace
        prototypes (dedup/cap), persists to disk, allocates the shared speaker_id,
        and updates the in-memory set. Returns the prototype count. ``augment``
        merges into an existing identity (else replaces)."""
        clean = (name or "").strip().lower()
        if not self._store.valid_name(clean):
            raise ValueError(f"refusing to enroll invalid face name {clean!r}")
        if not aligned_crops:
            raise ValueError("no crops to enroll")
        embs = embed_batch(aligned_crops)
        existing = next((k for k in self._known if k.name == clean), None)
        if existing is not None and augment:
            protos, _ = merge_prototypes(
                existing.prototypes, list(embs),
                dedup_dist=FACE_DEDUP_DIST, max_protos=MAX_FACE_PROTOTYPES)
            existing.prototypes = protos
        else:
            protos = build_prototypes(
                list(embs), dedup_dist=FACE_DEDUP_DIST, max_protos=MAX_FACE_PROTOTYPES)
            sid = self._id_map.allocate(clean)
            if existing is not None:
                existing.prototypes = protos
            else:
                self._known.append(KnownFace(speaker_id=sid, name=clean, prototypes=protos))
        self._store.persist(clean, protos)
        log.info("[FACE-ENROLL] %s: %d prototype(s) from %d crop(s)",
                 clean, protos.shape[0], len(aligned_crops))
        return protos.shape[0]

    # ---------- match ----------
    def match_embedding(self, emb: np.ndarray):
        """Return (KnownFace, distance) of the nearest identity, or (None, inf)."""
        best, best_d = None, float("inf")
        for k in self._known:
            d = k.distance(emb)
            if d < best_d:
                best_d, best = d, k
        return best, best_d

    def identify_crop(self, aligned_crop: np.ndarray, bbox: tuple):
        """Recognize one aligned 112x112 RGB crop -> FacePrediction (accepted) or
        None (below threshold / no gallery). bbox is (x0,y0,x1,y1) in frame px."""
        if not self._known:
            return None
        emb = extract_embedding(aligned_crop)
        return self._prediction_from_embedding(emb, bbox)

    def _prediction_from_embedding(self, emb: np.ndarray, bbox: tuple):
        best, d = self.match_embedding(emb)
        thr = accept_threshold()          # live, tunable at OpenSauce
        if best is None or d >= thr:
            return None
        return FacePrediction(
            user_id=best.name,
            confidence=max(0.0, 1.0 - d),
            bbox=tuple(int(v) for v in bbox),
            band=band_of(d, high=0.8 * thr, medium=thr),
            sticky=False,  # no Pi stabilizer on this side (deliberate)
        )

    def observe(self, crops_with_boxes: list, *, behavior=None,
                image_size=None, captured_at: float = 0.0) -> FaceObservation:
        """Build a FaceObservation from [(aligned_crop, bbox), ...] — the drop-in
        for IdentityFusion. Only accepted (recognized) faces become predictions;
        unrecognized crops are omitted (fusion treats absence as unknown)."""
        preds = []
        for crop, bbox in crops_with_boxes:
            p = self.identify_crop(crop, bbox)
            if p is not None:
                preds.append(p)
        return FaceObservation(
            captured_at=captured_at,
            predictions=tuple(preds),
            behavior=behavior,
            image_size=image_size,
        )
