"""Modality-agnostic K-prototype identity machinery.

Shared by VOICE (``speaker/identifier.py``, WeSpeaker embeddings) and FACE
(``presence/face_identifier.py``, EdgeFace embeddings). Holds ONLY vector-space
bookkeeping and on-disk persistence — no encoder, and none of the
modality-specific gating (voice continuity / open-set / drift; face LED-speaker
gating / landmark-quality reject). An identity is a SET of L2-normalized
prototype vectors; a probe matches on the MINIMUM cosine distance across the
set. This mirrors the multi-prototype face-ID and WeSpeaker designs: one
averaged vector can't span the pose/distance/loudness variation of a real room,
but min-over-K recognizes any covered look.

Thresholds are NOT defined here. Each modality passes its own calibrated
``dedup_dist`` / ``max_protos`` because WeSpeaker and EdgeFace live on different
cosine scales (see ``lt-wespeaker-threshold-calibration`` and the EdgeFace
calibration sweep). Extracted 2026-06-30 from ``speaker/identifier.py`` so the
subtle id-map allocation and prototype dedup/cap logic have exactly ONE
implementation — Postgres ``db/speakers.py`` reconciles against it.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

log = logging.getLogger(__name__)

# Shared identity-name rules (both modalities). Names reserved/refused as an
# enrolled identity, and the valid on-disk name pattern (alpha start, 2-32 chars).
RESERVED_NAMES = frozenset({
    "timmy", "system", "unknown", "the", "a", "an",
    "you", "me", "i", "user", "assistant",
})
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


def is_valid_enroll_name(name: str) -> bool:
    """Single module-level validator for enrollable identity names (code
    review C19 — the predicate was inlined at four call sites, inviting
    drift). Canonicalizes (lowercase/strip) before checking."""
    clean = (name or "").strip().lower()
    return clean not in RESERVED_NAMES and bool(NAME_RE.match(clean))


class RetiredNameError(ValueError):
    """Raised when ``IdMap.allocate`` is asked to mint an id for a RETIRED
    name. Retirement is a positive, persisted state (a tombstone) — not mere
    absence — precisely so no sync/loader path can silently resurrect a
    deleted identity (the 2026-07-02 "john" lesson). Revive explicitly via
    ``IdMap.revive`` if the name should come back."""


# ── Pure vector-space helpers ────────────────────────────────────────────────

def min_cosine_distance(emb: np.ndarray, prototypes: np.ndarray) -> float:
    """Minimum cosine distance from ``emb`` to any prototype row (K, D).

    0 = identical, 2 = opposite. Prototypes are assumed L2-normalized rows;
    ``emb`` need not be (cosine ignores magnitude)."""
    return float(min(cosine(emb, p) for p in prototypes))


def build_prototypes(embeddings: list, *, dedup_dist: float,
                     max_protos: int) -> np.ndarray:
    """Turn a list of raw embeddings into a deduped, capped (K, D) prototype set.

    Each embedding is L2-normalized; one is dropped if it sits within
    ``dedup_dist`` cosine of an already-kept prototype (near-duplicate
    pose/distance). Capped at ``max_protos`` (keeping the earliest kept). Falls
    back to the L2-normalized mean if every sample deduped away."""
    kept: list[np.ndarray] = []
    for e in embeddings:
        e = e / np.linalg.norm(e)
        if any(cosine(e, k) < dedup_dist for k in kept):
            continue
        kept.append(e)
        if len(kept) >= max_protos:
            break
    if not kept:
        m = np.mean(embeddings, axis=0)
        m /= np.linalg.norm(m)
        kept = [m]
    return np.vstack(kept)


def merge_prototypes(existing: np.ndarray, embeddings: list, *,
                     dedup_dist: float, max_protos: int) -> tuple[np.ndarray, int]:
    """Append new embeddings to an existing (K, D) prototype set as additional
    prototypes, deduping against existing rows (within ``dedup_dist``) and
    capping at ``max_protos`` (keeping the most recent beyond the cap).

    Pure: returns ``(new_prototypes, added_count)`` and does NOT persist. Caller
    owns persistence (so the same math serves the persist and dry-run paths)."""
    protos = existing
    added = 0
    for e in embeddings:
        e = e / np.linalg.norm(e)
        if any(cosine(e, p) < dedup_dist for p in protos):
            continue
        protos = np.vstack([protos, e[None, :]])
        added += 1
    if added == 0:
        return existing, 0
    if protos.shape[0] > max_protos:
        protos = protos[-max_protos:]
    return protos, added


# ── Prototype file store (.npy per identity) ─────────────────────────────────

class PrototypeFileStore:
    """Loads/persists ``<dir>/<name><suffix>.npy`` prototype sets for one modality.

    e.g. voice: dir=models/speaker, suffix=``_wespeaker``; face: dir=models/face,
    suffix=``_edgeface``. Names are validated/normalized (lowercase) and refused
    if reserved or malformed. Writes are atomic (np.save to a fresh file, backing
    up any prior file as ``.bak.<unix_ts>``)."""

    def __init__(self, directory: Path, suffix: str, *,
                 reserved_names: frozenset, name_re: re.Pattern):
        self.dir = Path(directory)
        self.suffix = suffix
        self.reserved_names = reserved_names
        self.name_re = name_re

    def path_for(self, name: str) -> Path:
        return self.dir / f"{name.strip().lower()}{self.suffix}.npy"

    def valid_name(self, name: str) -> bool:
        clean = (name or "").strip().lower()
        return clean not in self.reserved_names and bool(self.name_re.match(clean))

    def iter_prototype_files(self):
        """Yield (name, path) for every ``*<suffix>.npy`` (skips .bak/.disabled —
        the glob requires the path to end in ``<suffix>.npy`` exactly)."""
        for path in sorted(self.dir.glob(f"*{self.suffix}.npy")):
            name = path.stem[: -len(self.suffix)].lower() if self.suffix else path.stem.lower()
            yield name, path

    @staticmethod
    def load(path: Path) -> np.ndarray:
        """Load a prototype set, coercing a legacy 1-D single vector to (1, D)."""
        protos = np.load(path)
        if protos.ndim == 1:
            protos = protos[None, :]
        return protos

    def persist(self, name: str, prototypes: np.ndarray, *, backup: bool = True) -> Path:
        """Atomic write of a prototype set. Accepts (D,) or (K, D); always stored
        2-D as float32 so older callers stay valid. Refuses reserved/malformed
        names. Backs up any existing file as ``.bak.<unix_ts>`` when backup=True."""
        clean = (name or "").strip().lower()
        if not self.valid_name(clean):
            raise ValueError(f"refusing to persist prototypes as {clean!r}")
        if not isinstance(prototypes, np.ndarray) or prototypes.ndim not in (1, 2):
            raise ValueError("prototypes must be a 1-D or 2-D ndarray")
        if prototypes.ndim == 1:
            prototypes = prototypes[None, :]
        self.dir.mkdir(parents=True, exist_ok=True)
        out = self.path_for(clean)
        if backup and out.exists():
            bak = self.dir / f"{clean}{self.suffix}.npy.bak.{int(time.time())}"
            out.rename(bak)
            log.info("Backed up existing prototypes: %s -> %s", out.name, bak.name)
        np.save(out, prototypes.astype(np.float32))
        log.info("Persisted prototypes: %s (%d prototype(s), %d-dim)",
                 out, prototypes.shape[0], prototypes.shape[1])
        return out


# ── Shared name -> speaker_id map ────────────────────────────────────────────

class IdMap:
    """One canonical ``name -> speaker_id`` map, shared by ALL modalities.

    Deliberately a SINGLE id space across face + voice: a person has ONE
    speaker_id regardless of which biometric enrolled them, so Postgres
    ``facts.speaker_id`` / ``memories.speaker_id`` FKs (reconciled in
    ``db/speakers.py`` against :meth:`enrolled_ids`) bind both modalities to the
    same identity. Two id-maps would multiply the Devon/Devin split-identity
    failure class.

    Backed by a JSON file (default the existing ``models/speaker/_id_map.json``
    to avoid a migration). Reserved names keep fixed ids regardless of the file;
    a ``_next_id`` key tracks the next free id. Writes are atomic (tmp + fsync +
    rename).

    **Tombstones (2026-07-06):** a ``_retired`` section holds
    ``name -> {"id": int, "at": unix_ts}`` for deleted identities. Deletion
    used to be representable only as *absence*, and every healer
    (``db/speakers.py`` startup sync, ``load_voiceprints`` stray-file
    allocation, room-ledger reload) read absence-in-one-store as damage to
    repair — the 4-store resurrection class. A tombstone makes "deleted" a
    positive fact that syncs *propagate* instead of undo. Retired ids are
    NEVER reused (S1: ``facts``/``memories`` FKs stay unambiguous forever)
    and ``allocate`` refuses a retired name with :class:`RetiredNameError`."""

    _NEXT_KEY = "_next_id"
    _RETIRED_KEY = "_retired"

    def __init__(self, path: Path, *, reserved_ids: dict, first_free_id: int):
        self.path = Path(path)
        self.reserved_ids = {k.lower(): int(v) for k, v in reserved_ids.items()}
        self.first_free_id = int(first_free_id)

    def read(self) -> dict:
        """name(lower) -> int id (plus the ``_next_id`` / ``_retired``
        bookkeeping keys), or {}. ``_retired`` round-trips as a dict so a
        caller that read()s, mutates names, and write()s back (e.g.
        ``load_voiceprints``) can never drop tombstones."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path) as f:
                raw = json.load(f)
            out = {}
            for k, v in raw.items():
                if k == self._NEXT_KEY:
                    out[k] = int(v)
                elif k == self._RETIRED_KEY:
                    out[k] = {
                        str(n).lower(): {"id": int(info["id"]),
                                         "at": float(info.get("at", 0.0))}
                        for n, info in dict(v).items()
                    }
                else:
                    out[k.lower()] = int(v)
            return out
        except Exception as e:
            log.warning("Failed to read %s: %s; treating as empty", self.path.name, e)
            return {}

    def write(self, mapping: dict) -> None:
        """Persist atomically (tmp + fsync + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(mapping, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)

    def enrolled_ids(self) -> dict:
        """``name -> speaker_id`` for every registered identity WITHOUT loading any
        prototype/encoder. Reserved ids are always present; ``_next_id`` excluded.
        This is the source of truth the Postgres ``speakers`` table is reconciled
        against, so an enrolled biometric can never FK-fail a facts insert."""
        mapping = {n: i for n, i in self.read().items()
                   if n not in (self._NEXT_KEY, self._RETIRED_KEY)
                   and isinstance(i, int)}
        mapping.update(self.reserved_ids)
        return mapping

    # ── Retirement (tombstones) ──────────────────────────────────────────────

    def retired(self) -> dict:
        """``name -> {"id": int, "at": unix_ts}`` for every retired identity."""
        return self.read().get(self._RETIRED_KEY, {})

    def is_retired(self, name: str) -> bool:
        clean = (name or "").strip().lower()
        return clean in self.retired()

    def retire(self, name: str, *, at: float | None = None) -> int | None:
        """Move ``name`` from the active map into the ``_retired`` tombstone
        section, preserving its id (never reused). Idempotent: an already
        retired name returns its tombstoned id. Reserved names refuse
        (ValueError). Returns the retired id, or None if the name has no id
        anywhere (nothing to tombstone)."""
        clean = (name or "").strip().lower()
        if clean in self.reserved_ids:
            raise ValueError(f"refusing to retire reserved identity {clean!r}")
        m = self.read()
        retired = m.setdefault(self._RETIRED_KEY, {})
        if clean in retired:
            return retired[clean]["id"]
        sid = m.pop(clean, None)
        if not isinstance(sid, int):
            return None
        retired[clean] = {"id": sid, "at": float(at if at is not None else time.time())}
        self.write(m)
        log.info("Retired speaker_id=%d for %s (tombstoned)", sid, clean)
        return sid

    def revive(self, name: str) -> int | None:
        """Move ``name`` back from ``_retired`` to the active map under its
        ORIGINAL id. Returns the id, or None if no tombstone exists (an
        already-active name returns its active id, idempotently)."""
        clean = (name or "").strip().lower()
        m = self.read()
        retired = m.get(self._RETIRED_KEY, {})
        if clean not in retired:
            existing = m.get(clean)
            return existing if isinstance(existing, int) else None
        sid = retired.pop(clean)["id"]
        m[clean] = sid
        self.write(m)
        log.info("Revived speaker_id=%d for %s", sid, clean)
        return sid

    def id_for(self, name: str) -> int | None:
        """Return the existing id for ``name`` (reserved or mapped), else None."""
        clean = (name or "").strip().lower()
        if clean in self.reserved_ids:
            return self.reserved_ids[clean]
        v = self.read().get(clean)
        return int(v) if isinstance(v, int) else None

    def allocate(self, name: str) -> int:
        """Return the id for ``name``, allocating + persisting a new one if needed.
        Reserved names return their fixed id without touching the file. Idempotent:
        the FK-precondition step of a cross-store identity commit.

        Raises :class:`RetiredNameError` for a tombstoned name — deletion must
        never be silently undone by an allocation path (startup sync, stray-file
        load, re-enroll). Callers that can legitimately re-create the identity
        go through :meth:`revive` explicitly."""
        clean = (name or "").strip().lower()
        if clean in self.reserved_ids:
            return self.reserved_ids[clean]
        m = self.read()
        retired = m.get(self._RETIRED_KEY, {})
        if clean in retired:
            raise RetiredNameError(
                f"{clean!r} is retired (id={retired[clean]['id']}); "
                "revive explicitly before re-enrolling")
        existing = m.get(clean)
        if isinstance(existing, int):
            return existing
        next_id = max(int(m.get(self._NEXT_KEY, self.first_free_id)), self.first_free_id)
        # Skip any id already taken by a reserved, mapped, or RETIRED name —
        # a tombstoned id is burnt forever (S1: FK history stays unambiguous).
        taken = set(self.reserved_ids.values()) | {
            v for k, v in m.items()
            if k not in (self._NEXT_KEY, self._RETIRED_KEY) and isinstance(v, int)}
        taken |= {info["id"] for info in retired.values()}
        while next_id in taken:
            next_id += 1
        m[clean] = next_id
        m[self._NEXT_KEY] = next_id + 1
        # Keep reserved entries present so the file stays consistent if wiped.
        for rn, rid in self.reserved_ids.items():
            m.setdefault(rn, rid)
        self.write(m)
        log.info("Allocated new speaker_id=%d for %s", next_id, clean)
        return next_id
