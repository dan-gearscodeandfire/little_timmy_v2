"""Unified dual-modality identity commit (Phase B).

ONE cross-store writer that binds a *name* to a face and/or a voice under a
single ``speaker_id``. Replaces the three divergent enroll paths (Pi SFace
gallery via ``main._handle_enrollment``, the voiceprint face-hint streak, and
the standalone ``enroll_*.py`` scripts) with a single fail-safe commit.

Design invariants (from the plan's ranked edge cases):

* **S1 — one shared id space.** Face + voice keep ONE ``speaker_id`` from the
  shared ``models/speaker/_id_map.json`` (``IdMap``). A person is never
  renumbered, so ``facts.speaker_id`` / ``memories.speaker_id`` FKs survive an
  embedder swap or a re-enroll. Losing embeddings is fine; renumbering ids
  orphans memory.
* **S2 — partial commit is VALID, never an orphaned FK.** Order:
  (1) allocate/confirm ``speaker_id`` + upsert the Postgres ``speakers`` row
  FIRST (the FK precondition), (2) persist the voiceprint, (3) persist the face
  prototypes. A crash after (1) leaves a known-by-nothing id (harmless); after
  (2) a known-by-voice-only person; after (3) a full identity. No 2PC — partials
  are logged for reconcile, never rolled back into an FK violation.
* **S3 — already-known augments, does not recreate; a stranger cannot claim a
  known name.** If the name already exists we *augment* its prototype set. If
  the caller demands verification (``require_match_for_known``) the committing
  samples must themselves match the existing identity above threshold on at
  least one modality that already has prototypes — otherwise we refuse
  (``status="mismatch"``) so the caller can disambiguate.
* **S8 — casing.** The name is lowercased at this single boundary; callers may
  keep Title-case for display.
* **S9 — thin buffers.** Per-modality minimums (``MIN_ENROLL_SAMPLES``) are
  advisory here: we commit whatever we are given (>=1 sample) but warn below the
  minimum so the caller/log can flag a weak enroll.

The *core* (``commit_identity_stores``) is pure filesystem + numpy so it is
hermetically testable with plain embeddings and temp dirs — no encoder, no
Postgres. The *live* wrapper (``commit_identity``) adds crop embedding, the
shared in-memory identifier refresh, and the async Postgres reconcile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from presence.face_thresholds import (
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
    is_valid_enroll_name,
    merge_prototypes,
    min_cosine_distance,
)

# Voice-scale constants live in speaker.identifier (module top is light — the
# WeSpeaker/torch encoder is imported lazily inside its methods, so this import
# does NOT pull heavy deps into the hermetic test path).
from speaker.identifier import (
    KNOWN_SPEAKER_THRESHOLD,
    MAX_PROTOTYPES,
    MIN_ENROLL_SAMPLES,
    PROTOTYPE_DEDUP_DIST,
    VOICEPRINT_DIR,
)

log = logging.getLogger(__name__)

# The shared id space (must agree with speaker.identifier / face_identifier).
_SHARED_ID_MAP = VOICEPRINT_DIR / "_id_map.json"
_RESERVED_IDS = {"dan": 1, "timmy": 2}
_FIRST_FREE_ID = 3


@dataclass
class CommitResult:
    """Outcome of a (possibly partial) identity commit.

    ``status`` is the single field callers should branch on:
      ``ok``               — everything requested committed.
      ``partial``          — id/DB fine, but a modality write failed (see warnings).
      ``mismatch``         — refused: samples don't match the claimed known name.
      ``lookalike``        — refused: NEW name, but the samples match an
                             EXISTING identity (``lookalike_of``) — likely an
                             STT homophone forking a known person (test G,
                             2026-07-02). Caller should disambiguate.
      ``invalid_name``     — refused: reserved/malformed name.
      ``nothing_to_commit``— no voice and no face samples supplied.
    """

    name: str
    speaker_id: int | None = None
    created: bool = False           # True = new identity, False = augmented existing
    voice_committed: bool = False
    face_committed: bool = False
    voice_added: int = 0            # prototypes actually added (0 if all deduped)
    face_added: int = 0
    db_synced: bool = False
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    lookalike_of: str | None = None  # set when status == "lookalike"

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "partial")


def _voice_store() -> PrototypeFileStore:
    return PrototypeFileStore(
        VOICEPRINT_DIR, "_wespeaker",
        reserved_names=RESERVED_NAMES, name_re=NAME_RE)


def _face_store(face_dir: Path | None = None) -> PrototypeFileStore:
    from presence.face_identifier import FACE_DIR
    return PrototypeFileStore(
        Path(face_dir) if face_dir else FACE_DIR, "_edgeface",
        reserved_names=RESERVED_NAMES, name_re=NAME_RE)


def _nearest_identity(store: PrototypeFileStore, embeddings: list,
                      exclude: str) -> tuple[str, float] | None:
    """Scan every enrolled identity in ``store`` (except ``exclude``) and
    return (name, best_min_cosine_distance) for the closest one, or None if
    the store has no other identities. Cheap: ~20 small .npy loads."""
    best: tuple[str, float] | None = None
    for nm, path in store.iter_prototype_files():
        if nm == exclude:
            continue
        try:
            protos = store.load(path)
        except Exception as e:  # pragma: no cover - corrupt file
            # A silently skipped identity is a hole in the lookalike guard —
            # make the degradation visible (code review C23).
            log.warning("lookalike scan: failed to load %s: %s", path.name, e)
            continue
        d = min(min_cosine_distance(e, protos) for e in embeddings)
        if best is None or d < best[1]:
            best = (nm, d)
    return best


def find_lookalike(name: str, v_embs: list, f_embs: list, *,
                   voice_store: PrototypeFileStore | None,
                   face_store: PrototypeFileStore | None,
                   voice_thr: float = KNOWN_SPEAKER_THRESHOLD,
                   face_thr: float = KNOWN_FACE_THRESHOLD,
                   ) -> tuple[str, float] | None:
    """Return (existing_name, distance) if any supplied sample matches an
    EXISTING identity other than ``name`` below its modality threshold, else
    None. Shared by the core (hermetic refusal) and the live wrapper (which
    must refuse BEFORE the S2 id/DB pre-allocation, or a refused new name
    still leaks into the id-map and resurrects at every startup sync)."""
    for embs, store, thr in ((v_embs, voice_store, voice_thr),
                             (f_embs, face_store, face_thr)):
        if not embs or store is None:
            continue
        near = _nearest_identity(store, embs, exclude=name)
        if near is not None and near[1] < thr:
            return near
    return None


def unverified_modalities(name: str, v_embs: list, f_embs: list,
                          voice_store: PrototypeFileStore | None,
                          face_store: PrototypeFileStore | None,
                          ) -> tuple[list, list]:
    """Split the supplied samples down to the modalities in which ``name``
    has NO prototype file — exactly the samples the S3 verify cannot check.

    The lookalike guard must scan these (and only these): a modality WITH
    prototypes is already policed by S3, while a prototype-less modality is
    where a stranger can claim a known name with only an 'augment_unverified'
    warning. Gating per NAME instead (any file in either store, the original
    C9 shape) left every single-modality identity open in its missing
    modality — voice-only 'devon' would accept an unverified face because
    her voiceprint existed (code review R1). Covers the same C9 cases too:
    new names and prototype-less id-map leaks have no files anywhere, so
    every supplied modality is scanned."""
    scan_v = v_embs if (voice_store is None
                        or not voice_store.path_for(name).exists()) else []
    scan_f = f_embs if (face_store is None
                        or not face_store.path_for(name).exists()) else []
    return scan_v, scan_f


def _min_distance_to(store: PrototypeFileStore, name: str,
                     embeddings: list) -> float | None:
    """Best (min) cosine distance from any of ``embeddings`` to ``name``'s
    existing prototype set, or None if the identity has no file for this
    modality (nothing to verify against)."""
    path = store.path_for(name)
    if not path.exists():
        return None
    try:
        protos = store.load(path)
    except Exception as e:  # pragma: no cover - corrupt file
        log.warning("verify: failed to load %s: %s", path.name, e)
        return None
    return min(min_cosine_distance(e, protos) for e in embeddings)


def commit_identity_stores(
    name: str, *,
    id_map: IdMap,
    voice_store: PrototypeFileStore | None = None,
    face_store: PrototypeFileStore | None = None,
    voice_embeddings: list | None = None,
    face_embeddings: list | None = None,
    augment: bool | None = None,
    require_match_for_known: bool = True,
    voice_match_thr: float = KNOWN_SPEAKER_THRESHOLD,
    face_match_thr: float = KNOWN_FACE_THRESHOLD,
    voice_dedup: float = PROTOTYPE_DEDUP_DIST,
    voice_max: int = MAX_PROTOTYPES,
    face_dedup: float = FACE_DEDUP_DIST,
    face_max: int = MAX_FACE_PROTOTYPES,
    min_voice: int = MIN_ENROLL_SAMPLES,
    min_face: int = MIN_FACE_ENROLL_SAMPLES,
    on_voice=None,
    on_face=None,
) -> CommitResult:
    """Pure-store core: canonicalize, guard, allocate id, persist prototypes.

    No Postgres, no encoder — the caller supplies already-extracted embeddings
    and the ``IdMap`` / ``PrototypeFileStore`` targets, so this is fully
    hermetic. ``on_voice(name, sid, protos)`` / ``on_face(name, sid, protos)``
    are optional callbacks the live wrapper uses to refresh the shared in-memory
    identifiers. Persistence order within this core is voice-then-face; the
    id/DB FK precondition (S2) is enforced by the live wrapper BEFORE this runs.
    """
    clean = (name or "").strip().lower()
    res = CommitResult(name=clean)

    # S8 + S1: validate against the shared name rules (reserved / pattern).
    if not is_valid_enroll_name(clean):
        res.status = "invalid_name"
        res.error = f"invalid or reserved name {clean!r}"
        return res

    v_embs = [np.asarray(e, dtype=np.float32) for e in (voice_embeddings or [])]
    f_embs = [np.asarray(e, dtype=np.float32) for e in (face_embeddings or [])]
    if not v_embs and not f_embs:
        res.status = "nothing_to_commit"
        res.error = "no voice or face samples supplied"
        return res

    known = id_map.id_for(clean) is not None
    res.created = not known
    if augment is None:
        augment = known

    # S3: a stranger must not claim an existing name. Verify the committing
    # samples resemble the known identity on any modality that already has
    # prototypes. If nothing is verifiable (e.g. adding a face to a
    # known-by-voice-only person, with no voice sample), allow but warn.
    if known and require_match_for_known:
        checks = []  # (modality, distance, threshold)
        if v_embs and voice_store is not None:
            d = _min_distance_to(voice_store, clean, v_embs)
            if d is not None:
                checks.append(("voice", d, voice_match_thr))
        if f_embs and face_store is not None:
            d = _min_distance_to(face_store, clean, f_embs)
            if d is not None:
                checks.append(("face", d, face_match_thr))
        if checks:
            matched = any(d < thr for _, d, thr in checks)
            if not matched:
                detail = ", ".join(f"{m}={d:.3f}>={thr:.2f}"
                                   for m, d, thr in checks)
                res.status = "mismatch"
                res.error = (f"samples do not match known {clean!r} ({detail}); "
                             "refusing to overwrite — disambiguate")
                log.warning("[COMMIT] mismatch: %s", res.error)
                return res
        else:
            res.warnings.append("augment_unverified")

    # Lookalike guard (test G, 2026-07-02): samples that match an EXISTING
    # identity arriving under a different name are almost always an STT
    # homophone ("Jon" -> "John") about to fork a known person into a second
    # id. Refuse; the caller disambiguates with the user. Gated PER MODALITY
    # (code review C9 + R1): scan exactly the supplied modalities in which
    # ``clean`` has no prototype file — the ones the S3 verify above could
    # not check. That covers new names, prototype-less known names ('erin'
    # id 5), and a known single-modality identity receiving samples in its
    # missing modality (voice-only 'devon' handed a stranger's face).
    # Modalities WITH prototypes are covered by S3.
    if require_match_for_known:
        scan_v, scan_f = unverified_modalities(
            clean, v_embs, f_embs, voice_store, face_store)
        hit = find_lookalike(
            clean, scan_v, scan_f,
            voice_store=voice_store, face_store=face_store,
            voice_thr=voice_match_thr, face_thr=face_match_thr) \
            if (scan_v or scan_f) else None
        if hit is not None:
            res.status = "lookalike"
            res.lookalike_of = hit[0]
            res.error = (f"samples for {clean!r} match existing "
                         f"{hit[0]!r} (d={hit[1]:.3f}); refusing to fork — "
                         "disambiguate")
            log.warning("[COMMIT] lookalike: %s", res.error)
            return res

    # (1) FK precondition — allocate/confirm the shared speaker_id. Idempotent;
    # the live wrapper has already upserted the Postgres row against this id.
    res.speaker_id = id_map.allocate(clean)

    # (2) Voice.
    if v_embs and voice_store is not None:
        if len(v_embs) < min_voice:
            res.warnings.append(f"voice_thin:{len(v_embs)}<{min_voice}")
        try:
            path = voice_store.path_for(clean)
            if augment and path.exists():
                existing = voice_store.load(path)
                protos, added = merge_prototypes(
                    existing, v_embs, dedup_dist=voice_dedup, max_protos=voice_max)
                if added:
                    voice_store.persist(clean, protos)
            else:
                protos = build_prototypes(
                    v_embs, dedup_dist=voice_dedup, max_protos=voice_max)
                voice_store.persist(clean, protos)
                added = protos.shape[0]
            res.voice_committed = True
            res.voice_added = added
            if on_voice is not None:
                on_voice(clean, res.speaker_id, protos)
        except Exception as e:
            res.warnings.append(f"voice_failed:{e}")
            log.warning("[COMMIT] voice persist failed for %s: %s", clean, e)

    # (3) Face.
    if f_embs and face_store is not None:
        if len(f_embs) < min_face:
            res.warnings.append(f"face_thin:{len(f_embs)}<{min_face}")
        try:
            path = face_store.path_for(clean)
            if augment and path.exists():
                existing = face_store.load(path)
                protos, added = merge_prototypes(
                    existing, f_embs, dedup_dist=face_dedup, max_protos=face_max)
                if added:
                    face_store.persist(clean, protos)
            else:
                protos = build_prototypes(
                    f_embs, dedup_dist=face_dedup, max_protos=face_max)
                face_store.persist(clean, protos)
                added = protos.shape[0]
            res.face_committed = True
            res.face_added = added
            if on_face is not None:
                on_face(clean, res.speaker_id, protos)
        except Exception as e:
            res.warnings.append(f"face_failed:{e}")
            log.warning("[COMMIT] face persist failed for %s: %s", clean, e)

    wanted_voice = bool(v_embs and voice_store is not None)
    wanted_face = bool(f_embs and face_store is not None)
    if (wanted_voice and not res.voice_committed) or \
       (wanted_face and not res.face_committed):
        res.status = "partial"
    return res


async def commit_identity(
    name: str, *,
    voice_embeddings: list | None = None,
    face_crops: list | None = None,
    face_embeddings: list | None = None,
    speaker_identifier=None,
    face_identifier=None,
    augment: bool | None = None,
    require_match_for_known: bool = True,
    db_sync: bool = True,
) -> CommitResult:
    """Live dual-modality commit: embed crops, upsert the Postgres row FIRST,
    then persist voice + face and refresh the shared in-memory identifiers.

    ``face_crops`` are aligned 112x112 RGB crops (embedded here via the shared
    EdgeFace encoder); pass ``face_embeddings`` instead to skip embedding.
    Returns a :class:`CommitResult` describing the (possibly partial) outcome.
    """
    clean = (name or "").strip().lower()

    # Resolve the live singletons lazily (keeps this import light).
    if face_identifier is None:
        from presence.face_identifier import get_shared_identifier
        face_identifier = get_shared_identifier()
    if speaker_identifier is None:
        from speaker.identifier import SpeakerIdentifier
        speaker_identifier = SpeakerIdentifier()

    if face_crops and face_embeddings is None:
        from presence.face_encoder import embed_batch
        try:
            face_embeddings = list(embed_batch(face_crops))
        except Exception as e:
            log.warning("[COMMIT] face embed failed for %s: %s", clean, e)
            face_embeddings = None

    # Reuse the singletons' own stores + id-map so file paths + reserved ids can
    # never drift from the running recognizers.
    id_map = face_identifier._id_map
    voice_store = speaker_identifier._store
    face_store = face_identifier._store

    from presence.face_identifier import KnownFace, accept_threshold
    from speaker.identifier import KnownSpeaker

    def _refresh_voice(nm, sid, protos):
        for ks in speaker_identifier._known_speakers:
            if ks.name == nm:
                ks.prototypes = protos
                return
        speaker_identifier._known_speakers.append(
            KnownSpeaker(speaker_id=sid, name=nm, prototypes=protos))

    def _refresh_face(nm, sid, protos):
        for kf in face_identifier._known:
            if kf.name == nm:
                kf.prototypes = protos
                return
        face_identifier._known.append(
            KnownFace(speaker_id=sid, name=nm, prototypes=protos))

    # Lookalike pre-flight (test G, 2026-07-02): must run BEFORE the S2 id/DB
    # pre-allocation below — a refused-but-preallocated new name lands in the
    # id-map, and db.speakers.sync_speakers_from_id_map re-mints its row at
    # every startup (the "john" resurrection). The core repeats this check
    # hermetically; here it just gates the allocation. Scans the SAME
    # per-modality unverified subset as the core (code review C9 + R1) — the
    # pre-flight must never be looser than the core, or a core refusal after
    # allocation leaks the id.
    if require_match_for_known and (voice_embeddings or face_embeddings):
        _v = [np.asarray(e, dtype=np.float32) for e in (voice_embeddings or [])]
        _f = [np.asarray(e, dtype=np.float32) for e in (face_embeddings or [])]
        scan_v, scan_f = unverified_modalities(
            clean, _v, _f, voice_store, face_store)
        hit = find_lookalike(
            clean, scan_v, scan_f, voice_store=voice_store,
            face_store=face_store, face_thr=accept_threshold()) \
            if (scan_v or scan_f) else None
        if hit is not None:
            res = CommitResult(name=clean, status="lookalike",
                               lookalike_of=hit[0])
            res.error = (f"samples for {clean!r} match existing "
                         f"{hit[0]!r} (d={hit[1]:.3f}); refusing to fork")
            log.warning("[COMMIT] lookalike (pre-alloc): %s", res.error)
            return res

    # S2 (1): FK precondition BEFORE any biometric write — allocate the id and
    # upsert the Postgres speakers row so a fact extracted mid-commit can't
    # FK-fail. Only meaningful for a valid, brand-new-or-known name.
    pre_ok = is_valid_enroll_name(clean)
    db_synced = False
    if pre_ok and (voice_embeddings or face_embeddings):
        id_map.allocate(clean)  # idempotent
        if db_sync:
            try:
                from db.speakers import sync_speakers_from_id_map
                await sync_speakers_from_id_map()
                db_synced = True
            except Exception as e:
                log.warning("[COMMIT] speakers DB sync failed for %s: %s "
                            "(biometrics still persist; reconcile at restart)",
                            clean, e)

    res = commit_identity_stores(
        clean,
        id_map=id_map,
        voice_store=voice_store,
        face_store=face_store,
        voice_embeddings=voice_embeddings,
        face_embeddings=face_embeddings,
        augment=augment,
        require_match_for_known=require_match_for_known,
        face_match_thr=accept_threshold(),
        on_voice=_refresh_voice,
        on_face=_refresh_face,
    )
    res.db_synced = db_synced
    log.info("[COMMIT] %s -> id=%s status=%s voice=%s(+%d) face=%s(+%d) db=%s%s",
             res.name, res.speaker_id, res.status,
             res.voice_committed, res.voice_added,
             res.face_committed, res.face_added, res.db_synced,
             (" warn=" + ",".join(res.warnings)) if res.warnings else "")
    return res
