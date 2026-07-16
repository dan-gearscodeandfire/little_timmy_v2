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
      ``retired_name``     — refused: the name is tombstoned (deleted); a
                             retired identity never silently resurrects.
                             Revive explicitly (``revive_identity``) first.
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
                      exclude: str,
                      retired: frozenset = frozenset(),
                      ) -> tuple[str, float] | None:
    """Scan every enrolled identity in ``store`` (except ``exclude`` and any
    ``retired`` tombstoned name — a stray/hand-restored file must not refuse
    an enroll in a deleted persona's name, review 7-06; the loaders skip the
    same class) and return (name, best_min_cosine_distance) for the closest
    one, or None if the store has no other identities. Cheap: ~20 small
    .npy loads."""
    best: tuple[str, float] | None = None
    for nm, path in store.iter_prototype_files():
        if nm == exclude or nm in retired:
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
                   retired: frozenset = frozenset(),
                   ) -> tuple[str, float] | None:
    """Return (existing_name, distance) if any supplied sample matches an
    EXISTING identity other than ``name`` below its modality threshold, else
    None. ``retired`` tombstoned names are excluded from the scan (review
    7-06). Shared by the core (hermetic refusal) and the live wrapper (which
    must refuse BEFORE the S2 id/DB pre-allocation, or a refused new name
    still leaks into the id-map and resurrects at every startup sync)."""
    for embs, store, thr in ((v_embs, voice_store, voice_thr),
                             (f_embs, face_store, face_thr)):
        if not embs or store is None:
            continue
        near = _nearest_identity(store, embs, exclude=name, retired=retired)
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
    fork_on_lookalike: bool = False,
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

    # Tombstone guard: a retired name must not silently re-enroll (the
    # deletion analogue of the lookalike guard — same "resurrection" class).
    if id_map.is_retired(clean):
        res.status = "retired_name"
        res.error = (f"{clean!r} is retired; refusing to re-enroll — "
                     "revive explicitly if intended")
        log.warning("[COMMIT] retired_name: %s", res.error)
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
            voice_thr=voice_match_thr, face_thr=face_match_thr,
            retired=frozenset(id_map.retired())) \
            if (scan_v or scan_f) else None
        if hit is not None:
            if fork_on_lookalike:
                # Explicit name-tell (Dan 2026-07-15, Open Sauce spec 5): a
                # visitor who SAID "my name is X" gets X, even when their
                # samples resemble an enrolled Z — refusing trapped Tushar
                # under a mis-heard name with no escape. The caller speaks
                # the disclosure ("You look like Z, but I'll remember you as
                # X") via res.lookalike_of. Auto/implicit paths keep the
                # refusal: this flag is only ever set by a confirmed,
                # user-spoken name.
                res.lookalike_of = hit[0]
                res.warnings.append(f"lookalike_fork:{hit[0]}:d={hit[1]:.3f}")
                log.info("[COMMIT] lookalike FORK-ALLOWED (explicit "
                         "name-tell): %r resembles %r (d=%.3f) — committing "
                         "as new identity", clean, hit[0], hit[1])
            else:
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
    fork_on_lookalike: bool = False,
) -> CommitResult:
    """Live dual-modality commit: embed crops, upsert the Postgres row FIRST,
    then persist voice + face and refresh the shared in-memory identifiers.

    ``fork_on_lookalike``: commit a lookalike as a NEW identity instead of
    refusing — for EXPLICIT confirmed name-tells only (Dan 2026-07-15); the
    caller speaks the "you look like Z" disclosure via res.lookalike_of.

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
        import asyncio
        from presence.face_encoder import embed_batch
        try:
            # Off the event loop (F5 fix, review 7-07): embed_batch is sync
            # ONNX CPU (face_encoder.py's own docstring says to_thread) and
            # was running ON the loop here — up to 12 crops stalling every
            # concurrent coroutine mid-commit. Covers both callers
            # (introductions name-tell + main's unified enroll).
            face_embeddings = list(await asyncio.to_thread(embed_batch, face_crops))
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

    # Tombstone guard BEFORE any allocation or scan: a retired name refuses
    # outright (and allocate() below would raise RetiredNameError anyway —
    # this just turns it into a structured result the dialog can speak).
    if id_map.is_retired(clean):
        res = CommitResult(name=clean, status="retired_name")
        res.error = (f"{clean!r} is retired; refusing to re-enroll — "
                     "revive explicitly if intended")
        log.warning("[COMMIT] retired_name (pre-alloc): %s", res.error)
        return res

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
            face_store=face_store, face_thr=accept_threshold(),
            retired=frozenset(id_map.retired())) \
            if (scan_v or scan_f) else None
        if hit is not None and not fork_on_lookalike:
            res = CommitResult(name=clean, status="lookalike",
                               lookalike_of=hit[0])
            res.error = (f"samples for {clean!r} match existing "
                         f"{hit[0]!r} (d={hit[1]:.3f}); refusing to fork")
            log.warning("[COMMIT] lookalike (pre-alloc): %s", res.error)
            return res
        if hit is not None:
            log.info("[COMMIT] lookalike (pre-alloc) FORK-ALLOWED: %r "
                     "resembles %r (d=%.3f) — explicit name-tell proceeds",
                     clean, hit[0], hit[1])

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
        fork_on_lookalike=fork_on_lookalike,
    )
    res.db_synced = db_synced
    log.info("[COMMIT] %s -> id=%s status=%s voice=%s(+%d) face=%s(+%d) db=%s%s",
             res.name, res.speaker_id, res.status,
             res.voice_committed, res.voice_added,
             res.face_committed, res.face_added, res.db_synced,
             (" warn=" + ",".join(res.warnings)) if res.warnings else "")
    return res


# ── Retirement: the mirror of the sole writer ────────────────────────────────
#
# ONE cross-store deleter, same shape as the commit: a pure hermetic core
# (``retire_identity_stores``) plus a live async wrapper (``retire_identity``)
# that adds the in-memory identifier removal, the Postgres ``retired_at`` mark,
# and the room-ledger forget. Deletion is a TOMBSTONE (IdMap ``_retired``) plus
# an ARCHIVE (prototypes move to ``models/trash/``, never destroyed), so it is
# (a) propagated — every healer that used to resurrect-by-absence now sees a
# positive "deleted" fact — and (b) reversible (``revive_identity``).

# Every on-disk biometric suffix a persona may own. The PrototypeFileStore
# pair covers the ACTIVE embedders (_wespeaker/_edgeface); legacy voice files
# in VOICEPRINT_DIR would otherwise linger and confuse an audit.
_LEGACY_VOICE_SUFFIXES = ("_resemblyzer", "_pyannote")


@dataclass
class RetireResult:
    """Outcome of a persona retirement (or revival).

    ``status``: ``ok`` | ``not_found`` (no id and no files anywhere) |
    ``reserved`` (dan/timmy — refused) | ``error``.
    """

    name: str
    speaker_id: int | None = None
    status: str = "ok"
    files_moved: list[str] = field(default_factory=list)
    trash_dir: str | None = None
    db_synced: bool = False
    ledger_forgotten: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _persona_files(name: str, voice_store: PrototypeFileStore | None,
                   face_store: PrototypeFileStore | None) -> list[Path]:
    """Every on-disk file belonging to ``name``: active prototype sets, legacy
    voice embeddings, and their ``.bak.*`` backups. Exact-suffix matches only —
    ``dan`` must never sweep up ``dan_the_barbarian``'s files."""
    paths: list[Path] = []
    stores = [s for s in (voice_store, face_store) if s is not None]
    for store in stores:
        main = store.path_for(name)
        paths.append(main)
        paths.extend(store.dir.glob(f"{name}{store.suffix}.npy.bak.*"))
    if voice_store is not None:
        for suffix in _LEGACY_VOICE_SUFFIXES:
            paths.append(voice_store.dir / f"{name}{suffix}.npy")
            paths.extend(voice_store.dir.glob(f"{name}{suffix}.npy.bak.*"))
    return [p for p in paths if p.exists()]


def retire_identity_stores(
    name: str, *,
    id_map: IdMap,
    voice_store: PrototypeFileStore | None = None,
    face_store: PrototypeFileStore | None = None,
    trash_root: Path | None = None,
    at: float | None = None,
) -> RetireResult:
    """Pure-store core: archive the persona's biometric files into
    ``trash_root/<name>.<ts>/`` and tombstone its id-map entry.

    Hermetic (fs + json only). The tombstone is the root fix: with it in
    place, ``db/speakers.py`` marks instead of re-inserts, ``allocate``
    refuses instead of re-mints, and the loaders skip instead of re-adopt.
    """
    clean = (name or "").strip().lower()
    res = RetireResult(name=clean)

    if clean in id_map.reserved_ids:
        res.status = "reserved"
        res.error = f"refusing to retire reserved identity {clean!r}"
        return res

    already = id_map.is_retired(clean)
    sid = id_map.retired().get(clean, {}).get("id") if already \
        else id_map.id_for(clean)
    files = _persona_files(clean, voice_store, face_store)
    if sid is None and not files:
        res.status = "not_found"
        res.error = f"no id-map entry and no biometric files for {clean!r}"
        return res
    res.speaker_id = sid
    if already:
        res.warnings.append("already_retired")

    import time as _time
    ts = float(at if at is not None else _time.time())

    if files:
        dest_root = Path(trash_root) if trash_root else None
        if dest_root is None:
            res.warnings.append("no_trash_root:files_left_in_place")
        else:
            dest = dest_root / f"{clean}.{int(ts)}"
            moved: list[tuple[Path, Path]] = []
            try:
                dest.mkdir(parents=True, exist_ok=True)
                for p in files:
                    target = dest / p.name
                    p.rename(target)
                    moved.append((p, target))
                    res.files_moved.append(p.name)
                res.trash_dir = str(dest)
                log.info("[RETIRE] archived %d file(s) for %s -> %s",
                         len(res.files_moved), clean, dest)
            except Exception as e:
                # Roll the partial archive back (review 7-06): a mid-loop
                # failure used to leave files split between the live dirs
                # and trash — voice archived, face live — with no tombstone,
                # while the log claimed 'stores unchanged'.
                stuck: list[str] = []
                for orig, target in moved:
                    try:
                        target.rename(orig)
                    except Exception:
                        stuck.append(target.name)
                res.status = "error"
                res.error = f"archive failed: {e}"
                if stuck:
                    res.trash_dir = str(dest)
                    res.files_moved = [n for n in res.files_moved
                                       if n in stuck]
                    res.warnings.append(
                        "rollback_incomplete:" + ",".join(stuck))
                    log.warning("[RETIRE] archive failed for %s: %s — "
                                "ROLLBACK INCOMPLETE, %d file(s) stranded "
                                "in %s (no tombstone written)",
                                clean, e, len(stuck), dest)
                else:
                    res.files_moved.clear()
                    log.warning("[RETIRE] archive failed for %s: %s "
                                "(rolled back; stores unchanged, no "
                                "tombstone)", clean, e)
                return res

    if sid is not None and not already:
        id_map.retire(clean, at=ts)
    return res


def revive_identity_stores(
    name: str, *,
    id_map: IdMap,
    trash_root: Path,
) -> RetireResult:
    """Reverse a retirement: restore the most recent ``<name>.<ts>`` archive
    from ``trash_root`` and clear the tombstone (same id, per S1)."""
    clean = (name or "").strip().lower()
    res = RetireResult(name=clean)

    archives = sorted(Path(trash_root).glob(f"{clean}.*"),
                      key=lambda p: p.name)
    latest = archives[-1] if archives else None

    sid = id_map.revive(clean)
    res.speaker_id = sid
    if sid is None and latest is None:
        res.status = "not_found"
        res.error = f"no tombstone and no archive for {clean!r}"
        return res

    if latest is not None:
        from presence.face_identifier import FACE_DIR
        restored = 0
        for p in sorted(latest.iterdir()):
            # Restore by suffix: face files to the face dir, everything else
            # to the voice dir (both trees are flat .npy stores).
            target = FACE_DIR if "_edgeface" in p.name else VOICEPRINT_DIR
            dest = target / p.name
            if dest.exists():
                res.warnings.append(f"exists_skipped:{p.name}")
                continue
            p.rename(dest)
            res.files_moved.append(p.name)
            restored += 1
        if not any(latest.iterdir()):
            latest.rmdir()
        log.info("[REVIVE] restored %d file(s) for %s from %s",
                 restored, clean, latest)
    else:
        res.warnings.append("no_archive:tombstone_cleared_only")
    return res


async def retire_identity(
    name: str, *,
    speaker_identifier=None,
    face_identifier=None,
    room_ledger=None,
    db_sync: bool = True,
    purge_facts: bool = False,
) -> RetireResult:
    """Live persona retirement: stop recognition NOW (in-memory removal),
    archive + tombstone the stores, mark Postgres ``speakers.retired_at``,
    and forget the room-ledger record.

    ``purge_facts=True`` additionally HARD-DELETES the persona's ``facts``
    rows (speaker_id or subject match) — for scrubbing test-minted junk, not
    for real people (their facts stay as inert FK-intact history; a retired
    persona never resolves again, so they never surface)."""
    clean = (name or "").strip().lower()

    if face_identifier is None:
        from presence.face_identifier import get_shared_identifier
        face_identifier = get_shared_identifier()
    if speaker_identifier is None:
        from speaker.identifier import SpeakerIdentifier
        speaker_identifier = SpeakerIdentifier()

    id_map = face_identifier._id_map
    voice_store = speaker_identifier._store
    face_store = face_identifier._store

    # (1) In-memory first — recognition stops immediately, and a concurrent
    # identify() can't re-observe the persona mid-retire. Capture what we
    # remove: the guards (reserved/not_found/archive-error) live in the
    # store core below, and a REFUSED retire must put recognition back —
    # review 7-06: retire('dan') was refused as reserved yet left the live
    # process blind to Dan until restart.
    _removed_ks = [ks for ks in speaker_identifier._known_speakers
                   if ks.name == clean]
    _removed_kf = [kf for kf in face_identifier._known if kf.name == clean]
    speaker_identifier._known_speakers = [
        ks for ks in speaker_identifier._known_speakers if ks.name != clean]
    face_identifier._known = [
        kf for kf in face_identifier._known if kf.name != clean]
    _removed_embs = speaker_identifier._recent_confident_embs.pop(clean, None)
    _removed_drift = speaker_identifier._drift_buffers.pop(clean, None)
    _removed_reenroll = None
    active_re = speaker_identifier._active_reenrollment
    if active_re and active_re.get("name") == clean:
        _removed_reenroll = active_re
        speaker_identifier._active_reenrollment = None

    # (2) Stores: archive + tombstone.
    trash_root = VOICEPRINT_DIR.parent / "trash"
    res = retire_identity_stores(
        clean, id_map=id_map, voice_store=voice_store, face_store=face_store,
        trash_root=trash_root)
    if not res.ok:
        # Refused or failed — restore live recognition to match the stores.
        speaker_identifier._known_speakers.extend(_removed_ks)
        face_identifier._known.extend(_removed_kf)
        if _removed_embs is not None:
            speaker_identifier._recent_confident_embs[clean] = _removed_embs
        if _removed_drift is not None:
            speaker_identifier._drift_buffers[clean] = _removed_drift
        if _removed_reenroll is not None:
            speaker_identifier._active_reenrollment = _removed_reenroll
        return res

    # (3) Postgres: mark retired (facts/memories FKs stay intact as history).
    if db_sync and res.speaker_id is not None:
        try:
            from db.connection import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE speakers SET retired_at = NOW() "
                    "WHERE id = $1 AND retired_at IS NULL", res.speaker_id)
                if purge_facts:
                    # Clear self-FK references first, then delete.
                    deleted = await conn.execute(
                        "WITH doomed AS (SELECT id FROM facts "
                        "  WHERE speaker_id = $1 OR lower(subject) = $2), "
                        "unlinked AS (UPDATE facts SET superseded_by = NULL "
                        "  WHERE superseded_by IN (SELECT id FROM doomed)) "
                        "DELETE FROM facts WHERE id IN (SELECT id FROM doomed)",
                        res.speaker_id, clean)
                    res.warnings.append(f"facts_purged:{deleted}")
            res.db_synced = True
        except Exception as e:
            res.warnings.append(f"db_failed:{e}")
            log.warning("[RETIRE] speakers DB mark failed for %s: %s "
                        "(tombstone holds; startup sync reconciles)", clean, e)

    # (4) Presence: forget the ledger record so it can't re-mint at reload.
    if room_ledger is not None:
        try:
            res.ledger_forgotten = room_ledger.forget(clean)
        except Exception as e:
            res.warnings.append(f"ledger_failed:{e}")

    log.info("[RETIRE] %s -> id=%s status=%s files=%d db=%s ledger=%s%s",
             res.name, res.speaker_id, res.status, len(res.files_moved),
             res.db_synced, res.ledger_forgotten,
             (" warn=" + ",".join(res.warnings)) if res.warnings else "")
    return res


async def revive_identity(
    name: str, *,
    speaker_identifier=None,
    face_identifier=None,
    db_sync: bool = True,
) -> RetireResult:
    """Reverse a retirement live: restore the archive, clear the tombstone
    and the Postgres mark, and reload the restored prototypes into the
    in-memory identifiers (no restart)."""
    clean = (name or "").strip().lower()

    if face_identifier is None:
        from presence.face_identifier import get_shared_identifier
        face_identifier = get_shared_identifier()
    if speaker_identifier is None:
        from speaker.identifier import SpeakerIdentifier
        speaker_identifier = SpeakerIdentifier()

    id_map = face_identifier._id_map
    voice_store = speaker_identifier._store
    face_store = face_identifier._store
    trash_root = VOICEPRINT_DIR.parent / "trash"

    res = revive_identity_stores(clean, id_map=id_map, trash_root=trash_root)
    if not res.ok:
        return res

    from presence.face_identifier import KnownFace
    from speaker.identifier import KnownSpeaker

    sid = res.speaker_id
    vpath = voice_store.path_for(clean)
    if sid is not None and vpath.exists() and not any(
            ks.name == clean for ks in speaker_identifier._known_speakers):
        try:
            speaker_identifier._known_speakers.append(KnownSpeaker(
                speaker_id=sid, name=clean, prototypes=voice_store.load(vpath)))
        except Exception as e:
            res.warnings.append(f"voice_reload_failed:{e}")
    fpath = face_store.path_for(clean)
    if sid is not None and fpath.exists() and not any(
            kf.name == clean for kf in face_identifier._known):
        try:
            face_identifier._known.append(KnownFace(
                speaker_id=sid, name=clean, prototypes=face_store.load(fpath)))
        except Exception as e:
            res.warnings.append(f"face_reload_failed:{e}")

    if db_sync and sid is not None:
        try:
            from db.connection import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE speakers SET retired_at = NULL WHERE id = $1", sid)
            res.db_synced = True
        except Exception as e:
            res.warnings.append(f"db_failed:{e}")

    log.info("[REVIVE] %s -> id=%s status=%s files=%d db=%s%s",
             res.name, res.speaker_id, res.status, len(res.files_moved),
             res.db_synced,
             (" warn=" + ",".join(res.warnings)) if res.warnings else "")
    return res


# ── Rename: relabel in place, id preserved ───────────────────────────────────
#
# The correction path for a mis-heard enrolled name (Dan 2026-07-15, Open
# Sauce spec 9: "Tushar" stuck as "too_sharp" with no escape). Same core/
# wrapper split as retire: a hermetic store core plus a live async wrapper
# that adds the in-memory relabel, the Postgres name update, and the fact-
# subject rewrite. The speaker_id NEVER changes — that is the whole point
# over retire+re-enroll (facts/memories FK history stays valid).

@dataclass
class RenameResult:
    old: str
    new: str
    speaker_id: int | None = None
    status: str = "ok"
    error: str | None = None
    files_moved: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    db_synced: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def rename_identity_stores(
    old: str, new: str, *,
    id_map: IdMap,
    voice_store: PrototypeFileStore | None = None,
    face_store: PrototypeFileStore | None = None,
) -> RenameResult:
    """Hermetic core: relabel the id-map entry (id preserved) and rename the
    active prototype files. Rolls the files back if the id-map rename or a
    later file move fails. Historic ``.bak`` files keep the old name (they
    are timestamped archives, not live prototypes)."""
    o = (old or "").strip().lower()
    n = (new or "").strip().lower()
    res = RenameResult(old=o, new=n)

    moves: list[tuple] = []   # (src, dst) executed, for rollback
    for store in (voice_store, face_store):
        if store is None:
            continue
        src = store.path_for(o)
        dst = store.path_for(n)
        if not src.exists():
            continue
        if dst.exists():
            res.status = "conflict"
            res.error = f"target file already exists: {dst.name}"
            break
        moves.append((src, dst))

    if res.status == "ok":
        done: list[tuple] = []
        try:
            for src, dst in moves:
                src.rename(dst)
                done.append((src, dst))
                res.files_moved.append(f"{src.name} -> {dst.name}")
            res.speaker_id = id_map.rename(o, n)
        except Exception as e:
            for src, dst in reversed(done):
                try:
                    dst.rename(src)
                except Exception:
                    res.warnings.append(f"rollback_stuck:{dst.name}")
            res.files_moved.clear()
            res.status = "error"
            res.error = str(e)

    if res.status != "ok":
        log.warning("[RENAME] %s -> %s refused/failed: %s", o, n, res.error)
    return res


async def rename_identity(
    old: str, new: str, *,
    speaker_identifier=None,
    face_identifier=None,
    room_ledger=None,
    db_sync: bool = True,
) -> RenameResult:
    """Live rename: stores + id-map (core), in-memory identifiers relabelled
    in place (recognition flips to the new name on the very next turn — no
    restart), Postgres ``speakers.name`` updated, and the persona's fact
    SUBJECTS rewritten (``_normalize_subject`` keys facts by speaker name,
    so ``too_sharp.likes = robots`` must become ``tushar.likes = robots``)."""
    o = (old or "").strip().lower()
    n = (new or "").strip().lower()

    if face_identifier is None:
        from presence.face_identifier import get_shared_identifier
        face_identifier = get_shared_identifier()
    if speaker_identifier is None:
        from speaker.identifier import SpeakerIdentifier
        speaker_identifier = SpeakerIdentifier()

    id_map = face_identifier._id_map
    res = rename_identity_stores(
        o, n, id_map=id_map,
        voice_store=speaker_identifier._store,
        face_store=face_identifier._store)
    if not res.ok:
        return res

    # In-memory relabel — recognition continuity, no reload needed (the
    # loaded prototypes are unchanged, only the label moves).
    for ks in speaker_identifier._known_speakers:
        if ks.name == o:
            ks.name = n
    for kf in face_identifier._known:
        if kf.name == o:
            kf.name = n
    for attr in ("_recent_confident_embs", "_drift_buffers"):
        d = getattr(speaker_identifier, attr, None)
        if isinstance(d, dict) and o in d:
            d[n] = d.pop(o)

    if db_sync and res.speaker_id is not None:
        try:
            from db.connection import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE speakers SET name = $1 WHERE id = $2",
                    n, res.speaker_id)
                rewritten = await conn.execute(
                    "UPDATE facts SET subject = $1 WHERE lower(subject) = $2",
                    n, o)
                res.warnings.append(f"fact_subjects:{rewritten}")
            res.db_synced = True
        except Exception as e:
            res.warnings.append(f"db_failed:{e}")
            log.warning("[RENAME] DB update failed for %s->%s: %s "
                        "(stores renamed; startup sync reconciles speakers)",
                        o, n, e)

    if room_ledger is not None:
        try:
            room_ledger.forget(o)   # ages back in under the new name
        except Exception as e:
            res.warnings.append(f"ledger_failed:{e}")

    log.info("[RENAME] %s -> %s id=%s status=%s files=%d db=%s%s",
             o, n, res.speaker_id, res.status, len(res.files_moved),
             res.db_synced,
             (" warn=" + ",".join(res.warnings)) if res.warnings else "")
    return res
