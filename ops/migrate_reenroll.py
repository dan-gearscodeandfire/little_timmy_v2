"""Phase C — name-preserving identity audit + reconcile.

The embedder swap (SFace -> EdgeFace) invalidated the old FACE embeddings, and
the new noise-reducing mic changes the VOICE domain, so some identities need a
fresh biometric enroll. The one thing that must NEVER change is the
``speaker_id``: ``facts.speaker_id`` / ``memories.speaker_id`` FK into it, so a
renumber orphans a person's whole memory (the Devon/Devin failure class). This
tool migrates *names*, never vectors or ids.

What it does:
  * Reads the shared id-map (``models/speaker/_id_map.json``) — the single
    source of truth for ``name -> speaker_id``.
  * Reports, per identity, which stores currently back it: voice
    (``<name>_wespeaker.npy``), face (``<name>_edgeface.npy``), and the id.
  * Flags identities missing a modality (need a fresh enroll to become
    dual-modality) and any prototype files with no id-map entry (orphan vectors).
  * ``--reconcile-db`` upserts the Postgres ``speakers`` row for every id-map
    name (idempotent, id-stable, name-preserving — never renames/renumbers).
  * ``--backup <name>`` renames a person's stale prototype file(s) to
    ``.bak.<ts>`` so the next enroll rebuilds them from scratch WITHOUT touching
    the id-map (id stays, memory survives, biometric is refreshed).

Read-only by default. Run:
    python -m ops.migrate_reenroll                 # audit
    python -m ops.migrate_reenroll --json
    python -m ops.migrate_reenroll --reconcile-db
    python -m ops.migrate_reenroll --backup erin --modality voice
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.prototype_base import (  # noqa: E402
    NAME_RE,
    RESERVED_NAMES,
    IdMap,
    PrototypeFileStore,
)
from speaker.identifier import VOICEPRINT_DIR  # noqa: E402

_RESERVED_IDS = {"dan": 1, "timmy": 2}
_FIRST_FREE_ID = 3


@dataclass
class IdentityRow:
    name: str
    speaker_id: int | None
    has_voice: bool
    has_face: bool

    @property
    def modalities(self) -> str:
        m = []
        if self.has_voice:
            m.append("voice")
        if self.has_face:
            m.append("face")
        return "+".join(m) or "NONE"

    @property
    def complete(self) -> bool:
        return self.has_voice and self.has_face


def audit(id_map: IdMap, voice_store: PrototypeFileStore,
          face_store: PrototypeFileStore) -> dict:
    """Pure audit of the three name spaces. Returns a report dict:
    ``{identities: [IdentityRow...], orphan_voice: [...], orphan_face: [...]}``.
    An orphan is a prototype file whose name has no id-map entry."""
    ids = id_map.enrolled_ids()  # name -> id (reserved always present)
    voice_names = {n for n, _ in voice_store.iter_prototype_files()}
    face_names = {n for n, _ in face_store.iter_prototype_files()}

    names = sorted(set(ids) | voice_names | face_names)
    rows = []
    for name in names:
        rows.append(IdentityRow(
            name=name,
            speaker_id=ids.get(name),
            has_voice=name in voice_names,
            has_face=name in face_names,
        ))
    orphan_voice = sorted(voice_names - set(ids))
    orphan_face = sorted(face_names - set(ids))
    return {"identities": rows, "orphan_voice": orphan_voice,
            "orphan_face": orphan_face}


def _voice_store() -> PrototypeFileStore:
    return PrototypeFileStore(VOICEPRINT_DIR, "_wespeaker",
                              reserved_names=RESERVED_NAMES, name_re=NAME_RE)


def _face_store() -> PrototypeFileStore:
    from presence.face_identifier import FACE_DIR
    return PrototypeFileStore(FACE_DIR, "_edgeface",
                              reserved_names=RESERVED_NAMES, name_re=NAME_RE)


def _id_map() -> IdMap:
    return IdMap(VOICEPRINT_DIR / "_id_map.json",
                 reserved_ids=_RESERVED_IDS, first_free_id=_FIRST_FREE_ID)


def _backup(store: PrototypeFileStore, name: str) -> Path | None:
    path = store.path_for(name)
    if not path.exists():
        return None
    bak = path.with_name(f"{path.name}.bak.{int(time.time())}")
    path.rename(bak)
    return bak


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Name-preserving identity audit/reconcile")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--reconcile-db", action="store_true",
                    help="upsert Postgres speakers rows from the id-map (id-stable)")
    ap.add_argument("--backup", metavar="NAME",
                    help="rename a person's stale prototype file(s) to .bak (forces re-enroll; id/memory kept)")
    ap.add_argument("--modality", choices=["voice", "face", "both"], default="both",
                    help="which store --backup applies to (default both)")
    args = ap.parse_args(argv)

    id_map, voice_store, face_store = _id_map(), _voice_store(), _face_store()

    if args.backup:
        name = args.backup.strip().lower()
        done = []
        if args.modality in ("voice", "both"):
            b = _backup(voice_store, name)
            if b:
                done.append(f"voice -> {b.name}")
        if args.modality in ("face", "both"):
            b = _backup(face_store, name)
            if b:
                done.append(f"face -> {b.name}")
        sid = id_map.id_for(name)
        print(f"[backup] {name} (id={sid}, PRESERVED): "
              + ("; ".join(done) if done else "nothing to back up"))
        print("  id-map + Postgres row untouched; re-enroll to rebuild the vectors.")
        return 0

    report = audit(id_map, voice_store, face_store)
    rows = report["identities"]

    if args.json:
        print(json.dumps({
            "identities": [asdict(r) for r in rows],
            "orphan_voice": report["orphan_voice"],
            "orphan_face": report["orphan_face"],
        }, indent=2))
    else:
        print(f"{'name':<20} {'id':>4}  {'voice':<5} {'face':<5}  status")
        print("-" * 52)
        for r in rows:
            v = "yes" if r.has_voice else "-"
            f = "yes" if r.has_face else "-"
            status = "complete" if r.complete else "NEEDS ENROLL"
            sid = "?" if r.speaker_id is None else r.speaker_id
            print(f"{r.name:<20} {sid:>4}  {v:<5} {f:<5}  {status}")
        incomplete = [r.name for r in rows if not r.complete
                      and r.name not in RESERVED_NAMES]
        if incomplete:
            print("\nSingle-modality identities (enroll the missing side to complete):")
            print("  " + ", ".join(incomplete))
        for label, key in (("voice", "orphan_voice"), ("face", "orphan_face")):
            if report[key]:
                print(f"\nORPHAN {label} prototypes (file present, NO id-map entry): "
                      + ", ".join(report[key]))
                print(f"  -> allocate an id (load the recognizer) or remove the file.")

    if args.reconcile_db:
        print("\n[reconcile-db] upserting speakers rows (idempotent, id-stable)...")
        try:
            from db.speakers import ensure_rows_for_enrolled
            inserted = ensure_rows_for_enrolled()
            print(f"[reconcile-db] done: {inserted} row(s) inserted.")
        except Exception as e:
            print(f"[reconcile-db] FAILED (biometrics unaffected): {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
