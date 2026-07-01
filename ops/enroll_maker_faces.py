"""Enroll Dan + the maker galleries as EdgeFace face identities.

Reads the staged aligned galleries (ops/calib/dan + ops/calib/makers/<slug>) and
enrolls each as ``models/face/<name>_edgeface.npy`` via FaceIdentifier, allocating
ids in the SHARED name->speaker_id map (models/speaker/_id_map.json) so each
person keeps one speaker_id across face + voice. Inert until FaceIdentifier is
wired into the runtime; voice is unaffected (these names have no voiceprint).

Run (production venv), after building galleries:
    python -m ops.enroll_maker_faces
"""

from pathlib import Path

import cv2

from presence.face_identifier import FaceIdentifier
from presence.face_thresholds import MIN_FACE_ENROLL_SAMPLES

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "ops" / "calib" / "dan" / "aligned"
MAKERS = REPO / "ops" / "calib" / "makers"


def _load(d: Path):
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            for p in sorted(d.glob("*.png"))]


def main() -> int:
    fi = FaceIdentifier()
    fi.load()
    jobs = []
    if DAN.is_dir():
        jobs.append(("dan", DAN))
    for d in sorted(MAKERS.glob("*/aligned")):
        jobs.append((d.parent.name, d))
    print(f"{'name':16} crops protos  note")
    for name, d in jobs:
        crops = _load(d)
        if not crops:
            print(f"{name:16} 0     -       SKIP (no crops)")
            continue
        n = fi.enroll(name, crops, augment=False)
        note = "" if len(crops) >= MIN_FACE_ENROLL_SAMPLES else "THIN (<3)"
        print(f"{name:16} {len(crops):<5} {n:<7} {note}")
    print(f"\nEnrolled {len(fi.known_names)} face identities -> models/face/")
    print("id-map:", fi._id_map.enrolled_ids())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
