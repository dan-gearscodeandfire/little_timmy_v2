"""Build per-person face galleries from noisy web-crawled images.

Web image search returns junk + wrong-people; this keeps only the DOMINANT
identity cluster per person (the actual maker), discarding outliers. For each
raw dir <raw>/<slug>/: take the largest face per image, embed (EdgeFace), find
the largest cluster within KNOWN_FACE_THRESHOLD of a medoid, and save that
cluster's aligned crops to <out>/<slug>/aligned/. These galleries double as
(a) a future enrollment/classifier set and (b) harder in-community impostors.

Usage (production venv):
  python -m ops.build_maker_gallery --raw /tmp/mk --out ops/calib/makers

Reuses the detection/alignment/embedding from ops.edgeface_calibrate + the
calibrated face threshold. Prints per-person yield + cluster tightness so weak
crawls (obscure people / camera-shy) are visible rather than silently thin.
"""

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cosine

import ops.edgeface_calibrate as C
from presence.face_encoder import embed_batch
from presence.face_thresholds import KNOWN_FACE_THRESHOLD


def _largest_face(frame):
    """Return (bbox, landmarks) of the biggest detected face, or None."""
    faces = C._yunet_faces(frame)
    if not faces:
        return None
    return max(faces, key=lambda bl: bl[0][2] * bl[0][3])


def _dominant_cluster(embs: np.ndarray, thr: float):
    """Indices of the largest cluster: the face with the most neighbors within
    ``thr`` is the medoid; keep everything within ``thr`` of it."""
    n = len(embs)
    if n == 0:
        return []
    if n == 1:
        return [0]
    D = np.array([[cosine(embs[i], embs[j]) for j in range(n)] for i in range(n)])
    neighbor_counts = (D < thr).sum(axis=1)
    medoid = int(np.argmax(neighbor_counts))
    return [j for j in range(n) if D[medoid][j] < thr]


def process(raw_root: Path, out_root: Path, thr: float) -> int:
    slugs = sorted(d for d in raw_root.iterdir() if d.is_dir())
    print(f"{'slug':22} raw faces cluster  tight(mean/max)")
    for d in slugs:
        crops, srcs = [], []
        for p in sorted(d.iterdir()):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            bl = _largest_face(frame)
            if bl is None:
                continue
            aligned = C._aligned_from_frame(frame, *bl)
            if aligned is not None:
                crops.append(aligned)
                srcs.append(p.name)
        if not crops:
            print(f"{d.name:22} 0    0     0        -")
            continue
        E = embed_batch(crops)
        keep = _dominant_cluster(E, thr)
        Ek = E[keep]
        if len(keep) >= 2:
            dd = [cosine(Ek[a], Ek[b]) for a, b in itertools.combinations(range(len(keep)), 2)]
            tight = f"{np.mean(dd):.3f}/{np.max(dd):.3f}"
        else:
            tight = "n/a"
        outdir = out_root / d.name / "aligned"
        outdir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(keep):
            cv2.imwrite(str(outdir / f"{i:03d}.png"),
                        cv2.cvtColor(crops[idx], cv2.COLOR_RGB2BGR))
        n_raw = len(list(d.iterdir()))
        print(f"{d.name:22} {n_raw:<4} {len(crops):<5} {len(keep):<8} {tight}")
    print(f"\nGalleries -> {out_root}  (threshold {thr})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build maker face galleries via clustering")
    ap.add_argument("--raw", required=True, help="dir of <slug>/ raw crawled images")
    ap.add_argument("--out", default="ops/calib/makers")
    ap.add_argument("--threshold", type=float, default=KNOWN_FACE_THRESHOLD)
    a = ap.parse_args()
    return process(Path(a.raw), Path(a.out), a.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
