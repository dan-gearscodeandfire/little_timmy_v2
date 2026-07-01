"""Build per-person face galleries from a creator's own YouTube content.

Source = the channel's video THUMBNAILS + AVATAR (not web search), so the images
are guaranteed to be the creator's own content — the recurring dominant face IS
the creator, with no famous-namesake / cartoon contamination that plain-name web
search suffers (Prince William <- "William Osman", Tinker Bell <- "Simone
Giertz"). Two selection modes per person:

  ANCHOR   the avatar has exactly one face -> keep thumbnail faces within
           ``thr`` of that avatar embedding (disambiguates namesakes/collabs).
  CLUSTER  the avatar is a logo (no face) -> keep the largest face cluster
           across thumbnails (the creator recurs most in their own thumbnails).

For each <raw>/<slug>/ (thumbnails) + optional <anchors>/<slug>.jpg (avatar):
collect ALL faces, select by mode, and save aligned crops to <out>/<slug>/
aligned/. Galleries double as (a) a future enrollment/classifier set and (b)
in-community impostor / cross-condition-genuine calibration data.

Usage (production venv):
  python -m ops.build_maker_gallery --raw /tmp/thumbs --anchors /tmp/anchors \
      --out ops/calib/makers

Reuses detection/alignment/embedding from ops.edgeface_calibrate + the
calibrated threshold. Prints per-person mode + yield + tightness; a CLUSTER
person with a logo avatar (ben/kevin/nigel) or a 2-host avatar (ruth_amos)
warrants a visual check that the dominant face is the intended creator.
"""

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cosine

import ops.edgeface_calibrate as C
from presence.face_encoder import embed_batch, extract_embedding
from presence.face_thresholds import KNOWN_FACE_THRESHOLD

# Cross-condition (thumbnail vs avatar/other-thumbnail) is looser than the live
# accept threshold; a touch above KNOWN_FACE_THRESHOLD, still well below the
# impostor floor (~0.70), for gallery selection only.
GALLERY_THR = 0.55


def _all_faces(image_dir: Path):
    """Every alignable face across all images in a dir (thumbnails have the
    creator + collaborators/subjects; selection sorts out which is the creator)."""
    crops = []
    for p in sorted(image_dir.glob("*.jpg")):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        for bl in C._yunet_faces(frame):
            aligned = C._aligned_from_frame(frame, *bl)
            if aligned is not None:
                crops.append(aligned)
    return crops


def _anchor_embedding(anchor_path: Path):
    """Embedding of the single avatar face, or None (missing / logo / multi-face)."""
    if not anchor_path.exists():
        return None
    frame = cv2.imread(str(anchor_path))
    if frame is None:
        return None
    faces = C._yunet_faces(frame)
    if len(faces) != 1:
        return None
    aligned = C._aligned_from_frame(frame, *faces[0])
    return extract_embedding(aligned) if aligned is not None else None


# Below this anchor-kept count, expand to the anchor's cluster for volume.
_ANCHOR_MIN = 12
# A cluster medoid within this (looser) distance of the anchor confirms the
# cluster IS the anchor person (not a recurring collaborator) before expanding.
_ANCHOR_EXPAND = 0.62


def _dominant_cluster(embs: np.ndarray, thr: float):
    """(indices, medoid_idx) of the largest cluster: the face with the most
    neighbors within ``thr`` is the medoid; keep everything within ``thr``."""
    n = len(embs)
    if n <= 1:
        return list(range(n)), (0 if n else -1)
    D = np.array([[cosine(embs[i], embs[j]) for j in range(n)] for i in range(n)])
    medoid = int(np.argmax((D < thr).sum(axis=1)))
    return [j for j in range(n) if D[medoid][j] < thr], medoid


def _clean(idxs: list, E: np.ndarray, thr: float) -> list:
    """Drop injected wrong-person faces: keep a face only if its MEDIAN cosine
    distance to the other kept faces is < thr. Cosine distance isn't a metric, so
    a cluster/anchor set can admit an outlier that's near the medoid yet far from
    the bulk (nate's 0.958 pair); this removes them. No-op for tiny sets."""
    if len(idxs) <= 3:
        return idxs
    sub = E[idxs]
    D = np.array([[cosine(sub[i], sub[j]) for j in range(len(sub))]
                  for i in range(len(sub))])
    keep = []
    for i in range(len(idxs)):
        others = np.delete(D[i], i)
        if np.median(others) < thr:
            keep.append(idxs[i])
    return keep or idxs


def process(raw_root: Path, anchors_root: Path, out_root: Path, thr: float) -> int:
    slugs = sorted(d for d in raw_root.iterdir() if d.is_dir())
    print(f"{'slug':16} mode     faces kept  tight(mean/max)")
    for d in slugs:
        crops = _all_faces(d)
        if not crops:
            print(f"{d.name:16} -        0     0     -")
            continue
        E = embed_batch(crops)
        anchor = _anchor_embedding(anchors_root / f"{d.name}.jpg") if anchors_root else None
        if anchor is not None:
            keep = [i for i in range(len(E)) if cosine(E[i], anchor) < thr]
            mode = "ANCHOR"
            if len(keep) < _ANCHOR_MIN:
                # Strict avatar distance missed volume (avatar pose/era differs).
                # Expand to the dominant cluster IF its medoid confirms as the
                # anchor person -> anchor-verified recall.
                cl, med = _dominant_cluster(E, thr)
                if med >= 0 and cosine(E[med], anchor) < _ANCHOR_EXPAND:
                    keep = sorted(set(keep) | set(cl))
                    mode = "ANCHOR+CL"
        else:
            keep, _ = _dominant_cluster(E, thr)
            mode = "CLUSTER"
        keep = _clean(keep, E, thr)
        if len(keep) >= 2:
            Ek = E[keep]
            dd = [cosine(Ek[a], Ek[b]) for a, b in itertools.combinations(range(len(keep)), 2)]
            tight = f"{np.mean(dd):.3f}/{np.max(dd):.3f}"
        else:
            tight = "n/a"
        outdir = out_root / d.name / "aligned"
        outdir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(keep):
            cv2.imwrite(str(outdir / f"{i:03d}.png"),
                        cv2.cvtColor(crops[idx], cv2.COLOR_RGB2BGR))
        print(f"{d.name:16} {mode:8} {len(crops):<5} {len(keep):<5} {tight}")
    print(f"\nGalleries -> {out_root}  (threshold {thr})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build maker face galleries from channel content")
    ap.add_argument("--raw", required=True, help="dir of <slug>/ thumbnail images")
    ap.add_argument("--anchors", default=None, help="dir of <slug>.jpg avatar anchors")
    ap.add_argument("--out", default="ops/calib/makers")
    ap.add_argument("--threshold", type=float, default=GALLERY_THR)
    a = ap.parse_args()
    return process(Path(a.raw), Path(a.anchors) if a.anchors else None,
                   Path(a.out), a.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
