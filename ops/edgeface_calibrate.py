"""EdgeFace threshold calibration harness.

WeSpeaker's cosine constants do NOT transfer to EdgeFace (different scale), so the
face match / dedup / band cutoffs must be derived against a real
genuine/impostor split. Runs entirely on okDemerzel CPU using the YuNet detector
already present at models/face_detection_yunet_2023mar.onnx (bbox + 5 landmarks),
the pinned aligner (presence.face_align), and the production embedder
(presence.face_encoder).

Modes:
  capture  --name dan --n 24         grab frames from the Pi /capture endpoint,
                                      keep those with exactly one detected face,
                                      save raw JPEGs + aligned crops under
                                      ops/calib/<name>/ (GENUINE set = one person).
  ingest   --label foo --dir PATH    ingest an external image dir (e.g. an LFW
                                      subset) as an IMPOSTOR cohort.
  report                             embed genuine + impostor crops, print the
                                      genuine (same-person) vs impostor
                                      (cross-person) cosine-distance distributions
                                      and a suggested threshold (max genuine vs
                                      min impostor; EER midpoint).

YuNet landmark order is [right-eye, left-eye, nose, right-mouth, left-mouth];
the ArcFace template (presence.face_align.ARCFACE_REF_5PT) is
[left-eye, right-eye, nose, left-mouth, right-mouth] — reordered in _yunet_faces.
"""

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO / "ops" / "calib"
YUNET_PATH = REPO / "models" / "face_detection_yunet_2023mar.onnx"

# YuNet landmark order already matches the ArcFace template by IMAGE POSITION
# (both: image-left eye, image-right eye, nose, image-left mouth, image-right
# mouth), so NO reorder — YuNet's "right eye" (idx0) is the person's right eye,
# which sits on the image-LEFT, exactly where template point 0 (x=38) is. An
# earlier L/R swap forced a mirrored correspondence a similarity warp can't
# satisfy, collapsing the transform to scale~0.23 (tiny face) and flattening
# EdgeFace discrimination. Verified on-camera 2026-06-30.
_YUNET_TO_ARCFACE = [0, 1, 2, 3, 4]


def _detector(w: int, h: int):
    det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (w, h),
                                    score_threshold=0.7, nms_threshold=0.3)
    det.setInputSize((w, h))
    return det


_YUNET_DET_MAX = 640   # detect at this longer-side scale; YuNet misses faces too
                       # large in-frame (tight hi-res headshots) at native size.


def _yunet_faces(frame_bgr: np.ndarray):
    """Return list of (bbox_xywh, landmarks5_arcface_order) for detected faces.

    Downscales so the longer side is <= _YUNET_DET_MAX before detection (large
    in-frame faces fall outside YuNet's anchor range), then maps bbox+landmarks
    back to original-image coordinates."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, _YUNET_DET_MAX / max(h, w))
    if scale < 1.0:
        dw, dh = int(round(w * scale)), int(round(h * scale))
        det_img = cv2.resize(frame_bgr, (dw, dh))
    else:
        dw, dh, det_img = w, h, frame_bgr
    det = _detector(dw, dh)
    _, faces = det.detect(det_img)
    out = []
    if faces is None:
        return out
    inv = 1.0 / scale
    for f in faces:
        box = (f[0:4].astype(np.float32)) * inv
        lm = (f[4:14].reshape(5, 2).astype(np.float32) * inv)[_YUNET_TO_ARCFACE]
        out.append((box, lm))
    return out


def _loose_crop(frame_bgr, box, scale=1.3):
    """Crop ~scale*bbox around the face, clamped; return (crop_bgr, offset)."""
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * scale
    x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
    x1 = int(min(frame_bgr.shape[1], cx + side / 2))
    y1 = int(min(frame_bgr.shape[0], cy + side / 2))
    return frame_bgr[y0:y1, x0:x1], np.array([x0, y0], dtype=np.float32)


def _aligned_from_frame(frame_bgr, box, lm):
    """Loose-crop + align to 112x112 RGB (mirrors the planned Pi->okDemerzel path:
    Pi ships loose crop + landmarks, okDemerzel warps)."""
    from presence.face_align import align_face, landmarks_ok
    crop_bgr, off = _loose_crop(frame_bgr, box)
    lm_local = lm - off
    if not landmarks_ok(lm_local):
        return None
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return align_face(crop_rgb, lm_local)


def capture(name: str, n: int, interval: float) -> int:
    import time
    import httpx
    import config
    outdir = CALIB_DIR / name.lower()
    (outdir / "aligned").mkdir(parents=True, exist_ok=True)
    kept = 0
    with httpx.Client(verify=False, timeout=5.0) as client:
        for i in range(n):
            try:
                r = client.get(config.STREAMERPI_CAPTURE_URL)
                frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[{i}] capture failed: {e}"); time.sleep(interval); continue
            faces = _yunet_faces(frame) if frame is not None else []
            if len(faces) != 1:
                print(f"[{i}] {len(faces)} faces — skip"); time.sleep(interval); continue
            box, lm = faces[0]
            aligned = _aligned_from_frame(frame, box, lm)
            if aligned is None:
                print(f"[{i}] bad landmarks — skip"); time.sleep(interval); continue
            cv2.imwrite(str(outdir / f"frame_{i:03d}.jpg"), frame)
            cv2.imwrite(str(outdir / "aligned" / f"a_{i:03d}.png"),
                        cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            kept += 1
            print(f"[{i}] kept ({kept} total)")
            time.sleep(interval)
    print(f"\nGENUINE capture for {name}: {kept}/{n} usable frames -> {outdir}")
    return 0


def ingest(label: str, src_dir: str) -> int:
    outdir = CALIB_DIR / f"_impostor_{label.lower()}" / "aligned"
    outdir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for p in sorted(Path(src_dir).rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        faces = _yunet_faces(frame)
        if len(faces) != 1:
            continue
        aligned = _aligned_from_frame(frame, *faces[0])
        if aligned is None:
            continue
        cv2.imwrite(str(outdir / f"{kept:04d}.png"),
                    cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        kept += 1
    print(f"IMPOSTOR ingest '{label}': {kept} usable faces -> {outdir}")
    return 0


def lfw(n: int) -> int:
    """Download LFW (random frontal faces) once and ingest N distinct-person
    images as the impostor cohort. Reads the extracted JPGs directly (globs
    lfw_home) rather than sklearn's multi-GB array load; a high
    min_faces_per_person keeps sklearn's in-RAM slice tiny while still extracting
    the full funneled set to disk."""
    import random
    from sklearn.datasets import fetch_lfw_people
    print("Fetching LFW (one-time ~200MB download to ~/scikit_learn_data)...")
    fetch_lfw_people(min_faces_per_person=100, color=True)  # extracts all JPGs
    home = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"
    people = [d for d in home.iterdir() if d.is_dir()]
    random.seed(20260630)
    random.shuffle(people)
    # One image per distinct identity (distinct people = true impostors).
    picks = []
    for d in people:
        jpgs = sorted(d.glob("*.jpg"))
        if jpgs:
            picks.append(jpgs[0])
        if len(picks) >= n:
            break
    outdir = CALIB_DIR / "_impostor_lfw" / "aligned"
    outdir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for p in picks:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        faces = _yunet_faces(frame)
        if len(faces) != 1:
            continue
        aligned = _aligned_from_frame(frame, *faces[0])
        if aligned is None:
            continue
        cv2.imwrite(str(outdir / f"{kept:04d}.png"),
                    cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        kept += 1
    print(f"IMPOSTOR (LFW): {kept}/{len(picks)} distinct-person faces -> {outdir}")
    return 0


def _embed_dir(aligned_dir: Path) -> np.ndarray:
    from presence.face_encoder import embed_batch
    crops = []
    for p in sorted(aligned_dir.glob("*.png")):
        c = cv2.imread(str(p))
        if c is not None:
            crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    return embed_batch(crops) if crops else np.empty((0, 512), np.float32)


def report() -> int:
    from scipy.spatial.distance import cosine
    genuine_sets = {}
    impostor_embs = []
    for d in sorted(CALIB_DIR.glob("*")):
        aligned = d / "aligned"
        if not aligned.is_dir():
            continue
        embs = _embed_dir(aligned)
        if d.name.startswith("_impostor_"):
            if len(embs):
                impostor_embs.append(embs)
        elif len(embs) >= 2:
            genuine_sets[d.name] = embs
    if not genuine_sets:
        print("No genuine sets. Run: capture --name <you> --n 24"); return 2

    gen_d = []
    for name, E in genuine_sets.items():
        for a, b in itertools.combinations(range(len(E)), 2):
            gen_d.append(cosine(E[a], E[b]))
    gen_d = np.array(gen_d)

    imp_d = []
    allimp = np.vstack(impostor_embs) if impostor_embs else np.empty((0, 512))
    for name, E in genuine_sets.items():
        for e in E:
            for j in range(len(allimp)):
                imp_d.append(cosine(e, allimp[j]))
    # Also cross-genuine-identity pairs count as impostor if >1 genuine person.
    names = list(genuine_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a in genuine_sets[names[i]]:
                for b in genuine_sets[names[j]]:
                    imp_d.append(cosine(a, b))
    imp_d = np.array(imp_d)

    print(f"\nGENUINE pairs: {len(gen_d)}  mean={gen_d.mean():.3f} "
          f"p95={np.percentile(gen_d,95):.3f} max={gen_d.max():.3f}")
    if len(imp_d):
        print(f"IMPOSTOR pairs: {len(imp_d)}  mean={imp_d.mean():.3f} "
              f"p05={np.percentile(imp_d,5):.3f} min={imp_d.min():.3f}")
        gap_lo, gap_hi = gen_d.max(), imp_d.min()
        mid = (np.percentile(gen_d, 95) + np.percentile(imp_d, 5)) / 2
        print(f"\nSeparation: max(genuine)={gap_lo:.3f}  min(impostor)={gap_hi:.3f}"
              f"  {'CLEAN' if gap_hi > gap_lo else 'OVERLAP'}")
        print(f"Suggested KNOWN_FACE_THRESHOLD ~ {mid:.3f} "
              f"(midpoint of p95-genuine / p05-impostor)")
        print(f"Suggested FACE_DEDUP_DIST ~ {np.percentile(gen_d,50):.3f} "
              f"(median genuine; near-dup below this)")
    else:
        print("No impostor set yet — ingest one for a threshold "
              "(ingest --label lfw --dir <path>).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="EdgeFace threshold calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--name", required=True)
    c.add_argument("--n", type=int, default=24); c.add_argument("--interval", type=float, default=0.6)
    g = sub.add_parser("ingest"); g.add_argument("--label", required=True); g.add_argument("--dir", required=True)
    lf = sub.add_parser("lfw"); lf.add_argument("--n", type=int, default=200)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "capture":
        return capture(a.name, a.n, a.interval)
    if a.cmd == "ingest":
        return ingest(a.label, a.dir)
    if a.cmd == "lfw":
        return lfw(a.n)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
