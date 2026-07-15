"""YuNet face detection + ArcFace alignment on okDemerzel (runtime + ops).

The okDemerzel-side counterpart to the Pi's detector: given a full frame (e.g. a
/capture JPEG), find faces (bbox + 5 landmarks) and produce aligned 112x112 RGB
crops ready for the EdgeFace encoder. Lets okDemerzel self-serve face recognition
from the existing on-demand frame grab — no Pi change required.

Uses the YuNet model already shipped at models/face_detection_yunet_2023mar.onnx.
Landmark order + the <=640px detect-scale gotcha are handled here (see
memory/feedback_yunet_arcface_landmark_order): YuNet's 5 points already match the
ArcFace template by image position, so NO reorder; detect at <=640 longer side
then map coords back, since YuNet misses faces that fill a hi-res frame.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from presence.face_align import align_face, frontal_ratio, landmarks_ok

log = logging.getLogger(__name__)

YUNET_PATH = Path(os.getenv(
    "TIMMY_YUNET_ONNX",
    str(Path(__file__).resolve().parent.parent / "models"
        / "face_detection_yunet_2023mar.onnx")))

# Detect at <=640px longer side (YuNet misses faces too large in-frame), then
# scale bbox+landmarks back to original coords.
DET_MAX = 640
# YuNet order already matches the ArcFace template by image position -> identity.
_YUNET_TO_ARCFACE = [0, 1, 2, 3, 4]


def _detector(w: int, h: int):
    det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (w, h),
                                    score_threshold=0.7, nms_threshold=0.3)
    det.setInputSize((w, h))
    return det


def detect_faces(frame_bgr: np.ndarray):
    """Return [(bbox_xywh, landmarks5_arcface_order), ...] in ORIGINAL coords."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, DET_MAX / max(h, w))
    det_img = (cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))))
               if scale < 1.0 else frame_bgr)
    det = _detector(det_img.shape[1], det_img.shape[0])
    _, faces = det.detect(det_img)
    out = []
    if faces is None:
        return out
    inv = 1.0 / scale
    for f in faces:
        box = f[0:4].astype(np.float32) * inv
        lm = (f[4:14].reshape(5, 2).astype(np.float32) * inv)[_YUNET_TO_ARCFACE]
        out.append((box, lm))
    return out


def loose_crop(frame_bgr: np.ndarray, box, scale: float = 1.3):
    """Crop ~scale*bbox around the face, clamped; return (crop_bgr, offset_xy)."""
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * scale
    x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
    x1 = int(min(frame_bgr.shape[1], cx + side / 2))
    y1 = int(min(frame_bgr.shape[0], cy + side / 2))
    return frame_bgr[y0:y1, x0:x1], np.array([x0, y0], dtype=np.float32)


def align_one(frame_bgr: np.ndarray, box, lm):
    """Loose-crop + align a single detected face -> aligned 112x112 RGB, or None
    (degenerate landmarks / warp failure). box/lm from :func:`detect_faces`."""
    crop_bgr, off = loose_crop(frame_bgr, box)
    lm_local = lm - off
    if not landmarks_ok(lm_local):
        return None
    try:
        return align_face(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), lm_local)
    except Exception:
        return None


def aligned_crops(frame_bgr: np.ndarray):
    """Detect + align every face -> [(aligned_rgb_112, bbox_xyxy, frontal), ...].

    bbox_xyxy is (x0,y0,x1,y1) in original-frame pixels (for FacePrediction).
    frontal is the yaw-proxy ratio from :func:`face_align.frontal_ratio` (lower
    = more frontal), computed pre-alignment while the landmarks still exist —
    the frontality-gate shadow signal (Dan 2026-07-15). Faces with degenerate
    landmarks are skipped (landmark-quality reject)."""
    out = []
    for box, lm in detect_faces(frame_bgr):
        aligned = align_one(frame_bgr, box, lm)
        if aligned is None:
            continue
        x, y, bw, bh = box
        bbox_xyxy = (int(x), int(y), int(x + bw), int(y + bh))
        out.append((aligned, bbox_xyxy, frontal_ratio(lm)))
    return out
