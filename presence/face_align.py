"""EdgeFace preprocessing contract — the ONE place face crops become model input.

The Pi ships a LOOSE crop (~1.3x bbox) + 5 YuNet landmarks; okDemerzel owns the
affine warp to 112x112 and the normalization, so the embedder's exact input
contract lives next to the embedder (not split across two repos). Getting the
reference template or the normalization wrong produces confident-but-wrong
embeddings, so both are pinned here and asserted by the ONNX parity test.

EdgeFace (Idiap) inference contract, from otroshi/edgeface README:
    transforms.ToTensor()                      -> [0,1], RGB, CHW
    transforms.Normalize(mean=0.5, std=0.5)    -> (x-0.5)/0.5  == (px-127.5)/127.5
Input 112x112 RGB. Output 512-d embedding (L2-normalized by the caller).
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

INPUT_SIZE = 112
EMBED_DIM = 512

# Canonical ArcFace 5-point reference landmarks for a 112x112 aligned crop
# (left-eye, right-eye, nose, left-mouth, right-mouth). This is the de-facto
# standard template used by ArcFace/insightface/EdgeFace training pipelines;
# the crop MUST be warped to these coordinates for the embedding to be valid.
ARCFACE_REF_5PT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Landmark-quality floor: reject a crop whose eyes are implausibly close (tiny /
# far / bad detection) — a degenerate warp yields a garbage-but-confident vector.
# Measured in the ORIGINAL crop's pixel space (inter-ocular distance).
MIN_INTEROCULAR_PX = 12.0

# Frontality ceiling (yaw proxy) — SHADOW MODE (Dan 2026-07-15: enroll crops
# are frontal-only; frontality gate > pose diversity). Currently compute-and-
# log only ([FRONTAL-SHADOW] in face_recognize) so the threshold can be
# calibrated on real booth frames before it filters anything. 0 = dead-on;
# ~0.35 ≈ moderate yaw. Recognition-path crops are NEVER gated by this —
# it applies (once enforcing) to enroll-bound sole/anchored crops only.
FRONTAL_MAX_RATIO = 0.35


def frontal_ratio(landmarks5: np.ndarray) -> float:
    """Yaw proxy from the 5-point set: nose x-offset from the eye midpoint,
    normalized by inter-ocular distance (original crop pixel space). Lower is
    more frontal. Returns inf on degenerate geometry — callers should already
    have passed :func:`landmarks_ok`."""
    lm = np.asarray(landmarks5, dtype=np.float32)
    if lm.shape != (5, 2) or not np.isfinite(lm).all():
        return float("inf")
    inter_ocular = float(np.linalg.norm(lm[0] - lm[1]))
    if inter_ocular <= 0.0:
        return float("inf")
    mid_x = (lm[0][0] + lm[1][0]) / 2.0
    return abs(float(lm[2][0]) - mid_x) / inter_ocular


def landmarks_ok(landmarks5: np.ndarray) -> bool:
    """True if the 5-point set is geometrically plausible for alignment.

    Guards against degenerate detections (collinear points, sub-pixel eye
    separation) that warp to a valid-shaped but meaningless 112x112."""
    lm = np.asarray(landmarks5, dtype=np.float32)
    if lm.shape != (5, 2) or not np.isfinite(lm).all():
        return False
    inter_ocular = float(np.linalg.norm(lm[0] - lm[1]))
    if inter_ocular < MIN_INTEROCULAR_PX:
        return False
    # Eyes should sit above the mouth points (rejects upside-down / swapped).
    eye_y = (lm[0][1] + lm[1][1]) / 2.0
    mouth_y = (lm[3][1] + lm[4][1]) / 2.0
    return mouth_y > eye_y


def align_face(image_rgb: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
    """Similarity-warp a face crop to the canonical 112x112 template.

    ``image_rgb`` is the (loose) crop as RGB HxWx3 uint8; ``landmarks5`` are the
    5 points in that crop's pixel coordinates. Returns a 112x112x3 uint8 RGB
    aligned crop. Uses a partial affine (rotation+uniform scale+translation) —
    the same transform class insightface/EdgeFace alignment uses."""
    lm = np.asarray(landmarks5, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(lm, ARCFACE_REF_5PT, method=cv2.LMEDS)
    if M is None:
        raise ValueError("could not estimate alignment transform from landmarks")
    aligned = cv2.warpAffine(image_rgb, M, (INPUT_SIZE, INPUT_SIZE),
                             borderValue=0.0)
    return aligned


def preprocess(aligned_rgb: np.ndarray) -> np.ndarray:
    """112x112x3 uint8 RGB -> (1, 3, 112, 112) float32, EdgeFace-normalized.

    Normalization is (px - 127.5) / 127.5 per the pinned contract. Batch of one;
    use :func:`preprocess_batch` for many crops in a single embed call."""
    x = aligned_rgb.astype(np.float32)
    x = (x - 127.5) / 127.5
    x = np.transpose(x, (2, 0, 1))          # HWC -> CHW
    return x[None, :, :, :]                  # add batch


def preprocess_batch(aligned_rgb_list: list) -> np.ndarray:
    """List of 112x112x3 uint8 RGB crops -> (N, 3, 112, 112) float32."""
    if not aligned_rgb_list:
        return np.empty((0, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    return np.concatenate([preprocess(a) for a in aligned_rgb_list], axis=0)
