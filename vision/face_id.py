"""Face identification using OpenCV FaceDetectorYN + FaceRecognizerSF.

Detects faces in a frame, extracts 128-dim embeddings via SFace,
and compares against enrolled reference embeddings to identify people.

Usage:
    fid = FaceID("models/face_detection_yunet_2023mar.onnx",
                 "models/face_recognition_sface_2021dec.onnx")
    fid.load_db()  # load enrolled embeddings
    results = fid.identify(frame)
    # [{"name": "Dan", "distance": 0.25, "bbox": [x,y,w,h]}, ...]
"""

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = os.path.expanduser("~/.face_db/embeddings.json")
DEFAULT_YUNET = "models/face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE = "models/face_recognition_sface_2021dec.onnx"

# Thresholds
MATCH_THRESHOLD = 0.4       # cosine distance below this = match
UNKNOWN_THRESHOLD = 0.6     # above this = definitely unknown
CONFIDENCE_HIGH = 0.3       # below this = high confidence match


class FaceID:
    """Face detection + recognition pipeline."""

    def __init__(self, yunet_path: str = DEFAULT_YUNET,
                 sface_path: str = DEFAULT_SFACE,
                 db_path: str = DEFAULT_DB_PATH,
                 detection_confidence: float = 0.6):
        self._yunet_path = yunet_path
        self._sface_path = sface_path
        self._db_path = db_path
        self._detection_confidence = detection_confidence

        self._detector: cv2.FaceDetectorYN | None = None
        self._recognizer: cv2.FaceRecognizerSF | None = None
        self._last_input_size = (0, 0)

        # Enrolled identities: {"Dan": [embedding_array], "Erin": [...]}
        self._enrolled: dict[str, np.ndarray] = {}

        # Unknown face tracker: maps embedding cluster → stable ID
        self._unknown_counter = 0
        self._unknown_embeddings: list[tuple[str, np.ndarray]] = []

    def init_models(self):
        """Load detection and recognition models."""
        if not os.path.exists(self._yunet_path):
            log.error("YuNet model not found: %s", self._yunet_path)
            return False
        if not os.path.exists(self._sface_path):
            log.error("SFace model not found: %s", self._sface_path)
            return False

        self._recognizer = cv2.FaceRecognizerSF.create(self._sface_path, "")
        log.info("FaceID models loaded (YuNet + SFace)")
        return True

    def _get_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        """Get or recreate detector for the given frame size."""
        if self._detector is None or self._last_input_size != (width, height):
            self._detector = cv2.FaceDetectorYN.create(
                self._yunet_path, "",
                (width, height),
                self._detection_confidence,
                0.3,  # NMS threshold
                5000  # top_k
            )
            self._last_input_size = (width, height)
        return self._detector

    def load_db(self) -> int:
        """Load enrolled embeddings from disk. Returns number of identities."""
        if not os.path.exists(self._db_path):
            log.info("No face DB found at %s, starting empty", self._db_path)
            return 0

        with open(self._db_path, "r") as f:
            data = json.load(f)

        self._enrolled = {}
        for name, embedding_list in data.get("enrolled", {}).items():
            self._enrolled[name] = np.array(embedding_list, dtype=np.float32)

        log.info("Loaded %d enrolled identities from %s: %s",
                 len(self._enrolled), self._db_path,
                 ", ".join(self._enrolled.keys()))
        return len(self._enrolled)

    def save_db(self):
        """Save enrolled embeddings to disk."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        data = {
            "enrolled": {
                name: emb.tolist() for name, emb in self._enrolled.items()
            },
            "version": 1,
        }
        with open(self._db_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Saved face DB to %s (%d identities)",
                 self._db_path, len(self._enrolled))

    def enroll(self, name: str, embeddings: list[np.ndarray]):
        """Enroll a person with multiple face embeddings.

        Stores the mean embedding as the reference.
        """
        if not embeddings:
            log.warning("No embeddings to enroll for %s", name)
            return

        mean_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        self._enrolled[name] = mean_embedding
        self.save_db()
        log.info("Enrolled %s with %d samples (mean embedding stored)", name, len(embeddings))

    def extract_embedding(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray | None:
        """Extract a 128-dim face embedding from a detected face.

        Args:
            frame: BGR image
            face: Single face detection row from FaceDetectorYN
                  (x, y, w, h, ..., landmarks)

        Returns:
            128-dim normalized embedding, or None on failure
        """
        if self._recognizer is None:
            return None

        try:
            aligned = self._recognizer.alignCrop(frame, face)
            embedding = self._recognizer.feature(aligned)
            return embedding.flatten()
        except Exception as e:
            log.debug("Embedding extraction failed: %s", e)
            return None

    def compare(self, embedding: np.ndarray) -> tuple[str, float]:
        """Compare an embedding against all enrolled identities.

        Returns:
            (name, distance) — best match. Name is "unknown_X" if no match.
        """
        if not self._enrolled:
            return self._assign_unknown(embedding), 1.0

        best_name = None
        best_distance = float("inf")

        for name, ref_embedding in self._enrolled.items():
            # Cosine distance via OpenCV
            distance = 1.0 - np.dot(embedding, ref_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_embedding) + 1e-8
            )
            if distance < best_distance:
                best_distance = distance
                best_name = name

        if best_distance < MATCH_THRESHOLD:
            return best_name, float(best_distance)
        else:
            return self._assign_unknown(embedding), float(best_distance)

    def _assign_unknown(self, embedding: np.ndarray) -> str:
        """Assign a stable unknown ID by checking if this face matches any prior unknowns."""
        for uid, ref_emb in self._unknown_embeddings:
            distance = 1.0 - np.dot(embedding, ref_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_emb) + 1e-8
            )
            if distance < MATCH_THRESHOLD:
                return uid

        self._unknown_counter += 1
        uid = f"unknown_{self._unknown_counter}"
        self._unknown_embeddings.append((uid, embedding.copy()))

        # Keep only last 20 unknowns
        if len(self._unknown_embeddings) > 20:
            self._unknown_embeddings = self._unknown_embeddings[-20:]

        return uid

    def detect_faces(self, frame: np.ndarray) -> np.ndarray | None:
        """Run YuNet face detection on a frame.

        Returns:
            Nx15 array of face detections, or None if no faces found.
        """
        h, w = frame.shape[:2]
        detector = self._get_detector(w, h)
        _, faces = detector.detect(frame)
        return faces

    def identify(self, frame: np.ndarray) -> list[dict]:
        """Full pipeline: detect faces → extract embeddings → identify.

        Args:
            frame: BGR image (any size)

        Returns:
            List of dicts: [{"name": "Dan", "distance": 0.25,
                             "confidence": "high", "bbox": [x,y,w,h]}]
        """
        if self._recognizer is None:
            return []

        faces = self.detect_faces(frame)
        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            if w < 30 or h < 30:
                continue

            embedding = self.extract_embedding(frame, face)
            if embedding is None:
                continue

            name, distance = self.compare(embedding)

            confidence = "high" if distance < CONFIDENCE_HIGH else \
                         "medium" if distance < MATCH_THRESHOLD else "low"

            results.append({
                "name": name,
                "distance": round(distance, 3),
                "confidence": confidence,
                "bbox": [x, y, w, h],
            })

        return results

    def identify_from_jpeg(self, jpeg_bytes: bytes) -> list[dict]:
        """Convenience: identify faces from JPEG bytes."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return []
        return self.identify(frame)

    def get_enrolled_names(self) -> list[str]:
        """Return list of enrolled identity names."""
        return list(self._enrolled.keys())
