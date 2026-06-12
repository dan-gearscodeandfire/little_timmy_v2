"""Shared dataclasses for the presence module."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FacePrediction:
    user_id: str  # canonical lowercase name
    confidence: float  # 0..1 (higher = better)
    bbox: tuple  # (x_min, y_min, x_max, y_max) in pixels
    embedding_hash: Optional[str] = None  # short hash for unknown-face continuity


@dataclass(frozen=True)
class BehaviorSnapshot:
    mode: str
    face_visible: bool
    elapsed_ms: int
    last_face_pan: Optional[float] = None
    last_face_tilt: Optional[float] = None


@dataclass(frozen=True)
class FaceObservation:
    captured_at: float  # unix ts
    predictions: tuple  # tuple[FacePrediction]
    behavior: Optional[BehaviorSnapshot]
    image_size: Optional[tuple] = None  # (width, height) for bbox normalization


@dataclass(frozen=True)
class FusionVerdict:
    final_name: str
    resolution_source: str  # 'voice' | 'face_hint' (Slice B rides on stabilized)
    face_hint_name: Optional[str]
    face_hint_confidence: Optional[float]
    head_steady: bool
    gates: dict
    # Slice B (2026-06-12): provenance of face_hint_name. 'face' = a real face
    # prediction (the ONLY source allowed to train a voiceprint via auto-enroll);
    # 'voice' = synthesized from a confident voice; 'temporal' = held from a
    # recent frame. Defaults preserve today's behavior (every existing hint is a
    # real face). `stabilized` flags the two new synthesized/held paths.
    face_hint_source: str = "face"
    stabilized: bool = False


@dataclass
class PersonRecord:
    name: str  # canonical lowercase
    last_seen_face_ts: Optional[float] = None
    last_seen_voice_ts: Optional[float] = None
    on_camera_now: bool = False
    last_pose: Optional[dict] = None  # {pan, tilt, bbox_center_norm, ts}
    times_seen_face: int = 0
    times_heard_voice: int = 0
