"""Shared dataclasses for the presence module."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FacePrediction:
    user_id: str  # canonical lowercase name
    confidence: float  # 0..1 (higher = better); = 1 - cosine_distance (lossless)
    bbox: tuple  # (x_min, y_min, x_max, y_max) in pixels
    embedding_hash: Optional[str] = None  # short hash for unknown-face continuity
    # streamerpi's calibrated confidence band (camera.py:737): "high" = dist<0.30
    # (conf>0.70), "medium" = 0.30-0.45 (conf 0.55-0.70). Carried as provenance so
    # the fuse gate tracks streamerpi's own cutoff instead of re-deriving it; None
    # for synthetic/test predictions (the gate then derives the band from
    # confidence). See _convert_in_tree_results.
    band: Optional[str] = None
    # True when streamerpi's identity stabilizer is HOLDING a latched identity
    # across an uncertain frame (face_identity_stabilizer.py): the face is
    # confidently the held person even though this frame's raw distance drifted
    # into the medium band. Lets the streak trust "medium + sticky" as a confident
    # hold while still excluding fresh-uncertain mediums.
    sticky: bool = False


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
    # Total faces DETECTED in frame (recognized OR not), when the source knows it
    # (okDemerzel self-serve path counts YuNet detections; predictions only carry
    # RECOGNISED faces, so a lone stranger would otherwise be invisible). Drives
    # the "sole face in frame == the speaker" rule (2026-07-01, standing in for
    # the green-LED is_speaker signal). None => source didn't report it; fusion
    # falls back to counting predictions (Pi/legacy path).
    detected_face_count: Optional[int] = None


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
    # True when a promoted face clears the STRICTER voiceprint-streak band
    # ("high" OR "medium"+sticky) — i.e. confident enough to bind a voiceprint,
    # not just to attribute the turn. Attribution promotes on high+medium; the
    # streak only counts when this is set. Always False without promotion.
    streak_eligible: bool = False


@dataclass
class PersonRecord:
    name: str  # canonical lowercase
    last_seen_face_ts: Optional[float] = None
    last_seen_voice_ts: Optional[float] = None
    on_camera_now: bool = False
    last_pose: Optional[dict] = None  # {pan, tilt, bbox_center_norm, ts}
    times_seen_face: int = 0  # lifetime face sightings (monotonic)
    times_heard_voice: int = 0
    # Consecutive face sightings with no gap longer than the reconfirm window.
    # Resets to 1 when a sighting lands after a long absence, so a stray frame
    # re-hitting a record the person has left reverts it to provisional rather
    # than re-lighting the full presence TTL. (Lifetime count stays in
    # times_seen_face.)
    face_confirm_streak: int = 0
