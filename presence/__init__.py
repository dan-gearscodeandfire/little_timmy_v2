"""Presence: cross-pipeline identity (voice + face) and room ledger."""

from .types import (
    FacePrediction,
    BehaviorSnapshot,
    FaceObservation,
    FusionVerdict,
    PersonRecord,
)
from .identity import canonicalize, fuse_identity, translate_pose
from .ledger import RoomLedger
from .face_client import fetch_face_observation, FaceClientConfig
from .auto_enroll import FaceHintStreak
from .look_at import LookAtPolicy, LookAtVerdict

__all__ = [
    "FacePrediction",
    "BehaviorSnapshot",
    "FaceObservation",
    "FusionVerdict",
    "PersonRecord",
    "canonicalize",
    "fuse_identity",
    "translate_pose",
    "RoomLedger",
    "fetch_face_observation",
    "FaceClientConfig",
    "FaceHintStreak",
    "LookAtPolicy",
    "LookAtVerdict",
]
