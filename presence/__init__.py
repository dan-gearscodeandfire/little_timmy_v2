"""Presence: cross-pipeline identity (voice + face) and room ledger."""

from .types import (
    FacePrediction,
    BehaviorSnapshot,
    FaceObservation,
    FusionVerdict,
    PersonRecord,
)
from .identity import canonicalize, fuse_identity
from .ledger import RoomLedger
from .face_client import fetch_face_observation, FaceClientConfig

__all__ = [
    "FacePrediction",
    "BehaviorSnapshot",
    "FaceObservation",
    "FusionVerdict",
    "PersonRecord",
    "canonicalize",
    "fuse_identity",
    "RoomLedger",
    "fetch_face_observation",
    "FaceClientConfig",
]
