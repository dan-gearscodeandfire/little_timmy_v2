"""Identity canonicalization and voice+face fusion rule.

Pure logic, no I/O. Unit-testable in isolation.
"""

from typing import Optional

from .types import FaceObservation, FusionVerdict


_TRACKING_MODES = frozenset({"engage", "track"})


def canonicalize(name: Optional[str]) -> Optional[str]:
    """Lowercase + strip; returns None for None/empty."""
    if name is None:
        return None
    s = name.strip().lower()
    return s or None


def fuse_identity(
    *,
    voice_name: str,
    voice_confidence: float,
    voice_is_unknown: bool,
    face: Optional[FaceObservation],
    face_conf_threshold: float = 0.85,
    head_steady_min_ms: int = 2000,
) -> FusionVerdict:
    """Resolve speaker identity given voice result and optional face observation.

    Voice always wins for confident matches. Face only contributes a name when:
      - voice is unknown_N (below voiceprint threshold)
      - face returned exactly one prediction
      - face confidence >= face_conf_threshold
      - behavior status reports tracking mode + face_visible
      - head has been steady on the face for >= head_steady_min_ms

    Otherwise face is recorded as a presence hint but not promoted to speaker.
    """
    voice_name = canonicalize(voice_name) or voice_name
    face_hint_name = None
    face_hint_confidence = None
    head_steady = False

    gates = {
        "voice_unknown": voice_is_unknown,
        "face_present": False,
        "single_face": False,
        "face_above_threshold": False,
        "behavior_known": False,
        "tracking_mode": False,
        "face_visible_flag": False,
        "head_steady": False,
    }

    if face is not None and face.predictions:
        gates["face_present"] = True
        gates["single_face"] = len(face.predictions) == 1
        top = face.predictions[0]
        face_hint_name = canonicalize(top.user_id)
        face_hint_confidence = float(top.confidence)
        gates["face_above_threshold"] = face_hint_confidence >= face_conf_threshold

        beh = face.behavior
        if beh is not None:
            gates["behavior_known"] = True
            gates["tracking_mode"] = beh.mode in _TRACKING_MODES
            gates["face_visible_flag"] = bool(beh.face_visible)
            head_steady = beh.elapsed_ms >= head_steady_min_ms
            gates["head_steady"] = head_steady

    promote = (
        voice_is_unknown
        and gates["face_present"]
        and gates["single_face"]
        and gates["face_above_threshold"]
        and gates["behavior_known"]
        and gates["tracking_mode"]
        and gates["face_visible_flag"]
        and gates["head_steady"]
    )

    if promote and face_hint_name is not None:
        final_name = face_hint_name
        resolution_source = "face_hint"
    else:
        final_name = voice_name
        resolution_source = "voice"

    return FusionVerdict(
        final_name=final_name,
        resolution_source=resolution_source,
        face_hint_name=face_hint_name,
        face_hint_confidence=face_hint_confidence,
        head_steady=head_steady,
        gates=gates,
    )
