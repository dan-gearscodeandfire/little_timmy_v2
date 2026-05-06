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



def translate_pose(
    camera_pan: float,
    camera_tilt: float,
    bbox_center_norm,
    pan_fov_steps: float = 80.0,
    tilt_fov_steps: float = 50.0,
):
    """Translate camera pose + face bbox center to the absolute pose that
    would re-center the camera on that face.

    Mirrors streamerpi camera.py face-centering math (camera.py:400-403):
        pan_correction  = -offset_x_norm * (pan_fov_steps / 2)
        tilt_correction = -offset_y_norm * (tilt_fov_steps / 2)

    where offset_x_norm = (bbox_center_x - 0.5) * 2 (i.e. -1..+1 across image).

    Args:
        camera_pan: current commanded camera pan (UI steps, streamerpi convention)
        camera_tilt: current commanded camera tilt
        bbox_center_norm: (x, y) tuple in [0,1] over image, or None
        pan_fov_steps: total horizontal FoV in UI pan steps (default 80)
        tilt_fov_steps: total vertical FoV in UI tilt steps (default 50)

    Returns:
        (person_pan, person_tilt) tuple. If bbox_center_norm is None, returns
        (camera_pan, camera_tilt) unchanged.
    """
    if bbox_center_norm is None:
        return (camera_pan, camera_tilt)
    bx, by = bbox_center_norm
    person_pan = camera_pan - (bx - 0.5) * pan_fov_steps
    person_tilt = camera_tilt - (by - 0.5) * tilt_fov_steps
    return (person_pan, person_tilt)
