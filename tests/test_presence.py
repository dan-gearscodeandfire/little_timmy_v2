"""Unit tests for the presence module: canonicalize, fuse_identity, RoomLedger,
FaceHintStreak.

Pure-logic tests - no network, no LT services. Run:
    .venv/bin/pytest tests/test_presence.py -v
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.identity import canonicalize, fuse_identity
from presence.ledger import RoomLedger
from presence.auto_enroll import FaceHintStreak
from presence.types import (
    BehaviorSnapshot,
    FaceObservation,
    FacePrediction,
)


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_lowercases(self):
        assert canonicalize("Dan") == "dan"
        assert canonicalize("THEA") == "thea"

    def test_strips_whitespace(self):
        assert canonicalize("  Devon  ") == "devon"
        assert canonicalize("\tDan\n") == "dan"

    def test_none_returns_none(self):
        assert canonicalize(None) is None

    def test_empty_returns_none(self):
        assert canonicalize("") is None
        assert canonicalize("   ") is None

    def test_already_canonical(self):
        assert canonicalize("dan") == "dan"


# ---------------------------------------------------------------------------
# fuse_identity helpers
# ---------------------------------------------------------------------------


def _good_behavior(elapsed_ms=3000, mode="track", face_visible=True):
    return BehaviorSnapshot(
        mode=mode,
        face_visible=face_visible,
        elapsed_ms=elapsed_ms,
        last_face_pan=15.0,
        last_face_tilt=-3.0,
    )


def _face_obs(predictions, behavior=None, image_size=(640, 480)):
    return FaceObservation(
        captured_at=time.time(),
        predictions=tuple(predictions),
        behavior=behavior,
        image_size=image_size,
    )


def _pred(name="Devon", confidence=0.91, bbox=(100, 80, 220, 210)):
    return FacePrediction(user_id=name, confidence=confidence, bbox=bbox)


# ---------------------------------------------------------------------------
# fuse_identity
# ---------------------------------------------------------------------------


class TestFuseIdentity:
    def test_voice_confident_wins_over_face(self):
        v = fuse_identity(
            voice_name="dan",
            voice_is_unknown=False,
            face=_face_obs([_pred("Thea", 0.95)], _good_behavior()),
        )
        assert v.final_name == "dan"
        assert v.resolution_source == "voice"
        assert v.face_hint_name == "thea"

    def test_voice_unknown_plus_sole_steady_face_promotes(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs([_pred("Devon", 0.91)], _good_behavior()),
        )
        assert v.final_name == "devon"
        assert v.resolution_source == "face_hint"
        assert v.face_hint_confidence == pytest.approx(0.91)
        assert v.head_steady is True

    def test_voice_unknown_plus_multi_face_blocks_promotion(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs(
                [_pred("Devon", 0.91), _pred("Thea", 0.88)],
                _good_behavior(),
            ),
        )
        assert v.final_name == "unknown_3"
        assert v.resolution_source == "voice"
        assert v.gates["single_face"] is False

    def test_head_not_steady_blocks_promotion(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs([_pred("Devon", 0.91)], _good_behavior(elapsed_ms=800)),
        )
        assert v.final_name == "unknown_3"
        assert v.resolution_source == "voice"
        assert v.gates["head_steady"] is False

    def test_face_below_threshold_blocks_promotion(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs([_pred("Devon", 0.70)], _good_behavior()),
        )
        assert v.final_name == "unknown_3"
        assert v.resolution_source == "voice"
        assert v.gates["face_above_threshold"] is False

    def test_missing_behavior_blocks_promotion(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs([_pred("Devon", 0.91)], behavior=None),
        )
        assert v.final_name == "unknown_3"
        assert v.resolution_source == "voice"
        assert v.gates["behavior_known"] is False

    def test_face_not_visible_flag_blocks_promotion(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs(
                [_pred("Devon", 0.91)],
                _good_behavior(face_visible=False),
            ),
        )
        assert v.final_name == "unknown_3"
        assert v.gates["face_visible_flag"] is False

    def test_idle_mode_blocks_promotion(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs(
                [_pred("Devon", 0.91)],
                _good_behavior(mode="idle"),
            ),
        )
        assert v.final_name == "unknown_3"
        assert v.gates["tracking_mode"] is False

    def test_no_face_obs_falls_back_to_voice(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=None,
        )
        assert v.final_name == "unknown_3"
        assert v.resolution_source == "voice"
        assert v.face_hint_name is None

    def test_engage_mode_also_promotes(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs(
                [_pred("Devon", 0.91)],
                _good_behavior(mode="engage"),
            ),
        )
        assert v.resolution_source == "face_hint"

    def test_face_hint_name_is_canonicalized(self):
        v = fuse_identity(
            voice_name="unknown_3",
            voice_is_unknown=True,
            face=_face_obs([_pred("DEVON", 0.91)], _good_behavior()),
        )
        assert v.face_hint_name == "devon"
        assert v.final_name == "devon"


# ---------------------------------------------------------------------------
# RoomLedger
# ---------------------------------------------------------------------------


class TestRoomLedger:
    def test_face_update_marks_on_camera_now(self):
        led = RoomLedger(presence_ttl_sec=900)
        obs = _face_obs([_pred("Dan", 0.92)], _good_behavior())
        led.update_from_face(obs)
        state = led.current_state()
        assert any(p["name"] == "dan" and p["on_camera_now"] for p in state["present"])

    def test_second_face_update_clears_first_persons_on_camera(self):
        led = RoomLedger(presence_ttl_sec=900)
        led.update_from_face(_face_obs([_pred("Dan", 0.92)], _good_behavior()))
        led.update_from_face(_face_obs([_pred("Thea", 0.90)], _good_behavior()))
        state = led.current_state()
        names_on_camera = {p["name"] for p in state["present"] if p["on_camera_now"]}
        assert names_on_camera == {"thea"}
        dan = next(p for p in state["present"] if p["name"] == "dan")
        assert dan["on_camera_now"] is False

    def test_voice_only_creates_record(self):
        led = RoomLedger(presence_ttl_sec=900)
        led.update_from_voice("dan")
        state = led.current_state()
        assert any(p["name"] == "dan" and p["source"] == "voice" for p in state["present"])

    def test_voice_plus_face_source_is_both(self):
        led = RoomLedger(presence_ttl_sec=900)
        led.update_from_face(_face_obs([_pred("Dan", 0.92)], _good_behavior()))
        led.update_from_voice("dan")
        state = led.current_state()
        dan = next(p for p in state["present"] if p["name"] == "dan")
        assert dan["source"] == "both"

    def test_ttl_boundary_just_under_keeps_person(self):
        led = RoomLedger(presence_ttl_sec=900)
        t0 = 1000.0
        led.update_from_face(_face_obs([_pred("Dan", 0.92)], _good_behavior()), now_ts=t0)
        state = led.current_state(now_ts=t0 + 14 * 60)
        assert any(p["name"] == "dan" for p in state["present"])

    def test_ttl_boundary_just_over_drops_person(self):
        led = RoomLedger(presence_ttl_sec=900)
        t0 = 1000.0
        led.update_from_face(_face_obs([_pred("Dan", 0.92)], _good_behavior()), now_ts=t0)
        state = led.current_state(now_ts=t0 + 16 * 60)
        assert all(p["name"] != "dan" for p in state["present"])

    def test_unknown_voice_uses_tighter_ttl(self):
        led = RoomLedger(presence_ttl_sec=900, unknown_voice_ttl_sec=120)
        t0 = 1000.0
        led.update_from_voice("unknown_3", ts=t0)
        state_under = led.current_state(now_ts=t0 + 90)
        assert state_under["unknown_voices_recent"] == 1
        state_over = led.current_state(now_ts=t0 + 130)
        assert state_over["unknown_voices_recent"] == 0

    def test_pose_recorded_with_translation(self):
        """Pose is translated from camera pose using bbox offset.

        bbox (100,80,200,200) over 640x480 -> center (0.234, 0.292).
        With default fov 80x50:
          person_pan  = 42.0 - (0.234 - 0.5) * 80 = 42.0 + 21.25 = 63.25
          person_tilt = -7.5 - (0.292 - 0.5) * 50 = -7.5 + 10.42 = 2.92
        Camera pose preserved for debugging.
        """
        led = RoomLedger(presence_ttl_sec=900)
        beh = BehaviorSnapshot(
            mode="track",
            face_visible=True,
            elapsed_ms=3000,
            last_face_pan=42.0,
            last_face_tilt=-7.5,
        )
        obs = _face_obs([_pred("Dan", 0.92, bbox=(100, 80, 200, 200))], beh)
        led.update_from_face(obs)
        pose = led.find_pose_for("dan")
        assert pose is not None
        # Translated person pose
        assert pose["pan"] == pytest.approx(42.0 - (150 / 640 - 0.5) * 80)
        assert pose["tilt"] == pytest.approx(-7.5 - (140 / 480 - 0.5) * 50)
        # Camera pose preserved
        assert pose["camera_pan"] == 42.0
        assert pose["camera_tilt"] == -7.5
        # bbox_center_norm preserved
        assert pose["bbox_center_norm"] is not None
        cx, cy = pose["bbox_center_norm"]
        assert cx == pytest.approx(150 / 640)
        assert cy == pytest.approx(140 / 480)

    def test_two_faces_same_frame_get_distinct_poses(self):
        """The whole point: Dan on left and Thea on right get DIFFERENT poses
        even though the camera is at one orientation."""
        led = RoomLedger(presence_ttl_sec=900)
        beh = BehaviorSnapshot(
            mode="track",
            face_visible=True,
            elapsed_ms=3000,
            last_face_pan=0.0,
            last_face_tilt=0.0,
        )
        # Dan on left third, Thea on right third
        dan_bbox = (50, 200, 150, 350)    # center x=100, y=275 -> (0.156, 0.573)
        thea_bbox = (490, 200, 590, 350)  # center x=540, y=275 -> (0.844, 0.573)
        obs = _face_obs(
            [_pred("Dan", 0.92, bbox=dan_bbox), _pred("Thea", 0.90, bbox=thea_bbox)],
            beh,
        )
        led.update_from_face(obs)
        dan_pose = led.find_pose_for("dan")
        thea_pose = led.find_pose_for("thea")
        assert dan_pose is not None
        assert thea_pose is not None
        # Same camera pose but different person poses
        assert dan_pose["camera_pan"] == thea_pose["camera_pan"] == 0.0
        assert dan_pose["pan"] != thea_pose["pan"]
        # Dan on left of frame -> POSITIVE pan (per streamerpi inverted convention)
        assert dan_pose["pan"] > 0
        assert thea_pose["pan"] < 0

    def test_pose_none_when_no_behavior(self):
        led = RoomLedger(presence_ttl_sec=900)
        led.update_from_face(_face_obs([_pred("Dan", 0.92)], behavior=None))
        assert led.find_pose_for("dan") is None

    def test_voice_extends_face_only_presence(self):
        led = RoomLedger(presence_ttl_sec=900)
        t0 = 1000.0
        led.update_from_face(_face_obs([_pred("Dan", 0.92)], _good_behavior()), now_ts=t0)
        led.update_from_voice("dan", ts=t0 + 800)
        state = led.current_state(now_ts=t0 + 800 + 200)
        assert any(p["name"] == "dan" for p in state["present"])

    def test_unknown_face_prediction_with_hash_creates_distinct_entry(self):
        led = RoomLedger(presence_ttl_sec=900)
        pred1 = FacePrediction(
            user_id="unknown",
            confidence=0.5,
            bbox=(0, 0, 100, 100),
            embedding_hash="abc12345",
        )
        led.update_from_face(_face_obs([pred1], _good_behavior()))
        assert any(k.startswith("unknown_face_") for k in led._records.keys())

    def test_find_pose_for_unknown_name(self):
        led = RoomLedger(presence_ttl_sec=900)
        assert led.find_pose_for("nobody") is None
        assert led.find_pose_for(None) is None

    def test_present_list_sorted_visible_first(self):
        led = RoomLedger(presence_ttl_sec=900)
        led.update_from_voice("dan")
        led.update_from_face(_face_obs([_pred("Thea", 0.92)], _good_behavior()))
        state = led.current_state()
        assert state["present"][0]["name"] == "thea"
        assert state["present"][0]["on_camera_now"] is True


# ---------------------------------------------------------------------------
# FaceHintStreak (auto-enrollment trigger)
# ---------------------------------------------------------------------------


class TestFaceHintStreak:
    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            FaceHintStreak(threshold=0)
        with pytest.raises(ValueError):
            FaceHintStreak(threshold=-1)

    def test_no_promotion_no_streak(self):
        s = FaceHintStreak(threshold=3)
        assert s.observe(None, "unknown_3") is None
        assert s.observe(None, None) is None
        assert s.current is None

    def test_below_threshold_does_not_fire(self):
        s = FaceHintStreak(threshold=3)
        assert s.observe("Devon", "unknown_3") is None
        assert s.observe("Devon", "unknown_3") is None
        assert s.current.count == 2

    def test_at_threshold_fires(self):
        s = FaceHintStreak(threshold=3)
        s.observe("Devon", "unknown_3")
        s.observe("Devon", "unknown_3")
        result = s.observe("Devon", "unknown_3")
        assert result is not None
        assert result.face_hint_name == "devon"
        assert result.voice_temp_id == "unknown_3"
        assert result.count == 3

    def test_different_name_resets(self):
        s = FaceHintStreak(threshold=3)
        s.observe("Devon", "unknown_3")
        s.observe("Devon", "unknown_3")
        s.observe("Thea", "unknown_3")  # different face name
        assert s.current.face_hint_name == "thea"
        assert s.current.count == 1

    def test_different_temp_id_resets(self):
        s = FaceHintStreak(threshold=3)
        s.observe("Devon", "unknown_3")
        s.observe("Devon", "unknown_3")
        s.observe("Devon", "unknown_4")  # different unknown speaker
        assert s.current.voice_temp_id == "unknown_4"
        assert s.current.count == 1

    def test_no_promotion_clears_streak(self):
        s = FaceHintStreak(threshold=3)
        s.observe("Devon", "unknown_3")
        s.observe("Devon", "unknown_3")
        s.observe(None, None)  # voice was confident this turn
        assert s.current is None

    def test_canonicalizes_face_name(self):
        s = FaceHintStreak(threshold=3)
        s.observe("Devon", "unknown_3")
        s.observe("DEVON", "unknown_3")
        result = s.observe("  devon  ", "unknown_3")
        assert result is not None
        assert result.face_hint_name == "devon"

    def test_reset_clears_state(self):
        s = FaceHintStreak(threshold=2)
        s.observe("Devon", "unknown_3")
        result = s.observe("Devon", "unknown_3")
        assert result is not None  # threshold met
        s.reset()
        assert s.current is None
        # Next observation starts a fresh streak
        s.observe("Devon", "unknown_3")
        assert s.current.count == 1

    def test_threshold_one_fires_immediately(self):
        s = FaceHintStreak(threshold=1)
        result = s.observe("Devon", "unknown_3")
        assert result is not None
        assert result.count == 1

    def test_empty_face_name_treated_as_no_promotion(self):
        s = FaceHintStreak(threshold=3)
        s.observe("Devon", "unknown_3")
        s.observe("", "unknown_3")
        assert s.current is None



# ---------------------------------------------------------------------------
# translate_pose
# ---------------------------------------------------------------------------


from presence.identity import translate_pose


class TestTranslatePose:
    def test_center_bbox_no_offset(self):
        p, t = translate_pose(10.0, 5.0, (0.5, 0.5))
        assert p == 10.0
        assert t == 5.0

    def test_right_edge_negative_pan(self):
        """Right of image -> camera should pan toward more negative
        (streamerpi camera.py inverted convention)."""
        p, _ = translate_pose(0.0, 0.0, (1.0, 0.5))
        assert p == pytest.approx(-40.0)  # -(1.0 - 0.5) * 80

    def test_left_edge_positive_pan(self):
        p, _ = translate_pose(0.0, 0.0, (0.0, 0.5))
        assert p == pytest.approx(40.0)

    def test_top_edge_positive_tilt(self):
        """Top of image -> face above camera pointing -> need to tilt UP
        (UI positive in streamerpi convention)."""
        _, t = translate_pose(0.0, 0.0, (0.5, 0.0))
        assert t == pytest.approx(25.0)  # -(0.0 - 0.5) * 50

    def test_bottom_edge_negative_tilt(self):
        _, t = translate_pose(0.0, 0.0, (0.5, 1.0))
        assert t == pytest.approx(-25.0)

    def test_none_bbox_returns_camera_pose(self):
        p, t = translate_pose(15.0, -3.0, None)
        assert p == 15.0
        assert t == -3.0

    def test_offset_added_to_camera_pose(self):
        """Translation is camera_pose + offset, not just offset."""
        p, t = translate_pose(15.0, -3.0, (0.75, 0.25))
        assert p == pytest.approx(15.0 - (0.75 - 0.5) * 80)
        assert t == pytest.approx(-3.0 - (0.25 - 0.5) * 50)

    def test_custom_fov(self):
        """Custom FoV scales the offset."""
        p, _ = translate_pose(0.0, 0.0, (1.0, 0.5), pan_fov_steps=160.0)
        assert p == pytest.approx(-80.0)
