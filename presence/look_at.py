"""Decide when to look at a speaker who is not currently on camera.

The trigger is intentionally narrow:
  - voice resolution (not face_hint, not unknown_N)
  - the speaker is NOT visible right now
  - last face sighting (with pose) is fresh enough to be useful
  - per-speaker cooldown to avoid panning every turn
  - current behavior mode is not 'engage' (don't disrupt active engagement)

The policy is pure-logic: it returns a verdict; the caller actually sends
the motor command.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LookAtVerdict:
    should_look: bool
    reason: str
    target_pan: Optional[float] = None
    target_tilt: Optional[float] = None
    pose_age_sec: Optional[float] = None


class LookAtPolicy:
    def __init__(
        self,
        cooldown_sec: float = 30.0,
        max_pose_age_sec: float = 120.0,
        fresh_face_age_sec: float = 30.0,
    ):
        self._cooldowns: dict[str, float] = {}
        self._cooldown_sec = cooldown_sec
        self._max_pose_age_sec = max_pose_age_sec
        self._fresh_face_age_sec = fresh_face_age_sec

    def evaluate(
        self,
        name: Optional[str],
        present_record: Optional[dict],
        last_pose: Optional[dict],
        current_behavior_mode: Optional[str],
        now_ts: float,
    ) -> LookAtVerdict:
        if not name or name.startswith("unknown"):
            return LookAtVerdict(False, "speaker is unknown or empty")

        if last_pose is None:
            return LookAtVerdict(False, "no recorded pose for speaker")

        pose_ts = last_pose.get("ts")
        if pose_ts is None:
            return LookAtVerdict(False, "pose missing timestamp")

        pose_age = now_ts - pose_ts
        if pose_age > self._max_pose_age_sec:
            return LookAtVerdict(
                False,
                f"pose too old ({pose_age:.0f}s > {self._max_pose_age_sec:.0f}s)",
                pose_age_sec=pose_age,
            )

        if present_record is not None:
            face_age = present_record.get("last_seen_face_age_s")
            if face_age is not None and face_age <= self._fresh_face_age_sec:
                return LookAtVerdict(
                    False,
                    f"face still fresh ({face_age:.0f}s <= {self._fresh_face_age_sec:.0f}s)",
                    pose_age_sec=pose_age,
                )

        last_lookat = self._cooldowns.get(name, 0.0)
        time_since = now_ts - last_lookat
        if time_since < self._cooldown_sec:
            return LookAtVerdict(
                False,
                f"cooldown ({time_since:.0f}s < {self._cooldown_sec:.0f}s)",
                pose_age_sec=pose_age,
            )

        if current_behavior_mode == "engage":
            return LookAtVerdict(
                False,
                "behavior is in engage mode (don't disrupt)",
                pose_age_sec=pose_age,
            )

        target_pan = last_pose.get("pan")
        target_tilt = last_pose.get("tilt")
        if target_pan is None or target_tilt is None:
            return LookAtVerdict(
                False,
                "pose missing pan or tilt",
                pose_age_sec=pose_age,
            )

        return LookAtVerdict(
            True,
            f"voice-confident speaker {name} not on camera; pose {pose_age:.0f}s old",
            target_pan=float(target_pan),
            target_tilt=float(target_tilt),
            pose_age_sec=pose_age,
        )

    def record_look_at(self, name: str, ts: float) -> None:
        if name:
            self._cooldowns[name] = ts

    def cooldown_remaining(self, name: str, now_ts: float) -> float:
        last = self._cooldowns.get(name, 0.0)
        elapsed = now_ts - last
        return max(0.0, self._cooldown_sec - elapsed)
