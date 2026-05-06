"""In-memory room ledger: who is present, when last seen, where last seen."""

import time
from typing import Optional

from .identity import canonicalize, translate_pose
from .types import FaceObservation, PersonRecord


class RoomLedger:
    """Tracks presence of named (and stable-unknown) people in the room.

    Presence is TTL-windowed: someone is `present` while the most recent
    voice or face sighting is within PRESENCE_TTL. `on_camera_now` is a
    transient flag set only for people seen in the most recent face frame.

    Updates run on the orchestrator's event loop (no thread crossings),
    so no lock is needed.
    """

    def __init__(
        self,
        presence_ttl_sec: float = 900.0,
        unknown_voice_ttl_sec: float = 120.0,
        camera_pan_fov_steps: float = 80.0,
        camera_tilt_fov_steps: float = 50.0,
    ):
        self._records: dict[str, PersonRecord] = {}
        self._ttl = presence_ttl_sec
        self._unknown_ttl = unknown_voice_ttl_sec
        self._pan_fov = camera_pan_fov_steps
        self._tilt_fov = camera_tilt_fov_steps

    def _key_for_face(self, prediction) -> str:
        """Canonical lookup key for a face prediction.

        Known face: use the canonicalized user_id.
        Unknown face: use a synthetic key from embedding hash so the same
        unknown face stays a single record across sightings.
        """
        canon = canonicalize(prediction.user_id)
        if canon and not canon.startswith("unknown"):
            return canon
        if prediction.embedding_hash:
            return f"unknown_face_{prediction.embedding_hash}"
        return canon or "unknown_face"

    def _ensure(self, name: str) -> PersonRecord:
        rec = self._records.get(name)
        if rec is None:
            rec = PersonRecord(name=name)
            self._records[name] = rec
        return rec

    def update_from_face(
        self,
        observation: FaceObservation,
        now_ts: Optional[float] = None,
    ) -> None:
        """Apply a face observation. Resets on_camera_now for everyone first."""
        ts = now_ts if now_ts is not None else time.time()

        for rec in self._records.values():
            rec.on_camera_now = False

        if not observation.predictions:
            return

        beh = observation.behavior
        cam_pan = beh.last_face_pan if beh else None
        cam_tilt = beh.last_face_tilt if beh else None

        img_w = observation.image_size[0] if observation.image_size else None
        img_h = observation.image_size[1] if observation.image_size else None

        for pred in observation.predictions:
            key = self._key_for_face(pred)
            rec = self._ensure(key)
            rec.last_seen_face_ts = ts
            rec.on_camera_now = True
            rec.times_seen_face += 1

            bbox_center_norm = None
            if pred.bbox and img_w and img_h:
                cx = (pred.bbox[0] + pred.bbox[2]) / 2.0 / img_w
                cy = (pred.bbox[1] + pred.bbox[3]) / 2.0 / img_h
                bbox_center_norm = (cx, cy)

            if cam_pan is not None and cam_tilt is not None:
                person_pan, person_tilt = translate_pose(
                    cam_pan, cam_tilt, bbox_center_norm,
                    pan_fov_steps=self._pan_fov,
                    tilt_fov_steps=self._tilt_fov,
                )
                rec.last_pose = {
                    "pan": person_pan,
                    "tilt": person_tilt,
                    "camera_pan": cam_pan,
                    "camera_tilt": cam_tilt,
                    "bbox_center_norm": bbox_center_norm,
                    "ts": ts,
                }

    def update_from_voice(
        self,
        name: str,
        ts: Optional[float] = None,
    ) -> None:
        """Apply a voice sighting (any name, including unknown_N)."""
        canon = canonicalize(name)
        if not canon:
            return
        ts = ts if ts is not None else time.time()
        rec = self._ensure(canon)
        rec.last_seen_voice_ts = ts
        rec.times_heard_voice += 1

    def _is_present(self, rec: PersonRecord, now_ts: float) -> bool:
        """Within the presence TTL on either signal?

        Unknown_N voice records use a tighter TTL since they accumulate fast.
        """
        is_unknown = rec.name.startswith("unknown")
        ttl = self._unknown_ttl if is_unknown else self._ttl

        latest = max(
            rec.last_seen_face_ts or 0.0,
            rec.last_seen_voice_ts or 0.0,
        )
        if latest <= 0:
            return False
        return (now_ts - latest) <= ttl

    def current_state(self, now_ts: Optional[float] = None) -> dict:
        """Snapshot of who is present right now.

        Returns:
          {
            "now": ts,
            "present": [
              {
                name, on_camera_now,
                last_seen_face_age_s, last_seen_voice_age_s,
                source: 'face'|'voice'|'both',
                last_pose: {pan, tilt, bbox_center_norm, ts} | None,
                times_seen_face, times_heard_voice,
              }, ...
            ],
            "unknown_voices_recent": int,
          }
        """
        now = now_ts if now_ts is not None else time.time()
        present = []
        unknown_voices = 0

        for rec in self._records.values():
            if not self._is_present(rec, now):
                continue

            if rec.name.startswith("unknown") and not rec.name.startswith("unknown_face"):
                unknown_voices += 1
                continue

            face_age = (now - rec.last_seen_face_ts) if rec.last_seen_face_ts else None
            voice_age = (now - rec.last_seen_voice_ts) if rec.last_seen_voice_ts else None

            if face_age is not None and voice_age is not None:
                source = "both"
            elif face_age is not None:
                source = "face"
            else:
                source = "voice"

            present.append({
                "name": rec.name,
                "on_camera_now": rec.on_camera_now,
                "last_seen_face_age_s": round(face_age, 1) if face_age is not None else None,
                "last_seen_voice_age_s": round(voice_age, 1) if voice_age is not None else None,
                "source": source,
                "last_pose": rec.last_pose,
                "times_seen_face": rec.times_seen_face,
                "times_heard_voice": rec.times_heard_voice,
            })

        present.sort(key=lambda p: (
            not p["on_camera_now"],
            min(
                p["last_seen_face_age_s"] if p["last_seen_face_age_s"] is not None else 1e9,
                p["last_seen_voice_age_s"] if p["last_seen_voice_age_s"] is not None else 1e9,
            ),
        ))

        return {
            "now": now,
            "present": present,
            "unknown_voices_recent": unknown_voices,
        }

    def find_pose_for(self, name: str) -> Optional[dict]:
        """Look up the most recent recorded pose for a canonical name."""
        canon = canonicalize(name)
        if not canon:
            return None
        rec = self._records.get(canon)
        return rec.last_pose if rec else None
