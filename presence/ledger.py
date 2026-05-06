"""In-memory room ledger: who is present, when last seen, where last seen."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .identity import canonicalize, translate_pose
from .types import FaceObservation, PersonRecord

log = logging.getLogger(__name__)


class RoomLedger:
    """Tracks presence of named (and stable-unknown) people in the room.

    Presence is TTL-windowed: someone is `present` while the most recent
    voice or face sighting is within PRESENCE_TTL. `on_camera_now` is a
    smoothed flag (True if face seen within `on_camera_fresh_threshold_sec`)
    so it doesn't flicker between VLM ticks. Raw "in latest frame" remains
    available as `on_camera_in_latest_frame`.

    Updates run on the orchestrator's event loop (no thread crossings),
    so no lock is needed.

    If `save_path` is set, the ledger persists to that JSON file on every
    update and reloads on init (dropping entries past TTL).
    """

    def __init__(
        self,
        presence_ttl_sec: float = 900.0,
        unknown_voice_ttl_sec: float = 120.0,
        camera_pan_fov_steps: float = 80.0,
        camera_tilt_fov_steps: float = 50.0,
        on_camera_fresh_threshold_sec: float = 30.0,
        save_path: Optional[str] = None,
    ):
        self._records: dict[str, PersonRecord] = {}
        self._ttl = presence_ttl_sec
        self._unknown_ttl = unknown_voice_ttl_sec
        self._pan_fov = camera_pan_fov_steps
        self._tilt_fov = camera_tilt_fov_steps
        self._on_camera_fresh = on_camera_fresh_threshold_sec
        self._save_path = Path(save_path) if save_path else None
        if self._save_path and self._save_path.exists():
            self._load_from_disk()

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
            self._save_to_disk()
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

        self._save_to_disk()

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
        self._save_to_disk()

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
        """Snapshot of who is present right now."""
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

            # Smoothed on_camera_now: True if face seen within fresh threshold,
            # to avoid the dot flickering off between VLM ticks.
            on_camera_smoothed = (
                face_age is not None and face_age <= self._on_camera_fresh
            )

            present.append({
                "name": rec.name,
                "on_camera_now": on_camera_smoothed,
                "on_camera_in_latest_frame": rec.on_camera_now,
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

    def _save_to_disk(self) -> None:
        """Atomic JSON dump (.tmp + rename). Best-effort; logs on failure."""
        if not self._save_path:
            return
        try:
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "records": {
                    name: {
                        "name": rec.name,
                        "last_seen_face_ts": rec.last_seen_face_ts,
                        "last_seen_voice_ts": rec.last_seen_voice_ts,
                        "last_pose": rec.last_pose,
                        "times_seen_face": rec.times_seen_face,
                        "times_heard_voice": rec.times_heard_voice,
                    }
                    for name, rec in self._records.items()
                },
            }
            tmp = self._save_path.with_suffix(self._save_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self._save_path)
        except Exception as e:
            log.warning("RoomLedger save failed: %s", e)

    def _load_from_disk(self) -> None:
        """Best-effort load; drops entries whose latest signal is older than TTL."""
        try:
            payload = json.loads(self._save_path.read_text())
        except Exception as e:
            log.warning("RoomLedger load failed: %s; starting empty", e)
            return
        if not isinstance(payload, dict) or payload.get("version") != 1:
            log.warning("RoomLedger file format unknown; starting empty")
            return
        records = payload.get("records", {})
        if not isinstance(records, dict):
            return
        now = time.time()
        loaded = 0
        skipped = 0
        for key, data in records.items():
            try:
                latest = max(
                    data.get("last_seen_face_ts") or 0.0,
                    data.get("last_seen_voice_ts") or 0.0,
                )
                is_unknown = key.startswith("unknown")
                ttl = self._unknown_ttl if is_unknown else self._ttl
                if latest <= 0 or (now - latest) > ttl:
                    skipped += 1
                    continue
                rec = PersonRecord(
                    name=data.get("name", key),
                    last_seen_face_ts=data.get("last_seen_face_ts"),
                    last_seen_voice_ts=data.get("last_seen_voice_ts"),
                    on_camera_now=False,
                    last_pose=data.get("last_pose"),
                    times_seen_face=int(data.get("times_seen_face", 0)),
                    times_heard_voice=int(data.get("times_heard_voice", 0)),
                )
                self._records[key] = rec
                loaded += 1
            except Exception as e:
                log.warning("RoomLedger record %s load failed: %s", key, e)
                skipped += 1
        log.info("RoomLedger loaded %d records (%d expired/skipped)", loaded, skipped)
