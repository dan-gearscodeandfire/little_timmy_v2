"""Vision context manager.

Maintains the current structured scene record and provides it to the prompt builder.
Thread-safe: capture/analysis runs async, prompt builder reads latest record.

Uses the relevance classifier (Step 3) to filter what gets injected into
the prompt, rather than dumping every VLM output.
"""

import asyncio
import logging
import time
from collections import deque

import config
from vision.capture import FrameCapture
from vision.analyzer import analyze_frame, check_model_available, SceneRecord
from vision.relevance import classify, RelevanceResult, INJECT_THRESHOLD
from vision.face_id import FaceID

log = logging.getLogger(__name__)


class VisionContext:
    """Manages the current visual scene for prompt injection."""

    def __init__(self):
        self._capture = FrameCapture()
        self._current: SceneRecord | None = None
        self._last_update: float = 0.0
        self._lock = asyncio.Lock()
        self._enabled = False
        self._face_id = FaceID()
        self._face_id_ready = False
        # Rolling buffer of recent scene records (for temporal context)
        self._history: deque[SceneRecord] = deque(maxlen=10)
        # Last analyzed JPEG (for debug/dashboard display)
        self._last_jpeg: bytes | None = None
        # Last relevance result
        self._last_relevance: RelevanceResult | None = None
        self._passive_face_callback = None

    def set_passive_face_callback(self, fn):
        """Register a callback invoked after each face-id enrichment.

        Signature: fn(results: list[dict], image_size: tuple | None) -> None
        Each result dict has keys: name, distance, confidence, bbox (x,y,w,h).
        """
        self._passive_face_callback = fn

    async def start(self):
        """Initialize vision pipeline: check model, start capture."""
        if not config.VISION_ENABLED:
            log.info("Vision pipeline disabled by config")
            return

        available = await check_model_available()
        if not available:
            log.warning("Vision pipeline disabled -- vision model not available")
            return

        self._enabled = True

        # Initialize face identification
        if self._face_id.init_models():
            n = self._face_id.load_db()
            self._face_id_ready = True
            log.info("Face ID ready (%d enrolled identities)", n)
        else:
            log.warning("Face ID not available (models not found)")

        await self._capture.start(self._on_frame)
        log.info("Vision context started (Qwen2.5-VL structured pipeline + face ID)")

    async def stop(self):
        """Stop the vision pipeline."""
        if self._enabled:
            await self._capture.stop()
            self._enabled = False
        log.info("Vision context stopped")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _on_frame(self, jpeg_bytes: bytes):
        """Callback from FrameCapture -- analyze the frame and update record."""
        record = await analyze_frame(jpeg_bytes)
        if record:
            # Enrich with face identification
            if self._face_id_ready:
                self._enrich_with_face_id(record, jpeg_bytes)

            async with self._lock:
                # Run relevance classifier against history
                history_list = list(self._history)
                relevance = classify(record, history_list)

                self._current = record
                self._last_update = time.monotonic()
                self._last_jpeg = jpeg_bytes
                self._history.append(record)
                self._last_relevance = relevance

    async def trigger_capture(self, reason: str = "speech") -> SceneRecord | None:
        """Trigger an immediate capture + analysis. Returns the new record."""
        if not self._enabled:
            return None
        jpeg = await self._capture.trigger(reason)
        if jpeg:
            record = await analyze_frame(jpeg)
            if record:
                # Enrich with face identification
                if self._face_id_ready:
                    self._enrich_with_face_id(record, jpeg)

                async with self._lock:
                    history_list = list(self._history)
                    relevance = classify(record, history_list)

                    self._current = record
                    self._last_update = time.monotonic()
                    self._history.append(record)
                    self._last_relevance = relevance
                return record
        return self._current if self._current else None

    def _enrich_with_face_id(self, record, jpeg_bytes: bytes):
        """Run face identification and update record.people with real names."""
        try:
            results = self._face_id.identify_from_jpeg(jpeg_bytes)
            if not results:
                return

            identified = []
            for r in results:
                name = r["name"]
                conf = r["confidence"]
                if conf in ("high", "medium"):
                    identified.append(name)
                    log.info("[FACE_ID] %s (distance=%.3f, %s)",
                             name, r["distance"], conf)
                else:
                    identified.append("unidentified person")
                    log.debug("[FACE_ID] Unknown face (distance=%.3f)", r["distance"])

            if identified:
                vlm_people = [p for p in record.people
                              if not any(p.lower().startswith(w)
                                         for w in ("person", "man", "woman", "someone"))]
                record.people = identified + vlm_people
                log.info("[FACE_ID] People updated: %s", record.people)

            if self._passive_face_callback and results:
                try:
                    img_size = None
                    try:
                        import cv2
                        import numpy as np
                        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            img_size = (img.shape[1], img.shape[0])
                    except Exception:
                        pass
                    self._passive_face_callback(results, img_size)
                except Exception:
                    log.exception("[FACE_ID] passive callback failed")

        except Exception:
            log.exception("[FACE_ID] Enrichment failed")

    def get_scene_record(self) -> SceneRecord | None:
        """Get the current scene record, or None if stale/unavailable."""
        if not self._enabled or self._current is None:
            return None

        age = time.monotonic() - self._last_update
        if age > config.VISION_STALE_THRESHOLD:
            log.debug("Scene record stale (%.0fs old)", age)
            return None

        return self._current

    def get_description(self) -> str | None:
        """Get a filtered scene description for prompt injection.

        This is the interface used by prompt_builder. Returns the relevance-
        filtered summary rather than the raw VLM output. Returns None if
        the scene isn't relevant enough to inject.
        """
        record = self.get_scene_record()
        if record is None:
            return None

        # Use relevance classifier output
        if self._last_relevance and self._last_relevance.should_inject:
            summary = self._last_relevance.filtered_summary
            if summary:
                return summary

        # Fallback: if no relevance result yet, use raw summary
        # (first frame before history builds up)
        if self._last_relevance is None:
            return record.summary()

        # Relevance says don't inject
        return None

    def get_history(self, n: int = 5) -> list[SceneRecord]:
        """Get the last N scene records for temporal context."""
        return list(self._history)[-n:]

    def get_vision_debug(self) -> dict:
        """Get vision state for the OS dashboard.

        Returns dict with scene record, last JPEG (base64), change score,
        relevance scores, stats.
        """
        import base64
        result = {
            "enabled": self._enabled,
            "has_frame": self._last_jpeg is not None,
            "frame_b64": None,
            "record": None,
            "relevance": None,
            "age_s": None,
            "change_score": self._capture._last_change_score if self._enabled else 0.0,
            "stats": {
                "polled": self._capture._frames_polled if self._enabled else 0,
                "analyzed": self._capture._frames_analyzed if self._enabled else 0,
            },
        }
        if self._last_jpeg:
            result["frame_b64"] = base64.b64encode(self._last_jpeg).decode("ascii")
        # Face ID status
        if self._face_id_ready:
            result["face_id"] = {
                "enrolled": self._face_id.get_enrolled_names(),
                "ready": True,
            }

        if self._current:
            result["record"] = {
                "people": self._current.people,
                "objects": self._current.objects,
                "actions": self._current.actions,
                "scene_state": self._current.scene_state,
                "change_from_prior": self._current.change_from_prior,
                "novelty": self._current.novelty,
                "humor_potential": self._current.humor_potential,
                "store_as_memory": self._current.store_as_memory,
                "speak_now": self._current.speak_now,
                "memory_tags": self._current.memory_tags,
                "timestamp": self._current.timestamp,
            }
            result["age_s"] = round(time.monotonic() - self._last_update, 1)
        if self._last_relevance:
            result["relevance"] = {
                "overall": self._last_relevance.overall,
                "novelty_score": self._last_relevance.novelty_score,
                "persistence_score": self._last_relevance.persistence_score,
                "urgency_score": self._last_relevance.urgency_score,
                "should_inject": self._last_relevance.should_inject,
                "detail_level": self._last_relevance.detail_level,
                "filtered_summary": self._last_relevance.filtered_summary,
                "new_people": self._last_relevance.new_people,
                "new_actions": self._last_relevance.new_actions,
            }
        return result
