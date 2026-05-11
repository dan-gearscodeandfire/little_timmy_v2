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
from vision.capture import FrameCapture, LOW_RES
from vision.analyzer import analyze_frame, check_model_available, SceneRecord
from vision.relevance import classify, RelevanceResult, INJECT_THRESHOLD
from vision.face_remote import RemoteFaceClient

log = logging.getLogger(__name__)


class VisionContext:
    """Manages the current visual scene for prompt injection."""

    def __init__(self):
        self._capture = FrameCapture()
        self._current: SceneRecord | None = None
        self._last_update: float = 0.0
        self._lock = asyncio.Lock()
        self._enabled = False
        self._face_remote = RemoteFaceClient()
        self._face_remote_ready = False
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

        # Verify remote face state is reachable on streamerpi
        data = await self._face_remote.fetch()
        if data is not None:
            self._face_remote_ready = True
            log.info("Remote face state reachable (%s)", self._face_remote.url)
        else:
            log.warning("Remote /faces not reachable at %s -- face id disabled", self._face_remote.url)

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

    def pause_polling(self):
        """Pause the periodic VLM poll loop (delegates to FrameCapture)."""
        self._capture.pause()

    def resume_polling(self):
        """Resume the periodic VLM poll loop."""
        self._capture.resume()

    async def _on_frame(self, jpeg_bytes: bytes):
        """Callback from FrameCapture -- analyze the frame and update record."""
        record = await analyze_frame(jpeg_bytes)
        if record:
            # Enrich with remote face state from streamerpi
            if self._face_remote_ready:
                await self._enrich_with_face_id(record)

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
        """Trigger an immediate capture + analysis. Returns the new record.

        2026-05-08: collapsed to single LOW_RES (320x180) because OCR is
        not a target use case. Visual questions still force a fresh
        capture; they just don't escalate resolution.
        """
        if not self._enabled:
            return None
        jpeg = await self._capture.trigger(reason, LOW_RES)
        if jpeg:
            record = await analyze_frame(jpeg)
            if record:
                # Enrich with remote face state from streamerpi
                if self._face_remote_ready:
                    await self._enrich_with_face_id(record)

                async with self._lock:
                    history_list = list(self._history)
                    relevance = classify(record, history_list)

                    self._current = record
                    self._last_update = time.monotonic()
                    self._history.append(record)
                    self._last_relevance = relevance
                return record
        return self._current if self._current else None

    async def _enrich_with_face_id(self, record):
        """Fetch remote face state and update record.people with real names.

        Calls streamerpi /faces instead of running local YuNet+SFace.
        """
        try:
            data = await self._face_remote.fetch_full()
            if data is None:
                return
            results = data["faces"]
            img_size = data["image_size"] if data["image_size"][0] > 0 else None
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
        # Face ID status (data lives on streamerpi now)
        if self._face_remote_ready:
            result["face_id"] = {
                "enrolled": [],  # enrolled names not currently exposed by /faces
                "ready": True,
                "remote_url": self._face_remote.url,
            }

        if self._current:
            result["record"] = {
                "people": self._current.people,
                "objects": self._current.objects,
                "actions": self._current.actions,
                "scene_state": self._current.scene_state,
                "change_from_prior": self._current.change_from_prior,
                "novelty": self._current.novelty,
                "speak_now": self._current.speak_now,
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
