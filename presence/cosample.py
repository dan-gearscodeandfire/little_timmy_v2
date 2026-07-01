"""Passive co-sample buffer for dual-modality enrollment (Phase B).

Holds recent SOLE-in-frame face crops keyed by the turn's speaker (temp_id for
an unknown voice, or the resolved name), so a later "enroll me" commit can bind
the face that has been talking WITHOUT a separate face-capture dialog. Voice
embeddings are not buffered here — they already live on the tracked
``UnknownSpeaker.embeddings`` and are pulled at commit.

Crops are only ever added when the recognizer reports exactly one detected face
(``FaceObservation.detected_face_count == 1``), i.e. the sole-face == speaker
rule already holds, so a buffered crop is unambiguously the speaker. Embedding
is deferred to commit time (never per 2 Hz tick — that is the saturation vector
called out in the plan); this buffer only holds raw crops, capped.
"""

from __future__ import annotations

import logging
from collections import deque

log = logging.getLogger(__name__)


class CoSampleBuffer:
    def __init__(self, max_crops: int = 12, max_per_speaker: int = 8):
        # Global ring so total memory is bounded regardless of speaker churn.
        self._buf: deque = deque(maxlen=max_crops)  # (speaker_key, crop)
        self.max_per_speaker = max_per_speaker

    def add(self, speaker_key: str, crops) -> None:
        """Buffer sole-face crops for ``speaker_key`` (a temp_id or name).

        No-op when there is no key or no crops. Silently caps per-speaker so a
        stationary subject can't crowd out everyone else in the ring."""
        if not speaker_key or not crops:
            return
        have = sum(1 for k, _ in self._buf if k == speaker_key)
        for c in crops:
            if c is None:
                continue
            if have >= self.max_per_speaker:
                break
            self._buf.append((speaker_key, c))
            have += 1

    def crops_for(self, speaker_key: str) -> list:
        """Buffered crops for one speaker (most-recent last)."""
        return [c for k, c in self._buf if k == speaker_key]

    def clear_speaker(self, speaker_key: str) -> None:
        """Drop a speaker's crops (call after a successful commit)."""
        kept = [(k, c) for k, c in self._buf if k != speaker_key]
        self._buf.clear()
        self._buf.extend(kept)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
