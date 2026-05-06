"""Track consecutive face_hint promotions for the same unknown voice.

When face recognition repeatedly identifies the same unknown speaker as a
known face for N consecutive turns, the voiceprint can be auto-enrolled
under that face name. This module tracks the streak; the actual
enrollment is performed by the caller (speaker_id_module.assign_name).

The streak resets when:
  - face_hint fires for a different name
  - the underlying unknown voice_temp_id changes (different unknown speaker)
  - a turn passes with no face_hint promotion (voice was confident, or face
    didn't match a known person)
  - auto-enrollment fires (one-shot — caller must enroll then we clear)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StreakState:
    face_hint_name: str  # canonical lowercase
    voice_temp_id: str
    count: int


class FaceHintStreak:
    """Tracks per-name consecutive face_hint promotions.

    Only one streak is alive at a time — observing a new (name, temp_id)
    pair clears any prior streak. Designed for the orchestrator to call
    once per turn with the fusion verdict outcome.
    """

    def __init__(self, threshold: int = 3):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._state: Optional[StreakState] = None

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def current(self) -> Optional[StreakState]:
        return self._state

    def observe(
        self,
        face_hint_name: Optional[str],
        voice_temp_id: Optional[str],
    ) -> Optional[StreakState]:
        """Record this turn's outcome. Returns the streak state IFF it just
        crossed the threshold (the caller should enroll and call reset()).

        face_hint_name=None means no promotion this turn — clears streak.
        voice_temp_id=None means we don't have a stable unknown ID to enroll
            against — also clears streak.
        """
        if not face_hint_name or not voice_temp_id:
            self._state = None
            return None

        canon = face_hint_name.strip().lower()
        if not canon:
            self._state = None
            return None

        if (
            self._state is None
            or self._state.face_hint_name != canon
            or self._state.voice_temp_id != voice_temp_id
        ):
            self._state = StreakState(
                face_hint_name=canon,
                voice_temp_id=voice_temp_id,
                count=1,
            )
        else:
            self._state.count += 1

        if self._state.count >= self._threshold:
            return self._state
        return None

    def reset(self) -> None:
        """Clear the streak. Call after a successful enrollment."""
        self._state = None
