"""Speaker identification using Resemblyzer embeddings.

Identifies known speakers (Dan, Timmy) and tracks unknown voices.
When an unknown voice becomes stable (enough utterances), signals
the orchestrator to ask for their name.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

log = logging.getLogger(__name__)

VOICEPRINT_DIR = Path(os.path.expanduser("~/little_timmy/models/speaker"))

# Thresholds (cosine distance: 0 = identical, 2 = opposite)
KNOWN_SPEAKER_THRESHOLD = 0.30    # below this = confident match to known speaker
SAME_UNKNOWN_THRESHOLD = 0.30     # below this = same unknown speaker as before
STABLE_UTTERANCE_COUNT = 3        # utterances needed before asking for name


@dataclass
class KnownSpeaker:
    """A permanently enrolled speaker."""
    speaker_id: int
    name: str
    embedding: np.ndarray
    persistent: bool = True  # loaded from disk every session


@dataclass
class UnknownSpeaker:
    """A tracked but unnamed speaker within a session."""
    temp_id: str              # e.g. "unknown_1"
    embeddings: list = field(default_factory=list)
    avg_embedding: np.ndarray | None = None
    utterance_count: int = 0
    last_text: str = ""       # last thing they said (for name solicitation)
    name_asked: bool = False  # have we already asked their name?
    name: str | None = None   # assigned name once told


@dataclass
class SpeakerResult:
    """Result of identifying a speaker."""
    speaker_id: int | None     # DB speaker ID (if known)
    name: str                  # "dan", "timmy", "unknown_1", or assigned name
    is_timmy: bool             # shortcut flag
    is_new: bool               # first time seeing this voice
    should_ask_name: bool      # True when unknown becomes stable
    confidence: float          # 1 - cosine_distance to best match


class SpeakerIdentifier:
    def __init__(self):
        self._encoder = None
        self._known_speakers: list[KnownSpeaker] = []
        self._unknown_speakers: list[UnknownSpeaker] = []
        self._unknown_counter = 0

    def _load_encoder(self):
        """Lazy-load Resemblyzer encoder."""
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder("cpu")
            log.info("Resemblyzer encoder loaded")
        return self._encoder

    def load_voiceprints(self):
        """Load known speaker voiceprints from disk."""
        self._load_encoder()

        voiceprints = {
            "dan": (1, VOICEPRINT_DIR / "dan_resemblyzer.npy"),
            "timmy": (2, VOICEPRINT_DIR / "timmy_resemblyzer.npy"),
        }

        for name, (speaker_id, path) in voiceprints.items():
            if path.exists():
                emb = np.load(path)
                self._known_speakers.append(KnownSpeaker(
                    speaker_id=speaker_id,
                    name=name,
                    embedding=emb,
                ))
                log.info("Loaded voiceprint: %s (speaker_id=%d)", name, speaker_id)
            else:
                log.warning("Voiceprint not found: %s", path)

    def extract_embedding(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from 16kHz float32 audio."""
        from resemblyzer import preprocess_wav
        encoder = self._load_encoder()
        processed = preprocess_wav(audio_16k, source_sr=16000)
        if len(processed) < 8000:  # <0.5s of speech after trimming
            processed = audio_16k  # fall back to raw if too short
        emb = encoder.embed_utterance(processed)
        return emb

    def identify(self, audio_16k: np.ndarray, transcribed_text: str = "") -> SpeakerResult:
        """Identify who is speaking from an audio segment.

        Returns SpeakerResult with identification info.
        """
        t0 = time.time()
        emb = self.extract_embedding(audio_16k)
        extract_ms = (time.time() - t0) * 1000

        # Compare against known speakers
        best_known = None
        best_known_dist = float("inf")
        for ks in self._known_speakers:
            dist = cosine(emb, ks.embedding)
            if dist < best_known_dist:
                best_known_dist = dist
                best_known = ks

        # Always log distances for debugging
        log.info("Speaker distances: best=%s dist=%.4f threshold=%.2f audio_len=%d (%dms)",
                 best_known.name if best_known else "none", best_known_dist,
                 KNOWN_SPEAKER_THRESHOLD, len(audio_16k), extract_ms)

        # Is it a known speaker?
        if best_known and best_known_dist < KNOWN_SPEAKER_THRESHOLD:
            log.info("Speaker identified: %s (dist=%.3f, %dms)",
                     best_known.name, best_known_dist, extract_ms)
            return SpeakerResult(
                speaker_id=best_known.speaker_id,
                name=best_known.name,
                is_timmy=(best_known.name == "timmy"),
                is_new=False,
                should_ask_name=False,
                confidence=1 - best_known_dist,
            )

        # Not a known speaker — check against tracked unknowns
        best_unknown = None
        best_unknown_dist = float("inf")
        for us in self._unknown_speakers:
            if us.avg_embedding is not None:
                dist = cosine(emb, us.avg_embedding)
                if dist < best_unknown_dist:
                    best_unknown_dist = dist
                    best_unknown = us

        if best_unknown and best_unknown_dist < SAME_UNKNOWN_THRESHOLD:
            # Same unknown speaker as before — accumulate
            best_unknown.embeddings.append(emb)
            best_unknown.utterance_count += 1
            best_unknown.avg_embedding = np.mean(best_unknown.embeddings, axis=0)
            best_unknown.avg_embedding /= np.linalg.norm(best_unknown.avg_embedding)
            best_unknown.last_text = transcribed_text

            should_ask = (
                best_unknown.utterance_count >= STABLE_UTTERANCE_COUNT
                and not best_unknown.name_asked
                and best_unknown.name is None
            )

            name = best_unknown.name or best_unknown.temp_id
            log.info("Unknown speaker %s (utterance #%d, dist=%.3f, %dms)",
                     name, best_unknown.utterance_count, best_unknown_dist, extract_ms)

            return SpeakerResult(
                speaker_id=None,
                name=name,
                is_timmy=False,
                is_new=False,
                should_ask_name=should_ask,
                confidence=1 - best_unknown_dist,
            )

        # Brand new unknown speaker
        self._unknown_counter += 1
        temp_id = f"unknown_{self._unknown_counter}"
        new_unknown = UnknownSpeaker(
            temp_id=temp_id,
            embeddings=[emb],
            avg_embedding=emb / np.linalg.norm(emb),
            utterance_count=1,
            last_text=transcribed_text,
        )
        self._unknown_speakers.append(new_unknown)

        log.info("New unknown speaker: %s (known_best_dist=%.3f, %dms)",
                 temp_id, best_known_dist, extract_ms)

        return SpeakerResult(
            speaker_id=None,
            name=temp_id,
            is_timmy=False,
            is_new=True,
            should_ask_name=False,
            confidence=0.0,
        )

    def assign_name(self, temp_id: str, name: str) -> bool:
        """Assign a name to an unknown speaker after they tell us."""
        for us in self._unknown_speakers:
            if us.temp_id == temp_id or us.name == temp_id:
                us.name = name.strip().lower()
                us.name_asked = True
                log.info("Assigned name '%s' to %s", us.name, us.temp_id)
                return True
        return False

    def mark_name_asked(self, temp_id: str):
        """Mark that we've asked this unknown for their name (avoid re-asking)."""
        for us in self._unknown_speakers:
            if us.temp_id == temp_id:
                us.name_asked = True
                return

    def get_unknown_for_name_ask(self, temp_id: str) -> UnknownSpeaker | None:
        """Get unknown speaker info for name solicitation prompt."""
        for us in self._unknown_speakers:
            if us.temp_id == temp_id:
                return us
        return None

    def get_all_session_speakers(self) -> list[str]:
        """Get names of all identified speakers this session."""
        names = [ks.name for ks in self._known_speakers]
        for us in self._unknown_speakers:
            if us.name:
                names.append(us.name)
            else:
                names.append(us.temp_id)
        return names


    def set_last_text(self, name: str, text: str):
        """Set the last transcribed text for an unknown speaker (for name solicitation)."""
        for us in self._unknown_speakers:
            if us.temp_id == name or us.name == name:
                us.last_text = text
                return
    def undo_last_observation(self, name: str):
        """Roll back the last observation for an unknown speaker (e.g., noise/empty STT)."""
        for us in self._unknown_speakers:
            if us.temp_id == name or us.name == name:
                if us.utterance_count <= 1:
                    self._unknown_speakers.remove(us)
                    log.debug("Removed phantom unknown: %s", name)
                else:
                    us.utterance_count -= 1
                    if us.embeddings:
                        us.embeddings.pop()
                        us.avg_embedding = np.mean(us.embeddings, axis=0)
                        us.avg_embedding /= np.linalg.norm(us.avg_embedding)
                    log.debug("Rolled back observation for %s (now %d)", name, us.utterance_count)
                return
