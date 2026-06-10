"""Speaker identification using Resemblyzer embeddings.

Identifies known speakers (Dan, Timmy, plus any other ``*_resemblyzer.npy``
in VOICEPRINT_DIR) and tracks unknown voices. When an unknown voice
becomes stable (enough utterances), signals the orchestrator to ask for
their name. Three live-enrollment triggers persist voiceprints during
conversation:

  Trigger 1 (auto-persist on naming): when ``assign_name(temp_id, name)``
    fires from the existing in-conversation flow, the unknown speaker's
    avg embedding is written to ``models/speaker/<name>_resemblyzer.npy``
    and the speaker is promoted to a KnownSpeaker for the rest of the
    session. Survives restart.

  Trigger 2 (voice-command re-enrollment): start_reenrollment(name, dur)
    opens a collection window. Every confident match for that speaker
    during the window contributes its embedding. On expiry, the new
    samples are blended 50/50 with the existing voiceprint and persisted.

  Trigger 3 (continuous drift correction): off by default. Enabled via
    ``config.SPEAKER_DRIFT_LEARNING = True``. Tight matches (dist <
    TIGHT_DRIFT_THRESHOLD) feed a per-speaker rolling buffer. Every
    DRIFT_BATCH_SIZE samples, the buffer is folded into the voiceprint
    via EMA and persisted.
"""

import logging
import os
import re
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

# Short-audio continuity (see top-of-file docstring on identify()).
SHORT_AUDIO_SAMPLES = 80000        # ~5s @ 16kHz; below this is "short"
SHORT_AUDIO_DIST_CAP = 0.55        # max dist tolerated for short-audio continuity
CONTINUITY_WINDOW_SEC = 60.0       # last confident match must be this recent

# Trigger 3 (drift learning) constants.
TIGHT_DRIFT_THRESHOLD = 0.20       # only matches under this contribute to drift
DRIFT_BATCH_SIZE = 30              # samples per drift update
DRIFT_NEW_WEIGHT = 0.30            # EMA weight for new samples vs existing voiceprint

# K-prototype matching. An identity is a SET of prototype embeddings (shape
# (K, D)); a live utterance matches on the MINIMUM cosine distance across them.
# This mirrors the SFace face-ID fix (26bab94): one averaged prototype can't
# span the distance/loudness/pose variation of a real room, but min-over-K
# recognizes any covered look. Old single-vector .npy files load as (1, D).
PROTOTYPE_DEDUP_DIST = 0.05        # a new prototype within this of an existing one is a dup → skip
MAX_PROTOTYPES = 12               # cap per identity (keep most recent beyond this)

# Online (in-conversation) voiceprint learning master switch. When False, the
# three live-enrollment triggers (T1 auto-persist on naming, T2 voice-command
# re-enroll, T3 drift) do NOT write prototypes to disk or mutate an identity's
# prototype set — deliberate enrollment via enroll_prototypes.py is the only way
# voiceprints change. Kept False for the party so short/noisy utterances can't
# pollute identities mid-event. (T3 also independently gated by
# config.SPEAKER_DRIFT_LEARNING.)
ONLINE_LEARNING_ENABLED = False

# Minimum collected samples before T1 will auto-persist a freshly-named speaker
# (guards against enrolling a one-word filler like "Glad").
MIN_ENROLL_SAMPLES = 3

# Names we refuse to persist as voiceprints (reserved or easy footguns).
RESERVED_NAMES = {
    "timmy", "system", "unknown", "the", "a", "an",
    "you", "me", "i", "user", "assistant",
}

# Valid name pattern for live enrollment: alpha + underscore/hyphen, 2-32 chars.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


def _build_prototypes(embeddings: list) -> np.ndarray:
    """Turn a list of raw embeddings into a deduped, capped (K, D) prototype set.

    Each embedding is L2-normalized; one is dropped if it sits within
    PROTOTYPE_DEDUP_DIST cosine of an already-kept prototype (near-duplicate
    pose/distance). Capped at MAX_PROTOTYPES. Falls back to the mean if every
    sample deduped away.
    """
    kept: list[np.ndarray] = []
    for e in embeddings:
        e = e / np.linalg.norm(e)
        if any(cosine(e, k) < PROTOTYPE_DEDUP_DIST for k in kept):
            continue
        kept.append(e)
        if len(kept) >= MAX_PROTOTYPES:
            break
    if not kept:
        m = np.mean(embeddings, axis=0)
        m /= np.linalg.norm(m)
        kept = [m]
    return np.vstack(kept)


@dataclass
class KnownSpeaker:
    """A permanently enrolled speaker.

    ``prototypes`` is a 2-D array of shape (K, D) holding K L2-normalized
    embeddings that span this speaker's recorded variation. A live utterance
    is matched on the MINIMUM cosine distance across the set.
    """
    speaker_id: int
    name: str
    prototypes: np.ndarray   # shape (K, D), L2-normalized rows
    persistent: bool = True  # loaded from disk every session

    def distance(self, emb: np.ndarray) -> float:
        """Minimum cosine distance from ``emb`` to any of this speaker's prototypes."""
        return float(min(cosine(emb, p) for p in self.prototypes))


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
    # Reserved DB ids used by the orchestrator/router by name. Any other
    # *_resemblyzer.npy in VOICEPRINT_DIR is auto-loaded with an id from
    # _NEXT_ID onward.
    _RESERVED_IDS = {"dan": 1, "timmy": 2}
    _NEXT_ID = 3

    def __init__(self):
        self._encoder = None
        self._known_speakers: list[KnownSpeaker] = []
        self._unknown_speakers: list[UnknownSpeaker] = []
        self._unknown_counter = 0
        # Short-audio continuity state.
        self._last_known_speaker: KnownSpeaker | None = None
        self._last_known_seen_ts: float = 0.0
        # Trigger 2 — at most one active re-enrollment at a time.
        # {"name": str, "started_ts": float, "expiry_ts": float, "embeddings": [np.ndarray]}.
        self._active_reenrollment: dict | None = None
        # Trigger 3 — per-speaker rolling buffers.
        self._drift_buffers: dict[str, list[np.ndarray]] = {}

    def _load_encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder("cpu")
            log.info("Resemblyzer encoder loaded")
        return self._encoder

    _ID_MAP_FILENAME = "_id_map.json"

    def _read_id_map(self) -> dict:
        """Read the persisted name -> speaker_id map, or {} if missing/corrupt."""
        path = VOICEPRINT_DIR / self._ID_MAP_FILENAME
        if not path.exists():
            return {}
        try:
            import json as _json
            with open(path) as f:
                return {k.lower(): int(v) for k, v in _json.load(f).items()}
        except Exception as e:
            log.warning("Failed to read %s: %s; treating as empty", path.name, e)
            return {}

    def _write_id_map(self, mapping: dict) -> None:
        """Persist the name -> speaker_id map atomically."""
        import json as _json
        path = VOICEPRINT_DIR / self._ID_MAP_FILENAME
        tmp = path.with_suffix(".json.tmp")
        VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            _json.dump(mapping, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def load_voiceprints(self):
        """Load every ``*_resemblyzer.npy`` in VOICEPRINT_DIR.

        File ``<name>_resemblyzer.npy`` becomes a KnownSpeaker named
        ``<name>``. Speaker ids are pulled from a persisted name -> id
        map at ``VOICEPRINT_DIR/_id_map.json`` so they survive both
        restarts and the addition of new voiceprints. Reserved names
        (dan=1, timmy=2) keep their fixed ids regardless of the map;
        the map's ``_next_id`` field tracks the next free id.

        Backup files (``*.bak``, ``*.disabled``, ``*.pre_*``) are skipped
        because the glob requires the path to end in ``.npy`` exactly.
        """
        self._load_encoder()

        id_map = self._read_id_map()
        next_id = max(int(id_map.get("_next_id", self._NEXT_ID)), self._NEXT_ID)
        # Seed reserved entries so the map stays consistent even if the file
        # is wiped/recreated.
        for name, sid in self._RESERVED_IDS.items():
            id_map[name] = sid
        dirty = False

        paths = sorted(VOICEPRINT_DIR.glob("*_resemblyzer.npy"))
        for path in paths:
            name = path.stem.replace("_resemblyzer", "").lower()
            if name in self._RESERVED_IDS:
                speaker_id = self._RESERVED_IDS[name]
            elif name in id_map and isinstance(id_map[name], int):
                speaker_id = id_map[name]
            else:
                speaker_id = next_id
                next_id += 1
                id_map[name] = speaker_id
                dirty = True
                log.info("Allocated new speaker_id=%d for %s", speaker_id, name)
            try:
                protos = np.load(path)
            except Exception as e:
                log.warning("Failed to load voiceprint %s: %s", path.name, e)
                continue
            if protos.ndim == 1:           # legacy single-vector file → (1, D)
                protos = protos[None, :]
            self._known_speakers.append(KnownSpeaker(
                speaker_id=speaker_id,
                name=name,
                prototypes=protos,
            ))
            log.info("Loaded voiceprint: %s (speaker_id=%d, %d prototype(s)) from %s",
                     name, speaker_id, protos.shape[0], path.name)

        if id_map.get("_next_id") != next_id:
            id_map["_next_id"] = next_id
            dirty = True
        if dirty:
            try:
                self._write_id_map(id_map)
                log.info("Updated id-map: %s", {k: v for k, v in id_map.items() if k != "_next_id"})
            except Exception as e:
                log.warning("Failed to persist id-map: %s", e)

        loaded = {ks.name for ks in self._known_speakers}
        for reserved in self._RESERVED_IDS:
            if reserved not in loaded:
                log.warning("Reserved voiceprint not found: %s_resemblyzer.npy",
                            reserved)
        log.info("Voiceprint load complete: %d enrolled (%s)",
                 len(self._known_speakers),
                 ", ".join(ks.name for ks in self._known_speakers) or "none")

    def extract_embedding(self, audio_16k: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav
        encoder = self._load_encoder()
        processed = preprocess_wav(audio_16k, source_sr=16000)
        if len(processed) < 8000:  # <0.5s of speech after trimming
            processed = audio_16k
        emb = encoder.embed_utterance(processed)
        return emb

    # ---------- Persistence primitive (shared by all three triggers) ----------

    def persist_voiceprint(self, name: str, prototypes: np.ndarray,
                           *, backup: bool = True) -> Path:
        """Atomic write of a prototype set to ``VOICEPRINT_DIR/<name>_resemblyzer.npy``.

        Accepts a single embedding (shape (D,)) or a prototype set (shape
        (K, D)); always stored on disk as 2-D so older callers stay valid.
        Refuses reserved/empty/malformed names. Backs up any existing file
        as ``.bak.<unix_ts>`` if backup=True.
        """
        clean = (name or "").strip().lower()
        if clean in RESERVED_NAMES or not _NAME_RE.match(clean):
            raise ValueError(f"refusing to persist voiceprint as {clean!r}")
        if not isinstance(prototypes, np.ndarray) or prototypes.ndim not in (1, 2):
            raise ValueError("prototypes must be a 1-D or 2-D ndarray")
        if prototypes.ndim == 1:
            prototypes = prototypes[None, :]
        VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
        out = VOICEPRINT_DIR / f"{clean}_resemblyzer.npy"
        if backup and out.exists():
            bak = VOICEPRINT_DIR / f"{clean}_resemblyzer.npy.bak.{int(time.time())}"
            out.rename(bak)
            log.info("Backed up existing voiceprint: %s -> %s", out.name, bak.name)
        np.save(out, prototypes.astype(np.float32))
        log.info("Persisted voiceprint: %s (%d prototype(s), %d-dim)",
                 out, prototypes.shape[0], prototypes.shape[1])
        return out

    def _next_known_id(self) -> int:
        used = {ks.speaker_id for ks in self._known_speakers}
        candidate = self._NEXT_ID
        while candidate in used:
            candidate += 1
        return candidate

    # ---------- identify() ----------

    def identify(self, audio_16k: np.ndarray, transcribed_text: str = "") -> SpeakerResult:
        # First, finalize any expired re-enrollment window from a prior turn.
        self._maybe_finalize_reenrollment()

        t0 = time.time()
        emb = self.extract_embedding(audio_16k)
        extract_ms = (time.time() - t0) * 1000

        best_known = None
        best_known_dist = float("inf")
        for ks in self._known_speakers:
            dist = ks.distance(emb)   # min cosine across this speaker's prototypes
            if dist < best_known_dist:
                best_known_dist = dist
                best_known = ks

        log.info("Speaker distances: best=%s dist=%.4f threshold=%.2f audio_len=%d (%dms)",
                 best_known.name if best_known else "none", best_known_dist,
                 KNOWN_SPEAKER_THRESHOLD, len(audio_16k), extract_ms)

        if best_known and best_known_dist < KNOWN_SPEAKER_THRESHOLD:
            # Confident match — refresh continuity anchor + feed both
            # collection mechanisms (Trigger 2 if active for this speaker;
            # Trigger 3 only on tight match).
            self._last_known_speaker = best_known
            self._last_known_seen_ts = time.time()
            self._record_for_reenrollment(best_known.name, emb)
            if best_known_dist < TIGHT_DRIFT_THRESHOLD:
                self._record_for_drift(best_known.name, emb)
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

        # Short-audio continuity fallback.
        audio_len = len(audio_16k)
        if (audio_len < SHORT_AUDIO_SAMPLES
                and best_known is not None
                and self._last_known_speaker is not None
                and best_known.name == self._last_known_speaker.name
                and best_known.name != "timmy"
                and best_known_dist < SHORT_AUDIO_DIST_CAP
                and (time.time() - self._last_known_seen_ts) < CONTINUITY_WINDOW_SEC):
            elapsed = time.time() - self._last_known_seen_ts
            self._last_known_seen_ts = time.time()
            log.info("Speaker continuity applied: %s (dist=%.3f cap=%.2f, audio_len=%d short, last_seen %.0fs ago)",
                     best_known.name, best_known_dist, SHORT_AUDIO_DIST_CAP,
                     audio_len, elapsed)
            # Continuity-applied matches are explicitly excluded from drift
            # learning (T3) and re-enrollment (T2): they're borderline and
            # could pollute the voiceprint.
            return SpeakerResult(
                speaker_id=best_known.speaker_id,
                name=best_known.name,
                is_timmy=False,
                is_new=False,
                should_ask_name=False,
                confidence=1 - best_known_dist,
            )

        # Compare against tracked unknowns.
        best_unknown = None
        best_unknown_dist = float("inf")
        for us in self._unknown_speakers:
            if us.avg_embedding is not None:
                dist = cosine(emb, us.avg_embedding)
                if dist < best_unknown_dist:
                    best_unknown_dist = dist
                    best_unknown = us

        if best_unknown and best_unknown_dist < SAME_UNKNOWN_THRESHOLD:
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

    # ---------- Trigger 1 — auto-persist on name assignment ----------

    def assign_name(self, temp_id: str, name: str) -> bool:
        """Attach a name to an unknown speaker, persist voiceprint to disk,
        and promote them to a KnownSpeaker for the rest of the session.

        Returns True on success, False if the unknown is missing or the
        proposed name is invalid/reserved/already-taken.
        """
        clean = (name or "").strip().lower()
        if clean in RESERVED_NAMES or not _NAME_RE.match(clean):
            log.warning("Refusing to assign invalid name %r", clean)
            return False
        if any(ks.name == clean for ks in self._known_speakers):
            log.warning("Refusing to assign name %r: already a known speaker", clean)
            return False

        for us in self._unknown_speakers:
            if us.temp_id == temp_id or us.name == temp_id:
                us.name = clean
                us.name_asked = True
                log.info("Assigned name '%s' to %s", us.name, us.temp_id)

                # Trigger 1: promote in-memory ALWAYS (so the speaker is
                # recognized for the rest of the session), but only write to
                # disk when online learning is enabled AND we have enough
                # samples to trust — guards against persisting a one-word
                # filler. The unknown's collected embeddings become the
                # initial prototype set.
                if us.avg_embedding is not None and len(us.embeddings) >= 1:
                    protos = _build_prototypes(us.embeddings)
                    new_id = self._next_known_id()
                    self._known_speakers.append(KnownSpeaker(
                        speaker_id=new_id,
                        name=clean,
                        prototypes=protos,
                    ))
                    self._unknown_speakers.remove(us)
                    log.info("[T1] Promoted %s (was %s) to known speaker_id=%d "
                             "with %d prototype(s) from %d samples",
                             clean, us.temp_id, new_id, protos.shape[0], len(us.embeddings))
                    if ONLINE_LEARNING_ENABLED and len(us.embeddings) >= MIN_ENROLL_SAMPLES:
                        try:
                            self.persist_voiceprint(clean, protos)
                        except Exception as e:
                            log.warning("[T1] persist failed for %s: %s", clean, e)
                    else:
                        log.info("[T1] persist skipped for %s (online_learning=%s, "
                                 "samples=%d) — session-only",
                                 clean, ONLINE_LEARNING_ENABLED, len(us.embeddings))
                else:
                    log.warning("[T1] cannot promote %s: no avg_embedding yet", clean)
                return True
        return False

    # ---------- Trigger 2 — voice-command re-enrollment ----------

    def start_reenrollment(self, name: str, duration_s: float = 60.0) -> bool:
        """Open a collection window for an already-known speaker. Every
        confident match for that speaker during the window contributes its
        embedding. On expiry, samples are blended 50/50 with the existing
        voiceprint and persisted.

        Returns True if the window opened, False if the name isn't a known
        speaker or another re-enrollment is already in progress.
        """
        clean = (name or "").strip().lower()
        if not any(ks.name == clean for ks in self._known_speakers):
            log.warning("[T2] cannot re-enroll %r: not a known speaker", clean)
            return False
        if self._active_reenrollment is not None:
            log.warning("[T2] re-enrollment already active for %r",
                        self._active_reenrollment["name"])
            return False
        now = time.time()
        self._active_reenrollment = {
            "name": clean,
            "started_ts": now,
            "expiry_ts": now + duration_s,
            "embeddings": [],
        }
        log.info("[T2] re-enrollment opened for %s (window=%.0fs)", clean, duration_s)
        return True

    def get_active_reenrollment(self) -> dict | None:
        """Snapshot of the active re-enrollment, or None. Safe to expose via API."""
        if not self._active_reenrollment:
            return None
        a = self._active_reenrollment
        now = time.time()
        return {
            "name": a["name"],
            "started_ts": a["started_ts"],
            "expiry_ts": a["expiry_ts"],
            "remaining_s": max(0.0, a["expiry_ts"] - now),
            "samples_collected": len(a["embeddings"]),
        }

    def _record_for_reenrollment(self, name: str, emb: np.ndarray) -> None:
        a = self._active_reenrollment
        if not a:
            return
        if a["name"] != name:
            return
        if time.time() >= a["expiry_ts"]:
            return  # finalize on next identify() call
        a["embeddings"].append(emb)

    def _maybe_finalize_reenrollment(self) -> None:
        a = self._active_reenrollment
        if not a:
            return
        if time.time() < a["expiry_ts"]:
            return
        name = a["name"]
        new_embs = a["embeddings"]
        self._active_reenrollment = None
        if not ONLINE_LEARNING_ENABLED:
            log.info("[T2] re-enrollment for %s discarded (online_learning disabled)", name)
            return
        if not new_embs:
            log.warning("[T2] re-enrollment for %s expired with 0 samples", name)
            return
        target = next((ks for ks in self._known_speakers if ks.name == name), None)
        if target is None:
            log.warning("[T2] target %s no longer known at finalize", name)
            return
        added = self._add_prototype_set(target, new_embs, persist=True)
        log.info("[T2] re-enrollment finalized for %s: %d/%d new prototype(s) added (now %d)",
                 name, added, len(new_embs), target.prototypes.shape[0])

    def _add_prototype_set(self, ks: KnownSpeaker, embeddings: list,
                           *, persist: bool) -> int:
        """Append new embeddings to ``ks.prototypes`` as additional prototypes,
        deduping against existing ones (within PROTOTYPE_DEDUP_DIST) and capping
        at MAX_PROTOTYPES (keeping the most recent). Returns how many were added;
        persists the updated set to disk when ``persist`` is True."""
        protos = ks.prototypes
        added = 0
        for e in embeddings:
            e = e / np.linalg.norm(e)
            if any(cosine(e, p) < PROTOTYPE_DEDUP_DIST for p in protos):
                continue
            protos = np.vstack([protos, e[None, :]])
            added += 1
        if added == 0:
            return 0
        if protos.shape[0] > MAX_PROTOTYPES:
            protos = protos[-MAX_PROTOTYPES:]
        ks.prototypes = protos
        if persist:
            try:
                self.persist_voiceprint(ks.name, protos)
            except Exception as e:
                log.warning("prototype persist failed for %s: %s", ks.name, e)
        return added

    # ---------- Trigger 3 — continuous drift correction ----------

    def _is_drift_learning_enabled(self) -> bool:
        try:
            import config as _cfg
            return bool(getattr(_cfg, "SPEAKER_DRIFT_LEARNING", False))
        except Exception:
            return False

    def _record_for_drift(self, name: str, emb: np.ndarray) -> None:
        if not self._is_drift_learning_enabled():
            return
        buf = self._drift_buffers.setdefault(name, [])
        buf.append(emb)
        if len(buf) >= DRIFT_BATCH_SIZE:
            self._apply_drift(name)

    def _apply_drift(self, name: str) -> None:
        embs = self._drift_buffers.pop(name, [])
        if not embs:
            return
        if not ONLINE_LEARNING_ENABLED:
            return
        target = next((ks for ks in self._known_speakers if ks.name == name), None)
        if target is None:
            return
        added = self._add_prototype_set(target, embs, persist=True)
        log.info("[T3] drift learning applied for %s: %d/%d prototype(s) added (now %d)",
                 name, added, len(embs), target.prototypes.shape[0])

    # ---------- Existing helpers (unchanged) ----------

    def mark_name_asked(self, temp_id: str):
        for us in self._unknown_speakers:
            if us.temp_id == temp_id:
                us.name_asked = True
                return

    def get_unknown_for_name_ask(self, temp_id: str) -> UnknownSpeaker | None:
        for us in self._unknown_speakers:
            if us.temp_id == temp_id:
                return us
        return None

    def get_all_session_speakers(self) -> list[str]:
        names = [ks.name for ks in self._known_speakers]
        for us in self._unknown_speakers:
            if us.name:
                names.append(us.name)
            else:
                names.append(us.temp_id)
        return names

    def set_last_text(self, name: str, text: str):
        for us in self._unknown_speakers:
            if us.temp_id == name or us.name == name:
                us.last_text = text
                return

    def undo_last_observation(self, name: str):
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
