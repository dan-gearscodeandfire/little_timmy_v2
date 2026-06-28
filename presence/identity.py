"""Identity canonicalization and voice+face fusion rule.

Pure logic, no I/O. Unit-testable in isolation.
"""

import difflib
from typing import Iterable, Optional

from .types import FaceObservation, FusionVerdict


_TRACKING_MODES = frozenset({"engage", "track"})


def canonicalize(name: Optional[str]) -> Optional[str]:
    """Lowercase + strip; returns None for None/empty."""
    if name is None:
        return None
    s = name.strip().lower()
    return s or None


# --- Near-homophone name de-duplication --------------------------------------
# Devon's FACE got enrolled as the homophone "devin" while her VOICE identity is
# "devon", splitting one person into two records (2026-06-24). canonicalize()
# only lowercases, so "devin" and "devon" stay distinct keys. resolve_alias()
# snaps a fuzzy/STT/face-derived name onto an ALREADY-KNOWN name when they are
# phonetic near-duplicates, so a new identity is never minted next to an existing
# one. Deliberately scoped to a SMALL known set (the few enrolled speakers /
# ledger records), where high precision is easy; this is NOT a general English
# homonym resolver. Pure stdlib (no phonetic dependency to install).

_SOUNDEX_MAP = {}
for _digit, _letters in (("1", "BFPV"), ("2", "CGJKQSXZ"), ("3", "DT"),
                         ("4", "L"), ("5", "MN"), ("6", "R")):
    for _L in _letters:
        _SOUNDEX_MAP[_L] = _digit


def soundex(name: str) -> str:
    """Classic Soundex code (e.g. 'devon' and 'devin' both -> 'D150').

    Vowels (and y) separate consonant codes; h/w do not. Returns '' for a name
    with no alphabetic characters.
    """
    letters = [c for c in (name or "").upper() if c.isalpha()]
    if not letters:
        return ""
    first = letters[0]
    code = first
    prev = _SOUNDEX_MAP.get(first, "0")
    for ch in letters[1:]:
        if ch in "HW":
            continue  # ignored, and does NOT reset the previous code
        digit = _SOUNDEX_MAP.get(ch, "0")
        if digit != "0" and digit != prev:
            code += digit
        prev = digit  # vowels (digit '0') reset prev, acting as a separator
    return (code + "000")[:4]


def resolve_alias(name: Optional[str], known_names: Iterable[str],
                  *, min_ratio: float = 0.6) -> Optional[str]:
    """Return the existing known name a fuzzy ``name`` should snap to, else None.

    A snap requires BOTH the same Soundex key AND a string-similarity ratio
    >= ``min_ratio``, so near-homophones (devon/devin) collapse while genuinely
    different names (dan/devon) never do. If ``name`` canonicalizes to a name
    already in ``known_names`` it returns that name unchanged (a no-op snap).
    Names beginning with ``unknown`` and empty names never match. Inputs are
    canonicalized, so casing/whitespace don't matter.
    """
    canon = canonicalize(name)
    if not canon or canon.startswith("unknown"):
        return None
    known = {c for c in (canonicalize(k) for k in known_names)
             if c and not c.startswith("unknown")}
    if canon in known:
        return canon
    key = soundex(canon)
    if not key:
        return None
    best, best_ratio = None, 0.0
    for other in known:
        if soundex(other) != key:
            continue
        ratio = difflib.SequenceMatcher(None, canon, other).ratio()
        if ratio >= min_ratio and ratio > best_ratio:
            best, best_ratio = other, ratio
    return best


# Slice B: a confident voice (1 - distance) at or above this stabilizes an
# absent/low-conf face for the presence/proactive path. Mirrors the matcher's
# 0.30-distance known threshold (= 0.70 confidence).
VOICE_STABILIZE_CONF_FLOOR = 0.70

# Face band cutoffs, mirrored from streamerpi (camera.py:737, FACE_ID_MATCH_
# THRESHOLD=0.45). confidence = 1 - cosine_distance (lossless), so:
#   "high"   = dist < 0.30  -> conf >= 0.70
#   "medium" = 0.30-0.45    -> conf 0.55-0.70
# ATTRIBUTION (set speaker_name for the turn; ephemeral, reversible) trusts
# streamerpi's full match decision: high OR medium. The STREAK (binds a
# voiceprint that sticks for the session) trusts only a high match, or a medium
# that the stabilizer is confidently HOLDING (sticky) -- which excludes
# fresh-uncertain mediums while still firing through tracking holds.
FACE_ATTRIBUTION_CONF = 0.55   # high+medium floor (dist < 0.45)
FACE_STREAK_HIGH_CONF = 0.70   # high-band floor (dist < 0.30)


def band_of(
    pred,
    high_conf: float = FACE_STREAK_HIGH_CONF,
    med_conf: float = FACE_ATTRIBUTION_CONF,
) -> str:
    """streamerpi's confidence band for a prediction: prefer the carried band
    string (tracks streamerpi's own cutoff if it retunes), else derive from the
    lossless confidence using the given floors. Returns 'high'|'medium'|'low'.

    This is the SINGLE source of truth for banding — both the attribution and
    streak gates classify through here, so a band string can never contradict a
    parallel numeric check."""
    b = getattr(pred, "band", None)
    if b in ("high", "medium", "low"):
        return b
    c = float(pred.confidence)
    if c >= high_conf:
        return "high"
    if c >= med_conf:
        return "medium"
    return "low"


_PARTY_REGIMES = frozenset({"party", "expo"})


def fuse_identity(
    *,
    voice_name: str,
    voice_is_unknown: bool,
    face: Optional[FaceObservation],
    face_conf_threshold: float = FACE_ATTRIBUTION_CONF,
    streak_high_conf: float = FACE_STREAK_HIGH_CONF,
    head_steady_min_ms: int = 2000,
    # --- Slice B (2026-06-12), all default to today's exact behavior ----------
    voice_confidence: Optional[float] = None,
    symmetric_enabled: bool = False,
    continuity_enabled: bool = False,
    prior_identity: Optional[str] = None,
    prior_identity_fresh: bool = False,
    regime: str = "normal",
) -> FusionVerdict:
    """Resolve speaker identity given voice result and optional face observation.

    Voice always wins for confident matches. Face only contributes a name when:
      - voice is unknown_N (below voiceprint threshold)
      - face returned exactly one prediction
      - face band is high or medium (attribution floor face_conf_threshold)
      - behavior status reports tracking mode + face_visible
      - head has been steady on the face for >= head_steady_min_ms

    Promotion here is ATTRIBUTION (sets speaker_name for the turn, reversible).
    Binding a voiceprint is stricter: verdict.streak_eligible gates the streak on
    a high match, or a sticky-held medium (streak_high_conf).

    Otherwise face is recorded as a presence hint but not promoted to speaker.

    Slice B (default-OFF) adds two ways to fill face_hint_name for the presence/
    proactive path WITHOUT touching final_name (voice still always wins):
      - symmetric: a CONFIDENT voice synthesizes its own name as the face hint
        when the face gave us nothing (face_hint_source='voice').
      - temporal: a fresh prior identity is held across a 1-2 frame face dropout
        (face_hint_source='temporal').
    Both are disabled in PARTY/EXPO regime (a wrong bind beats an abstain there),
    and neither is ever allowed to train a voiceprint (auto-enroll gates on
    face_hint_source=='face').
    """
    voice_name = canonicalize(voice_name) or voice_name
    face_hint_name = None
    face_hint_confidence = None
    head_steady = False

    gates = {
        "voice_unknown": voice_is_unknown,
        "face_present": False,
        "single_face": False,
        "face_above_threshold": False,
        "behavior_known": False,
        "tracking_mode": False,
        "face_visible_flag": False,
        "head_steady": False,
    }

    top_band = "low"
    top_sticky = False
    if face is not None and face.predictions:
        gates["face_present"] = True
        gates["single_face"] = len(face.predictions) == 1
        top = face.predictions[0]
        face_hint_name = canonicalize(top.user_id)
        face_hint_confidence = float(top.confidence)
        # Single classification (band string wins; else derived from the floors).
        top_band = band_of(top, high_conf=streak_high_conf, med_conf=face_conf_threshold)
        top_sticky = bool(getattr(top, "sticky", False))
        # Attribution gate: trust streamerpi's match decision (high OR medium).
        gates["face_above_threshold"] = top_band in ("high", "medium")

        beh = face.behavior
        if beh is not None:
            gates["behavior_known"] = True
            gates["tracking_mode"] = beh.mode in _TRACKING_MODES
            gates["face_visible_flag"] = bool(beh.face_visible)
            head_steady = beh.elapsed_ms >= head_steady_min_ms
            gates["head_steady"] = head_steady

    promote = (
        voice_is_unknown
        and gates["face_present"]
        and gates["single_face"]
        and gates["face_above_threshold"]
        and gates["behavior_known"]
        and gates["tracking_mode"]
        and gates["face_visible_flag"]
        and gates["head_steady"]
    )

    if promote and face_hint_name is not None:
        final_name = face_hint_name
        resolution_source = "face_hint"
    else:
        final_name = voice_name
        resolution_source = "voice"

    # Streak eligibility (stricter than attribution): a promoted face may bind a
    # voiceprint only on a "high" match, or a "medium" the stabilizer is holding
    # (sticky). Always requires promotion first, so it's a strict subset of the
    # face_hint path. Classified through the same band_of() as attribution.
    streak_eligible = bool(
        promote and (top_band == "high" or (top_band == "medium" and top_sticky))
    )

    # face_hint_source provenance: a real face prediction is 'face' (the only
    # source auto-enroll may train from). Default preserves today's behavior.
    face_hint_source = "face"
    stabilized = False

    # --- Slice B: synthesize/hold a face hint when the face gave us nothing ---
    # INVARIANT: only fills face_hint_name (presence/proactive); final_name and
    # the face_hint promotion above are untouched. PARTY/EXPO disables both.
    party = str(regime or "normal").strip().lower() in _PARTY_REGIMES
    if not party and face_hint_name is None:
        voice_confident = (
            not voice_is_unknown
            and (voice_confidence is None or voice_confidence >= VOICE_STABILIZE_CONF_FLOOR)
        )
        # NOTE: resolution_source is deliberately LEFT as 'voice'/'face_hint'.
        # Downstream consumers branch on it (e.g. look-at-speaker fires on
        # 'voice'); rewriting it would silently change their behavior. The new
        # provenance rides on stabilized + face_hint_source instead.
        if symmetric_enabled and voice_confident:
            # A sure voice stands in for the missing face (e.g. Dan's face
            # flapped to 'unidentified' on a head-turn but his voice is certain).
            face_hint_name = voice_name
            face_hint_source = "voice"
            stabilized = True
        elif continuity_enabled and prior_identity is not None and prior_identity_fresh:
            # Carry the recently-seen identity across a brief face dropout.
            face_hint_name = prior_identity
            face_hint_source = "temporal"
            stabilized = True

    return FusionVerdict(
        final_name=final_name,
        resolution_source=resolution_source,
        face_hint_name=face_hint_name,
        face_hint_confidence=face_hint_confidence,
        head_steady=head_steady,
        gates=gates,
        face_hint_source=face_hint_source,
        stabilized=stabilized,
        streak_eligible=streak_eligible,
    )



class IdentityFusion:
    """Stateful wrapper around the pure fuse_identity() (Slice B, DARK).

    Holds the last REAL face-resolved identity for temporal continuity, reads
    the Slice B toggles live (so they take effect without a restart), computes
    prior_identity_fresh, and threads voice confidence through.

    Memory-update invariant: _last_identity is refreshed ONLY from a verdict
    whose face_hint_source == 'face' (a genuine face observation). It is NEVER
    refreshed from a stabilized/held verdict, so a wrong temporal hold can only
    persist up to the continuity window from the last real sighting — it can
    never self-perpetuate. The default knobs make resolve() identical to calling
    fuse_identity() with today's behavior (all toggles OFF).
    """

    def __init__(self, knobs=None):
        # knobs(key) -> value; defaults to the live runtime_toggles reader.
        if knobs is None:
            from persistence import runtime_toggles
            knobs = runtime_toggles.get
        self._knobs = knobs
        self._last_identity: Optional[str] = None
        self._last_seen_ts: float = 0.0

    def resolve(
        self,
        *,
        voice_name: str,
        voice_is_unknown: bool,
        face: Optional[FaceObservation],
        voice_confidence: Optional[float] = None,
        face_conf_threshold: float = FACE_ATTRIBUTION_CONF,
        streak_high_conf: float = FACE_STREAK_HIGH_CONF,
        head_steady_min_ms: int = 2000,
        now: Optional[float] = None,
    ) -> FusionVerdict:
        import time as _time
        now = now if now is not None else _time.time()

        symmetric = bool(self._knobs("identity_fusion_symmetric_enabled"))
        continuity = bool(self._knobs("identity_continuity_enabled"))
        window = self._knobs("identity_continuity_window_s")
        try:
            window = float(window)
        except (TypeError, ValueError):
            window = 2.5
        regime = self._knobs("identity_regime") or "normal"

        prior_fresh = (
            self._last_identity is not None
            and (now - self._last_seen_ts) < window
        )

        verdict = fuse_identity(
            voice_name=voice_name,
            voice_is_unknown=voice_is_unknown,
            face=face,
            face_conf_threshold=face_conf_threshold,
            streak_high_conf=streak_high_conf,
            head_steady_min_ms=head_steady_min_ms,
            voice_confidence=voice_confidence,
            symmetric_enabled=symmetric,
            continuity_enabled=continuity,
            prior_identity=self._last_identity,
            prior_identity_fresh=prior_fresh,
            regime=regime,
        )

        # Refresh memory ONLY from a real face observation — never from a
        # synthesized/held hint (would let a wrong hold self-perpetuate).
        if verdict.face_hint_source == "face" and verdict.face_hint_name is not None:
            self._last_identity = verdict.face_hint_name
            self._last_seen_ts = now

        return verdict


def translate_pose(
    camera_pan: float,
    camera_tilt: float,
    bbox_center_norm,
    pan_fov_steps: float = 80.0,
    tilt_fov_steps: float = 50.0,
):
    """Translate camera pose + face bbox center to the absolute pose that
    would re-center the camera on that face.

    Mirrors streamerpi camera.py face-centering math (camera.py:400-403):
        pan_correction  = -offset_x_norm * (pan_fov_steps / 2)
        tilt_correction = -offset_y_norm * (tilt_fov_steps / 2)

    where offset_x_norm = (bbox_center_x - 0.5) * 2 (i.e. -1..+1 across image).

    Args:
        camera_pan: current commanded camera pan (UI steps, streamerpi convention)
        camera_tilt: current commanded camera tilt
        bbox_center_norm: (x, y) tuple in [0,1] over image, or None
        pan_fov_steps: total horizontal FoV in UI pan steps (default 80)
        tilt_fov_steps: total vertical FoV in UI tilt steps (default 50)

    Returns:
        (person_pan, person_tilt) tuple. If bbox_center_norm is None, returns
        (camera_pan, camera_tilt) unchanged.
    """
    if bbox_center_norm is None:
        return (camera_pan, camera_tilt)
    bx, by = bbox_center_norm
    person_pan = camera_pan - (bx - 0.5) * pan_fov_steps
    person_tilt = camera_tilt - (by - 0.5) * tilt_fov_steps
    return (person_pan, person_tilt)
