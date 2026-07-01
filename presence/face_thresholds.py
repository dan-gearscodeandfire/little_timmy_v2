"""EdgeFace-S cosine-distance thresholds (calibrated, PROVISIONAL).

Derived by ops/edgeface_calibrate.py against a real genuine/impostor split on
2026-06-30: genuine = 30 on-camera captures of Dan (single session), impostor =
171 distinct-identity LFW strangers, EdgeFace-S (gamma 0.5) on okDemerzel CPU
through the pinned aligner (presence.face_align).

    GENUINE  (Dan vs Dan)      mean 0.088  p95 0.173  max 0.193
    IMPOSTOR (Dan vs stranger) mean 1.012  p05 0.847  min 0.697
    Separation CLEAN: max(genuine) 0.193 << min(impostor) 0.697

PROVISIONAL because genuine is single-session (one lighting/day) — that is the
FALSE-REJECT-relevant unknown, and the impostors are random strangers, not the
lookalikes/household who could actually be confused with an enrolled person.
Phase B passive co-sampling collects real in-domain impostors + cross-session
genuine through the same camera; recalibrate then (harness: ingest/report) and
tighten. WeSpeaker's voice constants are a DIFFERENT scale — do not copy across.
"""

# Accept an identity when a probe's MINIMUM cosine distance to its prototypes is
# below this. Midpoint of p95-genuine (0.173) and p05-impostor (0.847) is ~0.51;
# set slightly under to bias toward FAR-safety, leaving ~2.6x headroom over the
# single-session genuine max (0.193) for cross-session drift, still far below the
# impostor floor (0.697).
KNOWN_FACE_THRESHOLD = 0.50

# Confidence bands (mirrors the voice band_of pattern; consumed by fuse_identity
# via FacePrediction.band). Distances below the cutoff fall in the band.
FACE_BAND_HIGH = 0.35     # decisively this person
FACE_BAND_MEDIUM = 0.50   # probable (== accept threshold)

# A new prototype within this cosine distance of an existing one is a near-dup
# (same pose/lighting) -> skip. ~ median genuine (0.069) rounded up a touch.
FACE_DEDUP_DIST = 0.07

# Cap prototypes per identity (keep most recent beyond this). Mirrors voice.
MAX_FACE_PROTOTYPES = 12

# Minimum distinct face samples before a fresh face voiceprint persists durably
# (guards a one-frame fluke). Mirrors voice MIN_ENROLL_SAMPLES.
MIN_FACE_ENROLL_SAMPLES = 3
