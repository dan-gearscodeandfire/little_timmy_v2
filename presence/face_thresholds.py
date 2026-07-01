"""EdgeFace-S cosine-distance thresholds (calibrated). EdgeFace-S (gamma 0.5) on
okDemerzel CPU through the pinned aligner (presence.face_align).

Two calibration passes (ops/edgeface_calibrate.py + ops/build_maker_gallery.py),
2026-06-30:

  (1) same-session floor — 30 on-camera Dan captures vs 171 LFW strangers:
      GENUINE  mean 0.088  p95 0.173  max 0.193
      IMPOSTOR mean 1.012  p05 0.847  min 0.697   (clean, huge gap)

  (2) cross-condition + in-community — Dan + 14 maker identities (each across
      many varied YouTube thumbnails) as genuine; maker-vs-maker + LFW as
      impostor (the real confusable population, not just random strangers):
      GENUINE  mean 0.362  p95 0.597  max 0.794   (realistic cross-condition)
      IMPOSTOR mean 0.987  p05 0.802  min 0.558
      @0.50 FRR 17% FAR 0.00% | @0.55 FRR 9% FAR 0.00% | @0.45 FRR 28% FAR 0.00%

Key result: FAR ~0% through 0.55 even against real in-community faces — the
separation holds. Pass (2)'s genuine spread is PESSIMISTIC for live use: it is
thumbnail->thumbnail (compression + years + wild pose), whereas the booth camera
enrolls AND recognizes the household in ONE consistent domain (pass (1) regime,
FRR far lower). The makers, enrolled from thumbnails but recognized live, are the
hard cross-domain case. Recalibrate in-domain once Phase B co-sampling collects
real booth-camera crops. WeSpeaker's voice constants are a DIFFERENT scale.
"""

# Accept an identity when a probe's MINIMUM cosine distance to its prototypes is
# below this. 0.50 sits below the in-community impostor min (0.558) -> FAR ~0%,
# while accepting the same-camera household comfortably and most cross-domain
# maker shots. 0.55 trades a hair of FAR headroom for ~8pt lower FRR if booth
# recognition of the makers proves too strict.
KNOWN_FACE_THRESHOLD = 0.50

# Confidence bands (mirrors the voice band_of pattern; consumed by fuse_identity
# via FacePrediction.band). Distances below the cutoff fall in the band. HIGH is
# ~ the same-session genuine ceiling (0.19) rounded up; MEDIUM == accept.
FACE_BAND_HIGH = 0.40     # decisively this person (same-domain tight match)
FACE_BAND_MEDIUM = 0.50   # probable (== accept threshold)

# A new prototype within this cosine distance of an existing one is a near-dup
# (same pose/lighting) -> skip. ~ median genuine (0.069) rounded up a touch.
FACE_DEDUP_DIST = 0.07

# Cap prototypes per identity (keep most recent beyond this). Mirrors voice.
MAX_FACE_PROTOTYPES = 12

# Minimum distinct face samples before a fresh face voiceprint persists durably
# (guards a one-frame fluke). Mirrors voice MIN_ENROLL_SAMPLES.
MIN_FACE_ENROLL_SAMPLES = 3
