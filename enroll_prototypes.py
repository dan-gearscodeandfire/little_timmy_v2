"""K-prototype speaker enrollment (party-ready).

Records several short clips through Little Timmy's EXACT capture path (the same
device/SR/channel/resample/HP-filter chain as live identification, reused from
``enroll_from_pipeline.record_like_timmy``), embeds each separately, and saves a
deduped (K, 256) prototype set to ``models/speaker/<name>_wespeaker.npy``.

Why multiple clips: one averaged prototype can't span the distance/loudness
variation of a real room (this is the cause of "Dan -> unknown_1"). Capturing a
deliberate spread of poses and matching on the MINIMUM cosine distance across
the set (see ``KnownSpeaker.distance``) recognizes any covered look. Mirrors the
SFace face-ID multi-prototype fix.

The prototype build + atomic save reuse the runtime's own
``speaker.identifier`` code so enrollment and live matching never diverge.

Usage:
  ./.venv/bin/python enroll_prototypes.py dan
  ./.venv/bin/python enroll_prototypes.py devon --samples 6 --seconds 4
  ./.venv/bin/python enroll_prototypes.py dan --dry-run   # capture + report, don't save

After saving, restart Little Timmy to load the new voiceprint.
"""

import argparse
import sys
import time

import numpy as np
from scipy.spatial.distance import cosine

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")

from enroll_from_pipeline import record_like_timmy  # exact LT capture path
from speaker.identifier import (
    SpeakerIdentifier,
    KNOWN_SPEAKER_THRESHOLD,
    _build_prototypes,
)

# Spoken cues that elicit a useful spread of poses/distances/loudness. Cycled
# if --samples exceeds the list length.
POSE_CUES = [
    "normal voice, ~3 ft from the mic",
    "step back — talk from ~6-8 ft away",
    "quieter / casual, like across the room",
    "louder — your PARTY voice over noise",
    "turned away, talking over your shoulder",
    "from the side of the mic, not head-on",
]


def embed(audio_16k: np.ndarray) -> np.ndarray:
    """Embed one clip with the SAME WeSpeaker backend identify() uses."""
    from speaker import encoder as _enc
    return _enc.extract_embedding(audio_16k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="lowercase speaker name (a-z, 0-9, _, -)")
    ap.add_argument("--samples", type=int, default=6, help="number of clips to capture")
    ap.add_argument("--seconds", type=int, default=4, help="seconds per clip")
    ap.add_argument("--dry-run", action="store_true", help="capture + report, do not save")
    args = ap.parse_args()

    name = args.name.strip().lower()
    print(f"\n=== K-prototype enrollment: {name} "
          f"({args.samples} clips x {args.seconds}s) ===")
    print("Vary distance/loudness/angle between clips as prompted — that spread "
          "is exactly what makes you recognizable across the room.\n")

    from speaker import encoder as _enc
    print("Loading WeSpeaker encoder...")
    _enc.get_inference()

    raw_embs: list[np.ndarray] = []
    for i in range(args.samples):
        cue = POSE_CUES[i % len(POSE_CUES)]
        print(f"\n--- Clip {i + 1}/{args.samples}: {cue} ---")
        audio = record_like_timmy(args.seconds)
        try:
            raw_embs.append(embed(audio))
        except Exception as e:
            print(f"  ! embed failed for clip {i + 1}: {e} (skipping)")
        time.sleep(0.3)

    if not raw_embs:
        print("\nNo usable clips captured — aborting.")
        return 1

    protos = _build_prototypes(raw_embs)   # normalize + dedup + cap (runtime code)
    print(f"\n=== Built {protos.shape[0]} prototype(s) from {len(raw_embs)} clips "
          f"(near-duplicates deduped) ===")

    # Self-consistency report: how each captured clip would score at match time
    # (min distance to the kept set) vs the live threshold.
    print(f"\nPer-clip min-distance to the kept set (live threshold "
          f"{KNOWN_SPEAKER_THRESHOLD:.2f}):")
    worst = 0.0
    for i, e in enumerate(raw_embs):
        en = e / np.linalg.norm(e)
        d = min(float(cosine(en, p)) for p in protos)
        worst = max(worst, d)
        flag = "ok" if d < KNOWN_SPEAKER_THRESHOLD else "MISS at current threshold"
        print(f"  clip {i + 1}: {d:.3f}  {flag}")

    # Intra-set spread (how diverse the prototypes are).
    if protos.shape[0] > 1:
        spread = [float(cosine(protos[a], protos[b]))
                  for a in range(len(protos)) for b in range(a + 1, len(protos))]
        print(f"\nPrototype spread: min={min(spread):.3f} "
              f"max={max(spread):.3f} mean={sum(spread) / len(spread):.3f}")
    print(f"Worst-clip self-distance: {worst:.3f} "
          f"(threshold guidance: keep it comfortably under "
          f"{KNOWN_SPEAKER_THRESHOLD:.2f}; if several clips MISS, recapture)")

    if args.dry_run:
        print("\n--dry-run: nothing saved.")
        return 0

    # Reuse the runtime's validated atomic write (backs up any existing file).
    si = SpeakerIdentifier()
    out = si.persist_voiceprint(name, protos)
    print(f"\nSaved {protos.shape[0]} prototype(s) -> {out}")
    print("Restart Little Timmy to load the new voiceprint "
          "(`systemctl --user restart <lt-service>` or per startup notes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
