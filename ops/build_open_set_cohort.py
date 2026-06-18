#!/usr/bin/env python3
"""Build the open-set anti-model COHORT in the production WeSpeaker space.

The cohort is a set of embeddings of NON-enrolled speakers + room noise that the
open-set guard (speaker/open_set.py) normalizes against. It must live in the same
embedding space as the live matcher, so we embed each clip with
``SpeakerIdentifier.extract_embedding`` — the exact backend identify() uses, now
WeSpeaker (``pyannote/wespeaker-voxceleb-resnet34-LM``) since the 2026-06-17
encoder migration. (The earlier Resemblyzer cohort lives in models/speaker/cohort/
and is superseded — never mix the two spaces; cross-encoder sim is ~0.)

Held-out by construction: the clips used here are DISJOINT from the calibration
test set in open_set_calibrate.py (cohort uses the WeSpeaker-cal *enroll* takes
of the synthetic voices + two floor clips; calibration tests on the
multi_voice *vol1.0* takes, Timmy-TTS, far Dan, and the other floor clips), so
the reported separation is not optimistic.

Each embedding is saved as models/speaker/cohort/<tag>.npy (1-D (D,)); the guard
loads the whole directory via OpenSetScorer.from_dir.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/build_open_set_cohort.py
  .venv/bin/python ops/build_open_set_cohort.py --clear   # wipe cohort dir first
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.io import wavfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OPS = os.path.join(REPO, "ops")
# WeSpeaker cohort dir — must match speaker.identifier.COHORT_DIR.
COHORT_DIR = os.path.join(REPO, "models", "speaker", "cohort_wespeaker")

# Held-out cohort manifest (globs). Synthetic non-enrolled voices (the ENROLL
# takes from the WeSpeaker calibration run) + pure room floor (mic-off run).
COHORT_GLOBS = [
    os.path.join(OPS, "wespeaker_cal_captures", "*_enr0.wav"),
    os.path.join(OPS, "wespeaker_cal_captures", "*_enr1.wav"),
    os.path.join(OPS, "snr_captures", "20260617-151919_cued_6in.wav"),
    os.path.join(OPS, "snr_captures", "20260617-151919_cued_8ft.wav"),
]


def load16k(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if np.abs(x).max() > 1.5:      # int16 saved as PCM
        x /= 32768.0
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Build WeSpeaker open-set cohort")
    ap.add_argument("--globs", nargs="*", default=COHORT_GLOBS)
    ap.add_argument("--outdir", default=COHORT_DIR)
    ap.add_argument("--clear", action="store_true", help="remove existing cohort .npy first")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.clear:
        for p in glob.glob(os.path.join(args.outdir, "*.npy")):
            os.remove(p)
        print(f"cleared {args.outdir}")

    paths = []
    for g in args.globs:
        paths.extend(sorted(glob.glob(g)))
    paths = sorted(set(paths))
    if not paths:
        print("No cohort source clips matched."); return 1

    from speaker.identifier import SpeakerIdentifier
    si = SpeakerIdentifier()           # no load_voiceprints needed; just the encoder

    print(f"Embedding {len(paths)} cohort clips under WeSpeaker ...\n")
    saved = 0
    for p in paths:
        try:
            x = load16k(p)
            emb = np.asarray(si.extract_embedding(x), dtype=np.float32)
        except Exception as e:
            print(f"  ! {os.path.basename(p)}: {e}"); continue
        tag = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(args.outdir, f"{tag}.npy"), emb)
        saved += 1
        print(f"  + {tag}  (dim={emb.shape[0]})")

    print(f"\nSaved {saved} cohort embeddings -> {args.outdir}")
    print("Load via speaker.open_set.OpenSetScorer.from_dir(<dir>).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
