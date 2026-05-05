"""Voice enrollment with level-check and cross-speaker safety.

Runs through Little Timmy's audio path and computes the Resemblyzer
embedding the orchestrator would actually see at runtime. Default mode
is *preview* (no disk write). Commit only happens with --commit, after
you've seen the peak/RMS numbers and the cross-distance to every
already-enrolled speaker.

Usage:
  python enroll_voice.py NAME                      # preview only
  python enroll_voice.py NAME --commit             # preview + write
  python enroll_voice.py NAME --seconds 20         # longer take
  python enroll_voice.py NAME --warmup 5           # tweak warmup
  python enroll_voice.py --list                    # list enrolled

The warmup is a short 'how loud are you' pre-record that prints peak +
RMS so you can adjust mic distance before the real take. Verdict line
afterwards calls out CLIPPING (peak >= 0.95), TOO_QUIET (peak < 0.05),
and TIGHT cross-distances (< 0.30 to any other enrolled speaker).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NATIVE_SR = 48000
TARGET_SR = 16000
CHANNELS = 2
DEVICE = "default"

PEAK_CLIP = 0.95            # >= this is clipping risk
PEAK_TOO_QUIET = 0.05       # < this is too quiet for resemblyzer
PEAK_GOOD_LO = 0.20
PEAK_GOOD_HI = 0.80
TIGHT_CROSS_DIST = 0.30


def record(seconds: float, label: str) -> np.ndarray:
    """Record `seconds` of audio at NATIVE_SR/CHANNELS, resample to 16k mono."""
    print(f"  [{label}] recording {seconds:.1f}s...")
    audio = sd.rec(int(seconds * NATIVE_SR),
                   samplerate=NATIVE_SR, channels=CHANNELS,
                   dtype="float32", device=DEVICE)
    sd.wait()
    ch0 = audio[:, 0]
    return resample_poly(ch0, TARGET_SR, NATIVE_SR).astype(np.float32)


def stats(audio: np.ndarray) -> dict:
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    # Fraction of samples above a noise floor.
    nonsilent = float(np.mean(np.abs(audio) > 0.01))
    return {"peak": peak, "rms": rms, "nonsilent_frac": nonsilent}


def verdict(s: dict) -> str:
    p = s["peak"]
    if p >= PEAK_CLIP:
        return "CLIPPING — back off the mic, this take is unusable"
    if p < PEAK_TOO_QUIET:
        return "TOO_QUIET — get closer or speak up, embedding will be noise-dominated"
    if p < PEAK_GOOD_LO:
        return "QUIET — usable but low signal; consider re-recording closer"
    if p > PEAK_GOOD_HI:
        return "HOT — close to clipping; a bit more distance would help"
    return "GOOD — solid level"


def cross_distances(emb: np.ndarray, exclude_name: str) -> list[tuple[str, float]]:
    from speaker.identifier import SpeakerIdentifier
    si = SpeakerIdentifier()
    si.load_voiceprints()
    out = []
    for ks in si._known_speakers:
        if ks.name == exclude_name:
            continue
        out.append((ks.name, float(cosine(emb, ks.embedding))))
    return sorted(out, key=lambda t: t[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?", help="lowercase speaker name (a-z, 0-9, _, -)")
    p.add_argument("--commit", action="store_true",
                   help="actually write the voiceprint after preview (default: preview only)")
    p.add_argument("--seconds", type=float, default=15.0,
                   help="real-take duration in seconds (default 15)")
    p.add_argument("--warmup", type=float, default=4.0,
                   help="warmup duration in seconds, 0 to skip (default 4)")
    p.add_argument("--list", action="store_true", help="list enrolled speakers and exit")
    args = p.parse_args()

    if args.list:
        from speaker.identifier import SpeakerIdentifier
        si = SpeakerIdentifier()
        si.load_voiceprints()
        for ks in si._known_speakers:
            print(f"  {ks.name}  speaker_id={ks.speaker_id}")
        return 0

    if not args.name:
        p.error("name is required (or --list)")
    name = args.name.strip().lower()

    print(f"=== Voice enrollment: {name} (mode={'COMMIT' if args.commit else 'PREVIEW'}) ===\n")

    # Warmup: short pre-record so the speaker can see their level and adjust.
    if args.warmup > 0:
        print(f"WARMUP — speak at the level you plan to use; this is just a check.")
        time.sleep(0.5)
        warmup_audio = record(args.warmup, "warmup")
        ws = stats(warmup_audio)
        print(f"  warmup peak={ws['peak']:.3f}  rms={ws['rms']:.3f}  "
              f"nonsilent_frac={ws['nonsilent_frac']:.2f}")
        print(f"  verdict: {verdict(ws)}")
        if ws["peak"] >= PEAK_CLIP:
            print("  ABORT: warmup clipped. Move farther from the mic and re-run.")
            return 2
        if ws["peak"] < PEAK_TOO_QUIET:
            print("  ABORT: warmup too quiet. Get closer / speak up and re-run.")
            return 2
        print("  warmup OK; pausing 1s before real take.\n")
        time.sleep(1.0)

    print("REAL TAKE")
    for i in range(3, 0, -1):
        print(f"  starting in {i}...")
        time.sleep(1)
    audio_16k = record(args.seconds, "take")
    s = stats(audio_16k)
    print(f"  peak={s['peak']:.3f}  rms={s['rms']:.3f}  nonsilent_frac={s['nonsilent_frac']:.2f}")
    print(f"  verdict: {verdict(s)}")

    if s["peak"] >= PEAK_CLIP:
        print("  ABORT: take clipped, refusing to compute embedding. Re-run with more distance.")
        return 2
    if s["peak"] < PEAK_TOO_QUIET:
        print("  ABORT: take essentially silent, refusing to compute embedding.")
        return 2

    # Embedding + cross-distance.
    from speaker.identifier import SpeakerIdentifier, VOICEPRINT_DIR
    si = SpeakerIdentifier()
    si._load_encoder()
    emb = si.extract_embedding(audio_16k)
    print(f"\n  embedding shape={emb.shape}  norm={np.linalg.norm(emb):.3f}")

    crosses = cross_distances(emb, exclude_name=name)
    print("  cross-distances vs other enrolled speakers:")
    if not crosses:
        print("    (no other speakers enrolled)")
    for n, d in crosses:
        flag = "  <-- TIGHT" if d < TIGHT_CROSS_DIST else ""
        print(f"    {name} vs {n}: dist={d:.4f}{flag}")
    if any(d < TIGHT_CROSS_DIST for _, d in crosses):
        print("\n  WARNING: at least one TIGHT cross-distance. Mid-conversation "
              "misclassification likely until margin is widened (re-record at a "
              "different mic level / different voice tone).")

    out = VOICEPRINT_DIR / f"{name}_resemblyzer.npy"
    if not args.commit:
        print(f"\n  PREVIEW MODE — not writing. To keep, re-run with --commit.")
        return 0

    if out.exists():
        bak = VOICEPRINT_DIR / f"{name}_resemblyzer.npy.bak.{int(time.time())}"
        out.rename(bak)
        print(f"\n  backed up existing -> {bak.name}")
    np.save(out, emb)
    print(f"  wrote {out}")

    # Re-load to make sure id-map gets updated and report final state.
    si2 = SpeakerIdentifier()
    si2.load_voiceprints()
    print("\n  enrolled after write:")
    for ks in si2._known_speakers:
        print(f"    - {ks.name}  speaker_id={ks.speaker_id}")
    print("\n  Restart little-timmy.service to pick up the new voiceprint.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
