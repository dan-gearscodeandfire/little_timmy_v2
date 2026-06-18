#!/usr/bin/env python3
"""Hands-free LEVEL-swept speaker loopback (distance proxy) for the LT mic.

Constraints this honours (Dan, 2026-06-17): nobody in the room, the lav is left
FIXED ~6 in from Timmy's speaker, mic is NOT moved. The only controllable sound
source is Timmy's own speaker, so we vary the OUTPUT VOLUME of the default sink
to emulate a quieter / more distant source and watch how the captured voice's
SNR and WeSpeaker identity degrade. This reproduces the SNR axis of the P1
mic-chain misID without a human and without moving the mic.

Caveats (stated, not hidden):
  - Source is Timmy's Piper TTS voice, not a human voice.
  - Lowering playback volume drops SIGNAL LEVEL but adds no room REVERB, so it
    models the SNR-degradation axis of distance, not acoustic coloration.

Safety / no-contamination:
  - hearing muted the whole run (captures dropped pre-STT); restored in finally.
  - output driven via /api/announce (force channel, no conversation turn),
    synthetic phrase.
  - default-sink volume is read first and RESTORED in finally.

Reuses snr_vs_distance.py's pipeline-matched capture + identifier.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/level_sweep_loopback.py
  .venv/bin/python ops/level_sweep_loopback.py --levels "1.0,0.6,0.35,0.2,0.1,0.05"
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ops.snr_vs_distance import (  # noqa: E402
    ORCH, TARGET_SR, _ensure_audio_env, set_hearing, find_input_device,
    highpass_80, to16k, record, rms, dbfs, speech_rms, save_wav,
)

PHRASE = ("The quick brown fox jumps over the lazy dog. "
          "One two three four five six seven eight.")          # synthetic, no PII
SINK = "@DEFAULT_AUDIO_SINK@"


def _wpctl(*args):
    return subprocess.run(["wpctl", *args], capture_output=True, text=True, timeout=8)


def get_volume() -> float | None:
    try:
        out = _wpctl("get-volume", SINK).stdout
        m = re.search(r"Volume:\s*([0-9.]+)", out)
        return float(m.group(1)) if m else None
    except Exception as e:
        print(f"  [vol] WARN get: {e}"); return None


def set_volume(v: float) -> None:
    try:
        _wpctl("set-volume", SINK, f"{v:.3f}")
    except Exception as e:
        print(f"  [vol] WARN set {v}: {e}")


def announce(text: str) -> None:
    try:
        req = urllib.request.Request(
            f"{ORCH}/api/announce",
            data=json.dumps({"text": text, "no_prefix": True}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except Exception as e:
        print(f"  [announce] WARN: {e}")


def capture_async(device, secs):
    box = {}
    t = threading.Thread(target=lambda: box.update(x=record(device, secs)), daemon=True)
    t.start()
    return t, box


def main():
    ap = argparse.ArgumentParser(description="Level-swept speaker loopback")
    ap.add_argument("--levels", default="1.0,0.6,0.35,0.2,0.1,0.05",
                    help="comma list of output-sink volumes (1.0=full) to sweep")
    ap.add_argument("--phrase", default=PHRASE)
    ap.add_argument("--rec", type=float, default=10.0, help="record seconds per level")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "level_sweep_captures"))
    args = ap.parse_args()

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    from speaker.identifier import SpeakerIdentifier, KNOWN_SPEAKER_THRESHOLD
    si = SpeakerIdentifier()
    si.load_voiceprints()
    print(f"Enrolled: {', '.join(ks.name for ks in si._known_speakers)} "
          f"threshold={KNOWN_SPEAKER_THRESHOLD}")

    device = find_input_device(args.device)
    import sounddevice as sd
    dname = sd.query_devices(device)["name"] if device is not None else "(default)"
    print(f"Input device: [{device}] {dname}")

    orig_vol = get_volume()
    print(f"Default-sink volume (will restore): {orig_vol}")
    print(f"Levels: {levels}\n")

    results = []
    try:
        set_hearing(False)
        time.sleep(0.4)
        # baseline at silence (volume irrelevant; nothing playing)
        base = to16k(record(device, 2.5))
        base_hp = highpass_80(base)
        floor = rms(base_hp)
        print(f"baseline floor: raw={dbfs(base):.1f} dBFS  post-HP={dbfs(base_hp):.1f} dBFS\n")

        for v in levels:
            set_volume(v)
            time.sleep(0.3)
            src_db = 20.0 * np.log10(v) if v > 0 else float("-inf")
            t, box = capture_async(device, args.rec)
            time.sleep(0.4)
            announce(args.phrase)
            t.join()
            cap = box.get("x")
            if cap is None or len(cap) == 0:
                print(f"[vol {v}] WARN empty capture"); continue
            x16 = to16k(cap)
            x16_hp = highpass_80(x16)
            sp = speech_rms(x16_hp)
            snr = 20.0 * np.log10(sp / (floor + 1e-12))

            emb = si.extract_embedding(x16_hp)
            ranked = sorted(((ks.distance(emb), ks.name) for ks in si._known_speakers),
                            key=lambda u: u[0])
            best = ranked[0] if ranked else (float("nan"), "none")
            second = ranked[1] if len(ranked) > 1 else (float("inf"), "none")
            margin = second[0] - best[0]
            would_match = best[0] < KNOWN_SPEAKER_THRESHOLD

            wav = os.path.join(args.outdir, f"{stamp}_vol{v}.wav")
            save_wav(wav, x16_hp)
            row = dict(volume=v, src_atten_db=round(src_db, 1), snr_db=round(snr, 1),
                       peak_dbfs=round(dbfs(x16_hp), 1),
                       best=best[1], best_dist=round(best[0], 3),
                       second=second[1], second_dist=round(second[0], 3),
                       margin=round(margin, 3), matches_known=would_match, wav=wav)
            results.append(row)
            print(f"[vol {v:>4} | {row['src_atten_db']:>5} dB] SNR={row['snr_db']:>5} dB | "
                  f"nearest={row['best']} @ {row['best_dist']} | 2nd={row['second']} @ "
                  f"{row['second_dist']} | margin={row['margin']} | "
                  f"{'MATCH '+row['best'] if would_match else 'unknown(>thr)'}")

        print("\n==== SUMMARY (lower volume = quieter/'farther' source) ====")
        print(json.dumps(results, indent=2))
        print("\nInterpretation:")
        print("  - SNR falling with volume = the chain's level-degradation curve.")
        print("  - Watch best_dist climb + margin collapse: the SNR at which a")
        print("    quiet source's embedding starts landing on an enrolled print")
        print("    (a false-accept) vs safely abstaining to unknown(>0.30).")
        print("  - Source is Timmy's TTS, not a human; models SNR axis, not reverb.")
    finally:
        if orig_vol is not None:
            print(f"\n>>> Restoring sink volume to {orig_vol}")
            set_volume(orig_vol)
        print(">>> Restoring hearing")
        set_hearing(True)


if __name__ == "__main__":
    main()
