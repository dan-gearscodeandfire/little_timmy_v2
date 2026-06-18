#!/usr/bin/env python3
"""Speaker-loopback mic test for the Little Timmy lav chain.

CONTEXT (2026-06-17): Dan moved the lav mic CLOSE to Timmy's loudspeaker and
turned the servos OFF (no motor hum). The old SNR-vs-distance script's caveat
("speaker playback under-couples to a close-talk lav") no longer holds once the
mic is parked next to the speaker, so a loopback test is now valid. This script
plays a SYNTHETIC phrase out Timmy's own TTS/speaker (/api/announce, force-
bypass) and measures what the mic captures, WITHOUT contaminating the
conversation.

Contamination guards:
  - Timmy's HEARING is muted for the whole run (POST /api/hearing enabled=false)
    -> captured audio is dropped pre-STT, never reaches conversation/memory.
    Restored in finally (even on Ctrl-C / error).
  - Output is driven via /api/announce (supervisor force channel) which does NOT
    inject a conversation turn; phrase is blatantly synthetic (no PII).
  - We open a SECOND PipeWire reader (proven fine); independent of LT's own
    capture suppression while it speaks.

It reuses snr_vs_distance.py's helpers so the capture pipeline (48k -> ::3 16k
-> 80 Hz Butterworth HP) and the WeSpeaker identifier match production exactly.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/speaker_loopback_mictest.py
  .venv/bin/python ops/speaker_loopback_mictest.py --rec 12 --reps 1
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Reuse the exact, pipeline-matched helpers from the SNR tool.
from ops.snr_vs_distance import (  # noqa: E402
    ORCH, TARGET_SR, _ensure_audio_env, set_hearing, find_input_device,
    highpass_80, to16k, record, rms, dbfs, speech_rms, save_wav,
)

PHRASE = ("Microphone loopback test. The quick brown fox jumps over the lazy "
          "dog. One two three four five.")


def announce(text: str) -> bool:
    try:
        req = urllib.request.Request(
            f"{ORCH}/api/announce",
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return True
    except Exception as e:
        print(f"  [announce] WARN: {e}")
        return False


def capture_async(device, secs):
    """Run record() in a thread; return a {'x': ndarray} dict filled on join."""
    box = {}

    def _run():
        box["x"] = record(device, secs)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, box


def main():
    ap = argparse.ArgumentParser(description="Speaker-loopback mic test")
    ap.add_argument("--rec", type=float, default=12.0,
                    help="seconds to record per rep (must outlast TTS playback)")
    ap.add_argument("--reps", type=int, default=1, help="how many announce reps")
    ap.add_argument("--phrase", default=PHRASE, help="SYNTHETIC phrase (no PII)")
    ap.add_argument("--device", default=None, help="input device idx/name")
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "loopback_captures"))
    args = ap.parse_args()

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    device = find_input_device(args.device)
    import sounddevice as sd
    dname = sd.query_devices(device)["name"] if device is not None else "(library default)"
    print(f"Input device: [{device}] {dname}")

    # Optional: report nearest enrolled speaker to the captured TTS (timmy
    # voiceprint is disabled, so this is informational re: false-accept region).
    si = None
    try:
        from speaker.identifier import SpeakerIdentifier, KNOWN_SPEAKER_THRESHOLD
        si = SpeakerIdentifier()
        si.load_voiceprints()
        print(f"Enrolled: {', '.join(ks.name for ks in si._known_speakers) or 'none'} "
              f"threshold={KNOWN_SPEAKER_THRESHOLD}")
    except Exception as e:
        print(f"  [identifier] WARN: {e}")

    results = []
    try:
        print(">>> Muting hearing (captures dropped pre-STT; restored at end)")
        set_hearing(False)
        time.sleep(0.5)

        print(">>> Baseline: capturing ~2.5s room tone (servos off, Timmy silent)")
        base = to16k(record(device, 2.5))
        base_hp = highpass_80(base)
        floor = rms(base_hp)
        print(f"  baseline floor: raw={dbfs(base):.1f} dBFS  post-HP={dbfs(base_hp):.1f} dBFS")

        for i in range(args.reps):
            print(f"\n[rep {i+1}/{args.reps}] recording {args.rec:.0f}s; "
                  f"announcing through Timmy's speaker...")
            t, box = capture_async(device, args.rec)
            time.sleep(0.4)  # let the stream spin up before audio starts
            announce(args.phrase)
            t.join()
            cap = box.get("x")
            if cap is None or len(cap) == 0:
                print("  WARN: empty capture"); continue
            x16 = to16k(cap)
            x16_hp = highpass_80(x16)
            sp = speech_rms(x16_hp)
            snr = 20.0 * np.log10(sp / (floor + 1e-12))

            row = dict(rep=i + 1, snr_db=round(snr, 1),
                       peak_dbfs=round(dbfs(x16_hp), 1),
                       speech_rms=round(float(sp), 5),
                       captured=bool(snr > 6.0))
            if si is not None:
                try:
                    emb = si.extract_embedding(x16_hp)
                    ranked = sorted(((ks.distance(emb), ks.name)
                                     for ks in si._known_speakers), key=lambda u: u[0])
                    if ranked:
                        row["nearest"] = ranked[0][1]
                        row["nearest_dist"] = round(ranked[0][0], 3)
                except Exception as e:
                    row["nearest"] = f"err:{e}"

            wav = os.path.join(args.outdir, f"{stamp}_rep{i+1}.wav")
            save_wav(wav, x16_hp)
            row["wav"] = wav
            results.append(row)
            print(f"  SNR={row['snr_db']} dB | peak={row['peak_dbfs']} dBFS | "
                  f"{'MIC CAPTURED speaker' if row['captured'] else 'NO/weak capture'}"
                  + (f" | nearest={row.get('nearest')} @ {row.get('nearest_dist')}"
                     if 'nearest' in row else ""))

        print("\n==== SUMMARY ====")
        print(json.dumps(results, indent=2))
        print("\nInterpretation:")
        print("  - SNR is Timmy's-voice-into-mic vs the (now servo-off) floor.")
        print("  - High SNR + clean peak = mic functioning + good new placement.")
        print("  - This validates the CHAIN; it does NOT reproduce off-mic-guest")
        print("    SNR (that still needs a human voice at distance via snr_vs_distance.py).")
    finally:
        print("\n>>> Restoring hearing")
        set_hearing(True)


if __name__ == "__main__":
    main()
