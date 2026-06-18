#!/usr/bin/env python3
"""Measure Timmy's OWN TTS level at the lav, when the mic is seated near his
speaker. The inverse of mic_placement_cued.py: there the subject's voice is the
signal and we wait Timmy out; here TIMMY's Piper voice IS the signal, so we
record DURING playback.

announce() speaks with the production skeletor Piper voice (one of the Piper
voices) on the force channel. It is non-blocking and there's no TTS-done
signal, so the contract is inverted vs the cued sweep: fire the cue, then start
a generous record window so playback lands inside it. speech_rms() (loudest ~1s
window) isolates the spoken portion from the synth-latency silence at the head.

Reuses capture + 48k->::3 16k->80Hz HP from snr_vs_distance.py so the floor and
SNR are on the same scale as the worn-position run. Hearing is muted the whole
time (test speech never reaches conversation) and restored in finally.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/speaker_output_snr.py
  .venv/bin/python ops/speaker_output_snr.py --takes 3 --rec 10
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

from ops.snr_vs_distance import (  # noqa: E402
    ORCH, _ensure_audio_env, set_hearing,
    highpass_80, to16k, record, rms, dbfs, speech_rms, save_wav,
)

# Repeated so there is a sustained >=1s loud window for speech_rms. Synthetic.
PHRASE = ("the quick brown fox jumps over the lazy dog. "
          "the quick brown fox jumps over the lazy dog.")


def _announce_async(text: str) -> threading.Thread:
    """Fire the cue in a thread (POST returns fast; playback starts ~1-2s
    later) so the caller can open the record window immediately."""
    def _go():
        try:
            req = urllib.request.Request(
                f"{ORCH}/api/announce",
                data=json.dumps({"text": text}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read()
        except Exception as e:
            print(f"  ! announce failed: {e}")
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


def _resolve_device(hint=None):
    """Re-resolve the PipeWire input BY NAME (indices churn when TTS rebuilds
    the graph). Refresh PortAudio's list each call."""
    import sounddevice as sd
    try:
        sd._terminate(); sd._initialize()
    except Exception:
        pass
    devs = sd.query_devices()
    if hint is not None:
        try:
            return int(hint)
        except ValueError:
            pass
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and d["name"] == "default":
            return i
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and "pipewire" in d["name"].lower():
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description="Timmy-speaker output SNR at the lav")
    ap.add_argument("--takes", type=int, default=3, help="number of TTS takes")
    ap.add_argument("--phrase", default=PHRASE, help="SYNTHETIC phrase (no PII)")
    ap.add_argument("--rec", type=float, default=10.0,
                    help="record window per take (must cover synth latency + speech)")
    ap.add_argument("--gap", type=float, default=2.0, help="seconds between takes")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "speaker_snr_captures"))
    args = ap.parse_args()

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Takes={args.takes} rec={args.rec}s phrase=\"{args.phrase}\"\n")

    results = []
    try:
        set_hearing(False)

        # --- floor baseline at the speaker-side seat (silence) ---
        dev = _resolve_device(args.device)
        base = to16k(record(dev, 2.5))
        base_hp = highpass_80(base)
        floor = rms(base_hp)
        floor_db = dbfs(base_hp)
        print(f"floor: raw={dbfs(base):.1f} dBFS  post-HP={floor_db:.1f} dBFS\n")

        for take in range(1, args.takes + 1):
            dev = _resolve_device(args.device)
            _announce_async(args.phrase)          # Timmy starts speaking ~1-2s in
            cap = record(dev, args.rec)           # window opened immediately, covers it
            x16 = to16k(cap)
            x16_hp = highpass_80(x16)
            sp = speech_rms(x16_hp)
            snr = 20.0 * np.log10(sp / (floor + 1e-12))
            sp_db = 20.0 * np.log10(sp)

            wav = os.path.join(args.outdir, f"{stamp}_take{take}.wav")
            save_wav(wav, x16_hp)
            heard = bool((sp_db - floor_db) > 10.0)  # sanity: did playback land in window?
            row = dict(take=take, snr_db=round(snr, 1),
                       floor_dbfs=round(floor_db, 1),
                       tts_dbfs=round(sp_db, 1), heard=heard, wav=wav)
            results.append(row)
            flag = "" if heard else "  <-- LOW: TTS may have missed the window (raise --rec)"
            print(f"  [take {take}] TTS SNR={row['snr_db']:>5} dB | "
                  f"floor={row['floor_dbfs']} dBFS | tts={row['tts_dbfs']} dBFS{flag}")
            time.sleep(args.gap)

        print("\n==== SUMMARY ====")
        print(json.dumps(results, indent=2))
        heard_rows = [r for r in results if r["heard"]]
        if heard_rows:
            avg = sum(r["snr_db"] for r in heard_rows) / len(heard_rows)
            print(f"\nMean TTS SNR (valid takes): {avg:.1f} dB")
        print("Compare against the worn-position run (~33-34 dB SNR at normal "
              "speech). Similar => the speaker-side seat hears Timmy as well as "
              "the worn seat hears Dan.")
    finally:
        print("\n>>> Restoring hearing")
        set_hearing(True)


if __name__ == "__main__":
    main()
