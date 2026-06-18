#!/usr/bin/env python3
"""Hands-free MULTI-VOICE loopback sweep for the LT mic — off-site multi-speaker
speaker-ID testing without a human.

Generalizes level_sweep_loopback.py from one source (Timmy's TTS via /api/announce)
to N distinct Piper voices played through Timmy's speaker. Each voice is a distinct
synthetic "speaker", so this exercises multi-speaker discrimination, the
mic-handed-around case, and (later) the open-set anti-model — all with nobody in
the room. Voices live in models/tts/test_voices/ (see project_lt_test_voices_multispeaker).

What it measures
  - Per voice: through-the-air SNR, nearest ENROLLED print + distance + margin, and
    whether it would FALSE-ACCEPT (best_dist < 0.30) onto a real identity.
  - Across voices: the pairwise cosine-distance matrix of the CAPTURED embeddings
    (post speaker->air->mic). This is the honest separation the live system sees —
    the acoustic chain can compress the clean-synth gaps measured by
    voice_separation_ab.py. A pair that drops below 0.30 here would be confused live.
  - Optional --levels: also sweep output volume per voice (distance/SNR proxy), to
    find the SNR at which voices start collapsing onto each other or onto an enrolled
    print.

Why not /api/announce: that channel speaks only with LT's production skeletor voice.
To play OTHER voices we synthesize locally and play the WAV through the default sink
(pw-play), driving volume with wpctl exactly like level_sweep_loopback.

Safety / no-contamination (same contract as level_sweep_loopback):
  - hearing muted the whole run (captures dropped pre-STT, no memory/conv writes);
    restored in finally.
  - default-sink volume read first and RESTORED in finally.
  - synthetic phrase only, no PII. Production config.PIPER_MODEL is never touched.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/multi_voice_sweep.py
  .venv/bin/python ops/multi_voice_sweep.py --voices en_US-ryan-high,en_US-amy-medium
  .venv/bin/python ops/multi_voice_sweep.py --levels "1.0,0.35,0.1"   # voice x level grid
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ops.snr_vs_distance import (  # noqa: E402
    TARGET_SR, _ensure_audio_env, set_hearing, find_input_device,
    highpass_80, to16k, record, rms, dbfs, speech_rms, save_wav,
)

VOICE_DIR = os.path.join(REPO, "models", "tts", "test_voices")
PHRASE = ("The quick brown fox jumps over the lazy dog. "
          "One two three four five six seven eight.")          # synthetic, no PII
SINK = "@DEFAULT_AUDIO_SINK@"


def _wpctl(*args):
    return subprocess.run(["wpctl", *args], capture_output=True, text=True, timeout=8)


def get_volume() -> float | None:
    try:
        m = re.search(r"Volume:\s*([0-9.]+)", _wpctl("get-volume", SINK).stdout)
        return float(m.group(1)) if m else None
    except Exception as e:
        print(f"  [vol] WARN get: {e}"); return None


def set_volume(v: float) -> None:
    try:
        _wpctl("set-volume", SINK, f"{v:.3f}")
    except Exception as e:
        print(f"  [vol] WARN set {v}: {e}")


def synth_wav(onnx_path: str, text: str, out_path: str) -> float:
    """Synthesize `text` to a 22.05 kHz WAV on disk (Piper native). Returns seconds."""
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    import wave
    voice = PiperVoice.load(onnx_path)
    sr = voice.config.sample_rate
    chunks = [c.audio_float_array
              for c in voice.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0))]
    a = np.concatenate(chunks).astype(np.float32)
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return len(a) / sr


def play_wav(path: str) -> None:
    """Blocking playback through the default sink (PipeWire)."""
    try:
        subprocess.run(["pw-play", path], capture_output=True, timeout=60)
    except Exception as e:
        print(f"  [play] WARN: {e}")


def capture_async(device, secs):
    box = {}
    t = threading.Thread(target=lambda: box.update(x=record(device, secs)), daemon=True)
    t.start()
    return t, box


def main():
    ap = argparse.ArgumentParser(description="Multi-voice speaker loopback sweep")
    ap.add_argument("--voices", default=None,
                    help="comma list of voice stems in test_voices/ (default: all)")
    ap.add_argument("--levels", default="1.0",
                    help="comma list of output-sink volumes per voice (default just 1.0)")
    ap.add_argument("--phrase", default=PHRASE)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "multi_voice_captures"))
    args = ap.parse_args()

    if args.voices:
        voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    else:
        voices = sorted(os.path.basename(p)[:-5]
                        for p in glob.glob(os.path.join(VOICE_DIR, "*.onnx")))
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    if not voices:
        print("No voices found in test_voices/."); return 1

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    from speaker.identifier import SpeakerIdentifier, KNOWN_SPEAKER_THRESHOLD
    si = SpeakerIdentifier()
    si.load_voiceprints()
    print(f"Enrolled: {', '.join(ks.name for ks in si._known_speakers)} "
          f"threshold={KNOWN_SPEAKER_THRESHOLD}")
    print(f"Voices ({len(voices)}): {', '.join(voices)}")
    print(f"Levels: {levels}")

    device = find_input_device(args.device)
    import sounddevice as sd
    dname = sd.query_devices(device)["name"] if device is not None else "(default)"
    print(f"Input device: [{device}] {dname}")

    # pre-synthesize each voice's WAV once
    wav_paths = {}
    for v in voices:
        wp = os.path.join(args.outdir, f"{stamp}_{v}_src.wav")
        dur = synth_wav(os.path.join(VOICE_DIR, v + ".onnx"), args.phrase, wp)
        wav_paths[v] = (wp, dur)

    orig_vol = get_volume()
    print(f"Default-sink volume (will restore): {orig_vol}\n")

    results = []
    captured_emb = {}   # (voice, level) -> embedding, for the separation matrix
    try:
        set_hearing(False)
        time.sleep(0.4)
        base = to16k(record(device, 2.5))
        floor = rms(highpass_80(base))
        print(f"baseline floor post-HP={dbfs(highpass_80(base)):.1f} dBFS\n")

        for v in voices:
            src_wav, dur = wav_paths[v]
            for lv in levels:
                set_volume(lv)
                time.sleep(0.3)
                rec_s = dur + 1.2
                t, box = capture_async(device, rec_s)
                time.sleep(0.4)
                play_wav(src_wav)
                t.join()
                cap = box.get("x")
                if cap is None or len(cap) == 0:
                    print(f"[{v} vol{lv}] WARN empty capture"); continue
                x16_hp = highpass_80(to16k(cap))
                sp = speech_rms(x16_hp)
                snr = 20.0 * np.log10(sp / (floor + 1e-12))
                emb = si.extract_embedding(x16_hp)
                captured_emb[(v, lv)] = emb
                ranked = sorted(((ks.distance(emb), ks.name) for ks in si._known_speakers),
                                key=lambda u: u[0])
                best = ranked[0] if ranked else (float("nan"), "none")
                second = ranked[1] if len(ranked) > 1 else (float("inf"), "none")
                would = best[0] < KNOWN_SPEAKER_THRESHOLD
                cap_wav = os.path.join(args.outdir, f"{stamp}_{v}_vol{lv}.wav")
                save_wav(cap_wav, x16_hp)
                row = dict(voice=v, volume=lv, snr_db=round(snr, 1),
                           peak_dbfs=round(dbfs(x16_hp), 1),
                           nearest=best[1], nearest_dist=round(best[0], 3),
                           second=second[1], second_dist=round(second[0], 3),
                           margin=round(second[0] - best[0], 3),
                           false_accept=would, cap_wav=cap_wav)
                results.append(row)
                print(f"[{v:20s} vol{lv:>4}] SNR={row['snr_db']:>5} dB | "
                      f"nearest={best[1]} @ {row['nearest_dist']} | "
                      f"{'FALSE-ACCEPT '+best[1] if would else 'unknown(>thr) OK'}")

        # ---- post-acoustic-chain separation matrix (captured embeddings) ----
        from scipy.spatial.distance import cosine
        print("\n==== CAPTURED-VOICE separation matrix (post speaker->air->mic) ====")
        for lv in levels:
            present = [v for v in voices if (v, lv) in captured_emb]
            if len(present) < 2:
                continue
            sn = lambda v: v.split("-")[1][:8]
            print(f"-- level {lv} (cosine dist; <0.30 = would be confused live) --")
            print("         " + " ".join(f"{sn(v):>8}" for v in present))
            tight = (None, 9.0)
            for vi in present:
                cells = []
                for vj in present:
                    if vi == vj:
                        cells.append(f"{'-':>8}")
                    else:
                        dd = float(cosine(captured_emb[(vi, lv)], captured_emb[(vj, lv)]))
                        cells.append(f"{dd:8.3f}")
                        if dd < tight[1]:
                            tight = ((vi, vj), dd)
                print(f"{sn(vi):8s} " + " ".join(cells))
            if tight[0]:
                print(f"  tightest pair: {sn(tight[0][0])}/{sn(tight[0][1])} @ {tight[1]:.3f}"
                      f"{'  <-- BELOW 0.30, confusable live' if tight[1] < 0.30 else ''}")

        out_json = os.path.join(args.outdir, f"{stamp}_summary.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary JSON: {out_json}")
        print("Interpretation:")
        print("  - false_accept=True means a synthetic voice landed on an enrolled print")
        print("    (e.g. the Erin noise-centroid / Dan basin) -> matcher would misID it.")
        print("  - tightest captured pair < 0.30 = those two synthetic speakers are not")
        print("    separable through the live chain (clean-synth gap got compressed).")
    finally:
        if orig_vol is not None:
            print(f"\n>>> Restoring sink volume to {orig_vol}")
            set_volume(orig_vol)
        print(">>> Restoring hearing")
        set_hearing(True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
