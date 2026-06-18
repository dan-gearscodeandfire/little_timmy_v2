#!/usr/bin/env python3
"""SNR-vs-distance capture for the Little Timmy lav-mic chain.

PURPOSE
  Measure how fast a real voice degrades with distance from the current
  (wireless lav -> receiver -> 3.5mm -> SN6186 analog) capture chain, and
  what that does to the WeSpeaker speaker-ID distance. This is the
  misID-relevant property: a guest who is OFF-mic (not wearing the lav)
  reaches the mic at low SNR; their embedding rots toward the noise-shaped
  centroid and can collapse under KNOWN_SPEAKER_THRESHOLD (0.30) onto an
  enrolled voiceprint. This script quantifies that rot curve.

WHY A SCRIPT (vs the remote speaker-playback test)
  Playing tones out Timmy's speaker under-couples to a CLOSE-TALK lav, so it
  can't measure real-voice SNR. You have to speak into the lav at varied
  distances. Run this in the shop.

WHAT IT DOES
  1. Mutes Timmy's HEARING via POST :8893/api/hearing (segments are dropped
     pre-STT, so nothing you say here reaches his conversation/memory).
     ALWAYS restored on exit (finally), even on Ctrl-C / error.
  2. Records a room-tone baseline (you silent).
  3. For each distance you give, prompts you to speak a fixed phrase, captures
     via a read-loop InputStream (reliable; sd.rec + concurrent audio leaves
     uninitialised garbage in the buffer), runs the EXACT LT capture pipeline
     (48k -> decimate ::3 -> 16k -> 80 Hz Butterworth HP, matching
     audio/capture.py), then the WeSpeaker embedding.
  4. Reports per distance: post-HP SNR (dB over baseline), nearest known
     speaker + distance, 2nd-nearest + MARGIN (the P1 quantity), and the
     distance specifically to your enrolled 'dan' prototype set.
  5. Saves each capture as a 16 kHz WAV under ops/snr_captures/ for re-analysis.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/snr_vs_distance.py
  # custom distances + phrase:
  .venv/bin/python ops/snr_vs_distance.py --distances "6in,3ft,8ft,15ft" \
      --phrase "the quick brown fox jumps" --dur 4

  Use a SYNTHETIC phrase, never real auto-memory / PII values.

NOTES
  - Timmy (PID main.py) keeps the mic open; this opens a 2nd PipeWire reader,
    which is fine (proven). Hearing-mute is the safety, not exclusivity.
  - 'dan' must already be enrolled (models/speaker/dan_wespeaker.npy).
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

# --- repo on path so we reuse LT's own identifier (pipeline-matched) ---------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SR = 48000            # native capture rate (PipeWire/SN6186)
TARGET_SR = 16000     # LT working rate
DECIMATE = SR // TARGET_SR   # ::3, matches audio/capture.py _downsample
ORCH = "http://127.0.0.1:8893"
HP_CUTOFF = 80        # matches audio/capture.py _highpass


# --- ensure the user PipeWire/Pulse session env is present -------------------
def _ensure_audio_env():
    uid = os.getuid()
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    os.environ.setdefault("PULSE_SERVER", f"unix:/run/user/{uid}/pulse/native")


def set_hearing(enabled: bool) -> bool:
    """Flip Timmy's hearing via the orchestrator API. Returns True on success."""
    try:
        req = urllib.request.Request(
            f"{ORCH}/api/hearing",
            data=json.dumps({"enabled": bool(enabled)}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            out = json.load(r)
        print(f"  [hearing] enabled={out.get('enabled')} muted={out.get('muted')}")
        return True
    except Exception as e:
        print(f"  [hearing] WARN: could not set hearing={enabled}: {e}")
        return False


def find_input_device(hint: str | None):
    """Resolve the lav input device. Indices churn (PipeWire graph), so prefer
    by NAME: the 'default' PipeWire source routes to SN6186 analog capture.
    Returns a sounddevice device index or None (= library default)."""
    import sounddevice as sd
    if hint is not None:
        try:
            return int(hint)
        except ValueError:
            pass
    want = (hint or "default").lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["name"].lower() == want:
            return i
    # fallback: first input that mentions pipewire/default, else library default
    for i, d in enumerate(sd.query_devices()):
        n = d["name"].lower()
        if d["max_input_channels"] > 0 and ("pipewire" in n or "default" in n):
            return i
    return None


_HP = {"b": None, "a": None}


def highpass_80(x: np.ndarray, fs: int = TARGET_SR) -> np.ndarray:
    """2nd-order Butterworth high-pass at 80 Hz — identical to audio/capture.py."""
    from scipy.signal import butter, lfilter
    if _HP["b"] is None:
        _HP["b"], _HP["a"] = butter(2, HP_CUTOFF, btype="high", fs=fs)
    return lfilter(_HP["b"], _HP["a"], x)


def to16k(x: np.ndarray) -> np.ndarray:
    return x[::DECIMATE]


def record(device, secs: float) -> np.ndarray:
    """Read-loop capture (ch0) for `secs`. Only frames actually delivered are
    kept, so no uninitialised-buffer garbage. Returns float64 @ 48k."""
    import sounddevice as sd
    frames = []
    bs = 2048
    with sd.InputStream(samplerate=SR, channels=2, dtype="float32",
                        device=device, blocksize=bs) as s:
        end = time.time() + secs
        while time.time() < end:
            data, _ = s.read(bs)
            frames.append(data[:, 0].copy())
    x = np.concatenate(frames).astype(np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def dbfs(x: np.ndarray) -> float:
    return 20.0 * np.log10(rms(x))


def speech_rms(x16_hp: np.ndarray) -> float:
    """RMS of the loudest ~1s window — approximates the spoken portion,
    ignoring leading/trailing silence."""
    win = TARGET_SR
    if len(x16_hp) <= win:
        return rms(x16_hp)
    best = 0.0
    for i in range(0, len(x16_hp) - win, win // 2):
        r = rms(x16_hp[i:i + win])
        best = max(best, r)
    return best


def save_wav(path: str, x16: np.ndarray):
    try:
        from scipy.io import wavfile
        clipped = np.clip(x16, -1.0, 1.0).astype(np.float32)
        wavfile.write(path, TARGET_SR, clipped)
    except Exception as e:
        print(f"  [wav] WARN: save failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="SNR-vs-distance lav-mic capture")
    ap.add_argument("--distances", default="6in,3ft,8ft",
                    help="comma list of distance labels to test (default 6in,3ft,8ft)")
    ap.add_argument("--phrase", default="the quick brown fox jumps over the lazy dog",
                    help="SYNTHETIC phrase to read each time (no PII)")
    ap.add_argument("--dur", type=float, default=4.0, help="capture seconds per utterance")
    ap.add_argument("--device", default=None, help="input device index or name (default: PipeWire 'default')")
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "snr_captures"))
    args = ap.parse_args()

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    distances = [d.strip() for d in args.distances.split(",") if d.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # Reuse LT's identifier so embeddings + distances match production exactly.
    from speaker.identifier import SpeakerIdentifier, KNOWN_SPEAKER_THRESHOLD
    si = SpeakerIdentifier()
    si.load_voiceprints()
    enrolled = [ks.name for ks in si._known_speakers]
    if "dan" not in enrolled:
        print("WARN: 'dan' not enrolled; will still report nearest known.")
    print(f"Enrolled: {', '.join(enrolled) or 'none'}  threshold={KNOWN_SPEAKER_THRESHOLD}\n")

    device = find_input_device(args.device)
    import sounddevice as sd
    dname = sd.query_devices(device)["name"] if device is not None else "(library default)"
    print(f"Input device: [{device}] {dname}\n")

    results = []
    try:
        print(">>> Muting Timmy's hearing (test audio will NOT reach his memory)")
        set_hearing(False)

        input("\nBaseline: stay SILENT, press Enter to capture ~2s room tone...")
        base = to16k(record(device, 2.0))
        base_hp = highpass_80(base)
        base_floor = rms(base_hp)
        print(f"  baseline floor: raw={dbfs(base):.1f} dBFS  post-HP={dbfs(base_hp):.1f} dBFS")

        for dist in distances:
            input(f"\n[{dist}] place lav at ~{dist}, press Enter, then say: \"{args.phrase}\"")
            cap = record(device, args.dur)
            x16 = to16k(cap)
            x16_hp = highpass_80(x16)
            sp = speech_rms(x16_hp)
            snr = 20.0 * np.log10(sp / base_floor)

            emb = si.extract_embedding(x16_hp)
            ranked = sorted(((ks.distance(emb), ks.name) for ks in si._known_speakers),
                            key=lambda t: t[0])
            best = ranked[0] if ranked else (float("nan"), "none")
            second = ranked[1] if len(ranked) > 1 else (float("inf"), "none")
            margin = second[0] - best[0]
            dan_dist = next((d for d, n in ranked if n == "dan"), float("nan"))
            would_match = best[0] < KNOWN_SPEAKER_THRESHOLD

            wav = os.path.join(args.outdir, f"{stamp}_{dist}.wav")
            save_wav(wav, x16_hp)

            row = dict(distance=dist, snr_db=round(snr, 1),
                       speech_dbfs=round(dbfs(x16_hp), 1),
                       best=best[1], best_dist=round(best[0], 3),
                       second=second[1], second_dist=round(second[0], 3),
                       margin=round(margin, 3), dan_dist=round(dan_dist, 3),
                       matches_known=would_match, wav=wav)
            results.append(row)
            print(f"  SNR={row['snr_db']:>5} dB | nearest={row['best']} @ {row['best_dist']} "
                  f"| 2nd={row['second']} @ {row['second_dist']} | margin={row['margin']} "
                  f"| dan={row['dan_dist']} | {'MATCH' if would_match else 'unknown(>thr)'}")

        print("\n==== SUMMARY (watch best_dist rise + margin collapse with distance) ====")
        print(json.dumps(results, indent=2))
        print("\nInterpretation:")
        print("  - best_dist climbing toward/over 0.30 with distance = the rot curve.")
        print("  - margin shrinking = the P1 collapse window (off-mic guest lands on enrolled).")
        print(f"  - WAVs saved under {args.outdir} for re-analysis.")
    finally:
        print("\n>>> Restoring Timmy's hearing")
        set_hearing(True)


if __name__ == "__main__":
    main()
