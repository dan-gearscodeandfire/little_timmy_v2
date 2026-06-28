#!/usr/bin/env python3
"""Pipeline-matched voiceprint enrollment for a SYNTHETIC persona voice
(couples-therapist), driven THROUGH the running little-timmy service.

Why via the service: on this box the analog speaker only physically emits while
the service holds its output stream, and a second capture client doesn't get the
mic. So we reuse the exact mechanism enroll_voiced.py uses for humans:
  - keep the service RUNNING (speaker emits),
  - POST /api/hearing {enabled:false} to RELEASE the service's mic capture
    (so our own sd.rec gets it; ALWAYS restored in finally),
  - drive playback with POST /api/announce voice=couples_therapist — the SAME
    persona .onnx + TTS output + speaker the persona uses at runtime,
  - capture that playback off the (re-resolved) pipewire device through Timmy's
    exact chain (48k stereo -> ch0 -> ::3 -> 16k -> 80Hz HP),
  - build a deduped prototype set with the runtime's own _build_prototypes and
    save via persist_voiceprint.
This makes the enrolled print match what the service captures when the persona
speaks for real. Restart the service afterwards to load it.

Usage (service must be RUNNING):
  XDG_RUNTIME_DIR=/run/user/1000 PULSE_SERVER=unix:/run/user/1000/pulse/native \
    ./.venv/bin/python ops/enroll_persona_voice.py couples_therapist [--clips 6] [--dry-run]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("PULSE_SERVER", "unix:/run/user/1000/pulse/native")

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, resample_poly
from scipy.spatial.distance import cosine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

LT = "http://localhost:8893"
NATIVE_SR, TARGET_SR, CHANNELS = 48000, 16000, 2
CAPTURE_SECONDS = 6            # window per clip (covers synth latency + playback)
MIN_PEAK = 0.02               # speaker->mic level is modest; below this = miss

SENTENCES = [
    "Let's take a breath and talk about what just happened between you two.",
    "I'm here to help you both understand each other, calmly and without judgment.",
    "Tell me how that made you feel, and try to use your own words.",
    "It sounds like there's some tension here that we should work through together.",
    "Remember, the goal is to listen as much as you speak.",
    "Thank you for sharing that. Let's see if we can find some common ground.",
]


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(LT + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b"{}")


def set_hearing(enabled: bool) -> None:
    try:
        print(f"  hearing -> {_post('/api/hearing', {'enabled': enabled})}")
    except Exception as e:
        print(f"  ! set_hearing({enabled}) failed: {e}")


def _resolve_input_index() -> int:
    """Re-resolve the pipewire input BY NAME (TTS churns the graph → indices go
    stale). Mirrors enroll_voiced._resolve_input_index."""
    sd._terminate(); sd._initialize()
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and "pipewire" in d["name"]:
            return i
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and d["name"] == "default":
            return i
    raise RuntimeError("no pipewire input device found")


def _to_pipeline_16k(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio[:, 0]
    a16 = resample_poly(audio, up=1, down=3).astype(np.float32)
    hp_b, hp_a = butter(2, 80, btype="high", fs=TARGET_SR)
    a16 = lfilter(hp_b, hp_a, a16).astype(np.float32)
    return a16[int(TARGET_SR * 0.1):]


def _trim_active(a16: np.ndarray, floor: float = 0.02, pad_s: float = 0.15) -> np.ndarray:
    """Trim to the region above `floor` (speech), with a small pad. Keeps the
    embedding from being diluted by the leading/trailing room ambience in the
    fixed capture window."""
    idx = np.where(np.abs(a16) > floor)[0]
    if len(idx) == 0:
        return a16
    pad = int(TARGET_SR * pad_s)
    lo = max(0, idx[0] - pad)
    hi = min(len(a16), idx[-1] + pad)
    return a16[lo:hi]


class PersistentCapturer:
    """One InputStream kept open for the whole enrollment. Keeping the pipewire
    source live is what stops it suspending between clips (a per-call sd.rec
    captured only ambient on this box)."""
    def __init__(self):
        idx = _resolve_input_index()
        self.frames: list = []
        self.stream = sd.InputStream(device=idx, channels=CHANNELS, samplerate=NATIVE_SR,
                                     dtype="float32", callback=lambda d, n, t, s: self.frames.append(d.copy()))

    def __enter__(self):
        self.stream.start(); time.sleep(0.8); return self

    def __exit__(self, *a):
        self.stream.stop(); self.stream.close()

    def _count(self) -> int:
        return sum(len(f) for f in self.frames)

    def announce_and_capture(self, text: str, seconds: int) -> np.ndarray:
        start = self._count()
        _post("/api/announce", {"text": text, "voice": "couples_therapist", "no_prefix": True})
        time.sleep(seconds)
        allf = np.concatenate(self.frames) if self.frames else np.zeros((0, CHANNELS), np.float32)
        seg = allf[start:]
        return _trim_active(_to_pipeline_16k(seg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", nargs="?", default="couples_therapist")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from speaker import encoder as _enc
    from speaker.identifier import SpeakerIdentifier, KNOWN_SPEAKER_THRESHOLD, _build_prototypes
    print("Loading WeSpeaker encoder...")
    _enc.get_inference()

    dan = np.load(os.path.join(REPO, "models", "speaker", "dan_wespeaker.npy"))
    dan = dan if dan.ndim == 2 else dan[None, :]

    raw_embs = []
    try:
        set_hearing(False)          # release the service mic; speaker stays live
        time.sleep(0.5)
        with PersistentCapturer() as cap:
            for i in range(args.clips):
                text = SENTENCES[i % len(SENTENCES)]
                print(f"\n--- Clip {i+1}/{args.clips}: {text[:48]}... ---")
                a16 = cap.announce_and_capture(text, CAPTURE_SECONDS)
                peak = float(np.max(np.abs(a16))) if len(a16) else 0.0
                rms = float(np.sqrt(np.mean(a16**2))) if len(a16) else 0.0
                print(f"  captured {len(a16)/TARGET_SR:.1f}s peak={peak:.4f} rms={rms:.5f}")
                if peak >= MIN_PEAK and len(a16) > TARGET_SR:   # >=1s of real speech
                    raw_embs.append(_enc.extract_embedding(a16))
                else:
                    print("  -> too quiet/short, skipping")
                time.sleep(0.3)
    finally:
        set_hearing(True)           # ALWAYS restore

    if not raw_embs:
        print("\nNo usable clips — aborting.")
        return 1

    protos = _build_prototypes(raw_embs)
    print(f"\n=== Built {protos.shape[0]} prototype(s) from {len(raw_embs)} clips ===")
    worst = 0.0
    print(f"Per-clip self-distance (live thresh {KNOWN_SPEAKER_THRESHOLD:.2f}):")
    for i, e in enumerate(raw_embs):
        en = e / np.linalg.norm(e)
        d = min(float(cosine(en, p)) for p in protos)
        worst = max(worst, d)
        print(f"  clip {i+1}: {d:.3f}  {'ok' if d < KNOWN_SPEAKER_THRESHOLD else 'MISS'}")
    dan_dists = [min(float(cosine(p, dp)) for dp in dan) for p in protos]
    print(f"Prototype distance to DAN: min={min(dan_dists):.3f} (must be > {KNOWN_SPEAKER_THRESHOLD:.2f})")
    print(f"Separation margin (dan_min - worst_self): {min(dan_dists)-worst:.3f}")

    if args.dry_run:
        print("\n--dry-run: nothing saved.")
        return 0

    out = SpeakerIdentifier().persist_voiceprint(args.name, protos)
    print(f"\nSaved {protos.shape[0]} prototype(s) -> {out}")
    try:
        from db.speakers import ensure_rows_for_enrolled
        print(f"Synced speakers table ({ensure_rows_for_enrolled()} new row(s)).")
    except Exception as e:
        print(f"NOTE: speakers sync deferred to restart ({e}).")
    print("Restart little-timmy.service to load the new voiceprint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
