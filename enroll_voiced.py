"""Hands-free, voice-coached K-prototype enrollment.

Drives the whole enrollment through Timmy's speaker so the subject never has to
watch a screen — used from Supervisor Mode. For each clip it:

  1. mutes LT's hearing (POST /api/hearing enabled=false) so LT won't transcribe
     or respond to the subject mid-enrollment, and so Timmy's own voice can't
     bleed into the captured audio (we still wait out each spoken cue),
  2. speaks the pose cue through Timmy (POST /api/announce),
  3. waits for the cue to finish playing (timed lead — there is no TTS-done
     signal), then records one clip directly off device 7 (the same
     device/SR/resample/HP chain LT uses live),
  4. verifies the clip carries speech (peak gate) and retries once if not.

Then it builds a deduped (K, D) prototype set with the runtime's own
``_build_prototypes`` and saves it via ``persist_voiceprint`` (atomic, backs up
the old file), so enrollment and live matching can never diverge. Hearing is
ALWAYS restored in a finally block. Restart LT afterwards to load the result.

Usage:
  XDG_RUNTIME_DIR=/run/user/1000 PULSE_SERVER=unix:/run/user/1000/pulse/native \
    ./.venv/bin/python enroll_voiced.py dan
"""

import json
import os
import sys
import time
import urllib.request

# PipeWire/PortAudio need these to enumerate device 7 from a non-systemd shell.
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("PULSE_SERVER", "unix:/run/user/1000/pulse/native")

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, resample_poly
from scipy.spatial.distance import cosine

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")
from speaker.identifier import (
    SpeakerIdentifier,
    KNOWN_SPEAKER_THRESHOLD,
    _build_prototypes,
)

LT = "http://localhost:8893"
NATIVE_SR, TARGET_SR, CHANNELS, DEVICE = 48000, 16000, 2, 7
ANNOUNCE_PREFIX_CHARS = 24      # server prepends "This is Claude talking. "
SPEECH_CPS = 12.0               # rough chars/sec for lead timing (conservative)
LEAD_BUFFER_S = 2.0             # extra slack so Timmy fully finishes before we record
MIN_PEAK = 0.025                # below this the clip is silence/too-far → retry
CLIP_SECONDS = 4

POSES = [
    "normal voice, close to the mic",
    "now step back a few feet",
    "quieter, casual, like across the room",
    "now your loud party voice",
    "turned away from the mic, over your shoulder",
    "off to the side of the mic",
]


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        LT + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b"{}")


def announce(text: str) -> None:
    try:
        _post("/api/announce", {"text": text})
    except Exception as e:
        print(f"  ! announce failed: {e}")
    # Wait out the spoken cue (incl. the server's "This is Claude talking." prefix)
    # so the recording that follows captures the subject, not Timmy.
    lead = (ANNOUNCE_PREFIX_CHARS + len(text)) / SPEECH_CPS + LEAD_BUFFER_S
    time.sleep(lead)


def set_hearing(enabled: bool) -> None:
    try:
        r = _post("/api/hearing", {"enabled": enabled})
        print(f"  hearing -> {r}")
    except Exception as e:
        print(f"  ! set_hearing({enabled}) failed: {e}")


def _resolve_input_index() -> int:
    """Re-resolve the PipeWire input device BY NAME each call.

    Timmy's TTS playback churns the PipeWire graph and PortAudio renumbers
    devices, so a cached numeric index (e.g. 7) goes stale mid-run. Refresh
    PortAudio's device list and match by name instead.
    """
    sd._terminate()
    sd._initialize()
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and "pipewire" in d["name"]:
            return i
    for i, d in enumerate(devs):           # fallback: the "default" source
        if d["max_input_channels"] > 0 and d["name"] == "default":
            return i
    raise RuntimeError("no PipeWire input device found")


def capture_16k(seconds: int) -> np.ndarray:
    """Record one clip through LT's exact capture chain (no countdown).

    Retries with a fresh device resolution if PortAudio errors (stale index
    after a TTS-driven graph change)."""
    audio = None
    last_err = None
    for attempt in range(3):
        try:
            idx = _resolve_input_index()
            audio = sd.rec(int(seconds * NATIVE_SR), samplerate=NATIVE_SR,
                           channels=CHANNELS, dtype="float32", device=idx)
            sd.wait()
            break
        except Exception as e:
            last_err = e
            print(f"  ! capture attempt {attempt + 1} failed ({e}); refreshing")
            time.sleep(0.5)
    if audio is None:
        raise last_err
    if audio.ndim == 2:
        audio = audio[:, 0]
    a16 = resample_poly(audio, up=1, down=3).astype(np.float32)
    hp_b, hp_a = butter(2, 80, btype="high", fs=TARGET_SR)
    a16 = lfilter(hp_b, hp_a, a16).astype(np.float32)
    return a16[int(TARGET_SR * 0.1):]   # drop filter transient


def embed(a16: np.ndarray) -> np.ndarray:
    """Embed via the production WeSpeaker backend — identical to live identify()."""
    from speaker import encoder as _enc
    return _enc.extract_embedding(a16)


def main() -> int:
    name = (sys.argv[1] if len(sys.argv) > 1 else "dan").strip().lower()
    n_clips = len(POSES)
    print(f"=== voiced enrollment: {name} ({n_clips} clips) ===")

    from speaker import encoder as _enc
    print("loading WeSpeaker encoder...")
    _enc.get_inference()

    raw_embs: list[np.ndarray] = []
    try:
        set_hearing(False)   # LT stops responding; capture stays clean
        announce("Okay. I have muted Timmy's listening so it will not answer you. "
                 f"We will do {n_clips} short clips. Here we go.")
        for i, pose in enumerate(POSES):
            lead = "Good. " if i else ""
            for attempt in range(2):
                redo = "Let us try that one again. " if attempt else ""
                announce(f"{lead}{redo}Clip {i + 1} of {n_clips}. {pose}. "
                         "Start talking now and keep going.")
                a16 = capture_16k(CLIP_SECONDS)
                peak = float(np.max(np.abs(a16)))
                rms = float(np.sqrt(np.mean(a16 ** 2)))
                print(f"  clip {i + 1} (try {attempt + 1}): peak={peak:.4f} rms={rms:.5f} "
                      f"samples={len(a16)}")
                if peak >= MIN_PEAK:
                    try:
                        raw_embs.append(embed(a16))
                    except Exception as e:
                        print(f"  ! embed failed: {e}")
                    break
                print("  -> too quiet, retrying" if attempt == 0 else "  -> still quiet, keeping anyway")
        announce("Perfect, that is all of them. One moment while I save your voiceprint.")
    finally:
        set_hearing(True)    # ALWAYS restore listening

    if not raw_embs:
        print("no usable clips — nothing saved")
        announce("Something went wrong, I did not capture any clips. We will retry.")
        return 1

    protos = _build_prototypes(raw_embs)
    print(f"\nbuilt {protos.shape[0]} prototype(s) from {len(raw_embs)} clips")
    print(f"per-clip min-distance to kept set (live threshold {KNOWN_SPEAKER_THRESHOLD:.2f}):")
    misses = 0
    for i, e in enumerate(raw_embs):
        en = e / np.linalg.norm(e)
        d = min(float(cosine(en, p)) for p in protos)
        miss = d >= KNOWN_SPEAKER_THRESHOLD
        misses += miss
        print(f"  clip {i + 1}: {d:.3f}  {'MISS' if miss else 'ok'}")
    if protos.shape[0] > 1:
        spread = [float(cosine(protos[a], protos[b]))
                  for a in range(len(protos)) for b in range(a + 1, len(protos))]
        print(f"prototype spread: min={min(spread):.3f} max={max(spread):.3f} "
              f"mean={sum(spread) / len(spread):.3f}")

    out = SpeakerIdentifier().persist_voiceprint(name, protos)
    print(f"saved {protos.shape[0]} prototype(s) -> {out}")

    # Auto-create the postgres speakers row so this voiceprint can't FK-fail a
    # facts/memories insert before the next restart (db/speakers.py).
    try:
        from db.speakers import ensure_rows_for_enrolled
        n = ensure_rows_for_enrolled()
        print(f"synced speakers table ({n} new row(s))")
    except Exception as e:
        print(f"WARNING: could not sync speakers table ({e}); "
              "a restart will reconcile it.")

    print(f"RESULT misses={misses}/{len(raw_embs)} k={protos.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
