#!/usr/bin/env python3
"""Make the Couples Therapist persona speak through Timmy's speaker.

The therapist persona has its OWN voice (en_US-kristin-medium, chosen 2026-06-28
for maximum WeSpeaker voiceprint distance from Dan -- see
models/tts/personas/couples_therapist.json) so it is audibly NOT Timmy and NOT
Dan. This synthesizes the given text in that voice and plays it out Timmy's
default PipeWire sink.

Contrast with POST /api/announce: announce reuses Timmy's (skeletor) voice and
sets capture.suppressed so Timmy never hears it. This helper uses a DIFFERENT
voice and plays UN-GATED (pw-play), so Timmy CAN hear it -- which is what makes
the voice eligible for speaker-ID once the persona's voiceprint is enrolled
(pipeline-matched; not yet done). If you want the therapist to talk TO Dan
without Timmy reacting, gate first (see --quiet-timmy note below) or route
through a future announce voice param instead.

Usage:
  ./.venv/bin/python ops/therapist_say.py "Let's talk about what just happened."
  ./.venv/bin/python ops/therapist_say.py --length-scale 1.05 "Take a breath."
  ./.venv/bin/python ops/therapist_say.py --save /tmp/x.wav "..."   # synth only, no play
"""
import argparse
import os
import subprocess
import sys
import wave

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ONNX = os.path.join(REPO, "models", "tts", "personas", "couples_therapist.onnx")


def synth(text: str, length_scale: float = 1.0) -> tuple[np.ndarray, int]:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    v = PiperVoice.load(ONNX)
    sr = v.config.sample_rate
    chunks = [c.audio_float_array for c in
              v.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale))]
    return np.concatenate(chunks).astype(np.float32), sr


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", help="what the couples therapist should say")
    ap.add_argument("--length-scale", type=float, default=1.0,
                    help="Piper speaking-rate (>1 slower, <1 faster)")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="write WAV here and do NOT play (synth-only)")
    args = ap.parse_args()

    audio, sr = synth(args.text, args.length_scale)
    out = args.save or "/tmp/couples_therapist_say.wav"
    write_wav(out, audio, sr)
    print(f"[therapist] synth {len(audio)/sr:.1f}s @ {sr}Hz -> {out}")
    if args.save:
        return 0
    # Un-gated playback out Timmy's speaker. Needs XDG_RUNTIME_DIR + PULSE_SERVER
    # in the environment (the little-timmy.service unit sets both for uid 1000).
    subprocess.run(["pw-play", out], check=False, timeout=120)
    print("[therapist] played through Timmy's speaker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
