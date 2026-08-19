"""Render every subscriber-hype line through Piper and report duration +
internal pauses, so dramatic beats can be tuned by editing
prompts/subscriber_hype_lines.txt (no restart needed — lines re-read live).

Replicates tts.engine.speak()'s comma -> em-dash rewrite so the measured
audio is byte-identical to what the live path plays. Writes per-line wavs
plus a single concatenated review wav (1s gaps) for a one-listen audit.

Usage: .venv/bin/python ops/measure_subscriber_lines.py [outdir]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from conversation import subscriber_hype
from tts.engine import _synthesize_raw

SILENCE_THRESHOLD = 0.02  # RMS below this = silence (matches measure_piper_pauses)
MIN_SILENCE_MS = 60       # report only gaps that read as deliberate beats


def find_internal_pauses(audio, sr):
    window = int(sr * 0.01)
    total_ms = len(audio) / sr * 1000
    pauses, in_sil, start = [], False, 0
    for i in range(0, len(audio) - window, window):
        rms = float(np.sqrt(np.mean(audio[i:i + window] ** 2)))
        if rms < SILENCE_THRESHOLD:
            if not in_sil:
                in_sil, start = True, i
        elif in_sil:
            dur = (i - start) / sr * 1000
            s = start / sr * 1000
            if dur >= MIN_SILENCE_MS and s > 50 and s + dur < total_ms - 50:
                pauses.append((s, dur))
            in_sil = False
    return pauses


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/subscriber_hype_renders")
    outdir.mkdir(parents=True, exist_ok=True)
    lines = subscriber_hype._load_lines()
    print(f"model={config.PIPER_MODEL} length_scale={config.TTS_LENGTH_SCALE}")
    print(f"{len(lines)} lines -> {outdir}\n")

    import wave

    def write_wav(path, audio, sr):
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())

    clips, sr = [], None
    for i, line in enumerate(lines, 1):
        spoken = line.replace(",", " —")  # tts.engine.speak() rewrite
        audio, sr = _synthesize_raw(spoken, config.PIPER_MODEL)
        dur_s = len(audio) / sr
        pauses = find_internal_pauses(audio, sr)
        pstr = ", ".join(f"{d:.0f}ms" for _, d in pauses) or "-"
        print(f"{i:2d}. {dur_s:5.2f}s  pauses[{pstr:>20s}]  {line}")
        write_wav(outdir / f"{i:02d}.wav", audio, sr)
        clips.append(audio)

    gap = np.zeros(sr, dtype=np.float32)
    review = np.concatenate([np.concatenate([c, gap]) for c in clips])
    write_wav(outdir / "review_all.wav", review, sr)
    print(f"\nreview wav: {outdir / 'review_all.wav'} "
          f"({len(review) / sr:.0f}s total)")


if __name__ == "__main__":
    main()
