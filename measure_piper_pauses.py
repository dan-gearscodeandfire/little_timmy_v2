"""Measure Piper TTS pause durations for various punctuation marks.

Synthesizes test sentences with specific punctuation, then analyzes
the waveform to find silence gaps and their durations.
"""

import sys
sys.path.insert(0, "/home/gearscodeandfire/little_timmy")

import numpy as np
import time

# Test sentences designed to isolate punctuation pauses
TEST_CASES = [
    ("comma", "Hello, world"),
    ("period", "Hello. World"),
    ("exclamation", "Hello! World"),
    ("question", "Hello? World"),
    ("semicolon", "Hello; world"),
    ("colon", "Hello: world"),
    ("dash", "Hello - world"),
    ("em_dash", "Hello — world"),
    ("ellipsis", "Hello... world"),
    ("no_punct", "Hello world"),
]

SILENCE_THRESHOLD = 0.02  # RMS below this = silence
MIN_SILENCE_MS = 20       # ignore gaps shorter than this


def find_silences(audio, sr, threshold=SILENCE_THRESHOLD, min_ms=MIN_SILENCE_MS):
    """Find silence gaps in audio. Returns list of (start_ms, duration_ms)."""
    window_size = int(sr * 0.01)  # 10ms windows
    silences = []
    in_silence = False
    silence_start = 0

    for i in range(0, len(audio) - window_size, window_size):
        chunk = audio[i:i + window_size]
        rms = np.sqrt(np.mean(chunk ** 2))

        if rms < threshold:
            if not in_silence:
                in_silence = True
                silence_start = i
        else:
            if in_silence:
                duration_ms = (i - silence_start) / sr * 1000
                if duration_ms >= min_ms:
                    start_ms = silence_start / sr * 1000
                    silences.append((start_ms, duration_ms))
                in_silence = False

    # Handle trailing silence
    if in_silence:
        duration_ms = (len(audio) - silence_start) / sr * 1000
        if duration_ms >= min_ms:
            start_ms = silence_start / sr * 1000
            silences.append((start_ms, duration_ms))

    return silences


def main():
    from tts.engine import _synthesize_raw
    import config

    print(f"{'='*60}")
    print("PIPER PUNCTUATION PAUSE MEASUREMENT")
    print(f"{'='*60}")
    print(f"Model: {config.PIPER_MODEL}")
    print(f"Length scale: {config.TTS_LENGTH_SCALE}")
    print(f"Silence threshold: {SILENCE_THRESHOLD} RMS")
    print()

    results = {}

    for label, text in TEST_CASES:
        audio, sr = _synthesize_raw(text, config.PIPER_MODEL)
        total_ms = len(audio) / sr * 1000

        silences = find_silences(audio, sr)

        # Filter out leading/trailing silence (only want internal pauses)
        # "Internal" = not in first 50ms or last 50ms
        internal = [
            (s, d) for s, d in silences
            if s > 50 and s + d < total_ms - 50
        ]

        max_pause = max((d for _, d in internal), default=0)
        total_silence = sum(d for _, d in internal)

        results[label] = {
            "text": text,
            "total_ms": total_ms,
            "internal_silences": internal,
            "max_pause_ms": max_pause,
            "total_internal_silence_ms": total_silence,
        }

        pause_str = ", ".join(f"{d:.0f}ms@{s:.0f}" for s, d in internal) if internal else "none"
        print(f"  {label:12s} | {text:25s} | total={total_ms:6.0f}ms | "
              f"max_pause={max_pause:5.0f}ms | pauses: {pause_str}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    baseline = results["no_punct"]["max_pause_ms"]
    print(f"  {'no_punct':12s} (baseline): {baseline:.0f}ms")

    for label in ["comma", "period", "exclamation", "question", "semicolon",
                   "colon", "dash", "em_dash", "ellipsis"]:
        r = results[label]
        delta = r["max_pause_ms"] - baseline
        print(f"  {label:12s}: {r['max_pause_ms']:5.0f}ms (delta: {delta:+.0f}ms)")


if __name__ == "__main__":
    main()
