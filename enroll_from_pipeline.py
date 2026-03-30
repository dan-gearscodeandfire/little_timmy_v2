"""Re-enroll speaker voiceprint using Little Timmy's actual audio capture path.

This captures audio through the EXACT same sounddevice stream config
that Timmy uses (device 7, 48kHz native, stereo ch0, resampled to 16kHz),
so the voiceprint matches live capture characteristics.

Usage:
  python enroll_from_pipeline.py [name] [seconds]
  python enroll_from_pipeline.py dan 15
"""

import sys
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")

# Match Timmy's exact audio config
NATIVE_SR = 48000
TARGET_SR = 16000
CHANNELS = 2
DEVICE = 7  # PipeWire default


def record_like_timmy(seconds: int = 15) -> np.ndarray:
    """Record audio using Timmy's exact stream configuration."""
    # Match the chunk size Timmy uses
    chunk_frames = int(NATIVE_SR * 256 / 1000)  # ~256ms chunks at native rate

    print(f"\n>>> Recording {seconds}s through Timmy's audio path <<<")
    print(f"    Device: {DEVICE}, Native SR: {NATIVE_SR}, Channels: {CHANNELS}")
    print(f"    Will resample {NATIVE_SR} → {TARGET_SR} Hz")
    print()

    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    print("  RECORDING — speak naturally!")

    # Record at native sample rate, stereo, just like Timmy
    audio = sd.rec(
        int(seconds * NATIVE_SR),
        samplerate=NATIVE_SR,
        channels=CHANNELS,
        dtype="float32",
        device=DEVICE,
    )
    sd.wait()
    print("  Done recording.\n")

    # Take channel 0 (mono) — same as Timmy
    if audio.ndim == 2:
        audio = audio[:, 0]

    # Resample 48kHz → 16kHz — same as Timmy's pipeline
    audio_16k = resample_poly(audio, up=1, down=3).astype(np.float32)

    # 80Hz high-pass filter — matches Timmy's capture pipeline
    # Removes DC offset (0.014 on ch0) and 60Hz mains hum
    from scipy.signal import butter, lfilter
    hp_b, hp_a = butter(2, 80, btype="high", fs=TARGET_SR)
    audio_16k = lfilter(hp_b, hp_a, audio_16k).astype(np.float32)
    # Skip first 100ms to avoid filter transient spike
    audio_16k = audio_16k[int(TARGET_SR * 0.1):]

    peak = np.max(np.abs(audio_16k))
    rms = np.sqrt(np.mean(audio_16k ** 2))
    print(f"  Audio (16kHz): peak={peak:.4f}, RMS={rms:.6f}, samples={len(audio_16k)}")
    print(f"  Duration: {len(audio_16k)/TARGET_SR:.1f}s")

    if peak < 0.01:
        print("  WARNING: Very low audio — mic may not be capturing")
    if peak > 0.95:
        print("  WARNING: Audio clipping detected")

    return audio_16k


def create_voiceprint(audio_16k: np.ndarray) -> np.ndarray:
    """Create Resemblyzer embedding using Timmy's exact preprocessing."""
    from resemblyzer import VoiceEncoder, preprocess_wav

    print("  Loading Resemblyzer encoder...")
    encoder = VoiceEncoder("cpu")

    # Use preprocess_wav like Timmy's identifier.py does
    processed = preprocess_wav(audio_16k, source_sr=16000)
    print(f"  After VAD preprocessing: {len(processed)} samples ({len(processed)/16000:.1f}s)")

    if len(processed) < 8000:
        print("  WARNING: <0.5s after VAD, falling back to raw audio")
        processed = audio_16k

    embedding = encoder.embed_utterance(processed)
    embedding = embedding / np.linalg.norm(embedding)
    print(f"  Embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")

    return embedding


def test_against_live(embedding: np.ndarray, seconds: int = 5) -> list[float]:
    """Record a short test clip and check distance — simulates live matching."""
    from scipy.spatial.distance import cosine
    from resemblyzer import VoiceEncoder, preprocess_wav

    print(f"\n>>> Test: recording {seconds}s to check match quality <<<")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("  Recording test clip...")
    audio = sd.rec(int(seconds * NATIVE_SR), samplerate=NATIVE_SR,
                   channels=CHANNELS, dtype="float32", device=DEVICE)
    sd.wait()
    if audio.ndim == 2:
        audio = audio[:, 0]
    audio_16k = resample_poly(audio, up=1, down=3).astype(np.float32)

    encoder = VoiceEncoder("cpu")

    # Test with preprocess_wav (like identifier.py)
    processed = preprocess_wav(audio_16k, source_sr=16000)
    if len(processed) < 8000:
        processed = audio_16k
    test_emb = encoder.embed_utterance(processed)

    dist = cosine(embedding, test_emb)
    print(f"  Test distance: {dist:.4f}")

    # Also test without preprocessing (raw, like fallback)
    raw_emb = encoder.embed_utterance(audio_16k)
    raw_dist = cosine(embedding, raw_emb)
    print(f"  Raw distance:  {raw_dist:.4f}")

    return [dist, raw_dist]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "dan"
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    print(f"=== Pipeline-Matched Speaker Enrollment: {name} ===\n")

    audio = record_like_timmy(seconds)
    embedding = create_voiceprint(audio)

    # Compare with existing
    from scipy.spatial.distance import cosine
    existing_path = f"/home/gearscodeandfire/little_timmy/models/speaker/{name}_resemblyzer.npy"
    try:
        old = np.load(existing_path)
        dist = cosine(embedding, old)
        print(f"\n  Distance from current voiceprint: {dist:.4f}")
    except FileNotFoundError:
        pass

    # Test live match quality
    print("\n  Now say a short test phrase to verify match quality...")
    distances = test_against_live(embedding, seconds=5)

    # Save
    import shutil
    backup = existing_path + ".pre_pipeline"
    try:
        shutil.copy2(existing_path, backup)
        print(f"\n  Backed up old to {backup}")
    except FileNotFoundError:
        pass

    np.save(existing_path, embedding)
    print(f"  Saved new voiceprint to {existing_path}")

    avg_dist = sum(distances) / len(distances)
    if avg_dist < 0.25:
        print(f"\n  Excellent match ({avg_dist:.3f}) — threshold 0.35 would work")
    elif avg_dist < 0.35:
        print(f"\n  Good match ({avg_dist:.3f}) — threshold 0.45 recommended")
    elif avg_dist < 0.45:
        print(f"\n  Moderate match ({avg_dist:.3f}) — threshold 0.50 recommended")
    else:
        print(f"\n  Weak match ({avg_dist:.3f}) — audio path may still differ")

    print(f"\n  Restart Little Timmy to load the new voiceprint.")


if __name__ == "__main__":
    main()
