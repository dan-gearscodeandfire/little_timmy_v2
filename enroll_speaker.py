"""Re-enroll a speaker voiceprint from live mic capture.

Records audio from PipeWire default device (same path Little Timmy uses),
extracts a Resemblyzer embedding, and saves it.

Usage:
  python enroll_speaker.py [name] [seconds] [--save]
  python enroll_speaker.py dan 15 --save
"""

import sys
import time
import numpy as np
import sounddevice as sd

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")

SAMPLE_RATE = 16000
CHANNELS = 2  # SN6186 codec is stereo, take ch0


def record_audio(seconds: int = 15) -> np.ndarray:
    """Record audio from default PipeWire device."""
    print(f"\n>>> Recording {seconds} seconds — please speak naturally <<<")
    print("    (Talk as you normally would to Timmy — varied phrases, natural volume)")
    print()
    print("  Starting in 3...")
    time.sleep(1)
    print("  Starting in 2...")
    time.sleep(1)
    print("  Starting in 1...")
    time.sleep(1)

    print("  RECORDING NOW — speak!")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=None,  # PipeWire default
    )
    sd.wait()
    print("  Done recording.\n")

    # Take channel 0 (mono)
    if audio.ndim == 2:
        audio = audio[:, 0]

    # Report stats
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"  Audio stats: peak={peak:.4f}, RMS={rms:.6f}, samples={len(audio)}")

    if peak < 0.01:
        print("  WARNING: Very low audio level — mic may not be capturing")

    return audio


def simple_vad_trim(audio: np.ndarray, sr: int = 16000,
                    frame_ms: int = 30, threshold: float = 0.01) -> np.ndarray:
    """Simple energy-based VAD: keep frames above threshold RMS.

    More permissive than Resemblyzer's webrtcvad-based preprocessing.
    """
    frame_len = int(sr * frame_ms / 1000)
    voiced_frames = []

    for i in range(0, len(audio) - frame_len, frame_len):
        frame = audio[i:i + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold:
            voiced_frames.append(frame)

    if not voiced_frames:
        print(f"  VAD: no frames above threshold {threshold}, trying lower...")
        # Try with lower threshold
        for i in range(0, len(audio) - frame_len, frame_len):
            frame = audio[i:i + frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > threshold / 4:
                voiced_frames.append(frame)

    if not voiced_frames:
        print("  VAD: still nothing — using raw audio")
        return audio

    result = np.concatenate(voiced_frames)
    print(f"  VAD: kept {len(result)} samples ({len(result)/sr:.1f}s) "
          f"from {len(audio)} ({len(audio)/sr:.1f}s)")
    return result


def create_voiceprint(audio_16k: np.ndarray) -> np.ndarray:
    """Create Resemblyzer embedding from audio."""
    from resemblyzer import VoiceEncoder

    print("  Loading Resemblyzer encoder...")
    encoder = VoiceEncoder("cpu")

    # Use our own VAD instead of Resemblyzer's preprocess_wav
    # (Resemblyzer's webrtcvad is too aggressive with this mic setup)
    trimmed = simple_vad_trim(audio_16k)

    print("  Extracting embedding...")
    embedding = encoder.embed_utterance(trimmed)
    embedding = embedding / np.linalg.norm(embedding)

    print(f"  Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
    return embedding


def compare_with_existing(new_emb: np.ndarray, name: str):
    """Compare new embedding with existing one."""
    from scipy.spatial.distance import cosine

    existing_path = f"/home/gearscodeandfire/little_timmy/models/speaker/{name}_resemblyzer.npy"
    try:
        old_emb = np.load(existing_path)
        dist = cosine(new_emb, old_emb)
        print(f"\n  Distance from old voiceprint: {dist:.4f}")
        if dist < 0.15:
            print("  (Very similar to old — may not improve much)")
        elif dist < 0.30:
            print("  (Moderately different — should improve matching)")
        else:
            print("  (Significantly different — new mic setup captured differently)")
    except FileNotFoundError:
        print(f"\n  No existing voiceprint at {existing_path}")


def save_voiceprint(embedding: np.ndarray, name: str):
    """Save the new voiceprint, backing up the old one."""
    import shutil
    path = f"/home/gearscodeandfire/little_timmy/models/speaker/{name}_resemblyzer.npy"
    backup = path + ".bak3"

    try:
        shutil.copy2(path, backup)
        print(f"  Backed up old voiceprint to {backup}")
    except FileNotFoundError:
        pass

    np.save(path, embedding)
    print(f"  Saved new voiceprint to {path}")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "dan"
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    auto_save = "--save" in sys.argv

    print(f"=== Speaker Re-enrollment: {name} ===")
    print(f"Recording duration: {seconds}s")
    print(f"Auto-save: {auto_save}")

    audio = record_audio(seconds)
    embedding = create_voiceprint(audio)
    compare_with_existing(embedding, name)

    if auto_save:
        save_voiceprint(embedding, name)
        print(f"\n  Done! Restart Little Timmy to load the new voiceprint.")
    else:
        print("\n  Dry run — use --save to actually save.")


if __name__ == "__main__":
    main()
