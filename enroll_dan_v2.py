"""Re-enroll Dan's voiceprint with more varied samples.

Records 15 utterances with varied conditions:
- Different distances from mic
- Different energy levels (normal, quiet, loud)
- Different speaking styles
This produces a more robust voiceprint that handles natural variation.
"""

import sys
sys.path.insert(0, "/home/gearscodeandfire/little_timmy")

import numpy as np
import time
import sounddevice as sd

SAMPLE_RATE = 48000
TARGET_RATE = 16000
RECORD_SECONDS = 6
PAUSE_SECONDS = 2

SENTENCES = [
    # Normal distance, normal voice
    ("NORMAL VOICE, NORMAL DISTANCE",
     "Hey there Little Timmy, it's me, Dan, your creator."),
    ("NORMAL VOICE, NORMAL DISTANCE",
     "I have two cats named Dexter and Preston, they are Cornish Rexes."),
    ("NORMAL VOICE, NORMAL DISTANCE",
     "My wife's name is Erin and she supports me doing crazy things like this."),
    ("NORMAL VOICE, NORMAL DISTANCE",
     "I have a YouTube channel called Gears Code and Fire."),
    ("NORMAL VOICE, NORMAL DISTANCE",
     "I built a shop heater out of two hundred and fifty five gallon drums."),
    # Quieter / further from mic
    ("QUIETER VOICE — speak a bit softer or lean back",
     "This is a test of my voice at a lower volume level."),
    ("QUIETER VOICE — speak a bit softer or lean back",
     "Sometimes I talk quietly when I'm thinking through a problem."),
    ("QUIETER VOICE — speak a bit softer or lean back",
     "The quick brown fox jumps over the lazy dog near the riverbank."),
    # Louder / closer / more energetic
    ("LOUDER VOICE — speak with more energy",
     "I'm pretty impressed with how fast you respond to my questions!"),
    ("LOUDER VOICE — speak with more energy",
     "We need to add some pre-processing to filter out hallucinations!"),
    ("LOUDER VOICE — speak with more energy",
     "Alright, we're going to end this session for now, you did a good job!"),
    # Casual / relaxed speaking
    ("CASUAL — speak naturally like you're chatting with a friend",
     "So yeah, I was thinking about maybe adding some new features later."),
    ("CASUAL — speak naturally like you're chatting with a friend",
     "Okay so what else do we need to work on today."),
    # Short utterances (sighs, brief responses)
    ("SHORT — just a few words, natural",
     "Yeah, that sounds right to me."),
    ("SHORT — just a few words, natural",
     "Alright, let's do it."),
]


def find_usb_mic():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name = d["name"].lower()
        if d["max_input_channels"] > 0 and ("usb" in name or "mic" in name):
            print(f"Found USB mic: [{i}] {d['name']} (sr={d['default_samplerate']})")
            return i
    raise RuntimeError("No USB mic found")


def downsample(audio, from_rate, to_rate):
    ratio = from_rate // to_rate
    return audio[::ratio]


def record_utterance(device_idx, duration_s):
    n_samples = int(duration_s * SAMPLE_RATE)
    print("  >> RECORDING... speak now!", flush=True)
    audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1,
                   dtype='float32', device=device_idx)
    sd.wait()
    audio = audio.flatten()
    return downsample(audio, SAMPLE_RATE, TARGET_RATE)


def main():
    device_idx = find_usb_mic()

    print(f"\n{'='*60}")
    print("DAN VOICE RE-ENROLLMENT (v2 — varied samples)")
    print(f"{'='*60}")
    print(f"You will read {len(SENTENCES)} sentences.")
    print(f"Each recording is {RECORD_SECONDS} seconds.")
    print(f"Pay attention to the STYLE instruction for each one.")
    print(f"\nStarting in 5 seconds...\n")
    time.sleep(5)

    recordings = []
    for i, (style, sentence) in enumerate(SENTENCES):
        print(f"\n[{i+1}/{len(SENTENCES)}] Style: {style}")
        print(f'  "{sentence}"')
        time.sleep(1)

        audio = record_utterance(device_idx, RECORD_SECONDS)
        rms = np.sqrt(np.mean(audio**2))
        print(f"  Recorded {len(audio)/TARGET_RATE:.1f}s, RMS={rms:.4f}")

        if rms < 0.001:
            print("  WARNING: Very low audio level - might be silence!")

        recordings.append(audio)
        if i < len(SENTENCES) - 1:
            print(f"  (pause {PAUSE_SECONDS}s...)")
            time.sleep(PAUSE_SECONDS)

    print(f"\n{'='*60}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*60}")

    from resemblyzer import VoiceEncoder
    from scipy.spatial.distance import cosine

    encoder = VoiceEncoder("cpu")

    embeddings = []
    for i, audio in enumerate(recordings):
        t0 = time.time()
        emb = encoder.embed_utterance(audio)
        ms = (time.time() - t0) * 1000
        embeddings.append(emb)
        print(f"  [{i+1}] embedding extracted ({ms:.0f}ms)")

    # Average voiceprint
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # Self-similarity check
    similarities = []
    distances = []
    for i, emb in enumerate(embeddings):
        sim = 1 - cosine(avg_embedding, emb)
        dist = cosine(avg_embedding, emb)
        similarities.append(sim)
        distances.append(dist)
        style = SENTENCES[i][0].split("—")[0].strip()
        print(f"  [{i+1:2d}] {style:20s} sim={sim:.3f} dist={dist:.3f}")

    print(f"\n  Self-similarity: min={min(similarities):.3f} max={max(similarities):.3f} avg={np.mean(similarities):.3f}")
    print(f"  Max distance from avg: {max(distances):.3f}")

    # Compare against Timmy's voiceprint
    timmy_path = "/home/gearscodeandfire/little_timmy/models/speaker/timmy_resemblyzer.npy"
    timmy_emb = np.load(timmy_path)
    dan_vs_timmy = cosine(avg_embedding, timmy_emb)
    print(f"\n  Dan vs Timmy distance: {dan_vs_timmy:.3f} (higher = more different)")
    print(f"  Safety margin: {dan_vs_timmy - max(distances):.3f} (should be > 0)")

    # Compare against old voiceprint
    old_path = "/home/gearscodeandfire/little_timmy/models/speaker/dan_resemblyzer.npy"
    old_emb = np.load(old_path)
    old_vs_new = cosine(avg_embedding, old_emb)
    print(f"  Old vs new voiceprint distance: {old_vs_new:.3f}")

    # Save (backup old first)
    import shutil
    shutil.copy(old_path, old_path + ".bak")
    np.save(old_path, avg_embedding)
    print(f"\n  Old voiceprint backed up to {old_path}.bak")
    print(f"  New voiceprint saved to {old_path}")
    print(f"  Shape: {avg_embedding.shape}")
    print("\nDone!")


if __name__ == "__main__":
    main()
