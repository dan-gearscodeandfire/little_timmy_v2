"""Analyze audio signal from Timmy's capture path for artifacts.

Records silence and speech, computes FFT, identifies problematic frequencies.
"""

import sys
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")

NATIVE_SR = 48000
TARGET_SR = 16000
CHANNELS = 2
DEVICE = 7


def record_segment(label: str, seconds: int) -> np.ndarray:
    """Record and resample like Timmy's pipeline."""
    print(f"\n>>> Recording {seconds}s of {label} <<<")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print(f"  RECORDING {label.upper()}...")

    audio = sd.rec(int(seconds * NATIVE_SR), samplerate=NATIVE_SR,
                   channels=CHANNELS, dtype="float32", device=DEVICE)
    sd.wait()

    if audio.ndim == 2:
        audio = audio[:, 0]
    audio_16k = resample_poly(audio, up=1, down=3).astype(np.float32)

    peak = np.max(np.abs(audio_16k))
    rms = np.sqrt(np.mean(audio_16k ** 2))
    print(f"  peak={peak:.5f}, RMS={rms:.6f}, samples={len(audio_16k)}")
    return audio_16k


def analyze_spectrum(audio: np.ndarray, sr: int, label: str):
    """Compute and report FFT spectrum."""
    n = len(audio)
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    magnitude = np.abs(fft) / n

    # Convert to dB (relative to max)
    mag_db = 20 * np.log10(magnitude + 1e-10)
    max_db = np.max(mag_db)
    mag_db_norm = mag_db - max_db  # normalize so peak = 0dB

    print(f"\n=== {label} Spectrum Analysis ===")
    print(f"  Total samples: {n}, Duration: {n/sr:.1f}s")
    print(f"  Peak amplitude: {np.max(np.abs(audio)):.5f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"  DC offset: {np.mean(audio):.6f}")

    # Find dominant frequencies (top 10 peaks)
    # Smooth slightly to avoid noise peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(mag_db, height=-40, distance=10, prominence=3)

    if len(peaks) > 0:
        # Sort by magnitude
        sorted_idx = np.argsort(magnitude[peaks])[::-1][:15]
        top_peaks = peaks[sorted_idx]

        print(f"\n  Top frequency peaks:")
        print(f"  {'Freq (Hz)':>10} {'Magnitude':>12} {'dB (norm)':>10} {'Note':>20}")
        print(f"  {'-'*55}")
        for idx in top_peaks:
            freq = freqs[idx]
            mag = magnitude[idx]
            db = mag_db_norm[idx]
            note = ""
            if 55 <= freq <= 65:
                note = "** 60Hz MAINS **"
            elif 115 <= freq <= 125:
                note = "** 120Hz harmonic **"
            elif 175 <= freq <= 185:
                note = "** 180Hz harmonic **"
            elif 235 <= freq <= 245:
                note = "** 240Hz harmonic **"
            elif freq < 30:
                note = "DC/infrasonic"
            elif freq > 7000:
                note = "high freq noise"
            print(f"  {freq:10.1f} {mag:12.6f} {db:10.1f} {note:>20}")

    # Band energy analysis
    bands = [
        ("DC/infra", 0, 20),
        ("Sub-bass", 20, 60),
        ("Mains hum", 55, 65),
        ("Bass", 60, 250),
        ("Low-mid", 250, 500),
        ("Mid", 500, 2000),
        ("Upper-mid", 2000, 4000),
        ("Presence", 4000, 6000),
        ("Brilliance", 6000, 8000),
    ]

    print(f"\n  Band energy distribution:")
    print(f"  {'Band':>15} {'Hz range':>12} {'Energy':>12} {'%':>8}")
    print(f"  {'-'*50}")
    total_energy = np.sum(magnitude**2)
    for name, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy = np.sum(magnitude[mask]**2)
        pct = (band_energy / total_energy) * 100 if total_energy > 0 else 0
        print(f"  {name:>15} {lo:>5}-{hi:<5} {band_energy:12.8f} {pct:7.1f}%")

    return freqs, magnitude, mag_db_norm


def main():
    print("=== Little Timmy Audio Signal Analysis ===")
    print(f"Device: {DEVICE}, SR: {NATIVE_SR}→{TARGET_SR}, Stereo ch0\n")

    # Record silence
    print("Please stay QUIET for the silence recording.")
    silence = record_segment("silence (stay quiet!)", 5)

    # Record speech
    print("\nNow please SPEAK NORMALLY for the speech recording.")
    speech = record_segment("speech (talk naturally)", 8)

    # Analyze both
    analyze_spectrum(silence, TARGET_SR, "SILENCE (noise floor)")
    analyze_spectrum(speech, TARGET_SR, "SPEECH")

    # SNR estimate
    silence_rms = np.sqrt(np.mean(silence**2))
    speech_rms = np.sqrt(np.mean(speech**2))
    if silence_rms > 0:
        snr = 20 * np.log10(speech_rms / silence_rms)
        print(f"\n=== Signal-to-Noise Ratio ===")
        print(f"  Silence RMS: {silence_rms:.6f}")
        print(f"  Speech RMS:  {speech_rms:.6f}")
        print(f"  SNR: {snr:.1f} dB")
        if snr < 10:
            print("  WARNING: Very low SNR — speech barely above noise")
        elif snr < 20:
            print("  FAIR: Moderate SNR — filtering would help")
        else:
            print("  GOOD: Decent SNR")

    # Filter recommendations
    print(f"\n=== Filter Recommendations ===")

    # Check 60Hz
    silence_fft = np.abs(np.fft.rfft(silence)) / len(silence)
    silence_freqs = np.fft.rfftfreq(len(silence), 1.0 / TARGET_SR)
    mains_mask = (silence_freqs >= 55) & (silence_freqs <= 65)
    mains_energy = np.sum(silence_fft[mains_mask]**2)
    total_silence_energy = np.sum(silence_fft**2)
    mains_pct = (mains_energy / total_silence_energy) * 100 if total_silence_energy > 0 else 0

    if mains_pct > 5:
        print(f"  HIGH-PASS 80Hz: STRONGLY RECOMMENDED (60Hz mains = {mains_pct:.1f}% of noise)")
    elif mains_pct > 1:
        print(f"  HIGH-PASS 80Hz: Recommended (60Hz mains = {mains_pct:.1f}% of noise)")
    else:
        print(f"  HIGH-PASS 80Hz: Optional (60Hz mains = {mains_pct:.1f}% of noise)")

    # Check high frequency noise
    hf_mask = silence_freqs > 6000
    hf_energy = np.sum(silence_fft[hf_mask]**2)
    hf_pct = (hf_energy / total_silence_energy) * 100 if total_silence_energy > 0 else 0
    if hf_pct > 10:
        print(f"  LOW-PASS 6kHz: STRONGLY RECOMMENDED (HF noise = {hf_pct:.1f}% of noise)")
    elif hf_pct > 3:
        print(f"  LOW-PASS 8kHz: Recommended (HF noise = {hf_pct:.1f}% of noise)")
    else:
        print(f"  LOW-PASS: Optional (HF noise = {hf_pct:.1f}% of noise)")

    print("\nDone!")


if __name__ == "__main__":
    main()
