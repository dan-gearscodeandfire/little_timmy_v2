#!/usr/bin/env python3
"""A/B: Resemblyzer (current) vs pyannote WeSpeaker-ResNet34 (cached) on the
SAME captured WAVs from today's sweeps.

We can't reproduce "lands on Erin" (no Erin audio on disk), so we test the
*mechanism* behind it on the two identities we DO have real audio for — Dan and
Timmy-TTS — plus pure-floor (no-voice) clips:

  1. DEGRADATION ROBUSTNESS  within-Dan: dist(dan_clean, dan_degraded).
       smaller = the model keeps a degraded voice recognizable as the same person.
  2. IDENTITY SEPARATION      cross:      dist(dan_clean, tts_full).
       larger = the model keeps two different identities apart.
  3. DISCRIMINABILITY         ratio cross/within and margin cross-within.
       higher = more headroom before a degraded voice is mistaken for someone else.
  4. NOISE-FLOOR COLLAPSE     dist(dan_15ft, tts_quiet) and each vs pure floor.
       if two DIFFERENT identities, both at the floor, sit very close, the model
       has collapsed them onto a shared noise centroid (the P1 mechanism). If
       BOTH models collapse here, the bottleneck is the MIC, not the encoder.

Offline (uses the locally-cached wespeaker model). No enrollment needed.
"""
import os, sys, glob
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import cosine

REPO = "/home/gearscodeandfire/little_timmy"
sys.path.insert(0, REPO)


def load16k(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if np.abs(x).max() > 1.5:
        x /= 32768.0
    return x.astype(np.float32), sr


# ---- the captures (good cued run = real Dan voice; 151919 = pure floor/no-voice) ----
C = f"{REPO}/ops"
FILES = {
    "dan_clean_6in":  f"{C}/snr_captures/20260617-152423_cued_6in.wav",
    "dan_3ft":        f"{C}/snr_captures/20260617-152423_cued_3ft.wav",
    "dan_8ft":        f"{C}/snr_captures/20260617-152423_cued_8ft.wav",
    "dan_15ft":       f"{C}/snr_captures/20260617-152423_cued_15ft.wav",
    "floor_novoice":  f"{C}/snr_captures/20260617-151919_cued_6in.wav",   # mic-off run = pure floor
    "tts_full":       f"{C}/level_sweep_captures/20260617-152849_vol1.0.wav",
    "tts_mid":        f"{C}/level_sweep_captures/20260617-152849_vol0.35.wav",
    "tts_quiet":      f"{C}/level_sweep_captures/20260617-152849_vol0.05.wav",
}

# ---- Resemblyzer (production path) ----
from speaker.identifier import SpeakerIdentifier
si = SpeakerIdentifier()

def emb_resemblyzer(x):
    try:
        return np.asarray(si.extract_embedding(x), dtype=np.float64)
    except Exception as e:
        print(f"    resemblyzer embed err: {e}"); return None

# ---- pyannote WeSpeaker ResNet34 (cached) ----
import torch
from pyannote.audio import Model, Inference
wmodel = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
winf = Inference(wmodel, window="whole")
print(f"WeSpeaker model loaded (offline). Resemblyzer dim=256.\n")

def emb_wespeaker(x, sr=16000):
    try:
        out = winf({"waveform": torch.tensor(x).unsqueeze(0), "sample_rate": sr})
        return np.asarray(out, dtype=np.float64).ravel()
    except Exception as e:
        print(f"    wespeaker embed err: {e}"); return None


# ---- embed everything under both models ----
R, W = {}, {}
for name, path in FILES.items():
    x, sr = load16k(path)
    R[name] = emb_resemblyzer(x)
    W[name] = emb_wespeaker(x, sr)

def d(model, a, b):
    if model[a] is None or model[b] is None:
        return float("nan")
    return float(cosine(model[a], model[b]))

print(f"{'measure':28s} {'Resemblyzer':>12s} {'WeSpeaker':>12s}   winner")
def row(label, a, b, lower_better=True):
    rr, ww = d(R, a, b), d(W, a, b)
    win = ("Resemblyzer" if (rr < ww) == lower_better else "WeSpeaker")
    print(f"{label:28s} {rr:>12.3f} {ww:>12.3f}   {win}")
    return rr, ww

print("-- (1) within-Dan: smaller = robust to degradation --")
wr3, ww3 = row("dan_clean vs dan_3ft", "dan_clean_6in", "dan_3ft")
wr8, ww8 = row("dan_clean vs dan_8ft", "dan_clean_6in", "dan_8ft")
print("-- (2) cross-identity: larger = better separation (winner=higher) --")
cr, cw = row("dan_clean vs tts_full", "dan_clean_6in", "tts_full", lower_better=False)
print("-- (3) discriminability = cross / within(3ft) : higher = more headroom --")
print(f"{'ratio cross/within3ft':28s} {cr/wr3:>12.2f} {cw/ww3:>12.2f}   "
      f"{'WeSpeaker' if cw/ww3 > cr/wr3 else 'Resemblyzer'}")
print("-- (4) noise-floor collapse: larger = keeps diff identities apart at floor --")
row("dan_15ft vs tts_quiet", "dan_15ft", "tts_quiet", lower_better=False)
row("dan_15ft vs floor_novoice", "dan_15ft", "floor_novoice", lower_better=False)
row("tts_quiet vs floor_novoice", "tts_quiet", "floor_novoice", lower_better=False)

print("\nReference within-model distances (orientation):")
for m, nm in ((R, "Resemblyzer"), (W, "WeSpeaker")):
    print(f"  {nm}: dan_clean<->tts_full={d(m,'dan_clean_6in','tts_full'):.3f}  "
          f"dan_clean<->dan_15ft={d(m,'dan_clean_6in','dan_15ft'):.3f}  "
          f"dan_15ft<->tts_quiet={d(m,'dan_15ft','tts_quiet'):.3f}")
