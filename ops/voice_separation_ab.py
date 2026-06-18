#!/usr/bin/env python3
"""A/B: which encoder SEPARATES the synthetic test voices better —
Resemblyzer (production speaker-ID path) vs pyannote WeSpeaker-ResNet34 (cached).

Context (Dan, 2026-06-17): we play N distinct Piper voices through Timmy's
speaker to test multi-speaker speaker-ID hands-free / off-site. This tool asks,
purely on the CLEAN synthesized clips (no mic, no air): does swapping the
encoder buy us more headroom between those synthetic "speakers"?

Cosine distances from two different embedding spaces are NOT comparable in
absolute terms, so we normalise. For each voice we synthesize TWO different
phrases -> a within-voice distance (same speaker, different words). Then:

  within(enc)  = mean over voices of d(phraseA, phraseB)            [same speaker]
  cross(enc)   = mean over voice PAIRS of d(vi_A, vj_A)             [diff speakers]
  DISCRIMINABILITY = cross / within   (higher = more headroom; scale-free)
  clean-margin     = min_cross - max_within (in that encoder's own units)

A bigger discriminability ratio = that encoder pulls different voices apart
relative to how much it spreads one voice across phrasings. That is the honest
"separates better" metric. We also print each encoder's full pairwise matrix and
the tightest (hardest-to-separate) voice pair.

Companion to embed_ab_resemblyzer_vs_wespeaker.py (which A/B'd the misID
*mechanism* on real Dan/TTS captures; conclusion there: encoder swap doesn't fix
misID, ratio ~identical). This one scores the *synthetic test panel* instead.

Offline; uses locally-cached models. No enrollment, no mic, no contamination.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/voice_separation_ab.py
  .venv/bin/python ops/voice_separation_ab.py --voices en_US-ryan-high,en_US-amy-medium
"""
import argparse
import os
import sys
import glob
import itertools

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

VOICE_DIR = os.path.join(REPO, "models", "tts", "test_voices")
TARGET_SR = 16000

# Two distinct synthetic phrases (no PII) -> within-voice baseline.
PHRASE_A = ("The quick brown fox jumps over the lazy dog. "
            "One two three four five six seven eight.")
PHRASE_B = ("Pack my box with five dozen liquor jugs. "
            "Nine ten eleven twelve, the clock is ticking softly.")


def synth_16k(onnx_path: str, text: str) -> np.ndarray:
    """Synthesize `text` with a Piper voice, return float32 mono @ 16 kHz."""
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    voice = PiperVoice.load(onnx_path)
    sr = voice.config.sample_rate
    chunks = [c.audio_float_array
              for c in voice.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0))]
    a = np.concatenate(chunks).astype(np.float32)
    if sr != TARGET_SR:
        a = resample_poly(a, TARGET_SR, sr).astype(np.float32)
    return a


def main():
    ap = argparse.ArgumentParser(description="Encoder A/B for synthetic-voice separation")
    ap.add_argument("--voices", default=None,
                    help="comma list of voice stems in test_voices/ (default: all)")
    args = ap.parse_args()

    if args.voices:
        voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    else:
        voices = sorted(os.path.basename(p)[:-5]
                        for p in glob.glob(os.path.join(VOICE_DIR, "*.onnx")))
    if len(voices) < 2:
        print("Need >=2 voices to measure separation."); return 1
    print(f"Voices ({len(voices)}): {', '.join(voices)}\n")

    # ---- synthesize both phrases per voice, resampled to 16k ----
    print("Synthesizing 2 phrases per voice @ 16 kHz ...")
    wav = {}
    for v in voices:
        onnx = os.path.join(VOICE_DIR, v + ".onnx")
        wav[(v, "A")] = synth_16k(onnx, PHRASE_A)
        wav[(v, "B")] = synth_16k(onnx, PHRASE_B)

    # ---- Resemblyzer (production path) ----
    from speaker.identifier import SpeakerIdentifier
    si = SpeakerIdentifier()

    def emb_R(x):
        return np.asarray(si.extract_embedding(x), dtype=np.float64)

    # ---- pyannote WeSpeaker ResNet34 (cached, offline) ----
    import torch
    from pyannote.audio import Model, Inference
    wmodel = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    winf = Inference(wmodel, window="whole")

    def emb_W(x):
        out = winf({"waveform": torch.tensor(x).unsqueeze(0), "sample_rate": TARGET_SR})
        return np.asarray(out, dtype=np.float64).ravel()

    print("Embedding under Resemblyzer + WeSpeaker ...\n")
    R, W = {}, {}
    for key, x in wav.items():
        R[key] = emb_R(x)
        W[key] = emb_W(x)

    def analyze(name, E):
        within = {v: float(cosine(E[(v, "A")], E[(v, "B")])) for v in voices}
        pairs = list(itertools.combinations(voices, 2))
        cross = {(a, b): float(cosine(E[(a, "A")], E[(b, "A")])) for a, b in pairs}
        mean_w = float(np.mean(list(within.values())))
        max_w = max(within.values())
        mean_x = float(np.mean(list(cross.values())))
        min_x = min(cross.values())
        tightest = min(cross, key=cross.get)
        ratio = mean_x / mean_w if mean_w > 0 else float("inf")
        margin = min_x - max_w
        return dict(within=within, cross=cross, mean_w=mean_w, max_w=max_w,
                    mean_x=mean_x, min_x=min_x, tightest=tightest,
                    ratio=ratio, margin=margin)

    res = {"Resemblyzer": analyze("Resemblyzer", R), "WeSpeaker": analyze("WeSpeaker", W)}

    sn = lambda v: v.split("-")[1][:8]
    for enc in ("Resemblyzer", "WeSpeaker"):
        a = res[enc]
        print(f"==== {enc} — pairwise cross-voice cosine distance ====")
        hdr = "         " + " ".join(f"{sn(v):>8}" for v in voices)
        print(hdr)
        for vi in voices:
            cells = []
            for vj in voices:
                if vi == vj:
                    cells.append(f"{'-':>8}")
                else:
                    key = (vi, vj) if (vi, vj) in a["cross"] else (vj, vi)
                    cells.append(f"{a['cross'][key]:8.3f}")
            print(f"{sn(vi):8s} " + " ".join(cells))
        print(f"within-voice (same speaker, A vs B): " +
              ", ".join(f"{sn(v)}={a['within'][v]:.3f}" for v in voices))
        print(f"mean_within={a['mean_w']:.3f}  mean_cross={a['mean_x']:.3f}  "
              f"min_cross={a['min_x']:.3f} ({sn(a['tightest'][0])}/{sn(a['tightest'][1])})")
        print(f"DISCRIMINABILITY (cross/within) = {a['ratio']:.2f}   "
              f"clean-margin (min_cross - max_within) = {a['margin']:.3f}\n")

    rr, ww = res["Resemblyzer"], res["WeSpeaker"]
    better = "WeSpeaker" if ww["ratio"] > rr["ratio"] else "Resemblyzer"
    print("==== VERDICT ====")
    print(f"Discriminability ratio: Resemblyzer={rr['ratio']:.2f}  WeSpeaker={ww['ratio']:.2f}"
          f"  -> {better} separates the synthetic voices better (scale-free).")
    print(f"Tightest pair: Resemblyzer {sn(rr['tightest'][0])}/{sn(rr['tightest'][1])} "
          f"@ {rr['min_x']:.3f} (margin {rr['margin']:.3f});  "
          f"WeSpeaker {sn(ww['tightest'][0])}/{sn(ww['tightest'][1])} "
          f"@ {ww['min_x']:.3f} (margin {ww['margin']:.3f}).")
    print("Note: clean synth only. Through-the-speaker/air separation is measured by "
          "multi_voice_sweep.py (acoustic chain may compress these gaps).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
