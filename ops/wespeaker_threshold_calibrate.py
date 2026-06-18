#!/usr/bin/env python3
"""Calibrate a WeSpeaker decision THRESHOLD through the live acoustic chain.

The encoder A/B tools (voice_separation_ab.py, embed_ab_resemblyzer_vs_wespeaker.py)
showed WeSpeaker has "more headroom" than Resemblyzer at separating the synthetic
Piper panel -- but headroom is not a decision boundary. This tool turns it into one:
it ENROLLS a WeSpeaker voiceprint per test voice and measures same-vs-different
distances THROUGH THE CHAIN (speaker -> air -> mic), then derives the cosine
threshold T that cleanly splits genuine (same speaker) from impostor (different
speaker). WeSpeaker lives on a different distance scale than Resemblyzer's 0.30,
so this T is what the scale-free A/B ratio could not give us.

Protocol (all audio post speaker->air->mic, like multi_voice_sweep.py):
  - For each voice, synthesize a bank of K distinct synthetic phrases (no PII),
    play each through Timmy's speaker, capture via the lav, 80 Hz HP -> 16k.
  - ENROLL = L2-normalized centroid of the first --enroll captures of a voice
    (held-in phrases). Saved to models/speaker/wespeaker/<voice>.npy.
  - TEST  = the remaining captures (held-out phrases), so genuine scores are
    text-INDEPENDENT (enroll and test never share a phrase).
  - genuine[]  = cosine(test_capture, OWN-voice centroid)              -> want LOW
    impostor[] = cosine(test_capture, every OTHER-voice centroid)      -> want HIGH
  - Recommended T: if separable (max_genuine < min_impostor), midpoint with margin;
    always also report the equal-error-rate (EER) threshold and its error rate.
  - SEPARABILITY CHECK: every test capture's nearest centroid must be its own voice
    (100% closed-set ID), and min_impostor must clear max_genuine. If any pair is
    too tight it is flagged by name.

Same no-contamination contract as multi_voice_sweep.py: hearing muted the whole
run (captures dropped pre-STT) and restored in finally; default-sink volume read
first and restored in finally; production config.PIPER_MODEL untouched.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/wespeaker_threshold_calibrate.py
  .venv/bin/python ops/wespeaker_threshold_calibrate.py --enroll 2 --test 2
  .venv/bin/python ops/wespeaker_threshold_calibrate.py --voices en_US-ryan-high,en_US-amy-medium
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from scipy.spatial.distance import cosine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ops.snr_vs_distance import (  # noqa: E402
    _ensure_audio_env, set_hearing, find_input_device,
    highpass_80, to16k, record, rms, dbfs, speech_rms, save_wav,
)
from ops.multi_voice_sweep import (  # noqa: E402
    VOICE_DIR, synth_wav, play_wav, get_volume, set_volume, capture_async,
)

# Distinct synthetic phrases (no PII). enroll vs test draw from disjoint slices,
# so genuine scores are text-independent.
PHRASE_BANK = [
    "The quick brown fox jumps over the lazy dog. One two three four five.",
    "Pack my box with five dozen liquor jugs. Six seven eight nine ten.",
    "How vexingly quick daft zebras jump. Eleven twelve thirteen fourteen.",
    "The five boxing wizards jump quickly. Fifteen sixteen seventeen eighteen.",
    "Sphinx of black quartz, judge my vow. Nineteen twenty twenty one.",
    "Bright vixens jump; dozy fowl quack. Twenty two twenty three twenty four.",
]


def l2n(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def eer_threshold(genuine, impostor):
    """Equal-error-rate threshold: where FAR(impostor<T) == FRR(genuine>=T).
    Returns (T, eer)."""
    if not genuine or not impostor:
        return float("nan"), float("nan")
    cands = sorted(set(genuine) | set(impostor))
    grid = []
    for c in cands:
        grid.append(c)
        grid.append(c + 1e-6)
    best = None
    for t in grid:
        far = np.mean([i < t for i in impostor])   # impostor accepted as same
        frr = np.mean([g >= t for g in genuine])    # genuine rejected as diff
        gap = abs(far - frr)
        if best is None or gap < best[0]:
            best = (gap, t, (far + frr) / 2.0)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description="WeSpeaker through-chain threshold calibration")
    ap.add_argument("--voices", default=None,
                    help="comma list of voice stems in test_voices/ (default: all)")
    ap.add_argument("--enroll", type=int, default=2, help="phrases per voice for enrollment")
    ap.add_argument("--test", type=int, default=2, help="held-out phrases per voice for scoring")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "wespeaker_cal_captures"))
    ap.add_argument("--printdir", default=os.path.join(REPO, "models", "speaker", "wespeaker"))
    args = ap.parse_args()

    if args.voices:
        voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    else:
        voices = sorted(os.path.basename(p)[:-5]
                        for p in glob.glob(os.path.join(VOICE_DIR, "*.onnx")))
    if len(voices) < 2:
        print("Need >=2 voices."); return 1
    n_need = args.enroll + args.test
    if n_need > len(PHRASE_BANK):
        print(f"enroll+test={n_need} exceeds phrase bank ({len(PHRASE_BANK)})."); return 1
    enroll_phrases = PHRASE_BANK[:args.enroll]
    test_phrases = PHRASE_BANK[args.enroll:args.enroll + args.test]

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.printdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sn = lambda v: v.split("-")[1][:8]

    # ---- WeSpeaker (cached, offline) ----
    import torch
    from pyannote.audio import Model, Inference
    wmodel = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    winf = Inference(wmodel, window="whole")
    TARGET_SR = 16000

    def emb_W(x):
        out = winf({"waveform": torch.tensor(x.astype(np.float32)).unsqueeze(0),
                    "sample_rate": TARGET_SR})
        return l2n(np.asarray(out, dtype=np.float64).ravel())

    print(f"Voices ({len(voices)}): {', '.join(voices)}")
    print(f"enroll phrases={args.enroll}  test phrases={args.test}\n")

    device = find_input_device(args.device)
    import sounddevice as sd
    dname = sd.query_devices(device)["name"] if device is not None else "(default)"
    print(f"Input device: [{device}] {dname}")

    def capture_phrase(onnx, text, tag):
        """Synth -> play through sink -> capture via mic -> HP 16k -> WeSpeaker emb."""
        src = os.path.join(args.outdir, f"{stamp}_{tag}_src.wav")
        dur = synth_wav(onnx, text, src)
        t, box = capture_async(device, dur + 1.2)
        time.sleep(0.4)
        play_wav(src)
        t.join()
        cap = box.get("x")
        if cap is None or len(cap) == 0:
            return None, None, None
        x16_hp = highpass_80(to16k(cap))
        sp = speech_rms(x16_hp)
        snr = 20.0 * np.log10(sp / (floor + 1e-12))
        save_wav(os.path.join(args.outdir, f"{stamp}_{tag}.wav"), x16_hp)
        return emb_W(x16_hp), round(snr, 1), round(dbfs(x16_hp), 1)

    orig_vol = get_volume()
    print(f"Default-sink volume (will restore): {orig_vol}\n")

    centroids = {}        # voice -> enrolled WeSpeaker centroid (L2-normed)
    test_embs = {}        # voice -> list of held-out test embeddings
    snr_log = []
    global floor
    try:
        set_hearing(False)
        if orig_vol is not None:
            set_volume(1.0)
        time.sleep(0.4)
        base = to16k(record(device, 2.5))
        floor = rms(highpass_80(base))
        print(f"baseline floor post-HP={dbfs(highpass_80(base)):.1f} dBFS\n")

        for v in voices:
            onnx = os.path.join(VOICE_DIR, v + ".onnx")
            e_embs = []
            for i, ph in enumerate(enroll_phrases):
                emb, snr, pk = capture_phrase(onnx, ph, f"{v}_enr{i}")
                if emb is None:
                    print(f"  [{sn(v)} enr{i}] WARN empty capture"); continue
                e_embs.append(emb); snr_log.append(snr)
                print(f"  [{sn(v):8s} enr{i}] SNR={snr} dB  peak={pk} dBFS")
            t_embs = []
            for i, ph in enumerate(test_phrases):
                emb, snr, pk = capture_phrase(onnx, ph, f"{v}_tst{i}")
                if emb is None:
                    print(f"  [{sn(v)} tst{i}] WARN empty capture"); continue
                t_embs.append(emb); snr_log.append(snr)
                print(f"  [{sn(v):8s} tst{i}] SNR={snr} dB  peak={pk} dBFS")
            if not e_embs or not t_embs:
                print(f"  !! {v}: missing enroll/test captures, skipping"); continue
            cen = l2n(np.mean(np.stack(e_embs), axis=0))
            centroids[v] = cen
            test_embs[v] = t_embs
            np.save(os.path.join(args.printdir, f"{v}.npy"), cen)
            print(f"  -> enrolled {sn(v)} centroid ({len(e_embs)} phrases) saved\n")
    finally:
        if orig_vol is not None:
            print(f">>> Restoring sink volume to {orig_vol}")
            set_volume(orig_vol)
        print(">>> Restoring hearing")
        set_hearing(True)

    present = [v for v in voices if v in centroids and v in test_embs]
    if len(present) < 2:
        print("Not enough successfully-captured voices to calibrate."); return 1

    # ---- genuine vs impostor scores (held-out test vs centroids) ----
    genuine, impostor = [], []
    id_correct, id_total = 0, 0
    per_voice = {}
    for v in present:
        g_scores, worst_imp = [], (float("inf"), None)
        for emb in test_embs[v]:
            dg = float(cosine(emb, centroids[v]))
            genuine.append(dg); g_scores.append(dg)
            # closed-set nearest-centroid ID
            ranked = sorted(((float(cosine(emb, centroids[u])), u) for u in present),
                            key=lambda t: t[0])
            id_total += 1
            if ranked[0][1] == v:
                id_correct += 1
            for u in present:
                if u == v:
                    continue
                di = float(cosine(emb, centroids[u]))
                impostor.append(di)
                if di < worst_imp[0]:
                    worst_imp = (di, u)
        per_voice[v] = dict(max_genuine=round(max(g_scores), 3),
                            nearest_impostor=round(worst_imp[0], 3),
                            nearest_impostor_voice=worst_imp[1])

    max_gen = max(genuine); min_imp = min(impostor)
    mean_gen = float(np.mean(genuine)); mean_imp = float(np.mean(impostor))
    separable = max_gen < min_imp
    T_mid = (max_gen + min_imp) / 2.0
    T_eer, eer = eer_threshold(genuine, impostor)
    T_reco = T_mid if separable else T_eer

    # ---- centroid pairwise matrix (enrolled prints) ----
    print("==== WeSpeaker enrolled-centroid pairwise cosine distance (through chain) ====")
    print("         " + " ".join(f"{sn(v):>8}" for v in present))
    tight = (None, 9.0)
    for vi in present:
        cells = []
        for vj in present:
            if vi == vj:
                cells.append(f"{'-':>8}")
            else:
                dd = float(cosine(centroids[vi], centroids[vj]))
                cells.append(f"{dd:8.3f}")
                if dd < tight[1]:
                    tight = ((vi, vj), dd)
        print(f"{sn(vi):8s} " + " ".join(cells))

    print("\n==== GENUINE vs IMPOSTOR (held-out test captures) ====")
    print(f"  genuine  (same voice):   n={len(genuine):3d}  mean={mean_gen:.3f}  max={max_gen:.3f}")
    print(f"  impostor (diff voice):   n={len(impostor):3d}  mean={mean_imp:.3f}  min={min_imp:.3f}")
    print(f"  closed-set ID accuracy:  {id_correct}/{id_total}")
    print(f"  separable (max_gen<min_imp): {separable}  margin={min_imp - max_gen:+.3f}")
    print(f"  tightest centroid pair:  {sn(tight[0][0])}/{sn(tight[0][1])} @ {tight[1]:.3f}")

    print("\n==== THRESHOLD ====")
    if separable:
        print(f"  RECOMMENDED T = {T_reco:.3f}  (separable midpoint; "
              f"accepts genuine<= {max_gen:.3f}, rejects impostor>= {min_imp:.3f})")
    else:
        print(f"  RECOMMENDED T = {T_reco:.3f}  (EER point; distributions overlap)")
    print(f"  EER threshold = {T_eer:.3f}  EER = {eer*100:.1f}%")
    print(f"  (midpoint candidate = {T_mid:.3f})")

    verdict = ("ALL VOICES EASILY DISTINGUISHABLE" if separable and id_correct == id_total
               else "NOT fully separable -- see flags")
    print(f"\n==== VERDICT: {verdict} ====")
    if not (separable and id_correct == id_total):
        for v in present:
            pv = per_voice[v]
            if pv["max_genuine"] >= pv["nearest_impostor"]:
                print(f"  FLAG {sn(v)}: max_genuine {pv['max_genuine']} >= nearest impostor "
                      f"{pv['nearest_impostor']} ({sn(pv['nearest_impostor_voice'])})")

    out = dict(stamp=stamp, voices=present, enroll_phrases=args.enroll,
               test_phrases=args.test, scale="cosine_on_L2norm_wespeaker_resnet34",
               floor_dbfs=round(dbfs(highpass_80(base)), 1),
               snr_db=dict(min=min(snr_log), max=max(snr_log),
                           mean=round(float(np.mean(snr_log)), 1)),
               genuine=dict(n=len(genuine), mean=round(mean_gen, 3), max=round(max_gen, 3)),
               impostor=dict(n=len(impostor), mean=round(mean_imp, 3), min=round(min_imp, 3)),
               separable=separable, margin=round(min_imp - max_gen, 3),
               closed_set_id=f"{id_correct}/{id_total}",
               recommended_T=round(T_reco, 3), midpoint_T=round(T_mid, 3),
               eer_T=round(T_eer, 3), eer_pct=round(eer * 100, 2),
               tightest_centroid_pair=[sn(tight[0][0]), sn(tight[0][1]), round(tight[1], 3)],
               per_voice=per_voice, verdict=verdict,
               centroid_dir=args.printdir)
    out_json = os.path.join(args.outdir, f"{stamp}_wespeaker_threshold.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary JSON: {out_json}")
    print(f"Enrolled centroids: {args.printdir}/<voice>.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
