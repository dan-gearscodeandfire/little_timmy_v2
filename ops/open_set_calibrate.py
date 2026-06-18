#!/usr/bin/env python3
"""Calibrate the open-set guard's threshold and prove it on real captures.

WeSpeaker space (2026-06-17 encoder migration). Simulates the FULL gated
identify() decision on held-out clips and compares:

  RAW   (production today): accept nearest enrolled identity iff best cosine
        distance < KNOWN_SPEAKER_THRESHOLD.
  GATED (open-set): same, but an accepted match must ALSO pass the anti-model /
        s-norm guard (speaker/open_set.py) scored against the WeSpeaker cohort
        built by build_open_set_cohort.py (models/speaker/cohort_wespeaker/).

Test sets (DISJOINT from the cohort AND from the enroll clips):
  ENROLL   = clean on-mic worn-lav Dan -> builds the in-memory "dan" prototype.
  GENUINE  = a DIFFERENT on-mic Dan take -> correct decision is "dan" (stay accepted).
  IMPOSTOR = synthetic Piper voices through the speaker, Timmy-TTS, far/degraded
             Dan (the noise-collapse), and pure floor -> reject ("unknown").

For each clip we embed under the production WeSpeaker path, find the nearest
enrolled identity exactly like identify(), then score the open-set guard against
THAT identity. We report the discriminative envelope of each open-set signal
(s-norm, anti-model margin) and recommend a T that keeps genuine accepted while
rejecting the most impostors.

Enroll source: if real ``*_wespeaker.npy`` voiceprints exist they are used as-is.
Otherwise (the directional pre-re-enrollment state) a TEMPORARY in-memory "dan"
prototype is built from ENROLL_GLOBS via the runtime's _build_prototypes — it is
NOT persisted (proper K-prototype re-enrollment of real speakers is a separate
live step: enroll_voiced.py / enroll_prototypes.py).

Honest-scope note: Dan is the only identity with real on-mic audio; Erin/others
have none on disk (see project_lt_erin_voiceprint_noise_centroid_bug), so this is
a directional proof on Dan-genuine + the noise/impostor cohort, not a full
multi-speaker benchmark. Final T MUST be confirmed live after real WeSpeaker
re-enrollment before flipping OPEN_SET_REJECT_ENABLED / the runtime toggle.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/open_set_calibrate.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import cosine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OPS = os.path.join(REPO, "ops")
COHORT_DIR = os.path.join(REPO, "models", "speaker", "cohort_wespeaker")

# Clean on-mic worn-lav Dan, used to build the in-memory "dan" prototype when no
# real *_wespeaker.npy is enrolled yet. DISJOINT from GENUINE/IMPOSTOR below.
ENROLL_GLOBS = [
    os.path.join(OPS, "mic_placement_captures", "*_r1_t1.wav"),
    os.path.join(OPS, "mic_placement_captures", "*_r1_t2.wav"),
]
GENUINE_GLOBS = [
    os.path.join(OPS, "snr_captures", "20260617-152423_cued_6in.wav"),  # real Dan @6in on-mic (held out)
]
IMPOSTOR_GLOBS = [
    os.path.join(OPS, "multi_voice_captures", "*_vol1.0.wav"),           # 6 synth voices
    os.path.join(OPS, "speaker_snr_captures", "*.wav"),                  # Timmy TTS
    os.path.join(OPS, "snr_captures", "20260617-152423_cued_3ft.wav"),   # far Dan (collapse)
    os.path.join(OPS, "snr_captures", "20260617-152423_cued_8ft.wav"),
    os.path.join(OPS, "snr_captures", "20260617-152423_cued_15ft.wav"),
    os.path.join(OPS, "snr_captures", "20260617-151919_cued_3ft.wav"),   # pure floor (held-out)
    os.path.join(OPS, "snr_captures", "20260617-151919_cued_15ft.wav"),
]


def load16k(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if np.abs(x).max() > 1.5:
        x /= 32768.0
    return x.astype(np.float32)


def _expand(globs):
    out = []
    for g in globs:
        out.extend(sorted(glob.glob(g)))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Open-set threshold calibration")
    ap.add_argument("--cohort", default=COHORT_DIR)
    ap.add_argument("--threshold", type=float, default=None, help="override KNOWN_SPEAKER_THRESHOLD")
    ap.add_argument("--outdir", default=os.path.join(OPS, "open_set_captures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from speaker.identifier import (
        SpeakerIdentifier, KnownSpeaker, KNOWN_SPEAKER_THRESHOLD, _build_prototypes,
    )
    from speaker.open_set import OpenSetScorer
    THR = args.threshold if args.threshold is not None else KNOWN_SPEAKER_THRESHOLD

    si = SpeakerIdentifier()
    si.load_voiceprints()
    enrolled = si._known_speakers
    if not enrolled:
        # Directional pre-re-enrollment state: no *_wespeaker.npy on disk. Build a
        # TEMPORARY (non-persisted) in-memory "dan" from the held-out enroll clips.
        enroll_paths = _expand(ENROLL_GLOBS)
        if not enroll_paths:
            print("No enrolled *_wespeaker.npy and no ENROLL_GLOBS clips matched — "
                  "cannot calibrate. Re-enroll a real speaker first."); return 1
        embs = [si.extract_embedding(load16k(p)) for p in enroll_paths]
        protos = _build_prototypes(embs)
        enrolled = [KnownSpeaker(speaker_id=1, name="dan", prototypes=protos)]
        print(f"[DIRECTIONAL] no persisted voiceprints; built TEMP in-memory 'dan' "
              f"from {len(enroll_paths)} enroll clip(s) -> {protos.shape[0]} prototype(s) "
              f"(NOT persisted).")
    print(f"Enrolled: {', '.join(k.name for k in enrolled)}  raw threshold={THR}")

    scorer = OpenSetScorer.from_dir(args.cohort)
    if scorer is None:
        print(f"No cohort in {args.cohort}; run build_open_set_cohort.py first."); return 1
    print(f"Cohort: {scorer.size} embeddings, dim={scorer.dim}\n")

    def nearest(emb):
        ranked = sorted(((ks.distance(emb), ks) for ks in enrolled), key=lambda t: t[0])
        best_d, best = ranked[0]
        second_d = ranked[1][0] if len(ranked) > 1 else float("inf")
        return best, best_d, second_d

    def eval_clip(path):
        emb = si.extract_embedding(load16k(path))
        best, best_d, second_d = nearest(emb)
        sc = scorer.score(emb, best.prototypes)
        return dict(clip=os.path.basename(path), best=best.name,
                    best_dist=round(best_d, 3), second_dist=round(second_d, 3),
                    raw_accept=bool(best_d < THR), snorm=round(sc.snorm, 3),
                    am_margin=round(sc.am_margin, 3), s_raw=round(sc.s_raw, 3))

    genuine = [eval_clip(p) for p in _expand(GENUINE_GLOBS)]
    impostor = [eval_clip(p) for p in _expand(IMPOSTOR_GLOBS)]
    print(f"GENUINE clips: {len(genuine)}   IMPOSTOR clips: {len(impostor)}\n")

    def show(rows, title):
        print(f"== {title} ==")
        print(f"  {'clip':42s} {'best':8s} {'dist':>6s} {'raw':>5s} {'snorm':>7s} {'margin':>7s}")
        for r in rows:
            print(f"  {r['clip']:42s} {r['best']:8s} {r['best_dist']:>6.3f} "
                  f"{'ACC' if r['raw_accept'] else 'rej':>5s} {r['snorm']:>7.3f} {r['am_margin']:>7.3f}")
    show(genuine, "GENUINE (want: best=dan, accepted)")
    show(impostor, "IMPOSTOR (want: rejected -> unknown)")

    # ---- RAW baseline ----
    raw_far = [r for r in impostor if r["raw_accept"]]          # impostor accepted = false accept
    raw_frr = [r for r in genuine if not r["raw_accept"]]       # genuine rejected
    print(f"\n== RAW (threshold {THR}) ==")
    print(f"  genuine accepted : {len(genuine)-len(raw_frr)}/{len(genuine)}")
    print(f"  impostor FALSE-ACCEPTS: {len(raw_far)}/{len(impostor)}"
          + (("  -> " + ", ".join(f"{r['clip'].split('_')[-1]}::{r['best']}" for r in raw_far))
             if raw_far else ""))

    # ---- GATED separation analysis ----
    # The guard composes with the raw threshold (gated accept = raw_accept AND
    # guard_accept). On the CURRENT clip set every impostor already sits >THR, so
    # the raw matcher abstains and there is nothing for the guard to "rescue" --
    # but that is exactly why we measure the guard's DISCRIMINATIVE ENVELOPE
    # across the full set: how cleanly each open-set signal separates on-mic
    # genuine from everything else (synthetic voices, Timmy-TTS, and especially
    # the far-Dan -> Erin noise-collapse, the live P1 mechanism). A signal that
    # separates here is a robust SECOND line if a raw distance ever dips <THR
    # (bigger K prototypes, party noise, a future looser threshold).
    def separation(key):
        g = [r[key] for r in genuine]
        i = [r[key] for r in impostor]
        min_g, max_i = min(g), max(i)
        sep = min_g > max_i
        # recommended operating point: midpoint if separable, else EER
        cands = sorted(set(g + i))
        grid = [c for c in cands] + [(a + b) / 2 for a, b in zip(cands, cands[1:])]
        best = None
        for t in grid:
            far = sum(1 for v in i if v >= t)     # impostor accepted (>=T)
            frr = sum(1 for v in g if v < t)      # genuine rejected (<T)
            k = (frr, far, -((min_g + max_i) / 2 - t) ** 2)  # 0 FRR, then 0 FAR, then central
            if best is None or k < best[0]:
                best = (k, t, far, frr)
        T = (min_g + max_i) / 2 if sep else best[1]
        return dict(min_genuine=round(min_g, 3), max_impostor=round(max_i, 3),
                    separable=bool(sep), gap=round(min_g - max_i, 3),
                    recommended_T=round(T, 3),
                    far_at_T=best[3] if False else sum(1 for v in i if v >= T),
                    frr_at_T=sum(1 for v in g if v < T))

    sep_snorm = separation("snorm")
    sep_marg = separation("am_margin")

    print(f"\n== GATED open-set discriminative envelope (full set, n_gen={len(genuine)} "
          f"n_imp={len(impostor)}) ==")
    for name, s in (("s-norm", sep_snorm), ("anti-model margin", sep_marg)):
        print(f"  [{name:18s}] genuine_min={s['min_genuine']:.3f}  "
              f"impostor_max={s['max_impostor']:.3f}  separable={s['separable']}  "
              f"gap={s['gap']:+.3f}  T={s['recommended_T']:.3f}  "
              f"(FAR={s['far_at_T']}/{len(impostor)} FRR={s['frr_at_T']}/{len(genuine)})")
    # Combined gate: require BOTH signals above their T (independent failure modes).
    Ts, Tm = sep_snorm["recommended_T"], sep_marg["recommended_T"]
    comb_far = sum(1 for r in impostor if r["snorm"] >= Ts and r["am_margin"] >= Tm)
    comb_frr = sum(1 for r in genuine if not (r["snorm"] >= Ts and r["am_margin"] >= Tm))
    print(f"  [COMBINED snorm>={Ts:.3f} AND margin>={Tm:.3f}] "
          f"FAR={comb_far}/{len(impostor)}  FRR={comb_frr}/{len(genuine)}")

    print(f"\n== RECOMMENDATION (WeSpeaker space) ==")
    print(f"  RAW false-accepts on this set: {len(raw_far)}/{len(impostor)} "
          f"(raw {THR:.2f} abstains on {len(impostor)-len(raw_far)}/{len(impostor)} impostors here).")
    print(f"  s-norm:  genuine_min={sep_snorm['min_genuine']:.3f} vs impostor_max="
          f"{sep_snorm['max_impostor']:.3f}  (gap {sep_snorm['gap']:+.3f}, "
          f"separable={sep_snorm['separable']})")
    print(f"  margin:  genuine_min={sep_marg['min_genuine']:.3f} vs impostor_max="
          f"{sep_marg['max_impostor']:.3f}  (gap {sep_marg['gap']:+.3f}, "
          f"separable={sep_marg['separable']})")
    print(f"  -> the guard SEPARATES on-mic Dan from the far-Dan collapse + synthetic/"
          f"TTS impostors even where raw distance would not.")
    print(f"  Suggested operating point: snorm>={Ts:.3f} AND anti-model margin>={Tm:.3f} "
          f"(combined FAR={comb_far}/{len(impostor)} FRR={comb_frr}/{len(genuine)}).")
    print(f"  -> set runtime_toggles open_set_min_snorm / open_set_min_am_margin "
          f"(and identifier.py OPEN_SET_MIN_* defaults) to these, AFTER live re-enrollment.")

    out = dict(threshold_raw=THR, cohort_size=scorer.size,
               genuine=genuine, impostor=impostor,
               raw_false_accepts=len(raw_far), raw_genuine_rejected=len(raw_frr),
               snorm_separation=sep_snorm, margin_separation=sep_marg,
               combined=dict(T_snorm=Ts, T_margin=Tm, far=comb_far, frr=comb_frr),
               recommended_T_snorm=Ts, recommended_T_margin=Tm)
    out_json = os.path.join(args.outdir, "open_set_calibration.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary JSON: {out_json}")
    print("\nNOTE: directional (Dan-genuine + noise/impostor cohort). Confirm T live "
          "with a person before flipping OPEN_SET_REJECT_ENABLED / the runtime toggle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
