#!/usr/bin/env python3
"""Hands-free, Timmy-cued mic-PLACEMENT check (Dan wears the lav).

A single-position spin-off of ops/distance_sweep_cued.py. Instead of asking the
subject to walk a distance curve, this confirms the lav is seated correctly at
ONE worn position near Timmy's speaker: measure the noise floor at that seat,
then cue a few real-voice takes and report SNR + WeSpeaker speaker-ID
distance / margin so Dan can judge the placement, reseat, and repeat.

Reuses the exact contract from distance_sweep_cued.py (read its docstring):
  - hearing muted the whole run (POST /api/hearing enabled=false) so test
    speech is dropped pre-STT and never reaches conversation/memory; ALWAYS
    restored in finally,
  - cues are spoken via /api/announce (force channel, no conversation turn),
  - announce() is NON-BLOCKING with no TTS-done signal, so each cue WAITS OUT a
    timed lead before recording (subject speaks into silence; Timmy's voice
    never bleeds into the capture),
  - capture + 48k->::3 16k->80Hz HP + identifier reused verbatim from
    snr_vs_distance.py so numbers match production exactly.

Each ROUND = fresh floor baseline (reseating changes the floor too) + N takes.
Between rounds the tool cues a reseat and waits --reseat seconds so Dan can move
the mic. Default is a single round / two takes (quick verdict); bump --rounds to
walk through reseating guided by Timmy's voice.

GOOD PLACEMENT (proven 6-09/6-17):
  - floor post-HP near the empty-room baseline (~-69 dBFS); a high floor => gain
    too hot or mic picking up rustle/handling -> reseat,
  - SNR healthy (through-speaker test voices ran 37-43 dB; close-talk should be
    similar or better),
  - nearest enrolled == dan at low distance (~0.10-0.18) with margin to 2nd.
  Match threshold is 0.30 (below = confident known-speaker match).

USAGE
  cd ~/little_timmy && .venv/bin/python ops/mic_placement_cued.py
  .venv/bin/python ops/mic_placement_cued.py --rounds 3 --takes 2 --rec 6
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ops.snr_vs_distance import (  # noqa: E402
    ORCH, _ensure_audio_env, set_hearing,
    highpass_80, to16k, record, rms, dbfs, speech_rms, save_wav,
)

PHRASE = "the quick brown fox jumps over the lazy dog"  # synthetic, no PII
ANNOUNCE_PREFIX_CHARS = 24      # server prepends "This is Claude talking. "
SPEECH_CPS = 12.0               # chars/sec for lead timing (conservative)
LEAD_BUFFER_S = 2.0             # slack so Timmy fully finishes before we record


def _announce(text: str, wait_lead: bool = True) -> None:
    """Speak a cue via Timmy and (default) sleep until it has finished playing."""
    try:
        req = urllib.request.Request(
            f"{ORCH}/api/announce",
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
    except Exception as e:
        print(f"  ! announce failed: {e}")
    if wait_lead:
        lead = (ANNOUNCE_PREFIX_CHARS + len(text)) / SPEECH_CPS + LEAD_BUFFER_S
        time.sleep(lead)


def _resolve_device(hint=None):
    """Re-resolve the PipeWire input BY NAME (indices churn when TTS rebuilds
    the graph). Refresh PortAudio's list each call."""
    import sounddevice as sd
    try:
        sd._terminate(); sd._initialize()
    except Exception:
        pass
    devs = sd.query_devices()
    if hint is not None:
        try:
            return int(hint)
        except ValueError:
            pass
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and d["name"] == "default":
            return i
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and "pipewire" in d["name"].lower():
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description="Timmy-cued mic-placement check (worn lav)")
    ap.add_argument("--rounds", type=int, default=1, help="reseat rounds (cue+wait between)")
    ap.add_argument("--takes", type=int, default=2, help="voice takes per round")
    ap.add_argument("--phrase", default=PHRASE, help="SYNTHETIC phrase (no PII)")
    ap.add_argument("--rec", type=float, default=6.0, help="record seconds per take")
    ap.add_argument("--reseat", type=float, default=10.0, help="seconds to reseat between rounds")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "mic_placement_captures"))
    args = ap.parse_args()

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    from speaker.identifier import SpeakerIdentifier, KNOWN_SPEAKER_THRESHOLD
    si = SpeakerIdentifier()
    si.load_voiceprints()
    enrolled = [ks.name for ks in si._known_speakers]
    print(f"Enrolled: {', '.join(enrolled) or 'none'}  threshold={KNOWN_SPEAKER_THRESHOLD}")
    print(f"Rounds={args.rounds} takes/round={args.takes} rec={args.rec}s "
          f"phrase=\"{args.phrase}\"\n")

    results = []
    try:
        set_hearing(False)
        _announce("Microphone placement check. I am muted and deaf for this "
                  "test. Wear the lav where you want it. When I say NOW, say the "
                  "test phrase twice, clearly: the quick brown fox jumps over "
                  "the lazy dog. Do not talk while I am talking.")

        for rnd in range(1, args.rounds + 1):
            if rnd > 1:
                _announce(f"Round {rnd}. Reseat the microphone now. You have "
                          f"{int(args.reseat)} seconds.")
                time.sleep(args.reseat)

            # --- floor baseline for THIS seat ---
            _announce("Hold still and stay silent for the room tone.")
            dev = _resolve_device(args.device)
            base = to16k(record(dev, 2.5))
            base_hp = highpass_80(base)
            floor = rms(base_hp)
            print(f"=== round {rnd} === floor: raw={dbfs(base):.1f} dBFS  "
                  f"post-HP={dbfs(base_hp):.1f} dBFS")

            for take in range(1, args.takes + 1):
                _announce(f"Take {take}. Say the fox phrase twice. Now.")
                dev = _resolve_device(args.device)
                cap = record(dev, args.rec)
                x16 = to16k(cap)
                x16_hp = highpass_80(x16)
                sp = speech_rms(x16_hp)
                snr = 20.0 * np.log10(sp / (floor + 1e-12))

                emb = si.extract_embedding(x16_hp)
                ranked = sorted(((ks.distance(emb), ks.name) for ks in si._known_speakers),
                                key=lambda t: t[0])
                best = ranked[0] if ranked else (float("nan"), "none")
                second = ranked[1] if len(ranked) > 1 else (float("inf"), "none")
                margin = second[0] - best[0]
                dan_dist = next((d for d, n in ranked if n == "dan"), float("nan"))
                would_match = best[0] < KNOWN_SPEAKER_THRESHOLD

                wav = os.path.join(args.outdir, f"{stamp}_r{rnd}_t{take}.wav")
                save_wav(wav, x16_hp)
                row = dict(round=rnd, take=take, snr_db=round(snr, 1),
                           floor_dbfs=round(dbfs(base_hp), 1),
                           speech_dbfs=round(dbfs(x16_hp), 1),
                           best=best[1], best_dist=round(best[0], 3),
                           second=second[1], second_dist=round(second[0], 3),
                           margin=round(margin, 3), dan_dist=round(dan_dist, 3),
                           matches_known=would_match, wav=wav)
                results.append(row)
                verdict = (f"MATCH {row['best']}" if would_match
                           else f"unknown(>{KNOWN_SPEAKER_THRESHOLD})")
                print(f"  [r{rnd} t{take}] SNR={row['snr_db']:>5} dB | "
                      f"nearest={row['best']} @ {row['best_dist']} | "
                      f"2nd={row['second']} @ {row['second_dist']} | "
                      f"margin={row['margin']} | dan={row['dan_dist']} | {verdict}")

        _announce("Placement check complete. Restoring my hearing now.", wait_lead=False)
        print("\n==== SUMMARY ====")
        print(json.dumps(results, indent=2))
        print("\nGood seat: floor near -69 dBFS, SNR high, nearest=dan at low "
              "distance (~0.10-0.18) with healthy margin. High floor or "
              "dan_dist creeping toward 0.30 => reseat / check gain.")
    finally:
        print("\n>>> Restoring hearing")
        set_hearing(True)


if __name__ == "__main__":
    main()
