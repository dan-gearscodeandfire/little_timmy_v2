#!/usr/bin/env python3
"""Hands-free, Timmy-cued real-voice SNR-vs-distance sweep.

Same measurement as the interactive ops/snr_vs_distance.py (how a real voice
degrades with distance from the lav chain, and what that does to the
WeSpeaker speaker-ID distance / margin — the P1 misID quantity), but driven
entirely through Timmy's SPEAKER so no keyboard/TTY is needed (Claude Code's
shell is not a TTY, so input() prompts can't be used).

Per the proven enroll_voiced.py pattern:
  - hearing is muted the whole run (POST /api/hearing enabled=false) so the
    subject's speech is dropped pre-STT -> never reaches conversation/memory,
  - each cue is spoken via /api/announce (force channel, no conversation turn),
  - announce() is NON-BLOCKING and there is no TTS-done signal, so we WAIT OUT
    each cue with a timed lead before recording (subject speaks into silence;
    Timmy's voice never bleeds into the capture),
  - capture + 48k->::3 16k->80Hz HP + identifier are reused verbatim from
    snr_vs_distance.py so distances match production exactly,
  - hearing ALWAYS restored in finally.

Subject should be 'dan' (enrolled) so the curve shows his own voice rising in
distance and the margin to the 2nd-nearest collapsing.

USAGE
  cd ~/little_timmy && .venv/bin/python ops/distance_sweep_cued.py
  .venv/bin/python ops/distance_sweep_cued.py --distances "6in,3ft,8ft,15ft" \
      --rec 6 --move 7
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
    ORCH, TARGET_SR, _ensure_audio_env, set_hearing,
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
    ap = argparse.ArgumentParser(description="Timmy-cued real-voice distance sweep")
    ap.add_argument("--distances", default="6in,3ft,8ft,15ft",
                    help="comma list of distance labels")
    ap.add_argument("--phrase", default=PHRASE, help="SYNTHETIC phrase (no PII)")
    ap.add_argument("--rec", type=float, default=6.0, help="record seconds per distance")
    ap.add_argument("--move", type=float, default=7.0, help="seconds to walk to each distance")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=os.path.join(REPO, "ops", "snr_captures"))
    args = ap.parse_args()

    _ensure_audio_env()
    os.makedirs(args.outdir, exist_ok=True)
    distances = [d.strip() for d in args.distances.split(",") if d.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    from speaker.identifier import SpeakerIdentifier, KNOWN_SPEAKER_THRESHOLD
    si = SpeakerIdentifier()
    si.load_voiceprints()
    enrolled = [ks.name for ks in si._known_speakers]
    print(f"Enrolled: {', '.join(enrolled) or 'none'}  threshold={KNOWN_SPEAKER_THRESHOLD}")
    print(f"Distances: {distances}  phrase=\"{args.phrase}\"\n")

    results = []
    try:
        set_hearing(False)
        _announce("Distance sweep starting. I am muted and deaf for this test. "
                  "The test phrase is: the quick brown fox jumps over the lazy "
                  "dog. For each distance I will tell you where to stand, then "
                  "say the word NOW. Only after you hear NOW, say the fox phrase "
                  "twice, clearly. Do not talk while I am talking.")

        # --- baseline ---
        _announce("First, stay silent for the room tone.")
        dev = _resolve_device(args.device)
        base = to16k(record(dev, 2.5))
        base_hp = highpass_80(base)
        floor = rms(base_hp)
        print(f"baseline floor: raw={dbfs(base):.1f} dBFS  post-HP={dbfs(base_hp):.1f} dBFS\n")

        for dist in distances:
            _announce(f"Move to {dist} from the microphone.")
            time.sleep(args.move)  # walk time (cue lead already waited)
            # Short go-cue ending in "now"; do NOT speak the phrase (so the
            # subject can't talk over it). Window opens the instant Timmy stops.
            _announce(f"At {dist}. Say the fox phrase twice. Now.")
            # cue lead already waited inside _announce -> Timmy is silent; record subject
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

            wav = os.path.join(args.outdir, f"{stamp}_cued_{dist}.wav")
            save_wav(wav, x16_hp)
            row = dict(distance=dist, snr_db=round(snr, 1),
                       speech_dbfs=round(dbfs(x16_hp), 1),
                       best=best[1], best_dist=round(best[0], 3),
                       second=second[1], second_dist=round(second[0], 3),
                       margin=round(margin, 3), dan_dist=round(dan_dist, 3),
                       matches_known=would_match, wav=wav)
            results.append(row)
            print(f"[{dist}] SNR={row['snr_db']:>5} dB | nearest={row['best']} @ "
                  f"{row['best_dist']} | 2nd={row['second']} @ {row['second_dist']} | "
                  f"margin={row['margin']} | dan={row['dan_dist']} | "
                  f"{'MATCH '+row['best'] if would_match else 'unknown(>thr)'}")

        _announce("Sweep complete. Restoring my hearing now.", wait_lead=False)
        print("\n==== SUMMARY (best_dist rising + margin collapsing = the P1 rot curve) ====")
        print(json.dumps(results, indent=2))
        print("\nInterpretation:")
        print("  - best_dist climbing toward/over 0.30 with distance = embedding rot.")
        print("  - margin shrinking = the window where an off-mic guest collapses")
        print("    onto an enrolled voiceprint (the P1 misID).")
        print("  - dan_dist is the subject's own enrolled-set distance at each range.")
    finally:
        print("\n>>> Restoring hearing")
        set_hearing(True)


if __name__ == "__main__":
    main()
