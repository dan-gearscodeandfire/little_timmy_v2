#!/usr/bin/env python3
"""Acoustic synthetic-conversation driver for Little Timmy (2026-06-20).

Plays synthetic USER utterances (a Piper *test* voice, NOT skeletor) out Timmy's
own speaker (default PipeWire sink) so the lav mic — parked next to the speaker,
same placement Dan uses — picks them up and they run the FULL production path:
VAD/EOU -> whisper STT -> speaker-ID -> classifier/router -> brain -> skeletor TTS.

This is the OPPOSITE of ops/multi_voice_sweep.py / speaker_loopback_mictest.py:
those MUTE hearing to avoid contaminating memory. Here hearing stays ON on
purpose — we WANT real turns, to harden response/memory/reasoning. The caller is
responsible for cleaning synthetic facts/episodes afterward (id > baseline).

Ground truth comes from two places:
  - WS ws://localhost:8893/ws : transcribed user turn + Timmy's reply text.
  - the journal (parsed separately) : routing / store_fact / grounding internals.

Synthetic voice => speaker-ID = unknown/guest => store_fact subject is the guest,
NOT 'dan' -> real facts (esp. dan.name) are not the clobber target. No real PII.

USAGE
  .venv/bin/python ops/acoustic_convo_driver.py --calibrate "remember my robot is named Sparky"
  .venv/bin/python ops/acoustic_convo_driver.py --scenario /tmp/scenario.json [--voice en_US-ryan-high]
"""
import argparse, asyncio, glob, hashlib, json, os, subprocess, sys, time, wave
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(REPO, "models", "tts", "test_voices")
WS_URL = "ws://localhost:8893/ws"
CACHE = "/tmp/lt_acoustic_wavs"
os.makedirs(CACHE, exist_ok=True)


def synth_wav(onnx_path: str, text: str, out_path: str, length_scale: float = 1.0) -> float:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    voice = PiperVoice.load(onnx_path)
    sr = voice.config.sample_rate
    chunks = [c.audio_float_array
              for c in voice.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale))]
    a = np.concatenate(chunks).astype(np.float32)
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return len(a) / sr


def get_wav(voice: str, text: str, length_scale: float) -> tuple[str, float]:
    onnx = os.path.join(VOICE_DIR, voice + ".onnx")
    key = hashlib.md5(f"{voice}|{length_scale}|{text}".encode()).hexdigest()[:12]
    out = os.path.join(CACHE, f"{key}.wav")
    if os.path.exists(out):
        with wave.open(out, "rb") as w:
            return out, w.getnframes() / w.getframerate()
    dur = synth_wav(onnx, text, out, length_scale)
    return out, dur


def play_wav(path: str) -> None:
    subprocess.run(["pw-play", path], capture_output=True, timeout=120)


async def ws_collector(stop: asyncio.Event, sink: list):
    """Append (recv_time, parsed_or_raw) for every WS message until stop set."""
    import websockets
    try:
        async with websockets.connect(WS_URL, max_size=None) as ws:
            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
                try:
                    sink.append((time.time(), json.loads(raw)))
                except Exception:
                    sink.append((time.time(), {"_raw": raw}))
    except Exception as e:
        sink.append((time.time(), {"_ws_error": str(e)}))


def fmt_msg(m: dict) -> str:
    keys = ("type", "speaker", "speaker_id", "role", "name", "content", "text", "intent", "route")
    short = {k: m[k] for k in keys if k in m}
    return json.dumps(short, ensure_ascii=False) if short else json.dumps(m, ensure_ascii=False)[:200]


async def run_turn(idx, text, expect, voice, length_scale, msgs, reply_window):
    t0 = time.time()
    wav, dur = get_wav(voice, text, length_scale)
    print(f"\n--- TURN {idx}: said={text!r}  (voice={voice}, {dur:.1f}s)")
    if expect:
        print(f"    expect: {expect}")
    mark = len(msgs)
    play_wav(wav)                       # blocking; Timmy hears it after EOU silence
    t_played = time.time()
    deadline = t_played + reply_window
    reply = None
    while time.time() < deadline:
        await asyncio.sleep(0.2)
        for (_, m) in msgs[mark:]:
            if m.get("type") == "turn" and m.get("role") == "assistant":
                reply = m.get("content") or m.get("text") or ""
        if reply is not None:
            break
    # let skeletor TTS finish playing before the next utterance (avoid mic collision)
    if reply:
        words = len(reply.split())
        await asyncio.sleep(min(12.0, max(2.5, words * 0.42)))
    else:
        await asyncio.sleep(2.0)
    new = [m for (ts, m) in msgs[mark:]]
    heard = next((m.get("content") for m in new
                  if m.get("type") == "turn" and m.get("role") == "user"), None)
    route = next((m.get("route") for m in new if m.get("type") == "classifier_metric"), None)
    spk = next((m.get("speaker") for m in new
                if m.get("type") == "turn" and m.get("role") == "user"), None)
    print(f"    heard : {heard!r}  (speaker={spk}, route={route})")
    print(f"    timmy : {reply!r}")
    tool_msgs = [fmt_msg(m) for m in new
                 if m.get("type") not in ("token", "turn", "metrics", "retrieval", "classifier_metric")]
    if tool_msgs:
        print(f"    other : {tool_msgs}")
    return {"idx": idx, "said": text, "expect": expect, "heard": heard,
            "speaker": spk, "route": route, "reply": reply,
            "ws": new, "elapsed": round(time.time() - t0, 1)}


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", metavar="TEXT", help="single utterance, dump raw WS")
    ap.add_argument("--scenario", help="JSON list of {say, expect} (or [str,...])")
    ap.add_argument("--voice", default="en_US-ryan-high")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--reply-window", type=float, default=14.0)
    ap.add_argument("--out", default="/tmp/lt_acoustic_results.json")
    args = ap.parse_args()

    if args.calibrate:
        turns = [{"say": args.calibrate, "expect": "(calibration)"}]
    else:
        data = json.load(open(args.scenario))
        turns = [({"say": t} if isinstance(t, str) else t) for t in data]

    stop = asyncio.Event()
    msgs: list = []
    col = asyncio.create_task(ws_collector(stop, msgs))
    await asyncio.sleep(1.0)  # let WS connect

    results = []
    for i, t in enumerate(turns, 1):
        r = await run_turn(i, t["say"], t.get("expect", ""), args.voice,
                           args.length_scale, msgs, args.reply_window)
        results.append(r)

    await asyncio.sleep(1.0)
    stop.set()
    await col
    json.dump({"voice": args.voice, "results": results,
               "all_ws": [m for (_, m) in msgs]}, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[done] {len(results)} turns; raw -> {args.out}")


if __name__ == "__main__":
    asyncio.run(amain())
