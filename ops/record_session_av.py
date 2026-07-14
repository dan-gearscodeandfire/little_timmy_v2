#!/usr/bin/env python3
"""
record_session_av.py — full A/V session recorder for Little Timmy (okDemerzel).

Pulls the Pi's WebRTC H.264 camera feed (https://192.168.1.110:8080/offer) with
aiortc and muxes it, in a single system-ffmpeg process, with two PipeWire audio
tracks into one crash-resilient MPEG-TS file:

    stream 0 (video): scene camera   (from streamerpi WebRTC, VAAPI H.264 re-encode)
    stream 1 (audio): room microphone
    stream 2 (audio): Little Timmy's TTS output (.monitor loopback)

Why re-encode at all: aiortc hands us *decoded* frames, so a literal -c copy of the
Pi's bytes isn't available through this path. But the feed is 640x360@~8fps, so a
VAAPI encode (dedicated video block, not the LLM GPU compute units) is effectively
free. Falls back to libx264 ultrafast if VAAPI init fails.

Fully decoupled from the LT pipeline: separate process, and PipeWire tees the mic
so LT's own capture is unaffected. Video uses the Pi's single WebRTC slot (the Pi
enforces one client at a time), so recording and the live visitor screen can't pull
video simultaneously.

Usage:
    record_session_av.py [--duration SECS] [--out-dir DIR] [--size WxH]
                         [--force] [--no-video] [--enc vaapi|x264]
Runs until --duration elapses or SIGINT/SIGTERM (clean finalize either way).
"""
import argparse, asyncio, os, signal, ssl, subprocess, sys, time
from pathlib import Path

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

PI_OFFER = os.getenv("PI_OFFER_URL", "https://192.168.1.110:8080/offer")
MIC_SRC = os.getenv("MIC_SRC", "alsa_input.pci-0000_c6_00.6.analog-stereo")
MON_SRC = os.getenv("MON_SRC", "alsa_output.pci-0000_c6_00.6.analog-stereo.monitor")
RENDER_NODE = os.getenv("VAAPI_DEVICE", "/dev/dri/renderD128")


def _pulse_env() -> dict:
    env = os.environ.copy()
    uid = os.getuid()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    env.setdefault("PULSE_SERVER", f"unix:{env['XDG_RUNTIME_DIR']}/pulse/native")
    return env


def build_ffmpeg_cmd(out_path: str, w: int, h: int, enc: str, with_video: bool) -> list:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    maps = []
    if with_video:
        # Raw yuv420p frames arrive over the pipe as fast as WebRTC delivers them;
        # wall-clock timestamps make the variable framerate mux correctly and stay
        # aligned with the wall-clock-stamped audio.
        cmd += ["-use_wallclock_as_timestamps", "1",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                "-thread_queue_size", "512", "-i", "pipe:0"]
        maps.append("0:v")
    a_base = 1 if with_video else 0
    cmd += ["-f", "pulse", "-thread_queue_size", "1024", "-i", MIC_SRC,
            "-f", "pulse", "-thread_queue_size", "1024", "-i", MON_SRC]
    maps += [f"{a_base}:a", f"{a_base+1}:a"]
    for m in maps:
        cmd += ["-map", m]
    if with_video:
        if enc == "vaapi":
            cmd += ["-vaapi_device", RENDER_NODE, "-vf", "format=nv12,hwupload",
                    "-c:v", "h264_vaapi", "-b:v", "2M"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                    "-pix_fmt", "yuv420p", "-b:v", "2M"]
        cmd += ["-fps_mode:v", "vfr"]
    cmd += ["-c:a", "aac", "-b:a", "160k",
            "-metadata:s:a:0", "title=room-mic",
            "-metadata:s:a:1", "title=little-timmy-output"]
    cmd.append(out_path)
    return cmd


class Recorder:
    def __init__(self, args):
        self.args = args
        self.w, self.h = map(int, args.size.lower().split("x"))
        self.ff = None
        self.pc = None
        self.frames = 0
        self.stop_evt = asyncio.Event()

    def _start_ffmpeg(self, enc, with_video):
        cmd = build_ffmpeg_cmd(self.out_path, self.w, self.h, enc, with_video)
        stdin = subprocess.PIPE if with_video else subprocess.DEVNULL
        self.ff = subprocess.Popen(cmd, stdin=stdin, env=_pulse_env())
        self.enc = enc
        return cmd

    async def _connect_video(self):
        self.pc = RTCPeerConnection()
        self.pc.addTransceiver("video", direction="recvonly")
        first_frame = asyncio.get_event_loop().create_future()

        @self.pc.on("track")
        def on_track(track):
            async def pump():
                while not self.stop_evt.is_set():
                    try:
                        frame = await track.recv()
                    except Exception:
                        return
                    arr = frame.reformat(width=self.w, height=self.h,
                                         format="yuv420p").to_ndarray(format="yuv420p")
                    if not first_frame.done():
                        first_frame.set_result(True)
                    if self.ff and self.ff.stdin:
                        try:
                            self.ff.stdin.write(arr.tobytes())
                            self.frames += 1
                        except (BrokenPipeError, ValueError):
                            return
            asyncio.ensure_future(pump())

        await self.pc.setLocalDescription(await self.pc.createOffer())
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession() as s:
            async with s.post(PI_OFFER, json={
                "sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type,
                "client_id": self.args.client_id, "force": self.args.force,
            }, ssl=ctx) as r:
                if r.status == 409:
                    raise RuntimeError("Pi WebRTC slot busy (409) — another client holds it; "
                                       "re-run with --force to preempt.")
                if r.status != 200:
                    raise RuntimeError(f"/offer HTTP {r.status}: {(await r.text())[:200]}")
                ans = await r.json()
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        return first_frame

    async def run(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path(self.args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = str(out_dir / f"{ts}.ts")
        with_video = not self.args.no_video

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop_evt.set)

        if with_video:
            try:
                first_frame = await self._connect_video()
            except Exception as e:
                print(f"[video] connect failed: {e}\n[video] falling back to AUDIO-ONLY", flush=True)
                with_video = False

        enc = self.args.enc
        self._start_ffmpeg(enc, with_video)

        if with_video:
            # If VAAPI dies immediately (driver/init), relaunch with libx264.
            await asyncio.sleep(1.5)
            if self.ff.poll() is not None and enc == "vaapi":
                print("[enc] vaapi ffmpeg exited early — retrying with libx264", flush=True)
                self._start_ffmpeg("x264", True)
            try:
                await asyncio.wait_for(first_frame, timeout=8)
                print(f"[video] streaming {self.w}x{self.h} via {self.enc}", flush=True)
            except asyncio.TimeoutError:
                print("[video] no frames in 8s — video track will be empty", flush=True)

        print(f"[rec] -> {self.out_path} (video={'on' if with_video else 'OFF'}, enc={getattr(self,'enc','-')})",
              flush=True)

        try:
            if self.args.duration:
                await asyncio.wait_for(self.stop_evt.wait(), timeout=self.args.duration)
            else:
                await self.stop_evt.wait()
        except asyncio.TimeoutError:
            pass

        await self._shutdown()

    async def _shutdown(self):
        self.stop_evt.set()
        if self.pc:
            try:
                await self.pc.close()
            except Exception:
                pass
        if self.ff:
            if self.ff.stdin:
                try:
                    self.ff.stdin.close()
                except Exception:
                    pass
            # SIGINT => ffmpeg writes trailer / flushes cleanly.
            self.ff.send_signal(signal.SIGINT)
            try:
                self.ff.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.ff.terminate()
        sz = Path(self.out_path).stat().st_size if Path(self.out_path).exists() else 0
        print(f"[done] {self.out_path}  ({sz} bytes, {self.frames} video frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until signal")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/little_timmy/recordings"))
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--force", action="store_true", help="preempt the Pi's single WebRTC slot")
    ap.add_argument("--no-video", action="store_true", help="audio-only")
    ap.add_argument("--enc", choices=["vaapi", "x264"], default="vaapi")
    ap.add_argument("--client-id", default="okdemerzel-recorder")
    args = ap.parse_args()
    asyncio.run(Recorder(args).run())


if __name__ == "__main__":
    main()
