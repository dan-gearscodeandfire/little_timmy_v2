#!/usr/bin/env bash
# record_booth.sh — record the COMPOSITED booth display (okDemerzel monitor) + audio → .mp4
#
# Captures, as a single decoupled ffmpeg process (zero coupling to the LT pipeline):
#   video   : ffmpeg kmsgrab of the physical monitor scanout (/dev/dri/card1),
#             i.e. exactly what the booth_mockup (:8090, 40K overlay, scan-lines)
#             Chrome window shows — the overlay only exists as composited browser
#             pixels, so screen capture is the ONLY way to record it.
#             (record_session_av.py pulls the raw Pi WebRTC feed = no overlay.)
#   track 0 : room microphone            (alsa_input ...analog-stereo)
#   track 1 : Little Timmy's TTS output  (...analog-stereo.monitor loopback —
#             captures LT even while his mic is gated; PipeWire tees sources,
#             so LT's own capture is unaffected)
#
# Requirements / notes:
#   * kmsgrab needs CAP_SYS_ADMIN → ffmpeg runs via `sudo -n` (passwordless).
#     Do NOT setcap the system ffmpeg.
#   * The booth window must be OPEN + FULL-SCREEN on the monitor (whole-monitor
#     grab). Launch it with the dedicated profile (no cert interstitial, no
#     flag-warning banner in the recording):
#       google-chrome --user-data-dir=$HOME/.config/lt-booth-chrome --no-first-run \
#         --ignore-certificate-errors --test-type --ozone-platform=wayland \
#         --start-fullscreen --new-window https://192.168.1.156:8090/
#     (or the desktop launcher ops/launcher/booth_launch.sh + F11).
#   * ONE booth page only! Two open booth pages fight over the Pi's single
#     WebRTC slot (bounded-takeover ping-pong) and the on-screen one can end up
#     parked on "Live feed is in use on another display."
#   * Chrome runs native-Wayland → x11grab can NOT see it; kmsgrab is
#     compositor-agnostic (reads the DRM scanout).
#   * Records to crash-safe .mkv, then `stop` remuxes (-c copy) to .mp4 with
#     +faststart. A crash mid-recording still leaves a playable .mkv.
#   * 30 fps default (screen sampling rate — independent of the ~8fps WebRTC
#     feed inside the page; overlay animations stay smooth). h264_vaapi encode
#     is hardware and ~free. Override with FPS=/VBITRATE= env vars.
#
# Usage:  record_booth.sh start | stop | status
set -euo pipefail

MIC_SRC="${MIC_SRC:-alsa_input.pci-0000_c6_00.6.analog-stereo}"
MON_SRC="${MON_SRC:-alsa_output.pci-0000_c6_00.6.analog-stereo.monitor}"
REC_DIR="${REC_DIR:-$HOME/little_timmy/recordings}"
DRM_DEV="${DRM_DEV:-/dev/dri/card1}"
FPS="${FPS:-30}"
VBITRATE="${VBITRATE:-5M}"

U="$(id -u)"
ME="$(id -un)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$U}"
export PULSE_SERVER="${PULSE_SERVER:-unix:$XDG_RUNTIME_DIR/pulse/native}"

RUN_DIR="$XDG_RUNTIME_DIR/lt_booth_recorder"
PIDFILE="$RUN_DIR/ffmpeg.pid"
METAFILE="$RUN_DIR/current.path"
LOGFILE="$RUN_DIR/ffmpeg.log"
mkdir -p "$REC_DIR" "$RUN_DIR"

# ffmpeg runs as root (sudo for kmsgrab) → plain `kill -0` gets EPERM; use /proc.
is_running() { [[ -f "$PIDFILE" ]] && [[ -d "/proc/$(cat "$PIDFILE" 2>/dev/null)" ]]; }

start() {
  if is_running; then
    echo "already recording (pid $(cat "$PIDFILE")) -> $(cat "$METAFILE" 2>/dev/null)"
    return 0
  fi
  if ! pgrep -f 'lt-booth-chrome' >/dev/null; then
    echo "WARNING: no lt-booth-chrome window found — you'll record whatever is on the monitor." >&2
    echo "         Launch the booth first (see header of this script)." >&2
  fi
  local ts out
  ts="$(date +%Y%m%d-%H%M%S)"
  out="$REC_DIR/booth-$ts.mkv"

  # Root shell writes its own pid then execs into ffmpeg, so PIDFILE holds the
  # real (root) ffmpeg pid. -thread_queue_size on kmsgrab is REQUIRED: default 8
  # blocks the mux loop against the pulse inputs and drops video to ~12fps.
  sudo -n env XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" PULSE_SERVER="$PULSE_SERVER" \
    bash -c "echo \$\$ > '$PIDFILE'; exec ffmpeg -hide_banner -loglevel warning -y \
      -thread_queue_size 64 -device '$DRM_DEV' -framerate '$FPS' -f kmsgrab -i - \
      -f pulse -thread_queue_size 1024 -i '$MIC_SRC' \
      -f pulse -thread_queue_size 1024 -i '$MON_SRC' \
      -vf 'hwmap=derive_device=vaapi,scale_vaapi=format=nv12' \
      -c:v h264_vaapi -b:v '$VBITRATE' \
      -c:a aac -b:a 160k \
      -map 0:v -map 1:a -map 2:a \
      -metadata:s:a:0 title=room-mic \
      -metadata:s:a:1 title=little-timmy-output \
      -metadata title='LT booth session $ts' \
      '$out'" </dev/null >"$LOGFILE" 2>&1 &
  echo "$out" > "$METAFILE"

  sleep 1.5
  if is_running; then
    echo "recording -> $out (pid $(cat "$PIDFILE"), ${FPS}fps kmsgrab + 2 audio)"
  else
    echo "FAILED to start — last log lines:"; tail -n 8 "$LOGFILE" 2>/dev/null
    rm -f "$PIDFILE" "$METAFILE"; return 1
  fi
}

stop() {
  if ! is_running; then echo "not recording"; return 0; fi
  local pid out mp4; pid="$(cat "$PIDFILE")"; out="$(cat "$METAFILE" 2>/dev/null || true)"
  # SIGINT = ffmpeg finalizes cleanly (writes trailer, flushes).
  sudo -n kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 25); do [[ -d "/proc/$pid" ]] || break; sleep 0.2; done
  [[ -d "/proc/$pid" ]] && sudo -n kill -TERM "$pid" 2>/dev/null || true
  rm -f "$PIDFILE" "$METAFILE"
  echo "stopped."
  if [[ -n "$out" && -f "$out" ]]; then
    sudo -n chown "$ME" "$out"
    mp4="${out%.mkv}.mp4"
    # -map 0 is REQUIRED: default stream selection keeps only ONE audio track.
    if ffmpeg -hide_banner -loglevel error -y -i "$out" -map 0 -c copy -movflags +faststart "$mp4"; then
      rm -f "$out"
      echo "finalized -> $mp4"
      ls -l "$mp4"
    else
      echo "remux failed — crash-safe original kept: $out"
      ls -l "$out"
    fi
  fi
}

status() {
  if is_running; then
    local out; out="$(cat "$METAFILE" 2>/dev/null)"
    echo "recording (pid $(cat "$PIDFILE")) -> $out"
    [[ -n "$out" ]] && ls -l "$out" 2>/dev/null || true
  else
    echo "idle"
  fi
}

case "${1:-status}" in
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
