#!/usr/bin/env bash
# record_session.sh — server-side A/V session recorder for Little Timmy (okDemerzel)
#
# Captures, as a single decoupled process (zero coupling to the LT pipeline):
#   track 0 (audio): room microphone           (alsa_input ...analog-stereo)
#   track 1 (audio): Little Timmy's TTS output  (...analog-stereo.monitor)
#
# The .monitor source is a loopback of whatever LT plays to the speakers, so we
# capture LT's voice even though his mic is gated while he speaks. PipeWire tees
# each source to multiple readers, so LT's own capture is unaffected.
#
# Video is intentionally NOT handled here — the only scene video is the Pi's
# WebRTC H.264 feed (https://192.168.1.110:8080/offer), which needs a WebRTC
# client (aiortc), not ffmpeg. See record_session_av.py for that path. This
# script is the reliable, always-works audio core.
#
# Usage:  record_session.sh start | stop | status
#
# Container: MPEG-TS (.ts) — crash-resilient (a killed ffmpeg still leaves a
# playable file, unlike .mp4/.webm which need clean finalization) and carries
# H.264 + AAC, so the same file format extends cleanly when video is added.
set -euo pipefail

MIC_SRC="${MIC_SRC:-alsa_input.pci-0000_c6_00.6.analog-stereo}"
MON_SRC="${MON_SRC:-alsa_output.pci-0000_c6_00.6.analog-stereo.monitor}"
REC_DIR="${REC_DIR:-$HOME/little_timmy/recordings}"

U="$(id -u)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$U}"
export PULSE_SERVER="${PULSE_SERVER:-unix:$XDG_RUNTIME_DIR/pulse/native}"

RUN_DIR="$XDG_RUNTIME_DIR/lt_recorder"
PIDFILE="$RUN_DIR/ffmpeg.pid"
METAFILE="$RUN_DIR/current.path"
LOGFILE="$RUN_DIR/ffmpeg.log"
mkdir -p "$REC_DIR" "$RUN_DIR"

is_running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

start() {
  if is_running; then
    echo "already recording (pid $(cat "$PIDFILE")) -> $(cat "$METAFILE" 2>/dev/null)"
    return 0
  fi
  local ts out
  ts="$(date +%Y%m%d-%H%M%S)"
  out="$REC_DIR/$ts.ts"

  # -thread_queue_size guards against 'input buffer exhausted' warnings when one
  # source stalls (e.g. the sink suspends between LT utterances).
  setsid ffmpeg -hide_banner -loglevel warning -y \
    -f pulse -thread_queue_size 1024 -i "$MIC_SRC" \
    -f pulse -thread_queue_size 1024 -i "$MON_SRC" \
    -map 0:a -map 1:a \
    -c:a aac -b:a 160k \
    -metadata:s:a:0 title="room-mic" \
    -metadata:s:a:1 title="little-timmy-output" \
    -metadata title="LT session $ts" \
    "$out" </dev/null >"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  echo "$out" > "$METAFILE"

  sleep 1.2
  if is_running; then
    echo "recording -> $out (pid $(cat "$PIDFILE"))"
  else
    echo "FAILED to start — last log lines:"; tail -n 8 "$LOGFILE" 2>/dev/null
    rm -f "$PIDFILE" "$METAFILE"; return 1
  fi
}

stop() {
  if ! is_running; then echo "not recording"; return 0; fi
  local pid out; pid="$(cat "$PIDFILE")"; out="$(cat "$METAFILE" 2>/dev/null || true)"
  # SIGINT = ffmpeg finalizes cleanly (writes trailer, flushes).
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 25); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
  kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  rm -f "$PIDFILE" "$METAFILE"
  echo "stopped."
  [[ -n "$out" ]] && ls -l "$out" 2>/dev/null || true
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
