#!/usr/bin/env bash
# booth_launch.sh — desktop launcher for the Little Timmy booth setup.
#
# 1. Ensures the LT stack is running (systemctl start = no-op on already-active units).
# 2. Waits for LT-OS (:8894) and the booth display (:8090) to answer.
# 3. Opens two Chrome windows: LT-OS and the booth display.
#
# Booth display (:8090) uses a self-signed cert, so it opens in a DEDICATED
# Chrome profile with --ignore-certificate-errors. Never add that flag to the
# main profile.
set -u

HOST=192.168.1.156
LTOS_URL="http://${HOST}:8894/"
BOOTH_URL="https://${HOST}:8090/"
BOOTH_PROFILE="$HOME/.config/lt-booth-chrome"
LOG="$HOME/little_timmy/ops/launcher/booth_launch.log"

UNITS=(
  postgresql ollama
  whisper-server.service
  qwen36-server.service
  qwen36-vision-server.service
  qwen3-4b-classifier.service
  qwen3-4b-coref.service
  little-timmy.service
  little-timmy-os.service
  booth-mockup.service
  qwen36-watchdog.service
)

notify() { notify-send -i "$HOME/little_timmy/ops/launcher/lt-booth-icon.png" "LT Booth" "$1" 2>/dev/null; }
log()    { echo "$(date '+%F %T') $1" >>"$LOG"; }

log "launch requested"
notify "Starting Little Timmy stack..."

# Ensure-started (idempotent; does NOT restart anything already running).
if ! sudo -n systemctl start "${UNITS[@]}" 2>>"$LOG"; then
  log "systemctl start failed"
  notify "⚠ systemctl start failed — check ${LOG}"
fi

# Wait for readiness. LT-OS /api/timmy/toggles aggregates :8893, so it's a
# good proxy for the core stack; booth just needs to serve its page.
wait_for() { # url  curl_extra  label  timeout_s
  local url=$1 extra=$2 label=$3 timeout=$4 t=0
  until curl -so /dev/null -m 3 $extra "$url"; do
    sleep 2; t=$((t + 2))
    if (( t >= timeout )); then
      log "$label not ready after ${timeout}s"
      notify "⚠ $label not ready after ${timeout}s — opening anyway"
      return 1
    fi
  done
  log "$label ready after ${t}s"
}

wait_for "http://127.0.0.1:8894/api/timmy/toggles" ""   "LT-OS"         120
wait_for "https://127.0.0.1:8090/"                 "-k" "Booth display" 30

# Window 1: LT-OS in the normal profile.
google-chrome --new-window "$LTOS_URL" >>"$LOG" 2>&1 &

# Window 2: booth display in its own profile (self-signed cert accepted there).
google-chrome --user-data-dir="$BOOTH_PROFILE" --no-first-run \
  --ignore-certificate-errors --new-window "$BOOTH_URL" >>"$LOG" 2>&1 &

log "chrome windows launched"
notify "LT-OS + Booth display opened 🤖"
