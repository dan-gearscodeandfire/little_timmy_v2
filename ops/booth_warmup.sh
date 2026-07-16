#!/usr/bin/env bash
# Booth morning warm-up (2026-07-16): the first LT turn after idle re-prefills
# the FULL conversation context on :8083 (single slot, -np 1) at ~400-500 t/s
# — ~20k tokens = the live-observed 42s first-token spike (backlog 7-16 01:04).
# One injected turn before doors open eats that cost so the first visitor
# doesn't. Timmy will audibly reply once — run it before the crowd.
#
# NOTE: anything else that talks to :8083 (hermes-gateway) evicts LT's KV
# cache and re-introduces the spike mid-show. During show hours consider:
#   sudo systemctl stop hermes-gateway.service
#
# Usage: ./ops/booth_warmup.sh   (from ~/little_timmy on okdemerzel)
set -euo pipefail

echo "[warmup] injecting one turn (Timmy will reply out loud)..."
t0=$(date +%s)
curl -s -X POST http://localhost:8893/api/announce \
  -H 'Content-Type: application/json' \
  -d '{"text":"Good morning Timmy. Quick systems check before we open — say hello.","inject":true}' \
  && echo
t1=$(date +%s)
echo "[warmup] announce accepted after $((t1 - t0))s. Watch for Timmy's reply;"
echo "[warmup] if first token took tens of seconds, that was the cold prefill"
echo "[warmup] the visitors now won't pay. Re-run after any Hermes/:8083 use."
