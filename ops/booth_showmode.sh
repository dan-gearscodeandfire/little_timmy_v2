#!/usr/bin/env bash
# Booth show-mode slot protection (2026-07-16).
#
# WHY: LT's brain runs on :8083 as a SINGLE llama.cpp slot (-np 1, do NOT bump
# on Strix Halo Vulkan). Anything else that talks to :8083 -- chiefly
# hermes-gateway -- EVICTS LT's KV cache. The next LT turn then cold-re-prefills
# the full ~20-65K context, the live-observed 29-42s first-token spike (backlog
# 7-16 01:04 + Erin post-enroll dead-window 14:03). The prompt is already
# prefix-stable (roster sits in the regenerated tail, system[0] is cached
# forever) -- so the spike is EVICTION, not layout. The only cure during a show
# is to stop the evictor and keep the slot warm.
#
# 'on'  : stop hermes-gateway (free the slot for LT alone) + warm LT's KV once.
# 'off' : restart hermes-gateway (Hermes messaging back online after the show).
#
# Usage (from ~/little_timmy on okdemerzel; phone-friendly one-liner):
#   ./ops/booth_showmode.sh on     # before doors open
#   ./ops/booth_showmode.sh off    # after pack-up
set -euo pipefail

cmd="${1:-}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$cmd" in
  on)
    echo "[showmode] stopping hermes-gateway to free LT's :8083 slot..."
    sudo -n systemctl stop hermes-gateway.service
    echo "[showmode] hermes-gateway: $(systemctl is-active hermes-gateway.service || true)"
    echo "[showmode] warming LT's KV cache (Timmy replies once out loud)..."
    "$here/ops/booth_warmup.sh"
    echo "[showmode] ON. LT owns :8083. Re-run 'on' if anything else hit :8083."
    ;;
  off)
    echo "[showmode] restarting hermes-gateway (messaging back online)..."
    sudo -n systemctl start hermes-gateway.service
    echo "[showmode] hermes-gateway: $(systemctl is-active hermes-gateway.service || true)"
    echo "[showmode] OFF. Note: the next LT turn may pay one cold re-prefill as"
    echo "[showmode] Hermes touches :8083 again -- expected, post-show."
    ;;
  *)
    echo "usage: $0 on|off" >&2
    exit 2
    ;;
esac
