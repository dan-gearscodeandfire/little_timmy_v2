#!/usr/bin/env bash
# Qwen3.6 GPU brain watchdog.
#
# Why: okDemerzel has hard-frozen twice (2026-05-12, 2026-06-06) when the
# single-slot (-np 1) Strix Halo Vulkan brain on :8083 wedged under churning
# GPU load and took the whole box with it. RCA: thinking-ON memory-extraction
# cancel-churn stacking abandoned-but-still-computing generations on the one
# Vulkan slot. A soft wedge of llama-server blocks its main loop -- which also
# serves /health -- so a /health timeout is a reliable early signal. Catching
# it here and restarting the unit clears the Vulkan context BEFORE a soft wedge
# escalates to an unrecoverable amdgpu hard-hang.
#
# Scope/limits: once the kernel itself hard-hangs, nothing in userspace can
# help -- only a physical reboot. This watchdog acts only in the pre-escalation
# window (server wedged, box still alive), which is the only window we can act
# in. The MAX_RESTARTS cap exists so that if restarts don't help (GPU already
# hard-hung) we stop thrashing and log CRITICAL for a human.
#
# Detection is /health-only on purpose: this llama-server runs without --slots,
# and an inference probe on the shared -np1 slot would evict the conversation
# KV cache on every poll. If /health ever stops correlating with compute
# liveness, add a low-frequency inference probe behind its own flag.
#
# Ref: Obsidian Zettelkasten/okdemerzel-freeze-rca-extraction-cancel-churn-2026-06-06

set -uo pipefail

TARGETS="${WATCHDOG_TARGETS:-8083:qwen36-server.service 8084:qwen36-vision-server.service}"
INTERVAL="${WATCHDOG_INTERVAL:-15}"                  # seconds between poll rounds
TIMEOUT="${WATCHDOG_TIMEOUT:-5}"                     # per-probe curl timeout
FAIL_THRESHOLD="${WATCHDOG_FAIL_THRESHOLD:-4}"       # consecutive fails -> restart (~INTERVAL*THRESHOLD of non-response)
RECOVER_TIMEOUT="${WATCHDOG_RECOVER_TIMEOUT:-300}"   # max seconds to wait for /health after a restart (model reload)
MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-3}"           # restarts allowed per unit within MAX_WINDOW
MAX_WINDOW="${WATCHDOG_MAX_WINDOW:-1800}"            # rolling window (s) for the cap
STARTUP_GRACE="${WATCHDOG_STARTUP_GRACE:-120}"       # quiet period at start so a boot-time model load isn't flagged as a wedge
MARKER_DIR="${WATCHDOG_MARKER_DIR:-/var/log/qwen36-watchdog}"

mkdir -p "$MARKER_DIR" 2>/dev/null || true
log() { echo "$(date '+%F %T') [gpu-watchdog] $*"; }

declare -A FAILS
declare -A RESTART_TIMES   # unit -> space-separated epoch list of recent restarts

probe() { curl -fsS -m "$TIMEOUT" -o /dev/null "http://localhost:$1/health" 2>/dev/null; }

wait_for_recover() {  # $1=port $2=unit ; 0 if healthy within RECOVER_TIMEOUT
  local port="$1" unit="$2" start now
  start=$(date +%s)
  while true; do
    sleep "$INTERVAL"
    if probe "$port"; then
      log "$unit (port $port) healthy again after restart"
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= RECOVER_TIMEOUT )); then
      log "ERROR: $unit (port $port) still not healthy ${RECOVER_TIMEOUT}s after restart"
      return 1
    fi
  done
}

under_cap() {  # $1=unit ; prunes the window, returns 0 if a restart is still allowed
  local unit="$1" now t recent="" count
  now=$(date +%s)
  for t in ${RESTART_TIMES[$unit]:-}; do (( now - t < MAX_WINDOW )) && recent+="$t "; done
  RESTART_TIMES[$unit]="$recent"
  count=$(wc -w <<<"$recent")
  (( count < MAX_RESTARTS ))
}

handle_wedge() {  # $1=port $2=unit
  local port="$1" unit="$2" now
  if ! under_cap "$unit"; then
    log "CRITICAL: $unit wedged but already restarted ${MAX_RESTARTS}x within ${MAX_WINDOW}s -- NOT restarting. GPU likely hard-hung; manual reboot may be required."
    echo "$(date '+%F %T') giveup $unit" >>"$MARKER_DIR/events.log"
    FAILS[$port]=0
    return
  fi
  log "WARN: $unit (port $port) failed ${FAIL_THRESHOLD} consecutive /health checks -- restarting unit."
  echo "$(date '+%F %T') restart $unit (port $port)" >>"$MARKER_DIR/events.log"
  if systemctl restart "$unit"; then
    now=$(date +%s); RESTART_TIMES[$unit]="${RESTART_TIMES[$unit]:-}$now "
    log "restart $unit issued; waiting for model reload (up to ${RECOVER_TIMEOUT}s)"
    wait_for_recover "$port" "$unit" || true
  else
    log "ERROR: systemctl restart $unit failed (rc=$?)"
  fi
  FAILS[$port]=0
}

log "starting -- targets: [$TARGETS] interval=${INTERVAL}s timeout=${TIMEOUT}s fail_threshold=${FAIL_THRESHOLD} recover=${RECOVER_TIMEOUT}s cap=${MAX_RESTARTS}/${MAX_WINDOW}s grace=${STARTUP_GRACE}s"
for spec in $TARGETS; do FAILS["${spec%%:*}"]=0; done

# Boot/startup grace: don't flag a slow model load as a wedge. In-script (not
# ExecStartPre) so it doesn't count against systemd's start timeout.
if (( STARTUP_GRACE > 0 )); then
  log "startup grace: waiting ${STARTUP_GRACE}s before watching"
  sleep "$STARTUP_GRACE"
fi

while true; do
  for spec in $TARGETS; do
    port="${spec%%:*}"; unit="${spec##*:}"
    if probe "$port"; then
      if (( ${FAILS[$port]:-0} > 0 )); then log "$unit (port $port) recovered after ${FAILS[$port]} fail(s)"; fi
      FAILS[$port]=0
    else
      FAILS[$port]=$(( ${FAILS[$port]:-0} + 1 ))
      log "$unit (port $port) /health FAIL ${FAILS[$port]}/${FAIL_THRESHOLD}"
      (( ${FAILS[$port]} >= FAIL_THRESHOLD )) && handle_wedge "$port" "$unit"
    fi
  done
  sleep "$INTERVAL"
done
