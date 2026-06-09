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
# Detection is /health-only by default: this llama-server runs without --slots,
# and an inference probe on the shared -np1 slot would evict the conversation
# KV cache on every poll.
#
# BUT /health liveness != GPU-compute liveness. On 2026-06-08 okDemerzel froze a
# THIRD time (kernel hung_task: Xwayland + amdgpu kworkers blocked >122s) while
# /health on :8083/:8084 stayed 200 the whole time -- the HTTP accept loop kept
# answering while the Vulkan compute path wedged underneath it. This watchdog
# never fired (0 FAIL lines, empty events.log). That is exactly the "if /health
# stops correlating with compute liveness, add a low-frequency inference probe"
# caveat above, realized.
#
# So there is now an OPTIONAL, opt-in inference probe (WATCHDOG_INFERENCE_PROBE=1)
# that issues a low-frequency 1-token completion to catch a compute wedge that
# /health misses. It targets the :8084 VISION server by default, NOT :8083 --
# the vision server holds no long-lived conversation KV cache to evict, so a
# tiny periodic completion is safe there, and a wedge of the shared GPU shows up
# on :8084's compute path regardless. Off by default; behaviour is unchanged
# until the flag is set.
#
# Refs: Obsidian Zettelkasten/okdemerzel-freeze-rca-extraction-cancel-churn-2026-06-06
#       Obsidian Zettelkasten/okdemerzel-freeze-rca-watchdog-health-blindspot-2026-06-08

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

# --- Optional GPU-compute liveness probe (catches a wedge /health can't see) ---
# OFF by default; opt in with WATCHDOG_INFERENCE_PROBE=1. Probes the VISION
# server (:8084) so the :8083 conversation KV cache is never evicted.
INFERENCE_PROBE="${WATCHDOG_INFERENCE_PROBE:-0}"                 # 1 = enable the inference probe
INFERENCE_PORT="${WATCHDOG_INFERENCE_PORT:-8084}"               # which llama.cpp server to inference-probe (KV-cache-safe one)
INFERENCE_UNIT="${WATCHDOG_INFERENCE_UNIT:-qwen36-vision-server.service}"  # unit restarted if the inference probe wedges
INFERENCE_INTERVAL="${WATCHDOG_INFERENCE_INTERVAL:-60}"         # seconds between inference probes (a 1-token call still costs a GPU forward pass)
INFERENCE_TIMEOUT="${WATCHDOG_INFERENCE_TIMEOUT:-20}"           # per-probe timeout; healthy 1-token completion returns <2s, headroom for an in-flight VLM call
INFERENCE_FAIL_THRESHOLD="${WATCHDOG_INFERENCE_FAIL_THRESHOLD:-2}"  # consecutive fails -> restart (~INTERVAL*THRESHOLD of confirmed compute-dead)
INFERENCE_PATH="${WATCHDOG_INFERENCE_PATH:-/completion}"        # llama.cpp native completion endpoint
# JSON body kept as a literal (single-quoted) to avoid brace/quote escaping in ${:-} defaults; override wholesale with WATCHDOG_INFERENCE_PAYLOAD.
INFERENCE_PAYLOAD="${WATCHDOG_INFERENCE_PAYLOAD:-DEFAULT}"
[[ "$INFERENCE_PAYLOAD" == DEFAULT ]] && INFERENCE_PAYLOAD='{"prompt":"ping","n_predict":1,"temperature":0,"cache_prompt":false}'

mkdir -p "$MARKER_DIR" 2>/dev/null || true
log() { echo "$(date '+%F %T') [gpu-watchdog] $*"; }

declare -A FAILS
declare -A RESTART_TIMES   # unit -> space-separated epoch list of recent restarts

probe() { curl -fsS -m "$TIMEOUT" -o /dev/null "http://localhost:$1/health" 2>/dev/null; }

# GPU-compute liveness: a real 1-token forward pass. Hangs (and times out) if the
# Vulkan compute path is wedged even while /health still answers 200. -f makes a
# non-200 (e.g. wrong endpoint) count as a fail -- intentional, surfaces misconfig.
inference_probe() {  # $1=port ; 0 if the completion returns within INFERENCE_TIMEOUT
  curl -fsS -m "$INFERENCE_TIMEOUT" -H 'Content-Type: application/json' \
    -d "$INFERENCE_PAYLOAD" -o /dev/null "http://localhost:$1$INFERENCE_PATH" 2>/dev/null
}

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

handle_wedge() {  # $1=port $2=unit $3=reason (for logs). Caller resets its own fail counter afterward.
  local port="$1" unit="$2" reason="$3" now
  if ! under_cap "$unit"; then
    log "CRITICAL: $unit wedged ($reason) but already restarted ${MAX_RESTARTS}x within ${MAX_WINDOW}s -- NOT restarting. GPU likely hard-hung; manual reboot may be required."
    echo "$(date '+%F %T') giveup $unit" >>"$MARKER_DIR/events.log"
    return
  fi
  log "WARN: $unit (port $port) $reason -- restarting unit."
  echo "$(date '+%F %T') restart $unit (port $port)" >>"$MARKER_DIR/events.log"
  if systemctl restart "$unit"; then
    now=$(date +%s); RESTART_TIMES[$unit]="${RESTART_TIMES[$unit]:-}$now "
    log "restart $unit issued; waiting for model reload (up to ${RECOVER_TIMEOUT}s)"
    wait_for_recover "$port" "$unit" || true
  else
    log "ERROR: systemctl restart $unit failed (rc=$?)"
  fi
}

log "starting -- targets: [$TARGETS] interval=${INTERVAL}s timeout=${TIMEOUT}s fail_threshold=${FAIL_THRESHOLD} recover=${RECOVER_TIMEOUT}s cap=${MAX_RESTARTS}/${MAX_WINDOW}s grace=${STARTUP_GRACE}s"
if (( INFERENCE_PROBE )); then
  log "inference probe ENABLED -- ${INFERENCE_UNIT} (port ${INFERENCE_PORT}${INFERENCE_PATH}) every ${INFERENCE_INTERVAL}s timeout=${INFERENCE_TIMEOUT}s fail_threshold=${INFERENCE_FAIL_THRESHOLD}"
else
  log "inference probe disabled (set WATCHDOG_INFERENCE_PROBE=1 to catch GPU-compute wedges /health misses)"
fi
for spec in $TARGETS; do FAILS["${spec%%:*}"]=0; done
IFAILS=0          # consecutive inference-probe failures
LAST_INFERENCE=0  # epoch of last inference probe (set after grace below)

# Boot/startup grace: don't flag a slow model load as a wedge. In-script (not
# ExecStartPre) so it doesn't count against systemd's start timeout.
if (( STARTUP_GRACE > 0 )); then
  log "startup grace: waiting ${STARTUP_GRACE}s before watching"
  sleep "$STARTUP_GRACE"
fi
LAST_INFERENCE=$(date +%s)  # first inference probe fires one INFERENCE_INTERVAL after watching starts

while true; do
  for spec in $TARGETS; do
    port="${spec%%:*}"; unit="${spec##*:}"
    if probe "$port"; then
      if (( ${FAILS[$port]:-0} > 0 )); then log "$unit (port $port) recovered after ${FAILS[$port]} fail(s)"; fi
      FAILS[$port]=0
    else
      FAILS[$port]=$(( ${FAILS[$port]:-0} + 1 ))
      log "$unit (port $port) /health FAIL ${FAILS[$port]}/${FAIL_THRESHOLD}"
      if (( ${FAILS[$port]} >= FAIL_THRESHOLD )); then
        handle_wedge "$port" "$unit" "failed ${FAILS[$port]} consecutive /health checks"
        FAILS[$port]=0
      fi
    fi
  done

  # Optional GPU-compute liveness probe, on its own (lower) cadence.
  if (( INFERENCE_PROBE )); then
    now=$(date +%s)
    if (( now - LAST_INFERENCE >= INFERENCE_INTERVAL )); then
      LAST_INFERENCE=$now
      if inference_probe "$INFERENCE_PORT"; then
        if (( IFAILS > 0 )); then log "$INFERENCE_UNIT (port $INFERENCE_PORT) inference recovered after ${IFAILS} fail(s)"; fi
        IFAILS=0
      else
        IFAILS=$(( IFAILS + 1 ))
        log "$INFERENCE_UNIT (port $INFERENCE_PORT) inference probe FAIL ${IFAILS}/${INFERENCE_FAIL_THRESHOLD} (compute wedge? /health can still be 200)"
        if (( IFAILS >= INFERENCE_FAIL_THRESHOLD )); then
          handle_wedge "$INFERENCE_PORT" "$INFERENCE_UNIT" "failed ${IFAILS} consecutive inference probes (GPU compute wedge; /health green)"
          IFAILS=0
        fi
      fi
    fi
  fi

  sleep "$INTERVAL"
done
