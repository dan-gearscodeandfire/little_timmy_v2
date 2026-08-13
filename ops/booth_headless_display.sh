#!/usr/bin/env bash
# booth_headless_display.sh — render the booth mockup on a VIRTUAL 1920x1080 screen.
#
# WHY: the booth stream used to require the physical DP-1 monitor (kmsgrab of the
# DRM scanout). That coupled the RTSP feed to display hardware — unplugging the
# second monitor silently switched the capture to the 2560x1440 main desktop, and
# closing the browser silently streamed the wallpaper. This runs the booth on an
# Xvfb screen instead: no physical monitor required, nothing on the primary
# monitor, and the capture cannot wander onto the wrong output.
#
# Pairs with ops/booth_rtsp_publish.sh (SRC=x11), which x11grabs this display.
# Same Xvfb pattern already used on this box for headless Obsidian (:99).
#
# ⚠️ SINGLE WEBRTC SLOT: streamerpi serves ONE peer. This headless page takes the
#    feed. Opening a SECOND booth page (e.g. on a real monitor) will fight it —
#    whichever connects last wins and the other parks on "Live feed is in use on
#    another display." Run one or the other, not both.
#
# Env: DISPLAY_NUM (98) · BOOTH_URL · SCREEN_W/H (1920x1080)
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-98}"
SCREEN_W="${SCREEN_W:-1920}"
SCREEN_H="${SCREEN_H:-1080}"
BOOTH_URL="${BOOTH_URL:-https://192.168.1.156:8090/}"
PROFILE="${PROFILE:-$HOME/.config/lt-booth-chrome-headless}"

XVFB_PID=""
cleanup() { [[ -n "$XVFB_PID" ]] && kill "$XVFB_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

LOCKFILE="/tmp/.X${DISPLAY_NUM}-lock"

# ── Single-instance guard ────────────────────────────────────────────────────
# booth-rtsp has Requires=booth-headless, so a start job for this unit can be
# queued from two sources at once. Concurrent `Xvfb :98` attempts lose to
# "Server is already active" -- that is how this crash-looped 1388x on
# 2026-08-12. Hold an flock for the whole life of the process (the fd survives
# the exec into Chrome), so a second instance exits immediately and visibly
# instead of racing.
exec 9>"/tmp/booth-headless-${DISPLAY_NUM}.flock"
flock -n 9 || {
  echo "booth-headless: another instance already holds :${DISPLAY_NUM}; exiting" >&2
  exit 1
}

display_is_live() { xdpyinfo -display ":${DISPLAY_NUM}" >/dev/null 2>&1; }

# The X lock file holds the server's PID, space-padded.
lock_pid() { tr -dc '0-9' < "$LOCKFILE" 2>/dev/null; }

# ── Never hijack a display we do not own ─────────────────────────────────────
# The old code `rm -f`'d this lock unconditionally and then treated "the socket
# exists" as "our Xvfb is up". Point it at a display another server owns and it
# deletes that server's lock and drives its display. Proven by experiment, and
# found live on 2026-08-12: Xvfb :98 was running with its lock deleted.
if display_is_live; then
  echo "booth-headless: :${DISPLAY_NUM} already has a LIVE X server (lock pid $(lock_pid || echo unknown))." >&2
  echo "booth-headless: refusing to hijack it. Stop the owner first." >&2
  exit 1
fi

if [[ -e "$LOCKFILE" ]]; then
  lp="$(lock_pid || true)"
  if [[ -n "$lp" ]] && kill -0 "$lp" 2>/dev/null && [[ "$(cat /proc/$lp/comm 2>/dev/null)" =~ ^X ]]; then
    echo "booth-headless: :${DISPLAY_NUM} lock held by live X server pid $lp; refusing." >&2
    exit 1
  fi
  # Owner is gone (or was never an X server) -- a genuinely stale lock.
  echo "booth-headless: clearing stale lock $LOCKFILE (pid ${lp:-none} not alive)"
  rm -f "$LOCKFILE" 2>/dev/null || true
fi

Xvfb ":${DISPLAY_NUM}" -screen 0 "${SCREEN_W}x${SCREEN_H}x24" -nolisten tcp &
XVFB_PID=$!

# Readiness = the display ANSWERS, not merely that a socket file exists (any
# server's socket satisfies that test).
for _ in $(seq 1 50); do
  display_is_live && break
  kill -0 "$XVFB_PID" 2>/dev/null || { echo "Xvfb :${DISPLAY_NUM} died during startup" >&2; exit 1; }
  sleep 0.1
done
display_is_live || { echo "Xvfb :${DISPLAY_NUM} failed to start" >&2; exit 1; }

# ...and that the server answering is OURS.
lp="$(lock_pid || true)"
if [[ "$lp" != "$XVFB_PID" ]]; then
  echo "booth-headless: :${DISPLAY_NUM} is owned by pid ${lp:-unknown}, not our Xvfb ($XVFB_PID); refusing to drive it." >&2
  exit 1
fi

# --ignore-certificate-errors + --test-type: the booth is https with a self-signed
#   cert; without these Chrome parks on the interstitial and streams a grey page.
# --autoplay-policy: the booth's <video> must start with no user gesture.
# --disable-gpu: Xvfb has no GPU surface; Chrome software-renders (cheap here,
#   32 cores). The ENCODE is still hardware VAAPI in the publisher.
exec env DISPLAY=":${DISPLAY_NUM}" google-chrome \
  --user-data-dir="$PROFILE" \
  --no-first-run --no-default-browser-check --disable-session-crashed-bubble \
  --ignore-certificate-errors --test-type \
  --ozone-platform=x11 --disable-gpu --disable-dev-shm-usage \
  --autoplay-policy=no-user-gesture-required \
  --disable-features=Translate,InfiniteSessionRestore \
  --window-position=0,0 --window-size="${SCREEN_W},${SCREEN_H}" \
  --kiosk "$BOOTH_URL"
