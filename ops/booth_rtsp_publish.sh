#!/usr/bin/env bash
# booth_rtsp_publish.sh — publish the COMPOSITED booth display to MediaMTX as RTSP,
# for Blue Iris (okLLMBox 192.168.1.154) to subscribe to.
#
#   main : rtsp://192.168.1.156:8554/booth      1920x1080 @30fps  H.264 Main
#   sub  : rtsp://192.168.1.156:8554/booth_sub   640x360  @15fps  H.264 Main
#
# The 40K overlay only exists as composited browser pixels, so screen capture is
# the ONLY way to get it. Zero coupling to the LT pipeline.
#
# TWO CAPTURE MODES (SRC env):
#   SRC=x11  (DEFAULT) — x11grab of the VIRTUAL Xvfb screen driven by
#             ops/booth_headless_display.sh. NO physical monitor required, nothing
#             on the primary monitor, and the capture can never wander to another
#             output. Runs as an ordinary user (no root).
#   SRC=kms  (legacy)  — kmsgrab of the DP-1 physical scanout. Zero-copy and
#             GPU-composited, but REQUIRES the second monitor to be plugged in and
#             the booth fullscreen on it. Needs root (CAP_SYS_ADMIN). Kept as a
#             fallback; see ops/record_booth.sh which still uses this technique.
#
# NOTES
#   * -thread_queue_size 64 is REQUIRED (default 8 starves the mux loop -> ~12fps).
#   * x11 mode: frames land in system RAM, so `hwupload` pushes them to VAAPI.
#     Both encodes are still hardware (AMD VCN) — only the upload is new, and it
#     does not meaningfully contend with the two 35B models on the GPU.
#   * -bf 0 + -profile:v main: Blue Iris direct-to-disk and the QSV substream
#     preview are happiest without B-frames or exotic profiles.
#   * -g = 2x fps (2-second GOP) so BI gets frequent keyframes.
#   * -pkt_size 1400 keeps RTP under MediaMTX's 1440 limit (else it logs
#     "RTP packets are too big" and re-muxes, burning CPU).
#   * -draw_mouse 0: no pointer in the recording.
#   * Substream is 640x360 = 16:9, matching the source exactly (no distortion).
#
# Service: sudo systemctl {status,restart} booth-rtsp.service
set -euo pipefail

SRC="${SRC:-x11}"
DISPLAY_NUM="${DISPLAY_NUM:-98}"
DRM_DEV="${DRM_DEV:-/dev/dri/card1}"
VAAPI_DEV="${VAAPI_DEV:-/dev/dri/renderD128}"
FPS="${FPS:-30}"
SUB_FPS="${SUB_FPS:-15}"
MAIN_BR="${MAIN_BR:-6M}"
SUB_BR="${SUB_BR:-600k}"
MTX="${MTX:-rtsp://127.0.0.1:8554}"

COMMON_OUT=(
  -map "[m]"   -c:v h264_vaapi -profile:v main -bf 0 -g "$((FPS * 2))"
               -b:v "$MAIN_BR" -maxrate "$MAIN_BR" -bufsize 12M
               -pkt_size 1400 -f rtsp -rtsp_transport tcp "$MTX/booth"
  -map "[sub]" -c:v h264_vaapi -profile:v main -bf 0 -g "$((SUB_FPS * 2))"
               -b:v "$SUB_BR" -maxrate "$SUB_BR" -bufsize 1200k
               -pkt_size 1400 -f rtsp -rtsp_transport tcp "$MTX/booth_sub"
)

if [[ "$SRC" == "kms" ]]; then
  # Legacy: physical DP-1 scanout. Frames are already GPU-side (drm_prime), so
  # hwmap derives a VAAPI device from them instead of uploading.
  exec ffmpeg -hide_banner -loglevel warning -nostdin \
    -thread_queue_size 64 -device "$DRM_DEV" -framerate "$FPS" -f kmsgrab -i - \
    -filter_complex "[0:v]hwmap=derive_device=vaapi,scale_vaapi=w=1920:h=1080:format=nv12,split=2[m][s];[s]scale_vaapi=w=640:h=360:format=nv12,fps=${SUB_FPS}[sub]" \
    "${COMMON_OUT[@]}"
else
  # Default: virtual Xvfb screen. No monitor, no root.
  exec ffmpeg -hide_banner -loglevel warning -nostdin \
    -vaapi_device "$VAAPI_DEV" \
    -thread_queue_size 64 -f x11grab -draw_mouse 0 -framerate "$FPS" \
    -video_size 1920x1080 -i ":${DISPLAY_NUM}.0" \
    -filter_complex "[0:v]format=nv12,hwupload,split=2[m][s];[s]scale_vaapi=w=640:h=360:format=nv12,fps=${SUB_FPS}[sub]" \
    "${COMMON_OUT[@]}"
fi
