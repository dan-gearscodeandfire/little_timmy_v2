"""Shared amdgpu sysfs telemetry reader (Strix Halo / okDemerzel).

Single source of truth for "what GPU fields do we read, and from which card"
so the two consumers never drift:

  1. LT-OS  little_timmy_os/main.py  GET /api/host  -> live dashboard panel
  2. ops/gpu_watchdog.sh             ring-buffer    -> freeze forensics (dump-on-wedge)

Background: okDemerzel has hard-frozen three times (2026-05-12, -06-06, -06-08) on
amdgpu/Vulkan compute wedges. The 06-08 freeze left no fingerprints because nothing
was sampling the GPU. This helper feeds both the always-on dashboard read and the
watchdog's rolling buffer from one place.

Pure stdlib, no third-party deps (safe to import in any venv / run under any python3).
read_gpu() never raises: any unreadable field comes back None.

Notes on this card (amdgpu, /sys/class/drm/card1 on okDemerzel):
  - Strix Halo UMA: "VRAM" is a BIOS-partitioned slice of system RAM, reported in bytes.
  - mem_busy_percent is absent on this iGPU -> stays None (kept for portability).
  - hwmon index (hwmonN) can shift across reboots, so we glob device/hwmon/hwmon*
    rather than hardcode hwmon4. Leaf names (temp1/power1/freq1) are stable for amdgpu.
  - Raw units: temp m°C, power µW (power1_average = PPT), freq Hz (freq1 = sclk).

Refs: Obsidian Zettelkasten/okdemerzel-freeze-rca-watchdog-health-blindspot-2026-06-08
"""

from __future__ import annotations

import glob
import json
import os

# Override with GPU_SYSFS_CARD=cardN if the DRM enumeration ever changes.
CARD = os.environ.get("GPU_SYSFS_CARD", "card1")
DEVICE = f"/sys/class/drm/{CARD}/device"


def _read_int(path: str):
    """Read a single integer from a sysfs file, or None if unreadable/non-numeric."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _hwmon_dir(device: str) -> str | None:
    """The amdgpu hwmon directory for a card (glob; index varies across boots)."""
    for d in sorted(glob.glob(f"{device}/hwmon/hwmon*")):
        return d
    return None


def read_gpu(card: str | None = None) -> dict:
    """Return a flat telemetry snapshot. Every value is a number or None (never raises).

    Keys (the first four match LT-OS /api/host's historical schema exactly):
      vram_used_gb, vram_total_gb, vram_percent, gpu_busy_percent,
      mem_busy_percent, temp_c, power_w, sclk_mhz
    """
    device = f"/sys/class/drm/{card}/device" if card else DEVICE

    vram_used = _read_int(f"{device}/mem_info_vram_used")
    vram_total = _read_int(f"{device}/mem_info_vram_total")
    vram_used_gb = vram_total_gb = vram_percent = None
    if vram_used is not None and vram_total:
        vram_used_gb = round(vram_used / (1024 ** 3), 2)
        vram_total_gb = round(vram_total / (1024 ** 3), 2)
        vram_percent = round(100 * vram_used / vram_total, 1)

    # hwmon-backed fields; glob the dir so a reboot-shifted hwmonN still resolves.
    temp_c = power_w = sclk_mhz = None
    hwmon = _hwmon_dir(device)
    if hwmon:
        temp_milli = _read_int(f"{hwmon}/temp1_input")       # edge temp, m°C
        power_micro = _read_int(f"{hwmon}/power1_average")   # PPT, µW
        freq_hz = _read_int(f"{hwmon}/freq1_input")          # sclk, Hz
        if temp_milli is not None:
            temp_c = round(temp_milli / 1000, 1)
        if power_micro is not None:
            power_w = round(power_micro / 1_000_000, 2)
        if freq_hz is not None:
            sclk_mhz = round(freq_hz / 1_000_000)

    return {
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "vram_percent": vram_percent,
        "gpu_busy_percent": _read_int(f"{device}/gpu_busy_percent"),
        "mem_busy_percent": _read_int(f"{device}/mem_busy_percent"),  # absent on this iGPU
        "temp_c": temp_c,
        "power_w": power_w,
        "sclk_mhz": sclk_mhz,
    }


if __name__ == "__main__":
    # CLI: one compact JSON object on stdout, for the bash watchdog to ring-buffer.
    print(json.dumps(read_gpu(), separators=(",", ":")))
