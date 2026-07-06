#!/usr/bin/env python3
"""Booth-day LED calibration helper (read-only, LED-mic anchor 2026-07-06).

Grabs ONE /capture JPEG from streamerpi and prints every green-blob candidate
under the CURRENT anchor_led_* toggles — plus what the detector's exactly-one
rule decides. Tune the toggles live (POST :8893 or edit the JSON) and re-run;
no service restart, no state written.

Usage:
    .venv/bin/python ops/led_calibrate.py [--save /tmp/frame.jpg]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import httpx
import numpy as np

import config
from persistence import runtime_toggles
from presence.led_detect import find_green_led


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="also write the grabbed frame here")
    args = ap.parse_args()

    knobs = {k: int(runtime_toggles.get(f"anchor_led_{k}"))
             for k in ("h_lo", "h_hi", "s_min", "v_min",
                       "min_area_px", "max_area_px")}
    print(f"toggles: {knobs}")

    r = httpx.get(config.STREAMERPI_CAPTURE_URL, verify=False, timeout=5.0)
    r.raise_for_status()
    frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("frame decode failed", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    print(f"frame: {w}x{h}")
    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"saved: {args.save}")

    # Candidate dump: the same mask as find_green_led but WITHOUT the
    # exactly-one filter, so you can see everything the thresholds pass.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (knobs["h_lo"], knobs["s_min"], knobs["v_min"]),
                        (knobs["h_hi"], 255, 255))
    white = cv2.inRange(hsv, (0, 0, 250), (179, 60, 255))
    if cv2.countNonZero(white):
        near = cv2.dilate(green, np.ones((7, 7), np.uint8))
        green = cv2.bitwise_or(green, cv2.bitwise_and(white, near))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _l, stats, cents = cv2.connectedComponentsWithStats(green)
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        in_bounds = knobs["min_area_px"] <= area <= knobs["max_area_px"]
        cands.append((area, tuple(round(c, 1) for c in cents[i]), in_bounds))
    cands.sort(reverse=True)
    if not cands:
        print("candidates: none (nothing passed the HSV mask)")
    for area, cent, ok in cands[:15]:
        print(f"  blob area={area:5d} centroid={cent} "
              f"{'IN-BOUNDS' if ok else 'rejected(size)'}")
    if len(cands) > 15:
        print(f"  ... +{len(cands) - 15} more")

    hit = find_green_led(frame)
    print(f"verdict: {hit if hit else 'ABSTAIN (need exactly one in-bounds blob)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
