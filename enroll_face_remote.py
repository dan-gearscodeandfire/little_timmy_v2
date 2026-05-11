#!/usr/bin/env python3
"""Enroll / list / delete face identities via streamerpi's /face_db endpoints.

Replaces the legacy enroll_face.py (which captured frames locally on
okdemerzel and pushed embeddings to streamerpi). Streamerpi now runs the
entire enrollment pipeline against its live video buffer; this script is
a thin HTTP wrapper.

Stdlib only (urllib) so it runs under either system Python or the LT venv.

Usage:
    python3 enroll_face_remote.py Dan
    python3 enroll_face_remote.py Dan --count 20 --interval 0.5
    python3 enroll_face_remote.py --list
    python3 enroll_face_remote.py --delete Dan

Stand in front of the camera and slowly turn your head left/right and
tilt up/down during capture for diverse angle coverage.
"""
import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("STREAMERPI_BASE", "https://192.168.1.110:8080")
ENROLL_URL = BASE + "/face_db/enroll"
ENROLL_STREAM_URL = BASE + "/face_db/enroll/stream"
LIST_URL = BASE + "/face_db/list"
DELETE_URL = BASE + "/face_db/delete"

# streamerpi uses self-signed certs
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _request(method: str, url: str, body=None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except Exception:
            payload = {"error": str(e)}
        return e.code, payload


def cmd_list() -> int:
    status, data = _request("GET", LIST_URL)
    if status != 200:
        print(f"ERROR ({status}): {data}")
        return 1
    names = data.get("enrolled", [])
    print(f"Enrolled identities ({len(names)}):")
    for n in names:
        print(f"  - {n}")
    return 0


def cmd_delete(name: str) -> int:
    status, data = _request("POST", DELETE_URL, body={"name": name})
    if status == 404:
        print(f"Not enrolled: {name}")
        return 1
    if status != 200:
        print(f"ERROR ({status}): {data}")
        return 1
    print(f"Deleted '{name}'. Remaining: {data.get('remaining', [])}")
    return 0


def cmd_enroll(name: str, count: int, interval: float) -> int:
    print(f"\n=== Enrolling '{name}' on streamerpi ===")
    print(f"{count} samples at {interval}s intervals (~{count * interval:.0f}s total)")
    print("Stand in front of the camera and slowly turn your head left/right.")
    print("(Streamerpi tracking thread will pull samples from its live buffer.)\n")

    timeout_s = count * interval + 15
    try:
        status, data = _request(
            "POST",
            ENROLL_URL,
            body={"name": name, "count": count, "interval_s": interval},
            timeout=timeout_s,
        )
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach streamerpi at {BASE}: {e}")
        return 1

    if status >= 400:
        print(f"ERROR ({status}): {data.get('error', data)}")
        return 1

    captured = data.get("samples_captured", 0)
    skipped = data.get("samples_skipped", 0)
    saved = data.get("saved", False)
    enrolled = data.get("enrolled", [])

    print()
    print(f"Captured: {captured}")
    print(f"Skipped:  {skipped} (no face / extraction failed)")
    print(f"Saved:    {saved}")
    if not saved:
        print(f"Reason:   {data.get('error', 'unknown')}")
    print(f"Enrolled identities ({len(enrolled)}): {', '.join(enrolled)}")
    return 0 if saved else 1


def cmd_enroll_stream(name: str, count: int, interval: float) -> int:
    """Enroll using the SSE-streaming endpoint, printing progress as it goes."""
    print(f"\n=== Enrolling '{name}' on streamerpi (live progress) ===")
    print(f"{count} samples at {interval}s intervals (~{count * interval:.0f}s total)\n")

    timeout_s = count * interval + 30
    body = json.dumps({"name": name, "count": count, "interval_s": interval}).encode()
    req = urllib.request.Request(
        ENROLL_STREAM_URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    final_saved = False
    final_enrolled = []
    final_error = None
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout_s) as resp:
            evt_type = None
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    evt_type = None
                    continue
                if line.startswith("event:"):
                    evt_type = line[6:].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    if evt_type == "started":
                        pass
                    elif evt_type == "progress":
                        s = payload["sample"]
                        if payload.get("ok"):
                            bb = payload.get("bbox", [0, 0, 0, 0])
                            print(f"  [{s:>2}/{count}] OK   "
                                  f"captured={payload['captured']} skipped={payload['skipped']} "
                                  f"bbox={bb[2]}x{bb[3]}")
                        else:
                            print(f"  [{s:>2}/{count}] miss {payload.get('reason', '?')}")
                    elif evt_type == "complete":
                        final_saved = bool(payload.get("saved"))
                        final_enrolled = payload.get("enrolled", [])
                        final_error = payload.get("error")
                        print()
                        print(f"Captured: {payload.get('samples_captured', 0)}")
                        print(f"Skipped:  {payload.get('samples_skipped', 0)}")
                        print(f"Saved:    {final_saved}")
                        if final_error:
                            print(f"Reason:   {final_error}")
                        print(f"Enrolled identities ({len(final_enrolled)}): "
                              f"{', '.join(final_enrolled)}")
                    elif evt_type == "error":
                        final_error = payload.get("error", "unknown")
                        print(f"ERROR: {final_error}")
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach streamerpi: {e}")
        return 1
    return 0 if final_saved else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", help="Name to enroll")
    ap.add_argument("--count", type=int, default=15, help="number of samples (1-60)")
    ap.add_argument("--interval", type=float, default=0.7, help="seconds between samples (0.1-3.0)")
    ap.add_argument("--stream", action="store_true",
                    help="use SSE streaming endpoint with live per-sample progress")
    ap.add_argument("--list", action="store_true", help="list enrolled identities")
    ap.add_argument("--delete", metavar="NAME", help="delete an enrolled identity")
    args = ap.parse_args()

    if args.list:
        return cmd_list()
    if args.delete:
        return cmd_delete(args.delete)
    if not args.name:
        ap.print_help()
        return 1
    if args.stream:
        return cmd_enroll_stream(args.name, args.count, args.interval)
    return cmd_enroll(args.name, args.count, args.interval)


if __name__ == "__main__":
    sys.exit(main())
