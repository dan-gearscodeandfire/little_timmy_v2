"""Retire (or revive) a persona across every identity store — the CLI for
``presence/identity_commit.retire_identity``.

Endpoint-first: when little-timmy.service is RUNNING, direct store surgery is
unsafe (the live process re-flushes the room ledger and holds the in-memory
recognizers), so we go through ``POST :8893/api/identity/retire``. When the
service is DOWN the endpoint is unreachable and direct store mode is safe.
The pre-tombstone era took a live delete THREE stop-scrub-restart cycles
("john", 2026-07-02); this collapses it to one command either way.

    python -m ops.retire_identity --list [--since <unix_ts>]
    python -m ops.retire_identity sarah
    python -m ops.retire_identity sarah --purge-facts   # test junk only
    python -m ops.retire_identity sarah --revive
    python -m ops.retire_identity old_maker --pi        # also scrub Pi face_db

--pi is legacy-only: Phase 0/A moved enrollment to okDemerzel, so identities
minted by the unified path never reach the Pi face_db. Best-effort (self-signed
HTTPS, may be unreachable); failures warn, never block the okDemerzel retire.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LT_BASE = "http://localhost:8893"
PI_BASE = "https://192.168.1.110:8080"


def _endpoint(path: str, payload: dict | None = None, *, timeout: float = 30.0):
    """POST (or GET when payload is None) against the live LT API. Returns the
    parsed JSON, or None if the service is unreachable (→ direct mode)."""
    import requests
    url = LT_BASE + path
    try:
        if payload is None:
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None


def _direct(name: str, *, revive: bool, purge_facts: bool) -> dict:
    """Service-down path: run the live wrapper directly against the on-disk
    stores + Postgres. The in-memory/ledger steps are no-ops here (dead
    process). Any stale room-ledger record ages out on its TTL and can't
    re-mint: recognition no longer resolves the name once the prototypes are
    archived and the tombstone blocks re-allocation."""
    from presence.identity_commit import retire_identity, revive_identity
    from dataclasses import asdict

    async def _go():
        if revive:
            return await revive_identity(name)
        return await retire_identity(name, purge_facts=purge_facts)

    res = asyncio.run(_go())
    return {"ok": res.ok, **asdict(res)}


def _pi_scrub(name: str) -> None:
    """Best-effort removal of ``name`` from the Pi SFace face_db (legacy names
    only). Uses /face_db/list + /face_db/delete; the Pi hot-reloads its
    recognizer on update, no motor-service restart."""
    import requests
    try:
        listing = requests.get(f"{PI_BASE}/face_db/list", verify=False,
                               timeout=10).json()
        names = listing.get("names") or list(
            (listing.get("enrolled") or {}).keys())
        if name not in names:
            print(f"[pi] {name!r} not in Pi face_db — nothing to scrub")
            return
        r = requests.post(f"{PI_BASE}/face_db/delete", json={"name": name},
                          verify=False, timeout=10)
        r.raise_for_status()
        print(f"[pi] deleted {name!r} from Pi face_db")
    except Exception as e:
        print(f"[pi] WARNING: Pi face_db scrub failed ({e}) — "
              f"retire on okDemerzel still holds; scrub the Pi later")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", help="persona to retire/revive")
    ap.add_argument("--revive", action="store_true",
                    help="reverse a retirement (restore newest trash archive)")
    ap.add_argument("--purge-facts", action="store_true",
                    help="HARD-delete the persona's facts rows (test junk only)")
    ap.add_argument("--pi", action="store_true",
                    help="also scrub the Pi face_db (legacy names only)")
    ap.add_argument("--list", action="store_true", dest="list_",
                    help="inventory every persona (active + retired)")
    ap.add_argument("--since", type=float, default=None,
                    help="with --list: only identities minted after this unix ts")
    args = ap.parse_args()

    if args.list_:
        out = _endpoint("/api/identity/list"
                        + (f"?since={args.since}" if args.since else ""))
        if out is None:
            print("service down — direct inventory via ops.migrate_reenroll:")
            import subprocess
            return subprocess.call(
                [sys.executable, "-m", "ops.migrate_reenroll"],
                cwd=Path(__file__).resolve().parents[1])
        for p in out["identities"]:
            mods = "+".join(m for m in ("voice", "face") if p.get(m)) or "-"
            flags = "RETIRED" if p["retired"] else (
                "reserved" if p.get("reserved") else "")
            print(f"{p['speaker_id']:>4}  {p['name']:<24} {mods:<12} {flags}")
        return 0

    if not args.name:
        ap.error("name required (or --list)")
    name = args.name.strip().lower()

    path = "/api/identity/revive" if args.revive else "/api/identity/retire"
    out = _endpoint(path, {"name": name, "purge_facts": args.purge_facts})
    if out is None:
        print("service down — direct store mode")
        out = _direct(name, revive=args.revive, purge_facts=args.purge_facts)
    print(json.dumps(out, indent=2, default=str))

    if args.pi and not args.revive and out.get("ok"):
        _pi_scrub(name)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
