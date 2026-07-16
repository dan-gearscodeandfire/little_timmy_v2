#!/usr/bin/env python3
"""Enrollment stress-test rig runner (F1).

Orchestrates a synthetic face+voice enrollment test against the LIVE
little-timmy.service, then restores it:

  1. start ops/mock_streamerpi (synthetic /capture /faces /behavior /servo)
  2. install a reversible systemd drop-in pointing TIMMY_*_URL at the mock,
     restart the service, verify it came up healthy on the mock
  3. run a scenario: interleave mock frame changes (face + green LED) with
     synthetic voice turns (ops.acoustic_convo_driver), observing WS + journal
  4. cleanup memory: ops.retire_identity <persona> --purge-facts + synthtest_guard
  5. ALWAYS teardown: remove drop-in, restart, restore the real Pi URLs, stop mock

Subcommands (so the risky env-swap can be exercised in isolation):
  up               swap the live service onto the mock + restart (leaves it there)
  down             remove the drop-in + restart (restore the real Pi)
  status           show whether the rig override is installed + service health
  run <scenario>   full cycle: up -> scenario -> cleanup -> down

Restart discipline: sudo -n systemctl (feedback_lt_service_restart_systemd_not_api).
NEVER run during a live booth conversation (feedback_lt_test_suite_hits_live_8083).

Usage:
  cd ~/little_timmy && .venv/bin/python -m ops.enroll_rig up
  cd ~/little_timmy && .venv/bin/python -m ops.enroll_rig run ops/enroll_scenarios/f0_new_face_bind.json
  cd ~/little_timmy && .venv/bin/python -m ops.enroll_rig down
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SERVICE = "little-timmy.service"
DROPIN_DIR = f"/etc/systemd/system/{SERVICE}.d"
DROPIN = f"{DROPIN_DIR}/zz-rig-override.conf"
MOCK_HOST = "127.0.0.1"
MOCK_PORT = 8899
MOCK_BASE = f"http://{MOCK_HOST}:{MOCK_PORT}"
API = "http://localhost:8893"

DROPIN_CONTENT = f"""# INSTALLED BY ops/enroll_rig.py — safe to delete (restores the real Pi).
[Service]
Environment=TIMMY_CAPTURE_URL={MOCK_BASE}/capture
Environment=TIMMY_FACES_URL={MOCK_BASE}/faces
Environment=TIMMY_BEHAVIOR_URL={MOCK_BASE}/behavior/status
Environment=TIMMY_SERVO_MOVE_URL={MOCK_BASE}/servo/move
"""


# ---------------------------------------------------------------- shell utils
def sh(cmd: list[str], timeout=60) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def sudo(cmd: list[str], timeout=60) -> tuple[int, str]:
    return sh(["sudo", "-n"] + cmd, timeout=timeout)


def http_get(url: str, timeout=3.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as e:
        return 0, str(e).encode()


def http_post(url: str, obj: dict, timeout=5.0) -> tuple[int, bytes]:
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as e:
        return 0, str(e).encode()


# ------------------------------------------------------------- service control
def svc_start_ts() -> str:
    _, out = sh(["systemctl", "show", SERVICE, "-p", "ExecMainStartTimestamp"])
    return out


def svc_active() -> bool:
    rc, out = sh(["systemctl", "is-active", SERVICE])
    return out.strip() == "active"


def _audio_ready_since(t_wall: float, timeout=35.0) -> bool:
    """Wait for the audio capture thread to actually be listening — the web API
    answers ~13 s BEFORE 'Audio capture started', so turns fired on web-health
    alone play into a deaf service (observed 2026-07-15)."""
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_wall - 1))
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, jlog = sh(["journalctl", "-u", SERVICE, "--since", since, "-o", "cat"],
                     timeout=15)
        if "Audio capture started (always listening)" in jlog:
            return True
        time.sleep(1.0)
    return False


def restart_and_wait(prev_ts: str, label: str, health_timeout=45.0,
                     wait_audio=False) -> bool:
    """Restart the service, wait for the start timestamp to move + a healthy
    :8893 (and, when wait_audio, the audio capture thread). Returns True on
    healthy."""
    print(f"  [{label}] restarting {SERVICE} ...")
    t_restart = time.time()
    rc, out = sudo(["systemctl", "restart", SERVICE], timeout=60)
    if rc != 0:
        print(f"  [{label}] restart FAILED rc={rc}: {out}")
        return False
    deadline = time.time() + health_timeout
    while time.time() < deadline:
        time.sleep(1.0)
        if svc_start_ts() == prev_ts:
            continue  # timestamp hasn't moved yet
        if not svc_active():
            continue
        code, _ = http_get(f"{API}/api/identity/list")
        if code == 200:
            print(f"  [{label}] web healthy (start={svc_start_ts()})")
            if wait_audio:
                if _audio_ready_since(t_restart):
                    print(f"  [{label}] audio capture listening")
                else:
                    print(f"  [{label}] WARNING: audio not confirmed listening")
            return True
    print(f"  [{label}] did NOT become healthy within {health_timeout:.0f}s")
    return False


def dropin_installed() -> bool:
    return os.path.exists(DROPIN)


def install_dropin() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as f:
        f.write(DROPIN_CONTENT)
        tmp = f.name
    sudo(["mkdir", "-p", DROPIN_DIR])
    sudo(["cp", tmp, DROPIN])
    sudo(["chmod", "0644", DROPIN])
    os.unlink(tmp)
    sudo(["systemctl", "daemon-reload"])
    print(f"  installed drop-in -> {DROPIN}")


def remove_dropin() -> None:
    if dropin_installed():
        sudo(["rm", "-f", DROPIN])
    sudo(["systemctl", "daemon-reload"])
    print("  removed drop-in (restored real Pi URLs)")


# ---------------------------------------------------------------- mock control
class Mock:
    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        # Refuse to stomp an existing mock (orphan hygiene).
        code, _ = http_get(f"{MOCK_BASE}/rig/health", timeout=1.0)
        if code == 200:
            raise RuntimeError(f"a mock is already listening on {MOCK_BASE} "
                               "— pkill -f mock_streamerpi first")
        py = os.path.join(REPO, ".venv", "bin", "python")
        self.proc = subprocess.Popen(
            [py, "-m", "ops.mock_streamerpi", "--host", MOCK_HOST,
             "--port", str(MOCK_PORT)],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline:
            time.sleep(0.5)
            code, _ = http_get(f"{MOCK_BASE}/rig/health", timeout=1.0)
            if code == 200:
                print(f"  mock streamerpi up on {MOCK_BASE} (pid {self.proc.pid})")
                return
        raise RuntimeError("mock streamerpi failed to come up")

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            print("  mock streamerpi stopped")
        # belt-and-suspenders: kill any stragglers
        sh(["pkill", "-9", "-f", "ops.mock_streamerpi"])

    def set_frame(self, spec: dict) -> dict:
        code, body = http_post(f"{MOCK_BASE}/rig/frame", spec)
        res = json.loads(body) if code == 200 else {"ok": False, "raw": body.decode()}
        print(f"  frame <- {spec}  =>  {res}")
        return res

    def servo_log(self) -> list:
        code, body = http_get(f"{MOCK_BASE}/rig/servo_log")
        return json.loads(body) if code == 200 else []


# ------------------------------------------------------------- up / down verbs
def rig_up() -> None:
    print("[rig up] pointing the live service at the mock")
    prev = svc_start_ts()
    install_dropin()
    ok = restart_and_wait(prev, "up", wait_audio=True)
    if not ok:
        print("[rig up] service unhealthy on the mock — tearing back down")
        rig_down()
        raise SystemExit(2)


def rig_down() -> None:
    print("[rig down] restoring the real Pi")
    prev = svc_start_ts()
    remove_dropin()
    restart_and_wait(prev, "down")


def rig_status() -> None:
    print(f"override installed : {dropin_installed()}")
    print(f"service active     : {svc_active()}")
    print(f"service start ts   : {svc_start_ts()}")
    code, _ = http_get(f"{API}/api/identity/list")
    print(f"api :8893 healthy  : {code == 200}")
    code, body = http_get(f"{MOCK_BASE}/rig/health", timeout=1.0)
    print(f"mock listening     : {code == 200}")
    ffr = http_get(f"{API}/api/face_recognition")
    if ffr[0] == 200:
        print(f"face_recognition   : {ffr[1].decode()[:200]}")


async def wait_mic_open(max_wait: float = 25.0, settle: float = 1.2) -> None:
    """Block until Timmy's mic-gate is open (his TTS finished) + a settle beat.

    The acoustic driver's word-count sleep undershoots long replies — turn 3
    of rig f0b played into the gate (suppressed=True) and was never heard.
    /api/audio/diag.suppressed is the live gate state."""
    deadline = time.time() + max_wait
    clear_since = None
    while time.time() < deadline:
        code, body = http_get(f"{API}/api/audio/diag", timeout=2.0)
        suppressed = True
        if code == 200:
            try:
                suppressed = bool(json.loads(body).get("suppressed", True))
            except Exception:
                pass
        now = time.time()
        if not suppressed:
            if clear_since is None:
                clear_since = now
            if now - clear_since >= settle:
                return
        else:
            clear_since = None
        await asyncio.sleep(0.3)


# ------------------------------------------------------------------- scenario
async def run_scenario(mock: Mock, scenario: dict) -> dict:
    """A scenario = {persona, voice, reply_window, steps:[...]}. Each step is
    {"frame": {...}} (set mock scene) or {"say": "...", "expect": "..."} (voice
    turn) or {"sleep": secs} or {"note": "..."}."""
    from ops import acoustic_convo_driver as drv

    persona = scenario.get("persona", "oliver")
    voice = scenario.get("voice", "en_US-ryan-high")
    length_scale = scenario.get("length_scale", 1.0)
    reply_window = scenario.get("reply_window", 14.0)
    steps = scenario["steps"]

    # identity roster BEFORE the run, so we can surface a mis-heard name too.
    code, body = http_get(f"{API}/api/identity/list")
    pre_names = {r.get("name", "").lower()
                 for r in (json.loads(body).get("identities", []) if code == 200 else [])}

    stop = asyncio.Event()
    msgs: list = []

    async def _resilient_collector():
        # The plain collector exits when its socket drops (e.g. the restart
        # durability step) — reconnect until the scenario ends.
        while not stop.is_set():
            await drv.ws_collector(stop, msgs)
            if not stop.is_set():
                await asyncio.sleep(1.0)

    col = asyncio.create_task(_resilient_collector())
    await asyncio.sleep(1.0)

    t_start = time.time()
    turn_idx = 0
    for step in steps:
        if "note" in step:
            print(f"\n# {step['note']}")
        if "frame" in step:
            mock.set_frame(step["frame"])
        if "api" in step:
            # Mid-scenario API call against :8893 (e.g. a rename) — the
            # response is printed so asserts can eyeball it in the log.
            spec = step["api"]
            path = spec["path"]
            if spec.get("method", "GET").upper() == "POST":
                code, body = http_post(f"{API}{path}", spec.get("json") or {})
            else:
                code, body = http_get(f"{API}{path}")
            print(f"  api {path} -> {code}: {body.decode()[:400]}")
        if "restart" in step:
            # Durability step: bounce the service (still on the mock — the
            # drop-in holds) and wait for audio. In-memory state is wiped;
            # whatever survives is what's actually on disk.
            print("  [restart] bouncing service (durability check) ...")
            restart_and_wait(svc_start_ts(), "restart", wait_audio=True)
        if "sleep" in step:
            await asyncio.sleep(float(step["sleep"]))
        if "say" in step:
            turn_idx += 1
            await wait_mic_open()   # never play into Timmy's own TTS (f0b turn 3)
            await drv.run_turn(turn_idx, step["say"], step.get("expect", ""),
                               voice, length_scale, msgs, reply_window)

    await asyncio.sleep(1.0)
    stop.set()
    await col

    # --- observe binding state ---
    code, body = http_get(f"{API}/api/identity/list")
    ident = json.loads(body) if code == 200 else {}
    roster = ident.get("identities", [])
    persona_row = next((r for r in roster
                        if r.get("name", "").lower() == persona.lower()), None)
    new_identities = [r for r in roster
                      if r.get("name", "").lower() not in pre_names]
    code, abody = http_get(f"{API}/api/anchor")
    anchor = json.loads(abody) if code == 200 else {}

    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start - 2))
    _, jlog = sh(["journalctl", "-u", SERVICE, "--since", since, "-o", "cat"],
                 timeout=20)
    marks = [ln for ln in jlog.splitlines()
             if any(k in ln for k in ("[ANCHOR]", "[INTRO]", "[COMMIT]",
                                      "[FRONTAL-SHADOW]", persona.lower()))]

    report = {
        "persona": persona,
        "persona_identity": persona_row,
        "new_identities": new_identities,
        "anchor": anchor,
        "servo_moves": len(mock.servo_log()),
        "journal_marks": marks[-40:],
        "bound_face": bool(persona_row and persona_row.get("face")),
        "bound_voice": bool(persona_row and persona_row.get("voice")),
    }
    return report


def _purge_tombstone(name: str) -> None:
    """Drop a synthetic persona's tombstone from _id_map.json so the SAME test
    name is re-enrollable next run (rig f0b finding: retire_identity tombstones
    the name and commit_identity's retired_name guard then refuses it forever).
    The allocated id is NOT reused (the _next_id counter never rewinds — S1
    holds); only the name label is freed. Safe here because rig_down restarts
    the service right after, reloading the map from disk."""
    idmap_path = os.path.join(REPO, "models", "speaker", "_id_map.json")
    try:
        with open(idmap_path) as f:
            data = json.load(f)
        retired = data.get("_retired", {})
        if name in retired:
            del retired[name]
            tmp = idmap_path + ".rig_tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, idmap_path)
            print(f"  tombstone purged for {name!r} (id stays burned)")
    except Exception as e:
        print(f"  tombstone purge failed for {name!r}: {e}")


def cleanup_memory(names: list[str], baseline: str | None) -> None:
    py = os.path.join(REPO, ".venv", "bin", "python")
    for name in sorted({n.lower() for n in names if n}):
        print(f"\n[cleanup] retiring synthetic identity {name!r} + purging facts")
        rc, out = sh([py, "-m", "ops.retire_identity", name, "--purge-facts"],
                     timeout=60)
        print(f"  retire rc={rc}: {out[-400:]}")
        _purge_tombstone(name)
    if baseline:
        rc, out = sh([py, "-m", "ops.synthtest_guard", "cleanup", baseline],
                     timeout=60)
        print(f"  synthtest_guard cleanup rc={rc}: {out[-400:]}")


def snapshot_memory() -> str | None:
    py = os.path.join(REPO, ".venv", "bin", "python")
    base = f"/tmp/lt_rig_baseline_{int(time.time())}.json"
    rc, out = sh([py, "-m", "ops.synthtest_guard", "snapshot", base], timeout=60)
    if rc == 0:
        print(f"  memory snapshot -> {base}")
        return base
    print(f"  snapshot FAILED rc={rc}: {out[-300:]}")
    return None


def rig_run(scenario_path: str) -> None:
    scenario = json.load(open(scenario_path))
    persona = scenario.get("persona", "oliver")
    print(f"[rig run] {scenario_path}  persona={persona}")
    mock = Mock()
    baseline = None
    report = None
    try:
        mock.start()
        # initial frame BEFORE the service comes up so vision is never blind
        mock.set_frame(scenario.get("initial_frame",
                                    {"face": "scene", "led_below_face": True}))
        baseline = snapshot_memory()
        rig_up()
        report = asyncio.run(run_scenario(mock, scenario))
    finally:
        if report is not None:
            print("\n================ RIG REPORT ================")
            print(json.dumps({k: v for k, v in report.items()
                              if k != "journal_marks"}, indent=2))
            print("---- journal marks ----")
            for ln in report.get("journal_marks", []):
                print("  " + ln)
            print("===========================================")
        # memory cleanup happens while still on the mock (service up), then restore.
        # Retire the intended persona AND any mis-heard new identity the run minted.
        cleanup_names = [persona]
        if report is not None:
            cleanup_names += [r.get("name", "")
                              for r in report.get("new_identities", [])]
        try:
            cleanup_memory(cleanup_names, baseline)
        except Exception as e:
            print(f"  [cleanup] error: {e}")
        rig_down()
        mock.stop()
    if report:
        ok = report["bound_face"] and report["bound_voice"]
        print(f"\nRESULT: face={report['bound_face']} voice={report['bound_voice']} "
              f"-> {'PASS' if ok else 'INCOMPLETE (see journal marks)'}")


def main():
    ap = argparse.ArgumentParser(description="Enrollment stress-test rig")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up")
    sub.add_parser("down")
    sub.add_parser("status")
    pr = sub.add_parser("run")
    pr.add_argument("scenario")
    args = ap.parse_args()
    if args.cmd == "up":
        rig_up()
    elif args.cmd == "down":
        rig_down()
    elif args.cmd == "status":
        rig_status()
    elif args.cmd == "run":
        rig_run(args.scenario)


if __name__ == "__main__":
    main()
