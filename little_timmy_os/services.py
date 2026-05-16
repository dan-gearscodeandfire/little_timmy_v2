"""Service management — health checks, kill, launch, logging."""

import asyncio
import json
import logging
import os
import signal
import socket
import time
from datetime import datetime
from pathlib import Path

import httpx

import config

log = logging.getLogger("timmy_os.services")

_session_log: list[dict] = []
_session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
_status_callbacks: list = []


def register_status_callback(fn):
    _status_callbacks.append(fn)


async def _broadcast_status(msg: str, level: str = "info"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": msg,
    }
    _session_log.append(entry)
    log.info("[STATUS] %s", msg)
    for fn in _status_callbacks:
        try:
            await fn("status", entry)
        except Exception:
            pass


def _write_session_log():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    path = os.path.join(config.LOG_DIR, f"session_{_session_id}.json")
    with open(path, "w") as f:
        json.dump(_session_log, f, indent=2)


async def check_health(svc_id: str) -> dict:
    svc = config.SERVICES[svc_id]
    result = {"id": svc_id, "name": svc["name"], "status": "unknown", "detail": ""}

    health_url = svc.get("health_url")
    port_for_display = svc.get("port")

    if svc_id == "conversation_llm":
        model = config.CONVERSATION_MODELS.get(config.current_conversation_model, {})
        result["model"] = config.current_conversation_model
        result["model_name"] = model.get("name", "Unknown")
        # Shared-model entries (external_url) route the conversation tier
        # to an already-running server (e.g. Qwen3.6 brain @ :8083) instead
        # of spawning a duplicate on :8081. Override the static probe so the
        # dashboard reflects the real backing port + reachability.
        if model.get("external_url"):
            from urllib.parse import urlparse
            ext = model["external_url"].rstrip("/")
            health_url = ext + "/health"
            try:
                port_for_display = urlparse(ext).port or port_for_display
            except Exception:
                pass

    result["port"] = port_for_display

    if health_url:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(health_url)
                if r.status_code == 200:
                    result["status"] = "connected"
                    result["detail"] = "HTTP 200"
                else:
                    result["status"] = "disconnected"
                    result["detail"] = f"HTTP {r.status_code}"
        except Exception as e:
            result["status"] = "disconnected"
            result["detail"] = str(e)[:120]
    else:
        port = svc["port"]
        try:
            sock = socket.create_connection(("localhost", port), timeout=2)
            sock.close()
            result["status"] = "connected"
            result["detail"] = f"TCP :{port} open"
        except Exception as e:
            result["status"] = "disconnected"
            result["detail"] = str(e)[:120]

    return result


async def check_all_health() -> dict[str, dict]:
    tasks = {sid: check_health(sid) for sid in config.SERVICES}
    results = {}
    for sid, coro in tasks.items():
        results[sid] = await coro
    return results


async def find_pids_on_port(port: int) -> list[int]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "fuser", f"{port}/tcp",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if stdout:
            return [int(p) for p in stdout.decode().split() if p.strip().isdigit()]
    except Exception:
        pass
    return []


async def kill_service(svc_id: str) -> bool:
    svc = config.SERVICES[svc_id]
    port = svc["port"]

    await _broadcast_status(f"Stopping {svc['name']} (port {port})...")

    if svc.get("systemd"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "stop", svc["systemd"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                await _broadcast_status(f"Stopped {svc['name']} via systemctl")
                _write_session_log()
                return True
            else:
                detail = stderr.decode().strip()[:200]
                await _broadcast_status(
                    f"systemctl stop {svc['systemd']} failed: {detail}", "error"
                )
        except Exception as e:
            await _broadcast_status(f"systemctl error: {e}", "error")

    pids = await find_pids_on_port(port)
    if not pids:
        await _broadcast_status(f"No process found on port {port}")
        _write_session_log()
        return False

    for pid in pids:
        try:
            await _broadcast_status(f"Killing PID {pid} on port {port}")
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            await asyncio.create_subprocess_exec(
                "sudo", "kill", "-TERM", str(pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

    await asyncio.sleep(1)

    remaining = await find_pids_on_port(port)
    for pid in remaining:
        try:
            await _broadcast_status(f"Force-killing PID {pid} (SIGKILL)", "warning")
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            await asyncio.create_subprocess_exec(
                "sudo", "kill", "-9", str(pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

    await _broadcast_status(f"Stopped {svc['name']}")
    _write_session_log()
    return True


async def launch_service(svc_id: str) -> bool:
    svc = config.SERVICES[svc_id]

    await _broadcast_status(f"Launching {svc['name']}...")

    if svc.get("systemd"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "start", svc["systemd"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                await _broadcast_status(f"Started {svc['name']} via systemctl")
            else:
                detail = stderr.decode().strip()[:200]
                await _broadcast_status(
                    f"systemctl start {svc['systemd']} failed: {detail}", "error"
                )
                _write_session_log()
                return False
        except Exception as e:
            await _broadcast_status(f"systemctl error: {e}", "error")
            _write_session_log()
            return False
    else:
        if svc_id == "conversation_llm":
            start_cmd = config.get_conversation_start_cmd()
        else:
            start_cmd = svc.get("start_cmd")

        if not start_cmd:
            await _broadcast_status(f"No start command for {svc['name']}", "error")
            _write_session_log()
            return False

        cwd = svc.get("start_cwd")
        try:
            proc = await asyncio.create_subprocess_shell(
                start_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
            )
            await _broadcast_status(
                f"Launched {svc['name']} (PID {proc.pid})"
            )
        except Exception as e:
            await _broadcast_status(f"Launch failed: {e}", "error")
            _write_session_log()
            return False

    await _broadcast_status(f"Waiting for {svc['name']} to become healthy...")
    for attempt in range(10):
        await asyncio.sleep(2)
        health = await check_health(svc_id)
        if health["status"] == "connected":
            await _broadcast_status(
                f"{svc['name']} is connected ({health['detail']})"
            )
            _write_session_log()
            return True

    await _broadcast_status(
        f"{svc['name']} did not become healthy after 20s", "error"
    )
    _write_session_log()
    return False


async def toggle_service(svc_id: str, desired_state: bool) -> dict:
    if desired_state:
        current = await check_health(svc_id)
        if current["status"] == "connected":
            await _broadcast_status(
                f"{config.SERVICES[svc_id]['name']} is already connected"
            )
            return current

        await kill_service(svc_id)
        await asyncio.sleep(1)

        success = await launch_service(svc_id)
        return await check_health(svc_id)
    else:
        await kill_service(svc_id)
        return await check_health(svc_id)


_LLAMA_3B_UNIT = "llama-3b-server.service"


async def _systemctl(action: str, unit: str) -> bool:
    """Run sudo systemctl <action> <unit>. Returns True on success.

    The systemd unit auto-restarts on SIGKILL because Restart=on-failure,
    so a fuser/kill-by-PID is not durable. systemctl stop / start are.
    Requires NOPASSWD sudo for systemctl on this host (already in place).
    """
    proc = await asyncio.create_subprocess_exec(
        "sudo", "-n", "systemctl", action, unit,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        log.warning("systemctl %s %s failed: %s",
                    action, unit, err.decode(errors="replace").strip())
        return False
    return True


def _persist_model_choice(model_id: str, external_url: str | None) -> None:
    """Write conversation_model_id + conversation_url_override atomically
    (best-effort). Keeps the two keys in sync so a future LT-OS restart can
    restore state without ambiguity."""
    try:
        from persistence import runtime_toggles
        runtime_toggles.set("conversation_model_id", model_id)
        runtime_toggles.set("conversation_url_override", external_url or "")
    except Exception as e:
        log.warning("failed to persist conversation model choice: %s", e)


async def switch_conversation_model(model_id: str) -> dict:
    if model_id not in config.CONVERSATION_MODELS:
        await _broadcast_status(f"Unknown model: {model_id}", "error")
        return {"success": False, "error": "Unknown model"}

    model = config.CONVERSATION_MODELS[model_id]
    old_model = config.current_conversation_model
    old_name = config.CONVERSATION_MODELS.get(old_model, {}).get("name", old_model)
    new_is_external = bool(model.get("external_url"))
    old_is_external = bool(config.CONVERSATION_MODELS.get(old_model, {}).get("external_url"))

    if model_id == old_model:
        if new_is_external:
            await _broadcast_status(f"{model['name']} is already selected")
            return {"success": True, "model": model_id}
        health = await check_health("conversation_llm")
        if health["status"] == "connected":
            await _broadcast_status(f"{model['name']} is already loaded")
            return {"success": True, "model": model_id, "health": health}

    # Persist BEFORE broadcasting / killing anything. If the WS client
    # disconnects mid-flow (which crashes the broadcast helper), the
    # durable state is at least coherent on disk for the next LT-OS
    # startup to restore.
    _persist_model_choice(model_id, model.get("external_url") if new_is_external else None)
    config.current_conversation_model = model_id
    config.SERVICES["conversation_llm"]["name"] = model["name"]

    await _broadcast_status(f"Switching conversation LLM: {old_name} -> {model['name']}...")

    if new_is_external:
        # External path: stop the llama-3b systemd unit (NOT just kill the
        # process -- Restart=on-failure would respawn it). LT picks up the
        # URL override on next conversation call; no LT restart needed.
        ok = await _systemctl("stop", _LLAMA_3B_UNIT)
        if ok:
            await _broadcast_status(
                f"Conversation LLM routed to {model['name']} ({_LLAMA_3B_UNIT} stopped, no local server spawned)")
        else:
            await _broadcast_status(
                f"Conversation override set, but {_LLAMA_3B_UNIT} stop failed -- check sudoers", "warning")
        return {"success": True, "model": model_id, "external": True}

    # Switching FROM external -> spawnable: also ensure llama-3b systemd
    # unit is running so the existing kill_service / launch_service flow
    # below can take over for non-default models. For the default model_id
    # (llama3.2-3b) systemctl start is the right answer.
    if old_is_external:
        await _systemctl("start", _LLAMA_3B_UNIT)
        await asyncio.sleep(2)  # let the unit bind to :8081

    # Existing spawnable -> spawnable (or systemd-managed default) flow:
    # kill whatever is on :8081 then launch the new model_id via the
    # SERVICES dict start_cmd. The kill is via fuser-k (kill_service);
    # since the new launch uses the same port the restart-on-failure of
    # llama-3b-server.service is benign here -- the launched llama-server
    # will reclaim the port either way.
    if model_id != "llama3.2-3b":
        # For non-default models we override the systemd-managed Llama 3B.
        # Stop the unit first to avoid contention, then spawn manually.
        await _systemctl("stop", _LLAMA_3B_UNIT)
        await asyncio.sleep(1)
        success = await launch_service("conversation_llm")
    else:
        # Default model -- llama-3b-server.service is the canonical owner.
        # Make sure the unit is running (no-op if already started above).
        success = await _systemctl("start", _LLAMA_3B_UNIT)
        await asyncio.sleep(2)

    health = await check_health("conversation_llm")
    if success:
        await _broadcast_status(f"Conversation LLM switched to {model['name']}")
    else:
        await _broadcast_status(f"Failed to start {model['name']}", "error")

    return {"success": success, "model": model_id, "health": health}


def get_session_log() -> list[dict]:
    return _session_log


async def _tcp_connect(host: str, port: int, timeout: float) -> bool:
    """Open a TCP connection then immediately close. True iff the port
    accepts the connection within `timeout`. Cheap "is something listening
    there" probe — no auth, no TLS handshake, no HTTP parse."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def check_streamerpi_server_status() -> dict:
    """Probe the streamerpi main codebase (little-timmy-motor.service).

    Returns {reachable, running, error?} where:
      reachable: host responds on SSH (port 22). Determines whether we
                 could even attempt a remote start.
      running:   the main server is listening on STREAMERPI_HTTP_PORT.
                 Doesn't authenticate against any specific endpoint —
                 just "the process is bound to the port".

    No SSH, no sudo, no HTTP. Two short TCP-connect probes (~20 ms each
    when the host is up). Suitable for 10 s polling.
    """
    import config as cfg
    running = await _tcp_connect(cfg.STREAMERPI_HOST, cfg.STREAMERPI_HTTP_PORT, 1.5)
    if running:
        return {"reachable": True, "running": True}
    reachable = await _tcp_connect(cfg.STREAMERPI_HOST, cfg.STREAMERPI_SSH_PORT, 1.5)
    return {"reachable": reachable, "running": False}


async def toggle_streamerpi_server(enabled: bool) -> dict:
    """Start or stop little-timmy-motor.service on streamerpi via SSH +
    NOPASSWD sudo. Requires the broad sudoers drop-in on the Pi:
        echo "pi ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/pi-nopasswd
        sudo chmod 0440 /etc/sudoers.d/pi-nopasswd

    On enable, refuses to attempt the SSH call if SSH itself is
    unreachable — surfaces "unreachable" to the dashboard instead of
    burning the 3 s SSH ConnectTimeout.
    """
    import config as cfg
    action = "start" if enabled else "stop"
    await _broadcast_status(
        f"{'Starting' if enabled else 'Stopping'} streamerpi main service...")

    if enabled:
        reachable = await _tcp_connect(
            cfg.STREAMERPI_HOST, cfg.STREAMERPI_SSH_PORT, 2.0)
        if not reachable:
            await _broadcast_status("streamerpi unreachable, cannot start", "error")
            _write_session_log()
            return {"running": False, "reachable": False,
                    "error": "streamerpi unreachable"}

    cmd = [
        "ssh", "-i", cfg.STREAMERPI_SSH_KEY,
        "-o", "ConnectTimeout=3", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{cfg.STREAMERPI_SSH_USER}@{cfg.STREAMERPI_HOST}",
        "sudo", "-n", "systemctl", action, cfg.STREAMERPI_MAIN_UNIT,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        await _broadcast_status(f"systemctl {action} timed out", "error")
        _write_session_log()
        return {"running": False, "reachable": True, "error": "ssh timeout"}
    except Exception as e:
        await _broadcast_status(f"systemctl {action} failed: {e}", "error")
        _write_session_log()
        return {"running": False, "reachable": False, "error": str(e)[:200]}

    if proc.returncode != 0:
        err = (stderr.decode(errors="replace").strip()
               or stdout.decode(errors="replace").strip()
               or f"exit {proc.returncode}")[:200]
        await _broadcast_status(f"systemctl {action} failed: {err}", "error")
        _write_session_log()
        return {"running": False, "reachable": True, "error": err}

    # Give the daemon a moment to bind / unbind the listening socket before
    # the state probe — systemctl returns as soon as the process is forked.
    await asyncio.sleep(2.0)
    final = await check_streamerpi_server_status()
    state_str = "started" if final["running"] else "stopped"
    await _broadcast_status(f"streamerpi main service {state_str}")
    _write_session_log()
    return final


async def check_face_pipeline_status() -> dict:
    """Read the three-layer face-pipeline state on streamerpi.

    Returns {detection_enabled, tracking_enabled, motors_enabled,
    detection_alive, behavior_mode} or an error stub on failure.
    """
    import config as cfg
    try:
        async with httpx.AsyncClient(timeout=3.0, verify=False) as client:
            r = await client.get(f"{cfg.STREAMERPI_URL}/face_pipeline/status")
            return r.json()
    except Exception as e:
        return {
            "detection_enabled": False, "tracking_enabled": False, "motors_enabled": False,
            "error": str(e)[:120],
        }


async def _toggle_pipeline_layer(layer: str, enabled: bool) -> dict:
    """Shared backend for the three layer-specific toggles."""
    import config as cfg
    if layer not in ("detection", "tracking", "motors"):
        return {"error": f"unknown layer: {layer}"}
    label = {"detection": "Face detection", "tracking": "Face tracking", "motors": "Motors"}[layer]
    await _broadcast_status(f"{'Enabling' if enabled else 'Disabling'} {label.lower()} on streamerpi...")
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            r = await client.post(
                f"{cfg.STREAMERPI_URL}/{layer}/toggle",
                json={"enabled": enabled},
            )
            result = r.json()
            state_str = "enabled" if result.get("enabled") else "disabled"
            await _broadcast_status(f"{label} {state_str}")
            _write_session_log()
            return result
    except Exception as e:
        await _broadcast_status(f"{label} toggle failed: {e}", "error")
        _write_session_log()
        return {"enabled": False, "error": str(e)[:120]}


async def toggle_detection(enabled: bool) -> dict:
    return await _toggle_pipeline_layer("detection", enabled)


async def toggle_tracking(enabled: bool) -> dict:
    return await _toggle_pipeline_layer("tracking", enabled)


async def toggle_motors(enabled: bool) -> dict:
    return await _toggle_pipeline_layer("motors", enabled)


# Legacy aliases kept so existing callers in main.py keep working during
# the rename; remove once main.py is fully migrated.
async def check_face_tracking_status() -> dict:
    """LEGACY: returns the new face_pipeline_status shape with an `enabled`
    key bound to tracking_enabled for backwards compat with current callers."""
    p = await check_face_pipeline_status()
    if "error" not in p:
        p["enabled"] = p.get("tracking_enabled", False)
    return p


async def toggle_face_tracking(enabled: bool) -> dict:
    """LEGACY alias of toggle_tracking."""
    return await toggle_tracking(enabled)


# ---------- LT-side toggles (vision auto-poll + hearing) ----------

async def check_lt_toggles_status() -> dict:
    """Read both LT toggles in one round-trip-pair against :8893."""
    import config as cfg
    out = {
        "vision_auto_poll_enabled": False,
        "hearing_enabled": False,
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            v = await client.get(f"{cfg.TIMMY_BASE_URL}/api/vision/auto_poll")
            h = await client.get(f"{cfg.TIMMY_BASE_URL}/api/hearing")
            out["vision_auto_poll_enabled"] = bool(v.json().get("enabled", False))
            out["hearing_enabled"] = bool(h.json().get("enabled", False))
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


async def toggle_vision_auto_poll(enabled: bool) -> dict:
    """Enable/disable LT's periodic VLM poll loop. Event-driven trigger
    calls (speech, visual question) are unaffected."""
    import config as cfg
    await _broadcast_status(f"{'Enabling' if enabled else 'Disabling'} vision auto-poll...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{cfg.TIMMY_BASE_URL}/api/vision/auto_poll",
                json={"enabled": enabled},
            )
            result = r.json()
            state_str = "enabled" if result.get("enabled") else "disabled"
            await _broadcast_status(f"Vision auto-poll {state_str}")
            _write_session_log()
            return result
    except Exception as e:
        await _broadcast_status(f"Vision auto-poll toggle failed: {e}", "error")
        _write_session_log()
        return {"enabled": False, "error": str(e)[:120]}


async def toggle_hearing(enabled: bool) -> dict:
    """Mute/unmute LT's hearing. whisper-server stays running; only the
    mic-frame -> STT enqueue path is gated."""
    import config as cfg
    await _broadcast_status(f"{'Unmuting' if enabled else 'Muting'} Little Timmy's hearing...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{cfg.TIMMY_BASE_URL}/api/hearing",
                json={"enabled": enabled},
            )
            result = r.json()
            state_str = "unmuted" if result.get("enabled") else "muted"
            await _broadcast_status(f"Hearing {state_str}")
            _write_session_log()
            return result
    except Exception as e:
        await _broadcast_status(f"Hearing toggle failed: {e}", "error")
        _write_session_log()
        return {"enabled": False, "error": str(e)[:120]}
