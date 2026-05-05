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

    if svc_id == "conversation_llm":
        model = config.CONVERSATION_MODELS.get(config.current_conversation_model, {})
        result["model"] = config.current_conversation_model
        result["model_name"] = model.get("name", "Unknown")

    health_url = svc.get("health_url")

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


async def switch_conversation_model(model_id: str) -> dict:
    if model_id not in config.CONVERSATION_MODELS:
        await _broadcast_status(f"Unknown model: {model_id}", "error")
        return {"success": False, "error": "Unknown model"}

    model = config.CONVERSATION_MODELS[model_id]
    old_model = config.current_conversation_model
    old_name = config.CONVERSATION_MODELS.get(old_model, {}).get("name", old_model)

    if model_id == old_model:
        health = await check_health("conversation_llm")
        if health["status"] == "connected":
            await _broadcast_status(f"{model['name']} is already loaded")
            return {"success": True, "model": model_id, "health": health}

    await _broadcast_status(f"Switching conversation LLM: {old_name} -> {model['name']}...")

    await kill_service("conversation_llm")
    await asyncio.sleep(1)

    config.current_conversation_model = model_id
    config.SERVICES["conversation_llm"]["name"] = model["name"]

    success = await launch_service("conversation_llm")
    health = await check_health("conversation_llm")

    if success:
        await _broadcast_status(f"Conversation LLM switched to {model['name']}")
    else:
        await _broadcast_status(f"Failed to start {model['name']}", "error")

    return {"success": success, "model": model_id, "health": health}


def get_session_log() -> list[dict]:
    return _session_log


async def check_face_tracking_status() -> dict:
    """Check if face tracking is enabled on streamerpi."""
    import config as cfg
    try:
        async with httpx.AsyncClient(timeout=3.0, verify=False) as client:
            r = await client.get(f"{cfg.STREAMERPI_URL}/face_tracking/status")
            return r.json()
    except Exception as e:
        return {"enabled": False, "error": str(e)[:120]}


async def toggle_face_tracking(enabled: bool) -> dict:
    """Enable/disable face tracking on streamerpi."""
    import config as cfg
    await _broadcast_status(f"{'Enabling' if enabled else 'Disabling'} face tracking on streamerpi...")
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            r = await client.post(
                f"{cfg.STREAMERPI_URL}/face_tracking/toggle",
                json={"enabled": enabled},
            )
            result = r.json()
            state_str = "enabled" if result.get("enabled") else "disabled"
            await _broadcast_status(f"Face tracking {state_str}")
            _write_session_log()
            return result
    except Exception as e:
        await _broadcast_status(f"Face tracking toggle failed: {e}", "error")
        _write_session_log()
        return {"enabled": False, "error": str(e)[:120]}
