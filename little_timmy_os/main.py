"""Little Timmy OS — Standalone service manager and dashboard.

Runs independently of Little Timmy on port 8894.
Manages, monitors, and controls all Little Timmy stack services.
"""

# LT-OS needs the sibling persistence module from /home/gearscodeandfire/little_timmy/
# (Bundle D + Qwen3.6 conversation-tier swap). LT-OS cwd is little_timmy_os/ which
# does not have persistence/ in its search path; inject the parent dir.
import sys as _sys
_sys.path.append("/home/gearscodeandfire/little_timmy")

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import services

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timmy_os")


@asynccontextmanager
async def lifespan(app: FastAPI):
    poll_task = asyncio.create_task(health_poll_loop())
    timmy_ws_task = asyncio.create_task(timmy_ws_relay())
    log.info("Little Timmy OS started on http://%s:%d", config.WEB_HOST, config.WEB_PORT)
    # Restore persisted conversation model choice (Bundle: persistence for
    # Dan-requested toggles 2026-05-14). Lives inside lifespan() because
    # @app.on_event("startup") is silently no-op when lifespan is provided.
    try:
        from persistence import runtime_toggles
        persisted = runtime_toggles.get("conversation_model_id")
        if persisted and persisted in config.CONVERSATION_MODELS and persisted != config.current_conversation_model:
            log.info("Restoring conversation tier from persistence: %s", persisted)
            asyncio.create_task(services.switch_conversation_model(persisted))
        elif persisted and persisted not in config.CONVERSATION_MODELS:
            log.warning("persisted conversation_model_id %r unknown; keeping default %r",
                        persisted, config.current_conversation_model)
    except Exception as e:
        log.warning("runtime_toggles unavailable during startup restore: %s", e)
    yield
    poll_task.cancel()
    timmy_ws_task.cancel()


app = FastAPI(title="Little Timmy OS", version="0.2.0", lifespan=lifespan)

# Self-hosted vendor JS (Chart.js etc) so the dashboard renders without
# internet access — booth deploys may be on offline LANs.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_connected_websockets: list[WebSocket] = []


async def broadcast_event(event_type: str, data: dict):
    """Send an event to all connected WebSocket clients."""
    msg = json.dumps({"type": event_type, **data})
    disconnected = []
    for ws in _connected_websockets:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _connected_websockets:
            _connected_websockets.remove(ws)


# Register broadcast for service status updates
services.register_status_callback(broadcast_event)


async def timmy_ws_relay():
    """Connect to Little Timmy's WebSocket and relay turn/metrics/retrieval events."""
    import websockets

    while True:
        try:
            async with websockets.connect(
                config.TIMMY_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                log.info("Connected to Little Timmy WS relay")
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("type") in ("turn", "metrics", "retrieval", "tool_call", "classifier_metric", "resolution_metric"):
                            log.debug("Relay: %s event", data["type"])
                            await broadcast_event(data["type"], data)
                    except (json.JSONDecodeError, KeyError):
                        pass
            log.warning("Little Timmy WS closed, reconnecting in 3s...")
        except Exception as e:
            log.debug("WS relay error: %s, retrying in 3s...", e)
        await asyncio.sleep(3)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connected_websockets.append(websocket)

    # Send initial state
    health = await services.check_all_health()
    await websocket.send_text(json.dumps({"type": "health", "services": health}))

    # Send model list
    await websocket.send_text(json.dumps({
        "type": "models",
        "models": {k: v["name"] for k, v in config.CONVERSATION_MODELS.items()},
        "current": config.current_conversation_model,
    }))

    # Send LT-side toggles (vision auto-poll + hearing)
    try:
        toggles = await services.check_lt_toggles_status()
        await websocket.send_text(json.dumps({"type": "lt_toggles", **toggles}))
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "toggle":
                    svc_id = msg.get("service")
                    desired = msg.get("state", True)
                    if svc_id == "face_tracking":
                        result = await services.toggle_face_tracking(desired)
                        await broadcast_event("face_tracking", result)
                    elif svc_id == "streamerpi_main":
                        result = await services.toggle_streamerpi_server(desired)
                        await broadcast_event("streamerpi_main", result)
                    elif svc_id == "vision_auto_poll":
                        result = await services.toggle_vision_auto_poll(desired)
                        toggles = await services.check_lt_toggles_status()
                        await broadcast_event("lt_toggles", toggles)
                    elif svc_id == "hearing":
                        result = await services.toggle_hearing(desired)
                        toggles = await services.check_lt_toggles_status()
                        await broadcast_event("lt_toggles", toggles)
                    elif svc_id == "proactive_speech":
                        result = await services.toggle_proactive_speech(desired)
                        toggles = await services.check_lt_toggles_status()
                        await broadcast_event("lt_toggles", toggles)
                    elif svc_id == "classifier":
                        result = await services.toggle_classifier(desired)
                        toggles = await services.check_lt_toggles_status()
                        await broadcast_event("lt_toggles", toggles)
                    elif svc_id == "query_resolution":
                        result = await services.toggle_query_resolution(desired)
                        toggles = await services.check_lt_toggles_status()
                        await broadcast_event("lt_toggles", toggles)
                    elif svc_id in config.SERVICES:
                        result = await services.toggle_service(svc_id, desired)
                        health = await services.check_all_health()
                        await broadcast_event("health", {"services": health})
                elif msg.get("type") == "switch_model":
                    model_id = msg.get("model")
                    if model_id:
                        result = await services.switch_conversation_model(model_id)
                        # Broadcast updated health + model state
                        health = await services.check_all_health()
                        await broadcast_event("health", {"services": health})
                        await broadcast_event("models", {
                            "models": {k: v["name"] for k, v in config.CONVERSATION_MODELS.items()},
                            "current": config.current_conversation_model,
                        })
                elif msg.get("type") == "refresh":
                    health = await services.check_all_health()
                    await broadcast_event("health", {"services": health})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        if websocket in _connected_websockets:
            _connected_websockets.remove(websocket)


@app.get("/api/health")
async def get_health():
    return await services.check_all_health()


@app.get("/api/log")
async def get_log():
    return services.get_session_log()


@app.get("/api/host")
async def get_host_metrics():
    """Lightweight host snapshot for the LT-OS dashboard's Host Metrics panel.

    CPU %, system RAM, disk on /, load average (psutil), plus AMD GPU telemetry.
    GPU fields come from ops.gpu_sysfs.read_gpu() — the single source of truth
    shared with ops/gpu_watchdog.sh's freeze-forensics ring buffer, so the live
    dashboard and the wedge dump never drift on which card / which fields.
    The first four GPU keys (vram_used_gb, vram_total_gb, vram_percent,
    gpu_busy_percent) preserve this endpoint's historical schema; the helper
    also surfaces temp_c, power_w, sclk_mhz, mem_busy_percent.
    """
    import psutil
    import os
    from ops import gpu_sysfs
    try:
        cpu_pct = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        du = psutil.disk_usage("/")
        try:
            load1, load5, load15 = os.getloadavg()
        except (AttributeError, OSError):
            load1 = load5 = load15 = None
        return {
            "available": True,
            "cpu_percent": round(cpu_pct, 1),
            "cpu_count": psutil.cpu_count(logical=True) or 0,
            "ram_used_gb": round(vm.used / (1024 ** 3), 2),
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_percent": round(vm.percent, 1),
            "disk_used_gb": round(du.used / (1024 ** 3), 1),
            "disk_total_gb": round(du.total / (1024 ** 3), 1),
            "disk_percent": round(du.percent, 1),
            "load_1m": round(load1, 2) if load1 is not None else None,
            "load_5m": round(load5, 2) if load5 is not None else None,
            "load_15m": round(load15, 2) if load15 is not None else None,
            # GPU telemetry (amdgpu sysfs) via the shared helper.
            **gpu_sysfs.read_gpu(),
        }
    except Exception as e:
        return {"available": False, "error": str(e)[:120]}


# ---------- Recording control (Supervisor M6) ----------
# Thin proxy to booth-display's /api/recording/* endpoints. booth-display
# owns the actual subsystem (file writes, MediaRecorder signal, chunks);
# LT-OS just exposes operator-facing toggle + status so the recording
# button lives in the same dashboard as everything else.

@app.post("/api/recording/start")
async def recording_start():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.post(f"{config.BOOTH_DISPLAY_URL}/api/recording/start")
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"booth-display unreachable: {e}"}, status_code=502)


@app.post("/api/recording/stop")
async def recording_stop():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(f"{config.BOOTH_DISPLAY_URL}/api/recording/stop")
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"booth-display unreachable: {e}"}, status_code=502)


@app.get("/api/recording/status")
async def recording_status():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.BOOTH_DISPLAY_URL}/api/recording/status")
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"active": False, "error": f"booth-display unreachable: {e}"}, status_code=502)


@app.get("/api/models")
async def get_models():
    return {
        "models": {k: v["name"] for k, v in config.CONVERSATION_MODELS.items()},
        "current": config.current_conversation_model,
    }


@app.get("/api/timmy/metrics")
async def get_timmy_metrics():
    """Proxy metrics from Little Timmy if it is running."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_METRICS_URL)
            return r.json()
    except Exception:
        return {"error": "Little Timmy not reachable"}


@app.get("/api/timmy/conversation")
async def get_timmy_conversation():
    """Proxy conversation history from Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_CONVERSATION_URL)
            return r.json()
    except Exception:
        return {"error": "Little Timmy not reachable"}




@app.post("/api/service/{svc_id}/restart")
async def restart_service(svc_id: str):
    """Restart a service by killing and relaunching it."""
    if svc_id not in config.SERVICES:
        return {"error": f"Unknown service: {svc_id}"}
    result = await services.toggle_service(svc_id, True)
    health = await services.check_all_health()
    await broadcast_event("health", {"services": health})
    return result


@app.post("/api/service/{svc_id}/stop")
async def stop_service(svc_id: str):
    """Stop a service."""
    if svc_id not in config.SERVICES:
        return {"error": f"Unknown service: {svc_id}"}
    await services.kill_service(svc_id)
    health = await services.check_all_health()
    await broadcast_event("health", {"services": health})
    return await services.check_health(svc_id)


@app.post("/api/service/{svc_id}/start")
async def start_service(svc_id: str):
    """Start a service."""
    if svc_id not in config.SERVICES:
        return {"error": f"Unknown service: {svc_id}"}
    success = await services.launch_service(svc_id)
    health = await services.check_all_health()
    await broadcast_event("health", {"services": health})
    return await services.check_health(svc_id)


@app.get("/api/timmy/chatlog")
async def get_timmy_chatlog():
    """Proxy plain-text chat log from Little Timmy."""
    import httpx
    from fastapi.responses import PlainTextResponse
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/chatlog")
            return PlainTextResponse(r.text)
    except Exception:
        return PlainTextResponse("Little Timmy not reachable")


@app.get("/api/timmy/last_payload")
async def get_timmy_last_payload():
    """Proxy most-recent LLM payload (assembled ephemeral prompt) from Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/last_payload")
            return r.json()
    except Exception as e:
        return {"available": False, "error": f"timmy unreachable: {e}"}


@app.get("/api/timmy/mood")
async def get_timmy_mood():
    """Proxy current 2-axis mood state from Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/mood")
            return r.json()
    except Exception as e:
        return {"error": f"timmy unreachable: {e}"}


@app.get("/api/timmy/audio_diag")
async def get_timmy_audio_diag():
    """Proxy LT's live capture diagnostics (last_peak / last_vad_prob) for the
    mic VU meter. Polled by the dashboard at a few Hz, so keep the timeout
    short and never raise -- a missing reading just leaves the meter idle."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/audio/diag")
            return r.json()
    except Exception as e:
        return {"error": f"timmy unreachable: {e}"}


@app.post("/api/timmy/capture/energy_floor")
async def set_timmy_energy_floor(payload: dict | None = None):
    """Proxy: set LT's near-field onset energy floor. Body: {"value": 0.06}."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/capture/energy_floor",
                                  json=(payload or {}))
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"timmy unreachable: {e}"}, status_code=502)


# --- P4 face-flap debounce knobs (2026-06-11) -------------------------------
# Six knobs across two layers: A1/A2 sticky-identity lives on streamerpi
# (/face_id/config, persisted in face_id_tuning.json there); B3/C5 live in
# LT's runtime toggles (/api/vision/tuning). All apply live, no restarts.

@app.get("/api/timmy/face_id_tuning")
async def get_face_id_tuning():
    """Proxy: streamerpi A1/A2 sticky-identity tuning."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0, verify=False) as client:
            r = await client.get(config.STREAMERPI_URL + "/face_id/config")
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"streamerpi unreachable: {e}"}, status_code=502)


@app.post("/api/timmy/face_id_tuning")
async def set_face_id_tuning(payload: dict | None = None):
    """Proxy: update streamerpi A1/A2 tuning. Body: any subset of
    {face_unknown_debounce_frames, face_identity_acquire_dist,
     face_identity_release_dist}. The Pi validates + persists."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0, verify=False) as client:
            r = await client.post(config.STREAMERPI_URL + "/face_id/config",
                                  json=(payload or {}))
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"streamerpi unreachable: {e}"}, status_code=502)


@app.get("/api/timmy/vision/tuning")
async def get_vision_tuning():
    """Proxy: LT-side P4 knobs (B3 novelty persistence, C5 enroll gates)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/vision/tuning")
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"timmy unreachable: {e}"}, status_code=502)


@app.post("/api/timmy/vision/tuning")
async def set_vision_tuning(payload: dict | None = None):
    """Proxy: update LT-side P4 knobs. LT validates + persists."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/vision/tuning",
                                  json=(payload or {}))
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"timmy unreachable: {e}"}, status_code=502)


@app.get("/api/timmy/situation")
async def get_situation():
    """Proxy: LT-side situational-awareness regime (Slice A)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/situation")
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"timmy unreachable: {e}"}, status_code=502)


@app.post("/api/timmy/situation")
async def set_situation(payload: dict | None = None):
    """Proxy: set the regime. LT whitelists + persists ('' == OFF)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/situation",
                                  json=(payload or {}))
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"timmy unreachable: {e}"}, status_code=502)


@app.get("/api/timmy/tts_mute")
async def get_tts_mute():
    """Proxy: LT-side mouth-mute (Timmy's conversational voice + fillers)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/tts_mute")
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"timmy unreachable: {e}"}, status_code=502)


@app.post("/api/timmy/tts_mute")
async def set_tts_mute(payload: dict | None = None):
    """Proxy: mute/unmute Timmy's mouth. LT persists; /api/announce still speaks."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/tts_mute",
                                  json=(payload or {}))
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"timmy unreachable: {e}"}, status_code=502)


@app.post("/api/timmy/announce")
async def timmy_announce(payload: dict | None = None):
    """Proxy: speak text out of Timmy's speaker. Body: {"text": "...",
    "no_prefix": true}. Used by the auto-calibrate flow to voice its prompts
    (no_prefix -> Timmy just says the output, no "This is Claude" prefix)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/announce",
                                  json=(payload or {}))
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"spoken": False, "error": f"timmy unreachable: {e}"}, status_code=502)


@app.post("/api/timmy/mood/override")
async def set_timmy_mood_override(payload: dict | None = None):
    """Forward a manual mood override (or release) from the dashboard to Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/mood/override",
                                  json=payload or {})
            return r.json()
    except Exception as e:
        return {"ok": False, "error": f"timmy unreachable: {e}"}


@app.post("/api/timmy/speaker/reenroll")
async def proxy_speaker_reenroll(payload: dict | None = None):
    """Forward a UI re-enrollment click to Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/speaker/reenroll",
                                  json=payload or {})
            return r.json()
    except Exception as e:
        return {"ok": False, "error": f"timmy unreachable: {e}"}


@app.post("/api/timmy/feedback/manual_flag")
async def proxy_manual_flag(payload: dict | None = None):
    """Forward a UI flag click to Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(config.TIMMY_BASE_URL + "/api/feedback/manual_flag",
                                  json=payload or {})
            return r.json()
    except Exception as e:
        return {"ok": False, "error": f"timmy unreachable: {e}"}


@app.get("/api/timmy/feedback/last_flag")
async def proxy_last_flag():
    """Proxy the most recent flagged.jsonl entry from Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/feedback/last_flag")
            return r.json()
    except Exception as e:
        return {"available": False, "error": f"timmy unreachable: {e}"}


@app.get("/api/timmy/vision")
async def get_timmy_vision():
    """Proxy vision debug state from Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/vision")
            return r.json()
    except Exception:
        return {"enabled": False, "error": "Little Timmy not reachable"}


@app.get("/api/timmy/presence")
async def get_timmy_presence():
    """Proxy room-presence ledger from Little Timmy."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(config.TIMMY_BASE_URL + "/api/presence")
            return r.json()
    except Exception:
        return {"enabled": False, "present": [], "error": "Little Timmy not reachable"}


@app.get("/api/behavior")
async def get_behavior():
    """Proxy behavior state from streamerpi."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0, verify=False) as client:
            r = await client.get("https://192.168.1.110:8080/behavior/status")
            return r.json()
    except Exception:
        return {"mode": "unknown", "error": "streamerpi not reachable"}

@app.get("/api/face_pipeline")
async def get_face_pipeline():
    """Get all three face-pipeline flags from streamerpi (detection /
    tracking / motors) + diagnostics."""
    return await services.check_face_pipeline_status()


@app.post("/api/detection/toggle")
async def toggle_detection(data: dict):
    """Toggle YuNet+SFace detection on streamerpi."""
    return await services.toggle_detection(bool(data.get("enabled", True)))


@app.post("/api/tracking/toggle")
async def toggle_tracking(data: dict):
    """Toggle face→servo target binding on streamerpi."""
    return await services.toggle_tracking(bool(data.get("enabled", True)))


@app.post("/api/motors/toggle")
async def toggle_motors(data: dict):
    """Toggle the global motors enable on streamerpi (pan/tilt only;
    audio/jaw is on separate hardware and unaffected)."""
    return await services.toggle_motors(bool(data.get("enabled", True)))


# Legacy endpoint kept so any consumer still pointing at /api/face_tracking
# keeps working. Maps to the new face_pipeline status with a back-compat
# `enabled` key bound to tracking_enabled.
@app.get("/api/face_tracking")
async def get_face_tracking():
    return await services.check_face_tracking_status()


@app.post("/api/face_tracking/toggle")
async def toggle_face_tracking(data: dict):
    return await services.toggle_face_tracking(bool(data.get("enabled", True)))


@app.get("/api/streamerpi/main")
async def get_streamerpi_main():
    """Probe streamerpi main codebase (little-timmy-motor.service)."""
    return await services.check_streamerpi_server_status()


@app.post("/api/streamerpi/main/toggle")
async def toggle_streamerpi_main(data: dict):
    """Start / stop little-timmy-motor.service on streamerpi via SSH+sudo."""
    enabled = data.get("enabled", True)
    return await services.toggle_streamerpi_server(enabled)


@app.get("/api/timmy/toggles")
async def get_lt_toggles():
    """LT-side runtime toggles (vision auto-poll + hearing)."""
    return await services.check_lt_toggles_status()


@app.post("/api/vision/auto_poll/toggle")
async def toggle_vision_auto_poll(data: dict):
    """Toggle the LT periodic 1fps VLM poll loop. Event-driven trigger
    calls (speech, visual question) keep working independently."""
    return await services.toggle_vision_auto_poll(bool(data.get("enabled", True)))


@app.post("/api/hearing/toggle")
async def toggle_hearing(data: dict):
    """Mute/unmute LT's hearing. whisper-server stays running."""
    return await services.toggle_hearing(bool(data.get("enabled", True)))


@app.post("/api/proactive/toggle")
async def toggle_proactive_speech(data: dict):
    """Enable/disable LT's proactive (unprompted) speech. Live; no restart."""
    return await services.toggle_proactive_speech(bool(data.get("enabled", True)))


@app.post("/api/classifier/toggle")
async def toggle_classifier(data: dict):
    """Enable/disable LT's first-pass tool-call classifier (:8092). Live; no restart."""
    return await services.toggle_classifier(bool(data.get("enabled", True)))


@app.post("/api/query_resolution/toggle")
async def toggle_query_resolution(data: dict):
    """Enable/disable LT's elliptical-query coreference resolution (:8092). Live;
    no restart once the retrieval code is loaded."""
    return await services.toggle_query_resolution(bool(data.get("enabled", True)))


async def health_poll_loop():
    """Background task: poll service health and broadcast updates."""
    while True:
        await asyncio.sleep(config.HEALTH_POLL_INTERVAL)
        try:
            health = await services.check_all_health()
            await broadcast_event("health", {"services": health})
            toggles = await services.check_lt_toggles_status()
            await broadcast_event("lt_toggles", toggles)
        except Exception as e:
            log.warning("Health poll error: %s", e)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html>
<head>
<title>Little Timmy OS</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="/static/vendor/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Courier New', monospace;
  background: #0d1117;
  color: #c9d1d9;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
header {
  background: #161b22;
  border-bottom: 1px solid #30363d;
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
header h1 {
  color: #e94560;
  font-size: 20px;
  letter-spacing: 2px;
}
header .uptime {
  color: #8b949e;
  font-size: 12px;
}
.main-content {
  flex: 1;
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 16px;
  padding: 16px 24px;
}
.panel {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 16px;
}
.panel h2 {
  color: #e94560;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid #21262d;
}
/* Roll-up: each .panel is wrapped in <details open><summary>...h2...</summary>...
   Click the h2 to collapse the panel body. ▼ when open, ▶ when closed. */
.panel > summary {
  list-style: none;
  cursor: pointer;
  user-select: none;
}
.panel > summary::-webkit-details-marker { display: none; }
.panel > summary > h2 { margin-bottom: 0; border-bottom: none; padding-bottom: 0; }
.panel > summary::before {
  content: "\25BC";  /* ▼ */
  display: inline-block;
  width: 14px;
  color: #484f58;
  font-size: 10px;
  margin-right: 4px;
  transition: transform 0.15s ease;
}
.panel:not([open]) > summary::before { content: "\25B6"; /* ▶ */ }
.panel:not([open]) > summary > h2 { color: #8b949e; }
/* Restore the h2's bottom rule only when expanded; place it on the summary */
.panel[open] > summary {
  border-bottom: 1px solid #21262d;
  padding-bottom: 8px;
  margin-bottom: 14px;
}

/* Service cards */
.service-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  margin: 6px 0;
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 6px;
  transition: border-color 0.2s;
}
.service-card.connected { border-left: 3px solid #3fb950; }
.service-card.disconnected { border-left: 3px solid #f85149; }
.service-info {
  flex: 1;
}
.service-name {
  font-weight: bold;
  font-size: 13px;
  color: #e6edf3;
}
.service-detail {
  font-size: 11px;
  color: #8b949e;
  margin-top: 2px;
}
.service-port {
  font-size: 11px;
  color: #8b949e;
  margin-right: 12px;
}

/* M7: Services panel as horizontal strip across the top.
   The first column of .main-content (Services + streamerpi-controls +
   conversation LLM selector) spans the whole row, and service cards
   inside lay out as a flex-wrap of compact chips. */
.main-content > div:first-child {
  grid-column: 1 / -1;
}
.main-content > div:first-child #services,
.main-content > div:first-child #lt-runtime-toggles,
.main-content > div:first-child #streamerpi-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: stretch;
}
.main-content > div:first-child #streamerpi-controls > div:first-child {
  flex: 0 0 100%;
}
.main-content > div:first-child .service-card {
  flex: 0 1 240px;
  margin: 0;
  min-width: 0;
}
.main-content > div:first-child .service-card .service-name {
  font-size: 12px;
}
.main-content > div:first-child .service-card .service-detail {
  font-size: 10px;
}
.main-content > div:first-child #model-selector {
  margin-top: 12px;
}

/* Toggle switch */
.toggle {
  position: relative;
  width: 44px;
  height: 24px;
  flex-shrink: 0;
}
.toggle input {
  opacity: 0;
  width: 0;
  height: 0;
}
.toggle .slider {
  position: absolute;
  cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background: #21262d;
  border-radius: 24px;
  transition: 0.3s;
}
.toggle .slider:before {
  content: "";
  position: absolute;
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background: #8b949e;
  border-radius: 50%;
  transition: 0.3s;
}
.toggle input:checked + .slider {
  background: #238636;
}
.toggle input:checked + .slider:before {
  transform: translateX(20px);
  background: #3fb950;
}
.toggle.busy .slider {
  opacity: 0.5;
  cursor: wait;
}

/* Model selector */
.model-selector {
  margin: 12px 0 6px 0;
  padding: 10px 12px;
  background: #0d1117;
  border: 1px solid #21262d;
  border-left: 3px solid #58a6ff;
  border-radius: 6px;
}
.model-selector label {
  font-size: 11px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 1px;
  display: block;
  margin-bottom: 6px;
}
.model-selector select {
  width: 100%;
  background: #161b22;
  color: #e6edf3;
  border: 1px solid #30363d;
  border-radius: 4px;
  padding: 6px 8px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  cursor: pointer;
  outline: none;
}
.model-selector select:focus {
  border-color: #58a6ff;
}
.model-selector select:disabled {
  opacity: 0.5;
  cursor: wait;
}
.model-selector .model-status {
  font-size: 10px;
  color: #484f58;
  margin-top: 4px;
}

/* Conversation */
#conversation {
  max-height: 500px;
  overflow-y: auto;
}
.turn {
  margin: 6px 0;
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 13px;
  line-height: 1.4;
}
.turn.user {
  background: #0f3460;
  border-left: 3px solid #e94560;
}
.turn.assistant {
  background: #1a1a2e;
  border-left: 3px solid #00d2ff;
}
.turn .role {
  font-size: 10px;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 3px;
  font-weight: bold;
}
/* Speaker attribution colors: makes mis-classification immediately
   visible to the operator. Known names (Dan, Erin, etc.) appear in
   green; unknown_N or empty in amber so the operator sees the
   misclass without having to read the actual word. */
.turn .role.speaker-known   { color: #3fb950; }
.turn .role.speaker-unknown { color: #f0883e; }
.turn .role.speaker-assistant { color: #58a6ff; }

/* H9: prominent live speaker-ID badge in Body Behavior panel.
   Operator-glanceable indicator of the most recent user-turn speaker.
   Updates from the same WS turn-event flow that drives addTurn(). */
.speaker-badge {
  font-size: 18px;
  font-weight: bold;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  min-width: 80px;
  text-align: center;
}
.speaker-badge.speaker-known {
  color: #3fb950;
  background: #1f3a25;
  border-color: #3fb950;
}
.speaker-badge.speaker-unknown {
  color: #f0883e;
  background: #3a2a1f;
  border-color: #f0883e;
}
.speaker-badge-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.speaker-badge-label {
  font-size: 9px;
  letter-spacing: 1px;
  color: #8b949e;
  text-transform: uppercase;
}
.turn .content {
  color: #e6edf3;
}
#conv-offline {
  color: #8b949e;
  font-size: 12px;
  font-style: italic;
}

/* Retrieved context */
.ret-memory {
  padding: 6px 8px;
  margin: 4px 0;
  background: #0d1117;
  border: 1px solid #21262d;
  border-left: 3px solid #8b5cf6;
  border-radius: 4px;
  color: #c9d1d9;
  line-height: 1.3;
}
.ret-memory .ret-type {
  font-size: 10px;
  color: #8b5cf6;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.ret-memory .ret-score {
  font-size: 10px;
  color: #484f58;
  float: right;
}
.ret-fact {
  padding: 4px 8px;
  margin: 3px 0;
  background: #0d1117;
  border: 1px solid #21262d;
  border-left: 3px solid #d29922;
  border-radius: 4px;
  color: #d29922;
  font-size: 12px;
}

/* Metrics */
.metric-row {
  display: flex;
  justify-content: space-between;
  padding: 5px 0;
  border-bottom: 1px solid #21262d;
  font-size: 13px;
}
.metric-row .label { color: #8b949e; }
.metric-row .value { color: #58a6ff; font-weight: bold; }

/* Status bar */
#status-bar {
  background: #161b22;
  border-top: 1px solid #30363d;
  padding: 8px 24px;
  font-size: 12px;
  max-height: 140px;
  overflow-y: auto;
  display: flex;
  flex-direction: column-reverse;
}
.status-entry {
  padding: 3px 0;
  border-bottom: 1px solid #0d1117;
}
.status-entry .ts {
  color: #484f58;
  margin-right: 8px;
}
.status-entry.info { color: #8b949e; }
.status-entry.error { color: #f85149; }
.status-entry.warning { color: #d29922; }

/* Mood panel */
.mood-grid {
  display: grid;
  grid-template-columns: 72px repeat(3, 1fr);
  grid-template-rows: 22px repeat(3, 52px);
  gap: 4px;
  margin-top: 4px;
}
.mood-grid .corner {}
.mood-grid .axis-label-x,
.mood-grid .axis-label-y {
  color: #8b949e;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.mood-cell {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #484f58;
  font-size: 18px;
  transition: background 0.3s, color 0.3s, border-color 0.3s, box-shadow 0.3s;
}
.mood-cell.active {
  background: #e94560;
  color: #fff;
  border-color: #e94560;
  box-shadow: 0 0 8px rgba(233, 69, 96, 0.45);
}
/* Override mode: cells become clickable targets */
.mood-grid.override .mood-cell {
  cursor: pointer;
}
.mood-grid.override .mood-cell:hover {
  border-color: #58a6ff;
  color: #58a6ff;
}
.mood-grid.override .mood-cell.active {
  background: #d29922;            /* amber == human-driven, distinct from auto red */
  border-color: #d29922;
  box-shadow: 0 0 8px rgba(210, 153, 34, 0.5);
  color: #0d1117;
}
.mood-override-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0 2px;
  font-size: 11px;
  color: #8b949e;
  cursor: pointer;
  user-select: none;
}
.mood-override-toggle input { cursor: pointer; }
.mood-override-status {
  font-size: 10px;
  color: #d29922;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.mood-rendered {
  margin-top: 12px;
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 4px;
  padding: 8px;
  font-size: 11px;
  color: #bc8cff;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Courier New', monospace;
}
.mood-meta {
  margin-top: 8px;
  font-size: 10px;
  color: #484f58;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}

/* Responsive */
@media (max-width: 1000px) {
  .main-content { grid-template-columns: 1fr; }
}

/* PARTY MODE banner — the single most important party control. Loud on, dim off. */
#party-banner {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  margin: 0 0 16px 0; padding: 14px 22px; border-radius: 12px;
  border: 2px solid #30363d; background: #161b22; cursor: pointer;
  transition: all .2s ease; user-select: none;
}
#party-banner .pb-label {
  font-size: 22px; font-weight: 800; letter-spacing: 2px;
  text-transform: uppercase; color: #8b949e;
}
#party-banner .pb-sub { font-size: 12px; color: #6e7681; margin-top: 2px; }
#party-banner .pb-state {
  font-size: 13px; font-weight: 700; letter-spacing: 1px;
  padding: 6px 14px; border-radius: 999px;
  background: #21262d; color: #8b949e; white-space: nowrap;
}
/* ON: unmistakable — neon magenta, pulsing glow, bright text. */
#party-banner.on {
  border-color: #ff2d95;
  background: linear-gradient(100deg, #2a0a1f 0%, #3d0f2b 50%, #2a0a1f 100%);
  animation: partyPulse 1.4s ease-in-out infinite;
}
#party-banner.on .pb-label { color: #ff2d95; text-shadow: 0 0 12px rgba(255,45,149,.7); }
#party-banner.on .pb-sub { color: #ff8ec6; }
#party-banner.on .pb-state { background: #ff2d95; color: #fff; }
@keyframes partyPulse {
  0%,100% { box-shadow: 0 0 8px rgba(255,45,149,.35), inset 0 0 12px rgba(255,45,149,.12); }
  50%     { box-shadow: 0 0 26px rgba(255,45,149,.75), inset 0 0 20px rgba(255,45,149,.25); }
}
</style>
</head>
<body>

<!-- Tool-call flash banner (2026-06-18). Slides in when the classifier routes a
     turn to a tool, naming which tool fired; auto-hides after a few seconds. -->
<div id="tool-call-flash" style="display:none; position:fixed; top:16px; left:50%;
     transform:translateX(-50%); z-index:9999; background:#1f6feb; color:#fff;
     padding:10px 18px; border-radius:8px; font-weight:bold; font-size:14px;
     box-shadow:0 4px 16px rgba(0,0,0,0.5); border:1px solid #58a6ff;"></div>

<header>
  <h1>LITTLE TIMMY OS</h1>
  <span class="uptime" id="uptime">Connecting...</span>
</header>

<!-- PARTY MODE: one-tap situation_regime=PARTY. Disables the risky matcher
     continuity path + sets the 'assume strangers' prompt prior. The single
     most important control during the Open Sauce party. -->
<div id="party-banner" onclick="togglePartyMode()" title="Tap to toggle PARTY mode (situation_regime)">
  <div>
    <div class="pb-label" id="party-label">🎉 Party Mode — Off</div>
    <div class="pb-sub" id="party-sub">Tap to assume strangers + disable speaker auto-continuity</div>
  </div>
  <div class="pb-state" id="party-state">OFF</div>
</div>

<div class="main-content">
  <div>
    <details class="panel" open>
      <summary><h2>Services</h2></summary>
      <div id="services"></div>
      <!-- LT-side runtime toggles (relocated out of streamerpi Controls 2026-06-18):
           these gate Little Timmy itself, not the Pi, so they belong with the
           service cards above (whisper.cpp, ollama, brain), not under streamerpi. -->
      <div id="lt-runtime-toggles">
        <div class="service-card" id="hearing-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Hearing (mic → STT)</div>
            <div class="service-detail" id="hearing-detail">Checking...</div>
          </div>
          <label class="toggle" id="hearing-toggle">
            <input type="checkbox" onchange="toggleLTFlag('hearing', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <!-- Mic VU meter (relocated out of streamerpi Controls 2026-06-18): it's
             LT's own mic level, not a Pi control. Data from /api/audio/diag. -->
        <div class="service-card" id="mic-vu-card" style="border-left:3px solid #484f58; flex-direction:column; align-items:stretch;">
          <div class="service-info" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div class="service-name">
              Mic Level (VU)
              <span id="vu-speech-dot" title="VAD speech-active"
                    style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#30363d; margin-left:6px; vertical-align:middle;"></span>
            </div>
            <div class="service-detail" id="vu-readout" style="font-family:monospace; font-size:11px;">cur -- · pk -- · vad --</div>
          </div>
          <div id="vu-track" style="position:relative; height:14px; background:#161b22; border:1px solid #21262d; border-radius:3px; overflow:hidden;">
            <div id="vu-fill" style="position:absolute; left:0; top:0; bottom:0; width:0%; background:#3fb950; transition:width 90ms linear;"></div>
            <div id="vu-peak" style="position:absolute; top:0; bottom:0; width:2px; left:0%; background:#f0f6fc; box-shadow:0 0 3px #fff;"></div>
            <!-- Energy floor: onset below this peak is ignored as background. -->
            <div id="vu-floor" style="position:absolute; top:-2px; bottom:-2px; width:2px; left:0%; background:#f0883e; box-shadow:0 0 4px #f0883e;" title="Energy floor (onset gate)"></div>
          </div>
          <!-- Energy-floor control: peak-amplitude onset gate. 0 == off. Set
               between the room floor and your speaking peak (watch the bar). -->
          <div style="display:flex; align-items:center; gap:8px; margin-top:7px; font-size:11px; color:#8b949e;">
            <span style="white-space:nowrap;">Onset floor</span>
            <input type="range" id="vu-floor-slider" min="0" max="0.30" step="0.005" value="0"
                   oninput="onFloorInput(this.value)" onchange="commitFloor(this.value)"
                   style="flex:1; accent-color:#f0883e; cursor:pointer;">
            <span id="vu-floor-val" style="font-family:monospace; color:#f0883e; min-width:42px; text-align:right;">0.000</span>
          </div>
          <!-- Auto-calibrate: phase 1 measures the room (stay quiet), phase 2
               measures the voice (talk), then sets the floor between them. -->
          <div style="display:flex; align-items:center; gap:8px; margin-top:7px;">
            <button id="vu-cal-btn" onclick="autoCalibrateFloor()"
                    style="font-size:11px; padding:4px 10px; background:#3a2a1f; color:#f0883e; border:1px solid #f0883e; border-radius:4px; cursor:pointer; white-space:nowrap;">
              Auto-calibrate floor</button>
            <span id="vu-cal-status" style="font-size:11px; color:#8b949e;"></span>
          </div>
        </div>
        <div class="service-card" id="proactive-speech-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Proactive Speech (unprompted)</div>
            <div class="service-detail" id="proactive-speech-detail">Checking...</div>
          </div>
          <label class="toggle" id="proactive-speech-toggle">
            <input type="checkbox" onchange="toggleLTFlag('proactive_speech', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <!-- First-pass tool-call classifier (Qwen3-4B :8092, 2026-06-18). When ON,
             each utterance is routed by the classifier before the brain; store_fact
             commands execute as a tool instead of a chat reply. Detail line shows
             whether :8092 is reachable. -->
        <div class="service-card" id="classifier-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Tool-Call Classifier (Qwen3-4B :8092)</div>
            <div class="service-detail" id="classifier-detail">Checking...</div>
          </div>
          <label class="toggle" id="classifier-toggle">
            <input type="checkbox" onchange="toggleLTFlag('classifier', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <!-- Elliptical-query coreference resolution (:8092, 2026-06-18). When ON,
             a deictic follow-up ("what's its name again?") is rewritten to a
             standalone query via :8092 before semantic retrieval; clean queries
             skip it. Detail line shows whether :8092 is reachable. -->
        <div class="service-card" id="query_resolution-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Query Resolution (coref :8092)</div>
            <div class="service-detail" id="query_resolution-detail">Checking...</div>
          </div>
          <label class="toggle" id="query_resolution-toggle">
            <input type="checkbox" onchange="toggleLTFlag('query_resolution', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
      </div>
      <div id="streamerpi-controls" style="margin-top:12px; padding-top:10px; border-top:1px solid #21262d;">
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">streamerpi Controls</div>
        <div class="service-card" id="streamerpi-main-card" style="border-left:3px solid #484f58;"
             title="little-timmy-motor.service on streamerpi (port 8080)">
          <div class="service-info">
            <div class="service-name">Streamerpi Server</div>
            <div class="service-detail" id="streamerpi-main-detail">Checking...</div>
          </div>
          <label class="toggle" id="streamerpi-main-toggle">
            <input type="checkbox" onchange="toggleStreamerpiMain(this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <div class="service-card" id="detection-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Face Detection (YuNet+SFace)</div>
            <div class="service-detail" id="detection-detail">Checking...</div>
          </div>
          <label class="toggle" id="detection-toggle">
            <input type="checkbox" onchange="toggleFacePipeline('detection', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <div class="service-card" id="tracking-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Face Tracking (face → servo target)</div>
            <div class="service-detail" id="tracking-detail">Checking...</div>
          </div>
          <label class="toggle" id="tracking-toggle">
            <input type="checkbox" onchange="toggleFacePipeline('tracking', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <div class="service-card" id="motors-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Motors (pan / tilt)</div>
            <div class="service-detail" id="motors-detail">Checking...</div>
          </div>
          <label class="toggle" id="motors-toggle">
            <input type="checkbox" onchange="toggleFacePipeline('motors', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <div class="service-card" id="vision-auto-poll-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Vision Auto-Poll (1fps VLM)</div>
            <div class="service-detail" id="vision-auto-poll-detail">Checking...</div>
          </div>
          <label class="toggle" id="vision-auto-poll-toggle">
            <input type="checkbox" onchange="toggleLTFlag('vision_auto_poll', this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <!-- Mouth-mute (2026-06-12). Silences Timmy's replies + fillers (ears +
             matcher stay live; /api/announce still speaks). A clean bench for
             two-voice attribution tests and enrolling guests without Timmy
             talking over the cues. -->
        <div class="service-card" id="tts-mute-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Mute Mouth (lab)</div>
            <div class="service-detail" id="tts-mute-detail">Loading…</div>
          </div>
          <label class="toggle" id="tts-mute-toggle">
            <input type="checkbox" onchange="commitTtsMute(this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <!-- Slice A: manual situational-awareness regime (2026-06-12). Injects
             an NL [SITUATION] line into Timmy's prompt; "Off" emits nothing.
             Live (read per-turn) — no restart needed. -->
        <div class="service-card" id="situation-card" style="border-left:3px solid #484f58; flex-direction:column; align-items:stretch;">
          <div class="service-info" style="margin-bottom:6px;">
            <div class="service-name">Situation Regime</div>
            <div class="service-detail" id="situation-status">Loading…</div>
          </div>
          <select id="situation-select" onchange="commitSituation(this.value)"
                  style="width:100%; background:#0d1117; color:#c9d1d9; border:1px solid #30363d; border-radius:6px; padding:6px; font-size:12px;">
            <option value="">Off (no situation prior)</option>
            <option value="SOLO">Solo — alone with Dan</option>
            <option value="GUEST">Guest — one visitor</option>
            <option value="SMALL_GROUP">Small group — a few, some unknown</option>
            <option value="PARTY">Party — crowd of strangers</option>
            <option value="EXPO">Expo — show floor, constant strangers</option>
          </select>
        </div>
        <!-- P4 face-flap debounce knobs (2026-06-11). Two layers: Pi-side
             A1/A2 sticky identity (streamerpi /face_id/config) and LT-side
             B3/C5 gates (runtime toggles). All live, no restart needed. -->
        <div class="service-card" id="face-tuning-card" style="border-left:3px solid #484f58; flex-direction:column; align-items:stretch;">
          <div class="service-info" style="margin-bottom:6px;">
            <div class="service-name">Face ID Debounce (P4)</div>
            <div class="service-detail" id="face-tuning-status">Loading…</div>
          </div>
          <div style="display:grid; grid-template-columns:auto 70px; gap:4px 8px; font-size:11px; color:#8b949e; align-items:center;">
            <span title="A1: consecutive unknown frames before a held identity is released (Pi)">Unknown debounce (frames)</span>
            <input type="number" id="ft-debounce" min="1" max="20" step="1"
                   onchange="commitFaceTuning('pi','face_unknown_debounce_frames',this.value)" style="width:64px;">
            <span title="A2: latch an identity below this cosine distance (Pi; effective only ≤ the 0.45 match threshold)">Identity acquire dist</span>
            <input type="number" id="ft-acquire" min="0.05" max="1.5" step="0.01"
                   onchange="commitFaceTuning('pi','face_identity_acquire_dist',this.value)" style="width:64px;">
            <span title="A2: release a held identity only past this distance, sustained (Pi)">Identity release dist</span>
            <input type="number" id="ft-release" min="0.05" max="1.5" step="0.01"
                   onchange="commitFaceTuning('pi','face_identity_release_dist',this.value)" style="width:64px;">
            <span title="B3: a person must appear in ≥ ceil(this × 5) of the last 5 scene records to count as new (LT)">People novelty persistence</span>
            <input type="number" id="ft-persistence" min="0" max="1" step="0.05"
                   onchange="commitFaceTuning('lt','people_novelty_min_persistence',this.value)" style="width:64px;">
            <span title="C5: enroll-candidate samples must span at least this many seconds (LT)">Enroll min span (s)</span>
            <input type="number" id="ft-span" min="0" max="60" step="0.5"
                   onchange="commitFaceTuning('lt','enroll_candidate_min_span_s',this.value)" style="width:64px;">
            <span title="C5: most samples must be farther than this from every known identity (LT; shares A2 release value)">Enroll min dist</span>
            <input type="number" id="ft-enroll-dist" min="0.05" max="1.5" step="0.01"
                   onchange="commitFaceTuning('lt','enroll_candidate_min_dist',this.value)" style="width:64px;">
          </div>
        </div>
      </div>
      <!-- Conversation-model dropdown removed 2026-06-18 (Dan): the Llama-3B-vs-X
           switcher is retired; Qwen3.6 (:8083) is the permanent brain, shown as
           its own service card above. -->
    </details>
  </div>
  <div>
    <details class="panel" open>
      <summary><h2>Body Behavior</h2></summary>
      <!-- Bundle A 00:51: BodyBehavior reformatted. Speaker badge + Face
           label moved to the Who's-in-Room panel per Dan's 2026-05-14
           00:51 ask. This panel now focuses on what the SKULL is doing,
           with a recent-transitions log so operators can answer the
           "why did it just flip track->scan" question at a glance.
           Cause-attribution per transition is a follow-up: needs
           streamerpi behavior.py to publish trigger reasons. -->
      <div id="behavior-panel" style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div id="behavior-mode" style="font-size:28px; font-weight:bold; color:#e94560;">--</div>
        <div style="flex:1; min-width:150px;">
          <div id="behavior-info" style="font-size:11px; margin-bottom:4px;"></div>
          <div id="behavior-stats" style="font-size:10px; color:#484f58;"></div>
        </div>
      </div>
      <div style="margin-top:10px; padding-top:8px; border-top:1px solid #21262d;">
        <div style="font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Recent transitions <span style="color:#484f58; text-transform:none; letter-spacing:0;">(server-published with cause attribution)</span></div>
        <div id="behavior-transitions" style="font-size:11px; color:#8b949e; font-family:monospace; max-height:90px; overflow-y:auto;">
          <span style="color:#484f58; font-style:italic;">no transitions observed yet</span>
        </div>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px;">
      <summary><h2>Who's in the Room</h2></summary>
      <div id="presence-panel" style="font-size:12px; min-height:32px;">
        <div id="presence-empty" style="color:#8b949e; font-style:italic;">No one detected yet</div>
      </div>
      <div id="presence-meta" style="font-size:10px; color:#484f58; margin-top:6px;"></div>
    </details>
    <details class="panel" open style="margin-top:16px;">
      <summary><h2>Feedback</h2></summary>
      <div style="display:flex; gap:6px; flex-wrap:wrap;">
        <button id="praise-btn" type="button" data-kind="good"
                title="Mark the most recent assistant turn as a good (on-persona) example"
                style="font-size:12px; padding:4px 10px; background:#1f3a25; color:#3fb950; border:1px solid #3fb950; border-radius:4px; cursor:pointer;">
          👍 Good
        </button>
        <button id="flag-btn" type="button" data-kind="bad"
                title="Flag the most recent assistant turn as a persona-drift example for later audit"
                style="font-size:12px; padding:4px 10px; background:#3a1f1f; color:#f85149; border:1px solid #f85149; border-radius:4px; cursor:pointer;">
          👎 Bad
        </button>
        <button id="reenroll-btn" type="button"
                title="Re-enroll the most recent speaker for ~60s. Optional name prompt to refresh anyone."
                style="font-size:12px; padding:4px 10px; background:#1f2a3a; color:#58a6ff; border:1px solid #58a6ff; border-radius:4px; cursor:pointer;">
          🎤 Re-enroll voice
        </button>
        <button id="show-payload-btn" type="button"
                title="Show the most recent ephemeral system prompt + full payload sent to the LLM"
                style="font-size:12px; padding:4px 10px; background:#2a1f3a; color:#bc8cff; border:1px solid #bc8cff; border-radius:4px; cursor:pointer;">
          🔍 Last payload
        </button>
        <button id="show-last-flag-btn" type="button"
                title="Re-open the most recent 👍/👎 flag: system prompt, conversation, and your comment"
                style="font-size:12px; padding:4px 10px; background:#3a2a1f; color:#f0883e; border:1px solid #f0883e; border-radius:4px; cursor:pointer;">
          🗂 Last flag
        </button>
      </div>
      <div id="flag-status" style="font-size:11px; color:#8b949e; margin-top:8px; min-height:14px;"></div>
    </details>
    <details class="panel" open style="margin-top:16px;">
      <summary><h2>Booth Display</h2></summary>
      <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
        <button id="open-booth-btn" type="button"
                title="Open the Open Sauce visitor display (booth-mockup :8090) in a new browser window. Requires booth-mockup.service to be running — toggle it in the services table if not."
                style="font-size:12px; padding:4px 10px; background:#1f3a3a; color:#39c5cf; border:1px solid #39c5cf; border-radius:4px; cursor:pointer;">
          🎪 Open booth display
        </button>
        <span id="open-booth-hint" style="font-size:11px; color:#8b949e;">opens <code style="color:#39c5cf;">https://&lt;host&gt;:8090/</code> in a new window — accept the self-signed cert the first time</span>
      </div>
      <div id="open-booth-status" style="font-size:11px; color:#8b949e; margin-top:6px; min-height:14px;"></div>
    </details>
    <details class="panel" open style="margin-top:16px; display:flex; flex-direction:column;">
      <summary><h2>Conversation</h2></summary>
      <div id="conversation" style="max-height:380px; overflow-y:auto;">
        <div id="conv-offline">Waiting for Little Timmy...</div>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px;">
      <summary><h2>Mood</h2></summary>
      <label class="mood-override-toggle">
        <input type="checkbox" id="mood-override-chk">
        <span>Manual override</span>
        <span id="mood-override-status" class="mood-override-status"></span>
      </label>
      <div class="mood-grid">
        <div class="corner"></div>
        <div class="axis-label-x">Bored</div>
        <div class="axis-label-x">Neutral</div>
        <div class="axis-label-x">Interested</div>

        <div class="axis-label-y">Nice</div>
        <div class="mood-cell" data-x="-1" data-y="1" title="Bored & Begrudgingly Nice">●</div>
        <div class="mood-cell" data-x="0"  data-y="1" title="Neutral & Begrudgingly Nice">●</div>
        <div class="mood-cell" data-x="1"  data-y="1" title="Interested & Begrudgingly Nice">●</div>

        <div class="axis-label-y">Neutral</div>
        <div class="mood-cell" data-x="-1" data-y="0" title="Bored & Neutral">●</div>
        <div class="mood-cell" data-x="0"  data-y="0" title="Neutral & Neutral">●</div>
        <div class="mood-cell" data-x="1"  data-y="0" title="Interested & Neutral">●</div>

        <div class="axis-label-y">Mean</div>
        <div class="mood-cell" data-x="-1" data-y="-1" title="Bored & Mean">●</div>
        <div class="mood-cell" data-x="0"  data-y="-1" title="Neutral & Mean">●</div>
        <div class="mood-cell" data-x="1"  data-y="-1" title="Interested & Mean">●</div>
      </div>
      <pre id="mood-rendered" class="mood-rendered">--</pre>
      <div class="mood-meta">
        <span id="mood-signals">--</span>
        <span id="mood-updated">--</span>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px;">
      <summary><h2>Retrieved Context</h2></summary>
      <div id="retrieval" style="max-height:220px; overflow-y:auto; font-size:12px;">
        <div id="ret-empty" style="color:#8b949e; font-style:italic;">No retrievals yet</div>
      </div>
    </details>
  </div>
  <div>
    <!-- Bundle A 00:37: Latency split. Current-turn metrics + last-turn
         context live in their own card now; Rolling history is a sibling
         panel below so they can be collapsed / scanned independently. -->
    <details class="panel" open>
      <summary><h2>Latency (current turn)</h2></summary>
      <div id="metrics">
        <div class="metric-row"><span class="label">STT</span><span class="value" id="m-stt">--</span></div>
        <div class="metric-row"><span class="label">Tool filter (Qwen3-4B)</span><span class="value" id="m-classifier">--</span></div>
        <div class="metric-row"><span class="label">Query resolve (coref)</span><span class="value" id="m-resolution">--</span></div>
        <div class="metric-row"><span class="label">Retrieval</span><span class="value" id="m-retrieval">--</span></div>
        <div class="metric-row"><span class="label">LLM 1st token</span><span class="value" id="m-llm-ft">--</span></div>
        <div class="metric-row"><span class="label">LLM total</span><span class="value" id="m-llm">--</span></div>
        <div class="metric-row"><span class="label">TTS</span><span class="value" id="m-tts">--</span></div>
        <div class="metric-row"><span class="label">End-to-end</span><span class="value" id="m-e2e">--</span></div>
        <div class="metric-row"><span class="label">Turns</span><span class="value" id="m-turns">--</span></div>
      </div>
      <div id="latency-context" style="margin-top:10px; padding-top:8px; border-top:1px solid #21262d; font-size:11px; line-height:1.5;">
        <div style="font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Last token payload (sent to LLM)</div>
        <div><span style="color:#484f58;">prompt</span> <span id="latency-prompt-tokens" style="color:#c9d1d9; font-weight:bold;">--</span> <span style="color:#484f58;">tokens (estimated, ~4 chars/token)</span></div>
        <div><span style="color:#484f58;">completion</span> <span id="latency-completion-tokens" style="color:#bc8cff; font-weight:bold;">--</span> <span style="color:#484f58;">tokens (estimated)</span></div>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px;">
      <summary><h2>Latency History</h2></summary>
      <div style="display:flex; justify-content:flex-end; align-items:center; margin-bottom:4px;">
        <button id="latency-clear-btn" type="button"
                title="Clear chart history"
                style="font-size:10px; padding:2px 8px; background:#21262d; color:#8b949e; border:1px solid #30363d; border-radius:3px; cursor:pointer;">
          clear
        </button>
      </div>
      <div style="position:relative; height:200px; max-width:360px;">
        <canvas id="latency-chart"></canvas>
      </div>
      <div id="latency-chart-empty" style="color:#484f58; font-size:11px; font-style:italic; text-align:center; padding:6px 0;">
        waiting for first turn…
      </div>
    </details>
    <details class="panel" open style="margin-top:16px">
      <summary><h2>Host (okdemerzel)</h2></summary>
      <div id="host-panel" style="font-size:12px;">
        <div class="metric-row"><span class="label">CPU</span><span class="value" id="host-cpu">--</span></div>
        <div class="metric-row"><span class="label">RAM</span><span class="value" id="host-ram">--</span></div>
        <div class="metric-row"><span class="label">Disk /</span><span class="value" id="host-disk">--</span></div>
        <div class="metric-row"><span class="label">Load (1m / 5m / 15m)</span><span class="value" id="host-load">--</span></div>
        <div class="metric-row"><span class="label">VRAM</span><span class="value" id="host-vram">--</span></div>
        <div class="metric-row"><span class="label">GPU busy</span><span class="value" id="host-gpu">--</span></div>
        <div style="font-size:10px; color:#484f58; margin-top:6px;">VRAM/GPU read from amdgpu sysfs (card1). Strix Halo UMA — VRAM is partitioned from system RAM at BIOS.</div>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px">
      <summary><h2>Recording</h2></summary>
      <div id="recording-panel" style="font-size:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span>
            <span id="rec-dot" style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#484f58; margin-right:6px; vertical-align:middle;"></span>
            <span id="rec-state">idle</span>
          </span>
          <button id="rec-toggle-btn" type="button" onclick="toggleRecording()"
                  style="font-size:11px; padding:5px 12px; background:#1f6feb; color:#fff; border:1px solid #1f6feb; border-radius:4px; cursor:pointer;">
            Start
          </button>
        </div>
        <div id="rec-meta" style="font-size:11px; color:#8b949e;">No active recording</div>
        <div style="font-size:10px; color:#484f58; margin-top:6px;">
          Saves to <code>~/little_timmy/recordings/&lt;ts&gt;.{webm,jsonl,json}</code>. Audio is silent (visitor WebRTC stream is video-only).
        </div>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px">
      <summary><h2>Session Log</h2></summary>
      <div id="log-viewer" style="max-height:200px; overflow-y:auto; font-size:12px; color:#8b949e;">
        <em>No events yet</em>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px">
      <summary><h2>Vision</h2></summary>
      <div id="vision-panel">
        <div style="text-align:center; margin-bottom:8px;">
          <img id="vision-img" style="max-width:100%; border-radius:4px; border:1px solid #21262d; display:none;" />
          <div id="vision-no-img" style="color:#484f58; font-size:12px; padding:20px;">No frame yet</div>
        </div>
        <div id="vision-caption" style="font-size:12px; color:#c9d1d9; margin-bottom:8px;"></div>
        <div id="vision-scores" style="font-size:11px;"></div>
        <div id="vision-stats" style="font-size:10px; color:#484f58; margin-top:6px;"></div>
        <div id="vision-relevance" style="font-size:11px; margin-top:6px; padding-top:6px; border-top:1px solid #21262d;"></div>
      </div>
    </details>

  </div>
</div>

<div id="status-bar"></div>

<div id="flag-modal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:1000; align-items:center; justify-content:center;">
  <div style="background:#0d1117; border:1px solid #f0883e; border-radius:8px; width:min(960px, 92vw); max-height:88vh; display:flex; flex-direction:column; padding:18px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <h2 style="margin:0; color:#f0883e; font-size:14px; text-transform:uppercase;">
        <span id="flag-modal-kind-chip" style="display:inline-block; padding:2px 8px; border-radius:10px; font-size:13px; margin-right:8px;"></span>
        <span id="flag-modal-title">Flag Detail</span>
      </h2>
      <button id="flag-modal-close-btn" type="button"
              style="font-size:12px; padding:4px 10px; background:#3a1f1f; color:#f85149; border:1px solid #f85149; border-radius:4px; cursor:pointer;">
        Close
      </button>
    </div>
    <div id="flag-modal-meta" style="font-size:11px; color:#8b949e; margin-bottom:8px;"></div>
    <div style="display:flex; flex-direction:column; gap:10px; overflow-y:auto;">
      <div>
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">This flag</div>
        <div id="flag-modal-comment" style="background:#161b22; border:1px solid #30363d; border-radius:4px; padding:8px; font-size:12px; color:#e0e0e0; font-style:italic;"></div>
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; margin:8px 0 4px;">Flagged assistant response</div>
        <pre id="flag-modal-response" style="background:#161b22; border:1px solid #f0883e; border-radius:4px; padding:8px; white-space:pre-wrap; word-wrap:break-word; font-size:12px; color:#f0d8a8; margin:0;"></pre>
      </div>
      <div>
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">Conversation context (hot turns at finalization)</div>
        <div id="flag-modal-conversation" style="background:#161b22; border:1px solid #30363d; border-radius:4px; padding:8px; max-height:280px; overflow-y:auto; font-size:12px;"></div>
      </div>
      <details>
        <summary style="cursor:pointer; font-size:11px; color:#8b949e; text-transform:uppercase;">System prompt (ephemeral block the LLM saw)</summary>
        <pre id="flag-modal-system" style="background:#161b22; border:1px solid #30363d; border-radius:4px; padding:8px; white-space:pre-wrap; word-wrap:break-word; font-size:11px; color:#bc8cff; margin-top:6px;"></pre>
      </details>
    </div>
  </div>
</div>

<div id="payload-modal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:1000; align-items:center; justify-content:center;">
  <div style="background:#0d1117; border:1px solid #bc8cff; border-radius:8px; width:min(960px, 92vw); max-height:88vh; display:flex; flex-direction:column; padding:18px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <h2 style="margin:0; color:#bc8cff; font-size:14px; text-transform:uppercase;">Last LLM Payload</h2>
      <button id="payload-close-btn" type="button"
              style="font-size:12px; padding:4px 10px; background:#3a1f1f; color:#f85149; border:1px solid #f85149; border-radius:4px; cursor:pointer;">
        Close
      </button>
    </div>
    <div id="payload-meta" style="font-size:11px; color:#8b949e; margin-bottom:8px;"></div>
    <!-- Bundle A 00:39: header box showing Timmy's last assistant utterance.
         Sourced from the in-page conversation state (the last role=assistant
         turn rendered into #conversation), so it reflects the reply that
         followed the captured payload. -->
    <div style="background:#161b22; border:1px solid #bc8cff; border-radius:4px; padding:8px 10px; margin-bottom:12px;">
      <div style="font-size:10px; color:#bc8cff; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Last Timmy utterance</div>
      <div id="payload-last-asst" style="font-size:13px; color:#e0e0e0; white-space:pre-wrap; word-wrap:break-word;">(no reply yet)</div>
    </div>
    <div style="display:flex; flex-direction:column; gap:10px; overflow-y:auto;">
      <div>
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">User text</div>
        <pre id="payload-user" style="background:#161b22; border:1px solid #30363d; border-radius:4px; padding:8px; white-space:pre-wrap; word-wrap:break-word; font-size:12px; color:#e0e0e0; margin:0;"></pre>
      </div>
      <div>
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">Ephemeral system prompt (assembled fresh per turn)</div>
        <pre id="payload-ephemeral" style="background:#161b22; border:1px solid #30363d; border-radius:4px; padding:8px; white-space:pre-wrap; word-wrap:break-word; font-size:12px; color:#bc8cff; margin:0;"></pre>
      </div>
      <details>
        <summary style="cursor:pointer; font-size:11px; color:#8b949e; text-transform:uppercase;">Full message list (history + system + user)</summary>
        <pre id="payload-messages" style="background:#161b22; border:1px solid #30363d; border-radius:4px; padding:8px; white-space:pre-wrap; word-wrap:break-word; font-size:11px; color:#e0e0e0; margin-top:6px;"></pre>
      </details>
    </div>
  </div>
</div>

<script>
const SERVICE_ORDER = ["postgresql", "ollama", "qwen36", "qwen36_vision", "whisper", "little_timmy", "booth_mockup"];
let serviceState = {};
let busyServices = new Set();
let modelSwitching = false;
let currentModel = "";
let ws;
let metricsInterval;
const startTime = Date.now();

function connectWS() {
  ws = new WebSocket("ws://" + location.host + "/ws");

  ws.onopen = () => {
    document.getElementById("uptime").textContent = "Connected";
  };

  ws.onclose = () => {
    document.getElementById("uptime").textContent = "Disconnected - reconnecting...";
    setTimeout(connectWS, 3000);
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === "health") {
      serviceState = msg.services;
      renderServices();
    } else if (msg.type === "models") {
      renderModelSelector(msg.models, msg.current);
    } else if (msg.type === "status") {
      addStatusEntry(msg);
      // Clear busy states when we get terminal status messages
      busyServices.forEach(sid => {
        if (msg.message.includes("is connected") || msg.message.includes("already connected") ||
            msg.message.includes("not become healthy") || msg.message.includes("failed") ||
            msg.message.includes("No process found") || msg.message.includes("Stopped") ||
            msg.message.includes("already loaded")) {
          busyServices.delete(sid);
        }
      });
      // Clear model switching on terminal messages
      if (modelSwitching && (msg.message.includes("switched to") || msg.message.includes("Failed to start") ||
          msg.message.includes("already loaded"))) {
        modelSwitching = false;
        const sel = document.getElementById("model-select");
        if (sel) sel.disabled = false;
      }
      renderServices();
    } else if (msg.type === "face_tracking") {
      faceTrackingEnabled = msg.enabled || false;
      faceTrackingBusy = false;
      updateFaceTrackingUI();
    } else if (msg.type === "streamerpi_main") {
      streamerpiMainState = {
        running: !!msg.running,
        reachable: msg.reachable !== false,
        error: msg.error || null,
      };
      streamerpiMainBusy = false;
      updateStreamerpiMainUI();
    } else if (msg.type === "lt_toggles") {
      ltFlagBusy.vision_auto_poll = false;
      ltFlagBusy.hearing = false;
      ltFlagBusy.proactive_speech = false;
      ltFlagBusy.classifier = false;
      ltFlagBusy.query_resolution = false;
      applyLTToggles(msg);
    } else if (msg.type === "turn") {
      addTurn(msg.role, msg.content, msg.speaker);
    } else if (msg.type === "metrics") {
      updateMetricsFromWS(msg);
    } else if (msg.type === "retrieval") {
      renderRetrieval(msg);
    } else if (msg.type === "tool_call") {
      flashToolCall(msg);
    } else if (msg.type === "classifier_metric") {
      const el = document.getElementById("m-classifier");
      if (el && typeof msg.ms === "number") el.textContent = msg.ms + "ms";
    } else if (msg.type === "resolution_metric") {
      const el = document.getElementById("m-resolution");
      if (el && typeof msg.ms === "number") el.textContent = msg.ms + "ms" + (msg.resolved ? "" : " (no-op)");
    }
  };
}

function renderModelSelector(models, current) {
  // Dropdown retired 2026-06-18 (Dan). Element removed; no-op if absent.
  currentModel = current;
  const sel = document.getElementById("model-select");
  if (!sel) return;
  sel.innerHTML = "";
  for (const [id, name] of Object.entries(models)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = name;
    if (id === current) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.disabled = modelSwitching;
  const st = document.getElementById("model-status");
  if (st) st.textContent = modelSwitching ? "Switching..." : "";
}

function switchModel(modelId) {
  const sel = document.getElementById("model-select");
  if (!sel) return;  // dropdown retired
  if (!modelId || modelId === currentModel || modelSwitching) return;
  modelSwitching = true;
  sel.disabled = true;
  document.getElementById("model-status").textContent = "Switching...";
  ws.send(JSON.stringify({type: "switch_model", model: modelId}));
}

function addTurn(role, content, speaker) {
  const conv = document.getElementById("conversation");
  const offline = document.getElementById("conv-offline");
  if (offline) offline.remove();

  const div = document.createElement("div");
  div.className = "turn " + role;
  const label = role === "user" ? (speaker ? speaker.toUpperCase() : "USER") : "TIMMY";
  // Color-code the role label so the operator can see at a glance whether
  // a user turn was attributed to a known speaker (green) vs a transient
  // unknown_N cluster (amber). Supervisor H9 — operator was previously
  // unable to tell live which utterance was tagged Dan vs unknown_N.
  let speakerClass = "speaker-assistant";
  if (role === "user") {
    const s = (speaker || "").toLowerCase();
    speakerClass = (!s || s === "unknown" || s.startsWith("unknown_"))
      ? "speaker-unknown" : "speaker-known";
    // H9: also push the speaker to the live badge in Body Behavior so the
    // operator has a glance-distance read on Dan-vs-unknown_N without
    // scanning the conversation log.
    const badge = document.getElementById("speaker-badge");
    if (badge) {
      badge.textContent = label;
      badge.classList.remove("speaker-known", "speaker-unknown");
      badge.classList.add(speakerClass);
    }
  }
  // Bundle A 00:43: assistant content goes through normalizeForDebug on
  // operator surfaces so trolling-ellipses don't clutter the debug view.
  // User text stays raw so we see exactly what was heard.
  const renderedContent = (role === "assistant")
    ? normalizeForDebug(content)
    : content;
  div.innerHTML =
    '<div class="role ' + speakerClass + '">' + label + '</div>' +
    '<div class="content">' + escapeHtml(renderedContent) + '</div>';
  conv.appendChild(div);
  conv.scrollTop = conv.scrollHeight;

  // Bundle A 00:39: keep the payload-modal header in sync with the most
  // recent assistant utterance. The latency-context strip is no longer
  // text-based (00:37 reframe to token counts via updateMetricsFromWS).
  if (role === "assistant") {
    _setLastAssistantTurn(renderedContent);
  }
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

// Bundle A 00:43: normalize trailing/embedded ellipses out of ASSISTANT text
// for operator-debug surfaces only (conversation panel, payload modal,
// last-turn-context strip). The model still emits whatever it likes to the
// user-facing visitor.html + TTS path; this only collapses the rendering on
// dashboards Dan reads for debugging. Distinct from suppressing the persona
// behavior (which Dan said "was good, it was good" — funny + on-character).
function normalizeForDebug(text) {
  if (text == null) return "";
  // Collapse Unicode ellipsis, ASCII triplet, and spaced variants into a
  // single space (so words on either side don't fuse), then trim trailing.
  return String(text)
    .replace(/(\u2026|\.{3,}|\.\s\.\s\.)/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}

// Bundle A 00:39 prerequisite: track the most recent assistant utterance so
// the payload-modal opener can surface it in the header box.
let _lastAssistantTurn = "";
function _setLastAssistantTurn(text) { _lastAssistantTurn = text || ""; }

function updateMetricsFromWS(msg) {
  document.getElementById("m-stt").textContent = msg.stt_ms != null ? msg.stt_ms + "ms" : "--";
  document.getElementById("m-retrieval").textContent = msg.retrieval_ms != null ? msg.retrieval_ms + "ms" : "--";
  document.getElementById("m-llm-ft").textContent = msg.llm_first_token_ms != null ? msg.llm_first_token_ms + "ms" : "--";
  document.getElementById("m-llm").textContent = msg.llm_total_ms != null ? msg.llm_total_ms + "ms" : "--";
  document.getElementById("m-tts").textContent = msg.tts_ms != null ? msg.tts_ms + "ms" : "--";
  document.getElementById("m-e2e").textContent = msg.e2e_ms != null ? msg.e2e_ms + "ms" : "--";
  document.getElementById("m-turns").textContent = msg.turns || "--";
  // Bundle A 00:37 reframe: prompt/completion token estimates from main.py.
  const pTok = document.getElementById("latency-prompt-tokens");
  const cTok = document.getElementById("latency-completion-tokens");
  if (pTok) pTok.textContent = msg.est_prompt_tokens != null ? msg.est_prompt_tokens.toLocaleString() : "--";
  if (cTok) cTok.textContent = msg.est_completion_tokens != null ? msg.est_completion_tokens.toLocaleString() : "--";
  pushLatencySample({
    stt: msg.stt_ms, retrieval: msg.retrieval_ms,
    llm_ft: msg.llm_first_token_ms, llm: msg.llm_total_ms,
    tts: msg.tts_ms, e2e: msg.e2e_ms,
  });
}

// ---- Latency rolling chart ----
// Each metrics WS event (or poll) appends one sample; chart shows last
// LATENCY_BUFFER_MAX. Lines are toggleable via Chart.js legend clicks.
const LATENCY_BUFFER_MAX = 60;
const LATENCY_SERIES = [
  {key: "e2e",       label: "End-to-end", color: "#e94560"},
  {key: "llm",       label: "LLM total",  color: "#bc8cff"},
  {key: "tts",       label: "TTS",        color: "#3fb950"},
  {key: "stt",       label: "STT",        color: "#58a6ff"},
  {key: "retrieval", label: "Retrieval",  color: "#d29922"},
  {key: "llm_ft",    label: "LLM 1st tok", color: "#79c0ff"},
];
let latencyBuffer = [];  // [{ts, stt, retrieval, llm_ft, llm, tts, e2e}, ...]
let latencyChart = null;
let lastLatencyTs = 0;

function initLatencyChart() {
  const canvas = document.getElementById("latency-chart");
  if (!canvas || typeof Chart === "undefined") return;
  latencyChart = new Chart(canvas, {
    type: "line",
    data: {
      labels: [],
      datasets: LATENCY_SERIES.map(s => ({
        label: s.label, data: [], borderColor: s.color,
        backgroundColor: s.color + "22", borderWidth: 1.5,
        pointRadius: 2, tension: 0.25, spanGaps: true,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#c9d1d9", font: { size: 10 }, boxWidth: 10 } },
        tooltip: { titleColor: "#c9d1d9", bodyColor: "#c9d1d9",
                   backgroundColor: "#161b22", borderColor: "#30363d", borderWidth: 1,
                   callbacks: {
                     // Hover title shows wall-clock time for the sample,
                     // since the X axis itself is now turn-indexed (#N).
                     title: (items) => {
                       const i = items && items[0] ? items[0].dataIndex : -1;
                       const b = (i >= 0 && i < latencyBuffer.length) ? latencyBuffer[i] : null;
                       const turnLabel = '#' + (i + 1);
                       return b ? turnLabel + ' · ' + new Date(b.ts).toLocaleTimeString([], {hour12: false}) : turnLabel;
                     },
                   },
        },
      },
      scales: {
        x: { ticks: { color: "#484f58", font: { size: 9 }, maxRotation: 0,
                       autoSkipPadding: 12 },
              grid: { color: "#21262d" } },
        y: { beginAtZero: true, ticks: { color: "#484f58", font: { size: 9 },
              callback: v => v + "ms" },
              grid: { color: "#21262d" } },
      },
    },
  });
}

function pushLatencySample(s) {
  // Skip duplicate samples (same metrics object delivered via both WS + poll
  // within a few seconds). Dedupe key = nonzero metric tuple.
  const fingerprint = LATENCY_SERIES.map(x => s[x.key] || 0).join("|");
  const now = Date.now();
  if (latencyBuffer.length && latencyBuffer[latencyBuffer.length - 1].fp === fingerprint
      && now - lastLatencyTs < 2000) return;
  if (fingerprint === "0|0|0|0|0|0") return;  // nothing to plot
  lastLatencyTs = now;
  // Use ?? not || so a genuine 0ms metric still plots as 0 instead of a gap.
  latencyBuffer.push({
    ts: now, fp: fingerprint,
    stt: s.stt ?? null, retrieval: s.retrieval ?? null,
    llm_ft: s.llm_ft ?? null, llm: s.llm ?? null,
    tts: s.tts ?? null, e2e: s.e2e ?? null,
  });
  while (latencyBuffer.length > LATENCY_BUFFER_MAX) latencyBuffer.shift();
  redrawLatencyChart();
}

function redrawLatencyChart() {
  if (!latencyChart) return;
  const empty = document.getElementById("latency-chart-empty");
  if (empty && latencyBuffer.length) empty.style.display = "none";
  // Turn-indexed X axis: one notch per sample (response), evenly spaced.
  // Was wall-clock-time-derived, which left dead gaps during idle stretches
  // and made it hard to read "last N responses" trends. Wall-clock time is
  // still available on hover via the tooltip title callback below.
  const labels = latencyBuffer.map((_, i) => '#' + (i + 1));
  latencyChart.data.labels = labels;
  LATENCY_SERIES.forEach((s, i) => {
    latencyChart.data.datasets[i].data = latencyBuffer.map(b => b[s.key]);
  });
  latencyChart.update("none");
}

function clearLatencyChart() {
  latencyBuffer = []; lastLatencyTs = 0;
  const empty = document.getElementById("latency-chart-empty");
  if (empty) empty.style.display = "";
  redrawLatencyChart();
}

function renderRetrieval(msg) {
  const el = document.getElementById("retrieval");
  const empty = document.getElementById("ret-empty");
  if (empty) empty.remove();
  el.innerHTML = "";

  if (msg.memories && msg.memories.length > 0) {
    msg.memories.forEach(m => {
      const div = document.createElement("div");
      div.className = "ret-memory";
      div.innerHTML = '<span class="ret-type">' + m.type + '</span>' +
        '<span class="ret-score">RRF ' + m.score + '</span>' +
        '<div style="margin-top:3px">' + escapeHtml(m.content) + '</div>';
      el.appendChild(div);
    });
  }

  if (msg.facts && msg.facts.length > 0) {
    msg.facts.forEach(f => {
      const div = document.createElement("div");
      div.className = "ret-fact";
      div.textContent = f.subject + "." + f.predicate + " = " + f.value;
      el.appendChild(div);
    });
  }

  if ((!msg.memories || msg.memories.length === 0) && (!msg.facts || msg.facts.length === 0)) {
    el.innerHTML = '<div style="color:#8b949e; font-style:italic;">No context retrieved</div>';
  }
}

function renderServices() {
  const container = document.getElementById("services");
  container.innerHTML = SERVICE_ORDER.map(sid => {
    const svc = serviceState[sid];
    if (!svc) return "";
    const connected = svc.status === "connected";
    const busy = busyServices.has(sid);
    const port = svc.port || getPort(sid);
    return '<div class="service-card ' + (connected ? 'connected' : 'disconnected') + '">' +
      '<div class="service-info">' +
        '<div class="service-name">' + svc.name + '</div>' +
        '<div class="service-detail">' + (svc.detail || svc.status) + '</div>' +
      '</div>' +
      '<span class="service-port">:' + port + '</span>' +
      '<label class="toggle ' + (busy ? 'busy' : '') + '">' +
        '<input type="checkbox" ' + (connected ? 'checked' : '') + ' ' + (busy ? 'disabled' : '') +
          ' onchange="toggleService(\'' + sid + '\', this.checked)">' +
        '<span class="slider"></span>' +
      '</label>' +
    '</div>';
  }).join("");
}

function getPort(sid) {
  const ports = {postgresql:5432, ollama:11434, gptoss120b:8080, qwen36:8083, whisper:8891, little_timmy:8893};
  return ports[sid] || "?";
}

function toggleService(sid, state) {
  busyServices.add(sid);
  renderServices();
  ws.send(JSON.stringify({type: "toggle", service: sid, state: state}));
}

function addStatusEntry(msg) {
  const ts = msg.timestamp ? msg.timestamp.split("T")[1].split(".")[0] : "";
  const level = msg.level || "info";
  const text = msg.message || "";

  // Live ticker at the bottom of the page (existing surface).
  const bar = document.getElementById("status-bar");
  if (bar) {
    const div = document.createElement("div");
    div.className = "status-entry " + level;
    div.innerHTML = '<span class="ts">' + ts + "</span>" + text;
    bar.prepend(div);
    while (bar.children.length > 50) bar.removeChild(bar.lastChild);
  }

  // Dedicated Session Log panel — previously never populated despite the
  // WS broadcast pipeline firing for every service action. The panel sat
  // showing "No events yet" all session. Now mirrors the status-bar
  // entries with a higher cap so the user can scroll session history.
  const logViewer = document.getElementById("log-viewer");
  if (logViewer) {
    // Clear the "No events yet" placeholder on first real entry.
    const placeholder = logViewer.querySelector("em");
    if (placeholder) logViewer.innerHTML = "";
    const row = document.createElement("div");
    row.className = "status-entry " + level;
    row.innerHTML = '<span class="ts">' + ts + "</span>" + text;
    logViewer.prepend(row);
    // Cap history at 500 entries so memory stays bounded across long demos.
    while (logViewer.children.length > 500) logViewer.removeChild(logViewer.lastChild);
  }
}

async function loadSessionLog() {
  // Backfill the Session Log panel on page (re)load. /api/log returns
  // services.get_session_log() — the full in-memory _session_log list.
  // Without this, every refresh would empty the panel until the next WS
  // status event arrives.
  try {
    const r = await fetch("/api/log");
    const entries = await r.json();
    if (!Array.isArray(entries) || entries.length === 0) return;
    entries.forEach(e => addStatusEntry(e));
  } catch(e) {}
}

async function pollMetrics() {
  try {
    const r = await fetch("/api/timmy/metrics");
    const m = await r.json();
    if (!m.error) {
      document.getElementById("m-stt").textContent = m.last_stt_ms != null ? m.last_stt_ms + "ms" : "--";
      document.getElementById("m-classifier").textContent = m.last_classifier_ms != null ? m.last_classifier_ms + "ms" : (m.classifier_enabled ? "--" : "off");
      document.getElementById("m-resolution").textContent = m.last_resolution_ms != null ? m.last_resolution_ms + "ms" : (m.query_resolution_enabled ? "--" : "off");
      document.getElementById("m-retrieval").textContent = m.last_retrieval_ms != null ? m.last_retrieval_ms + "ms" : "--";
      document.getElementById("m-llm-ft").textContent = m.last_llm_first_token_ms != null ? m.last_llm_first_token_ms + "ms" : "--";
      document.getElementById("m-llm").textContent = m.last_llm_total_ms != null ? m.last_llm_total_ms + "ms" : "--";
      document.getElementById("m-tts").textContent = m.last_tts_ms != null ? m.last_tts_ms + "ms" : "--";
      document.getElementById("m-e2e").textContent = m.last_e2e_ms != null ? m.last_e2e_ms + "ms" : "--";
      document.getElementById("m-turns").textContent = m.turns || "--";
      // NOTE: do NOT pushLatencySample from this 30s poller. The chart is
      // turn-indexed and the canonical sample source is the WS metrics event
      // (updateMetricsFromWS), which fires exactly once per real turn. Pushing
      // from the poll loop replayed the same last-turn metrics every 30s,
      // producing phantom samples during quiet stretches (H7 fix 2026-05-13).
      // Text labels above still update from poll — that part is useful.
    }
  } catch(e) {}
}

async function loadConversationHistory() {
  try {
    const r = await fetch("/api/timmy/conversation");
    const data = await r.json();
    if (data.error) return;
    const conv = document.getElementById("conversation");
    const offline = document.getElementById("conv-offline");
    if (offline) offline.remove();
    if (data.hot) {
      data.hot.forEach(t => addTurn(t.role, t.content));
    }
  } catch(e) {}
}

// Update uptime display
setInterval(() => {
  const secs = Math.floor((Date.now() - startTime) / 1000);
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  document.getElementById("uptime").textContent =
    "Up " + (h > 0 ? h + "h " : "") + m + "m " + s + "s";
}, 1000);

// Vision panel polling
async function pollVision() {
  try {
    const r = await fetch('/api/timmy/vision');
    const v = await r.json();
    const img = document.getElementById('vision-img');
    const noImg = document.getElementById('vision-no-img');
    const caption = document.getElementById('vision-caption');
    const scores = document.getElementById('vision-scores');
    const stats = document.getElementById('vision-stats');

    if (!v.enabled) {
      caption.textContent = 'Vision pipeline disabled';
      return;
    }

    if (v.frame_b64) {
      img.src = 'data:image/jpeg;base64,' + v.frame_b64;
      img.style.display = 'block';
      noImg.style.display = 'none';
    }

    if (v.record) {
      const r = v.record;
      let parts = [];
      if (r.people && r.people.length) parts.push('<b>People:</b> ' + r.people.join(', '));
      if (r.actions && r.actions.length) parts.push('<b>Activity:</b> ' + r.actions.join(', '));
      if (r.objects && r.objects.length) parts.push('<b>Objects:</b> ' + r.objects.slice(0,5).join(', '));
      if (r.scene_state) parts.push('<b>Scene:</b> ' + r.scene_state);
      if (r.change_from_prior && r.change_from_prior.toLowerCase() !== 'none')
        parts.push('<b>Changed:</b> ' + r.change_from_prior);
      caption.innerHTML = parts.join('<br>');

      // 2026-05-07: removed Humor / Store / Tags rows alongside the VLM
      // schema fields they rendered (limited utility; +600 ms VLM cycle).
      const scoreItems = [
        '<span style="color:' + (r.novelty > 0.5 ? '#d29922' : '#484f58') + '">Novelty: ' + r.novelty.toFixed(2) + '</span>',
        '<span style="color:' + (r.speak_now ? '#f85149' : '#484f58') + '">Speak: ' + (r.speak_now ? 'YES' : 'no') + '</span>',
      ];
      scores.innerHTML = scoreItems.join(' &middot; ');
    }

    // Frame age + which capture path produced it. This is the exact frame the
    // VLM last consumed, so the thumbnail above can be trusted as ground-truth.
    const age = v.age_s != null ? v.age_s.toFixed(0) + 's ago' : '--';
    const src = v.frame_source ? ' (' + v.frame_source + ')' : '';
    const chg = v.change_score != null ? v.change_score.toFixed(1) : '--';
    const ratio = v.stats.polled > 0 ? ((v.stats.analyzed / v.stats.polled) * 100).toFixed(0) : '0';
    stats.textContent = 'Frame: ' + age + src + ' | Change: ' + chg + ' | VLM: ' + v.stats.analyzed + '/' + v.stats.polled + ' (' + ratio + '%)';

    // Relevance classifier scores
    const relDiv = document.getElementById('vision-relevance');
    if (v.relevance) {
      const rl = v.relevance;
      const injColor = rl.should_inject ? '#3fb950' : '#f85149';
      const lvlColor = rl.detail_level === 'full' ? '#3fb950' : rl.detail_level === 'minimal' ? '#d29922' : '#484f58';
      relDiv.innerHTML =
        '<span style="color:' + injColor + '">Inject: ' + (rl.should_inject ? 'YES' : 'no') + '</span>' +
        ' &middot; <span style="color:' + lvlColor + '">Detail: ' + rl.detail_level + '</span>' +
        ' &middot; <span style="color:#58a6ff">Score: ' + rl.overall.toFixed(2) + '</span>' +
        '<br><span style="color:#484f58">nov=' + rl.novelty_score.toFixed(2) +
        ' pers=' + rl.persistence_score.toFixed(2) +
        ' urg=' + rl.urgency_score.toFixed(2) + '</span>';
      if (rl.filtered_summary) {
        relDiv.innerHTML += '<br><span style="color:#8b949e; font-style:italic">' + rl.filtered_summary + '</span>';
      }
    } else {
      relDiv.innerHTML = '<span style="color:#484f58">No relevance data</span>';
    }
  } catch(e) {}
}

pollVision();
setInterval(pollVision, 5000);

// Behavior panel polling
async function pollBehavior() {
  try {
    const r = await fetch('/api/behavior');
    const b = await r.json();
    const modeEl = document.getElementById('behavior-mode');
    const infoEl = document.getElementById('behavior-info');
    const statsEl = document.getElementById('behavior-stats');

    if (b.error) {
      modeEl.textContent = 'OFFLINE';
      modeEl.style.color = '#484f58';
      infoEl.textContent = b.error;
      return;
    }

    const modeColors = {
      idle: '#e94560', scan: '#d29922', track: '#3fb950',
      engage: '#3fb950', look_around: '#8b5cf6', hold: '#484f58', sleep: '#484f58'
    };
    modeEl.textContent = b.mode.toUpperCase();
    modeEl.style.color = modeColors[b.mode] || '#c9d1d9';

    const elapsed = (b.elapsed_ms / 1000).toFixed(0);
    const timeout = b.timeout_ms > 0 ? ' / timeout ' + (b.timeout_ms / 1000).toFixed(0) + 's' : '';
    // Bundle A 00:51: face label removed -- now lives in Who's-in-Room.
    const prev = b.previous_mode ? ' | Prev: ' + b.previous_mode : '';
    infoEl.innerHTML =
      '<span style="color:#8b949e">Elapsed: ' + elapsed + 's' + timeout + '</span>' +
      '<span style="color:#484f58">' + prev + '</span>';

    if (b.stats) {
      statsEl.textContent = 'Transitions: ' + b.stats.transitions +
        ' | Faces found: ' + b.stats.faces_found +
        ' | Faces lost: ' + b.stats.faces_lost;
    }

    // Bundle A 00:51 follow-up: render server-side recent_transitions
    // from streamerpi (one entry per actual flip, with WHY tag). Replaces
    // the earlier client-side poll-based observer; that one had no cause
    // attribution and could miss flips that landed between poll ticks.
    if (Array.isArray(b.recent_transitions)) {
      const trEl = document.getElementById("behavior-transitions");
      if (trEl) {
        if (b.recent_transitions.length === 0) {
          trEl.innerHTML = '<span style="color:#484f58; font-style:italic;">no transitions observed yet</span>';
        } else {
          // Server returns oldest-first; reverse for newest-on-top display.
          const ordered = b.recent_transitions.slice().reverse();
          trEl.innerHTML = ordered.map(t => {
            const d = new Date(t.ts * 1000);
            const hh = String(d.getHours()).padStart(2,"0");
            const mm = String(d.getMinutes()).padStart(2,"0");
            const ss = String(d.getSeconds()).padStart(2,"0");
            const reason = (t.reason && t.reason.trim()) ? t.reason : "(no reason)";
            return '<div>' +
              '<span style="color:#484f58;">' + hh + ':' + mm + ':' + ss + '</span> ' +
              '<span>' + String(t.from || '?').toUpperCase() + '</span>' +
              ' <span style="color:#bc8cff;">&rarr;</span> ' +
              '<span style="color:#3fb950;">' + String(t.to || '?').toUpperCase() + '</span>' +
              ' <span style="color:#8b949e;">(why: ' + reason + ')</span>' +
            '</div>';
          }).join("");
        }
      }
    }
  } catch(e) {}
}

pollBehavior();
setInterval(pollBehavior, 2000);


// Face pipeline (three independent layers: detection / tracking / motors).
// Each has its own card with its own toggle. State arrives via /api/face_pipeline
// poll, mutated via /api/{detection,tracking,motors}/toggle.
const FACE_PIPELINE_LAYERS = {
  detection: { detail: 'YuNet+SFace inference inside the thread' },
  tracking:  { detail: 'face → servo target binding' },
  motors:    { detail: 'pan/tilt actuation (audio/jaw unaffected)' },
};
const facePipelineState = { detection: null, tracking: null, motors: null };
// Detection-layer wedge signal: streamerpi /face_pipeline/status now
// returns detection_alive (data-age honest) + data_age_s. When detection
// is enabled but data is stale, the card flips to amber + WEDGED label
// so the operator doesn't trust a frozen pipeline (2026-05-12 incident:
// 2 h stale while the old detection_alive lied True).
let detectionAlive = null;
let detectionDataAgeS = null;
const facePipelineBusy  = { detection: false, tracking: false, motors: false };

async function pollFaceTracking() {  // kept the function name so existing setInterval still wires up
  try {
    const r = await fetch('/api/face_pipeline');
    const data = await r.json();
    facePipelineState.detection = !!data.detection_enabled;
    facePipelineState.tracking  = !!data.tracking_enabled;
    facePipelineState.motors    = !!data.motors_enabled;
    detectionAlive     = (typeof data.detection_alive === 'boolean') ? data.detection_alive : null;
    detectionDataAgeS  = (typeof data.data_age_s === 'number') ? data.data_age_s : null;
    for (const layer of Object.keys(FACE_PIPELINE_LAYERS)) updateFacePipelineUI(layer);
  } catch(e) {}
}

function updateFacePipelineUI(layer) {
  const enabled = facePipelineState[layer];
  const busy    = facePipelineBusy[layer];
  const card    = document.getElementById(layer + '-card');
  const toggle  = document.querySelector('#' + layer + '-toggle input');
  const tlabel  = document.getElementById(layer + '-toggle');
  const detail  = document.getElementById(layer + '-detail');
  // Wedge state only applies to the detection layer (the only one whose
  // health we can verify by data-age on streamerpi). enabled+alive=false
  // means the YuNet thread is up but the inference loop has stopped
  // publishing -- exactly the failure mode that hid for 2 h on 2026-05-12.
  const wedged = (layer === 'detection') && enabled && (detectionAlive === false);
  if (card)   card.style.borderLeftColor = wedged ? '#d29922' : (enabled ? '#3fb950' : '#484f58');
  if (toggle) { toggle.checked = !!enabled; toggle.disabled = busy; }
  if (tlabel) tlabel.classList.toggle('busy', busy);
  if (detail) {
    if (busy)                  detail.textContent = 'Toggling...';
    else if (enabled === null) detail.textContent = 'Checking...';
    else if (wedged) {
      const ageStr = (typeof detectionDataAgeS === 'number')
        ? `${detectionDataAgeS.toFixed(1)}s stale`
        : 'no data';
      detail.textContent = `WEDGED -- ${ageStr} (motor service may need restart)`;
    }
    else if (enabled)          detail.textContent = FACE_PIPELINE_LAYERS[layer].detail;
    else                       detail.textContent = 'Disabled';
  }
}

async function toggleFacePipeline(layer, enabled) {
  if (!FACE_PIPELINE_LAYERS[layer]) return;
  facePipelineBusy[layer] = true;
  updateFacePipelineUI(layer);
  try {
    const r = await fetch('/api/' + layer + '/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !!enabled }),
    });
    const data = await r.json();
    if (typeof data.enabled === 'boolean') facePipelineState[layer] = data.enabled;
  } catch(e) { /* fall back to next poll */ }
  facePipelineBusy[layer] = false;
  updateFacePipelineUI(layer);
}

// LT-side runtime flags (vision auto-poll + hearing). State arrives via
// /api/timmy/toggles poll AND lt_toggles WS broadcasts.
const LT_FLAGS = {
  vision_auto_poll: {
    cardId: 'vision-auto-poll-card',
    toggleId: 'vision-auto-poll-toggle',
    detailId: 'vision-auto-poll-detail',
    route: '/api/vision/auto_poll/toggle',
    enabledDetail: '1fps poll loop active (~3-4 VLM calls/min)',
    disabledDetail: 'Polling paused (event-driven trigger still fires)',
  },
  hearing: {
    cardId: 'hearing-card',
    toggleId: 'hearing-toggle',
    detailId: 'hearing-detail',
    route: '/api/hearing/toggle',
    enabledDetail: 'Mic frames feeding whisper.cpp',
    disabledDetail: 'Muted (whisper-server still running)',
  },
  proactive_speech: {
    cardId: 'proactive-speech-card',
    toggleId: 'proactive-speech-toggle',
    detailId: 'proactive-speech-detail',
    route: '/api/proactive/toggle',
    enabledDetail: 'Reacts to visual events (cooldown 120s)',
    disabledDetail: 'Silent unless spoken to',
  },
  classifier: {
    cardId: 'classifier-card',
    toggleId: 'classifier-toggle',
    detailId: 'classifier-detail',
    route: '/api/classifier/toggle',
    enabledDetail: 'Routing utterances (:8092 up)',
    disabledDetail: 'Off — all utterances go straight to the brain',
  },
  query_resolution: {
    cardId: 'query_resolution-card',
    toggleId: 'query_resolution-toggle',
    detailId: 'query_resolution-detail',
    route: '/api/query_resolution/toggle',
    enabledDetail: 'Resolving deictic follow-ups before retrieval (:8092 up)',
    disabledDetail: 'Off — semantic query uses the context blend',
  },
};
const ltFlagState = { vision_auto_poll: null, hearing: null, proactive_speech: null, classifier: null, query_resolution: null };
const ltFlagBusy  = { vision_auto_poll: false, hearing: false, proactive_speech: false, classifier: false, query_resolution: false };
let ltProactiveMaster = true;  // config kill-switch; false => toggle is inert
let ltClassifierUp = false;    // :8092 reachable? surfaced in the classifier detail line
let ltQueryResolutionUp = false;  // same :8092 server; surfaced in the query-resolution detail line

function applyLTToggles(data) {
  if (typeof data.vision_auto_poll_enabled === 'boolean') ltFlagState.vision_auto_poll = data.vision_auto_poll_enabled;
  if (typeof data.hearing_enabled === 'boolean') ltFlagState.hearing = data.hearing_enabled;
  if (typeof data.proactive_speech_enabled === 'boolean') ltFlagState.proactive_speech = data.proactive_speech_enabled;
  if (typeof data.proactive_speech_master === 'boolean') ltProactiveMaster = data.proactive_speech_master;
  if (typeof data.classifier_enabled === 'boolean') ltFlagState.classifier = data.classifier_enabled;
  if (typeof data.classifier_up === 'boolean') ltClassifierUp = data.classifier_up;
  if (typeof data.query_resolution_enabled === 'boolean') ltFlagState.query_resolution = data.query_resolution_enabled;
  if (typeof data.query_resolution_up === 'boolean') ltQueryResolutionUp = data.query_resolution_up;
  for (const flag of Object.keys(LT_FLAGS)) updateLTFlagUI(flag);
}

async function pollLTToggles() {
  try {
    const r = await fetch('/api/timmy/toggles');
    const data = await r.json();
    applyLTToggles(data);
  } catch(e) {}
}

let _toolFlashTimer = null;
function flashToolCall(msg) {
  const el = document.getElementById('tool-call-flash');
  if (!el) return;
  const name = msg.name || 'tool';
  let detail = '';
  if (name === 'store_fact' && msg.subject) {
    detail = ' — ' + msg.subject + '.' + (msg.predicate || '') + ' = ' + (msg.value || '');
  }
  el.textContent = '🛠 tool call: ' + name + detail;
  el.style.display = 'block';
  if (_toolFlashTimer) clearTimeout(_toolFlashTimer);
  _toolFlashTimer = setTimeout(() => { el.style.display = 'none'; }, 4000);
}

function updateLTFlagUI(flag) {
  const def = LT_FLAGS[flag];
  if (!def) return;
  const enabled = ltFlagState[flag];
  const busy    = ltFlagBusy[flag];
  const card    = document.getElementById(def.cardId);
  const toggle  = document.querySelector('#' + def.toggleId + ' input');
  const tlabel  = document.getElementById(def.toggleId);
  const detail  = document.getElementById(def.detailId);
  // Proactive speech is inert when the config kill-switch (master) is off,
  // even if the runtime toggle reads on -- surface that so the switch isn't
  // misleading.
  const masterOff = (flag === 'proactive_speech' && !ltProactiveMaster);
  // Classifier enabled but its :8092 server is unreachable: surface as amber —
  // LT still works (degrades to normal pipeline), but the tool path is inert.
  const clsDown = (flag === 'classifier' && enabled && !ltClassifierUp);
  // Query resolution enabled but :8092 unreachable: amber — retrieval still
  // works (degrades to the context blend), but resolution is inert.
  const qrDown = (flag === 'query_resolution' && enabled && !ltQueryResolutionUp);
  if (card)   card.style.borderLeftColor = (masterOff || clsDown || qrDown) ? '#d29922' : (enabled ? '#3fb950' : '#484f58');
  if (toggle) { toggle.checked = !!enabled; toggle.disabled = busy; }
  if (tlabel) tlabel.classList.toggle('busy', busy);
  if (detail) {
    if (busy)                  detail.textContent = 'Toggling...';
    else if (enabled === null) detail.textContent = 'Checking...';
    else if (masterOff)        detail.textContent = 'Disabled by config (TIMMY_PROACTIVE_SPEECH_ENABLED=false)';
    else if (clsDown)          detail.textContent = 'ON, but :8092 unreachable — utterances fall through to the brain';
    else if (qrDown)           detail.textContent = 'ON, but :8092 unreachable — falls back to the context blend';
    else if (enabled)          detail.textContent = def.enabledDetail;
    else                       detail.textContent = def.disabledDetail;
  }
}

async function toggleLTFlag(flag, enabled) {
  if (!LT_FLAGS[flag]) return;
  ltFlagBusy[flag] = true;
  updateLTFlagUI(flag);
  try {
    const route = LT_FLAGS[flag].route;
    const r = await fetch(route, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !!enabled }),
    });
    const data = await r.json();
    if (typeof data.enabled === 'boolean') ltFlagState[flag] = data.enabled;
  } catch(e) { /* fall back to next poll */ }
  ltFlagBusy[flag] = false;
  updateLTFlagUI(flag);
}

// Streamerpi main codebase (little-timmy-motor.service on the Pi).
// State is two independent signals so the UI can distinguish
// "host is down" from "host is up but service is stopped".
let streamerpiMainState = { running: false, reachable: false, error: null };
let streamerpiMainBusy = false;

async function pollStreamerpiMain() {
  try {
    const r = await fetch('/api/streamerpi/main');
    const data = await r.json();
    streamerpiMainState = {
      running: !!data.running,
      reachable: data.reachable !== false,
      error: data.error || null,
    };
    updateStreamerpiMainUI();
  } catch(e) {}
}

function updateStreamerpiMainUI() {
  const card = document.getElementById('streamerpi-main-card');
  const toggle = document.querySelector('#streamerpi-main-toggle input');
  const detail = document.getElementById('streamerpi-main-detail');
  const toggleLabel = document.getElementById('streamerpi-main-toggle');
  const { running, reachable, error } = streamerpiMainState;

  // Border: green = running, gray = stopped-but-reachable, red = unreachable.
  let border = '#484f58';
  if (running) border = '#3fb950';
  else if (!reachable) border = '#f85149';
  if (card) card.style.borderLeftColor = border;

  if (toggle) {
    toggle.checked = running;
    // Refuse the toggle if the host is unreachable AND the user is trying
    // to start it. Stop is still possible (no-op SSH will fail explicitly).
    toggle.disabled = streamerpiMainBusy || (!reachable && !running);
  }
  if (toggleLabel) {
    toggleLabel.classList.toggle('busy', streamerpiMainBusy);
  }
  if (detail) {
    if (streamerpiMainBusy) {
      detail.textContent = 'Toggling...';
    } else if (!reachable) {
      detail.textContent = 'Host unreachable';
    } else if (running) {
      detail.textContent = 'Active on :8080';
    } else if (error) {
      detail.textContent = 'Stopped — ' + error;
    } else {
      detail.textContent = 'Stopped';
    }
  }
}

function toggleStreamerpiMain(enabled) {
  streamerpiMainBusy = true;
  updateStreamerpiMainUI();
  ws.send(JSON.stringify({type: "toggle", service: "streamerpi_main", state: enabled}));
}

connectWS();
// Thumbs-up / thumbs-down: each button carries data-kind. Optional reason via prompt().
async function submitFlag(kind) {
  const promptText = kind === "good"
    ? "Why was this response good? (optional)"
    : "Why is this response off-persona? (optional)";
  const reason = window.prompt(promptText, "");
  if (reason === null) return;  // Cancel
  const status = document.getElementById("flag-status");
  status.textContent = (kind === "good" ? "saving good example..." : "flagging...");
  status.style.color = "#8b949e";
  try {
    const r = await fetch("/api/timmy/feedback/manual_flag", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({kind: kind, reason: reason})
    });
    const j = await r.json();
    if (j.ok) {
      const verb = kind === "good" ? "saved" : "flagged";
      status.textContent = verb + " ✓ (" + j.persona_tuning_file + ")";
      status.style.color = (kind === "good" ? "#3fb950" : "#f0883e");
      // Auto-pop the detail modal so the user can immediately see what
      // the LLM saw when it produced the just-flagged response.
      populateFlagModal({
        kind: kind,
        comment: j.comment || reason,
        response: j.flagged_assistant,
        conversation_history: j.conversation_history || [],
        system_prompt: j.system_prompt || "",
        speaker: j.speaker,
        ts: Date.now() / 1000,
      });
      showFlagModal();
    } else {
      status.textContent = "failed: " + (j.error || "unknown");
      status.style.color = "#f85149";
    }
  } catch (e) {
    status.textContent = "network error: " + e;
    status.style.color = "#f85149";
  }
  setTimeout(() => { status.textContent = ""; status.style.color = "#8b949e"; }, 8000);
}
document.getElementById("praise-btn").addEventListener("click", () => submitFlag("good"));
document.getElementById("flag-btn").addEventListener("click", () => submitFlag("bad"));

// ---- Flag detail modal (System Prompt / Conversation / This Flag) ----
// Renders the most recent flagged.jsonl entry in plain English. Auto-pops
// after a 👍/👎 click; 🗂 Last flag re-opens it for the last persisted flag.
const flagModal = document.getElementById("flag-modal");

function populateFlagModal(data) {
  const kind = data.kind || "bad";
  const isGood = kind === "good";
  const chip = document.getElementById("flag-modal-kind-chip");
  chip.textContent = isGood ? "👍 GOOD" : "👎 BAD";
  chip.style.background = isGood ? "#1f3a25" : "#3a1f1f";
  chip.style.color      = isGood ? "#3fb950" : "#f85149";
  chip.style.border     = "1px solid " + (isGood ? "#3fb950" : "#f85149");

  document.getElementById("flag-modal-title").textContent =
    isGood ? "Saved as positive example" : "Flagged for persona-drift audit";

  const ts = data.ts ? new Date(data.ts * 1000).toLocaleString() : "—";
  const speaker = data.speaker || "—";
  const histLen = (data.conversation_history || []).length;
  document.getElementById("flag-modal-meta").textContent =
    ts + " · speaker: " + speaker + " · " + histLen + " hot turn" +
    (histLen === 1 ? "" : "s") + " in context";

  const comment = (data.comment || "").trim();
  document.getElementById("flag-modal-comment").textContent =
    comment ? "“" + comment + "”" : "(no comment provided)";

  document.getElementById("flag-modal-response").textContent =
    data.response || "(empty)";

  document.getElementById("flag-modal-system").textContent =
    data.system_prompt || "(empty)";

  const convEl = document.getElementById("flag-modal-conversation");
  convEl.innerHTML = "";
  const turns = data.conversation_history || [];
  if (turns.length === 0) {
    convEl.innerHTML = '<div style="color:#8b949e; font-style:italic;">No prior hot turns captured.</div>';
  } else {
    turns.forEach((t, i) => {
      const row = document.createElement("div");
      row.style.cssText = "margin-bottom:8px; padding:6px 8px; border-radius:4px; " +
        (t.role === "assistant"
          ? "background:#1a1f1f; border-left:2px solid #f0d8a8;"
          : "background:#181b22; border-left:2px solid #58a6ff;");
      // Highlight the final pair (the actually-flagged turn) so the user
      // sees which one their 👍/👎 attached to.
      if (i >= turns.length - 2) {
        row.style.outline = "1px dashed #f0883e";
      }
      const ts = t.timestamp ? new Date(t.timestamp * 1000).toLocaleTimeString() : "";
      const speaker = t.speaker ? " (" + t.speaker + ")" : "";
      const label = (t.role === "assistant" ? "TIMMY" : "USER") + speaker;
      row.innerHTML =
        '<div style="font-size:10px; color:#8b949e; text-transform:uppercase; margin-bottom:2px;">' +
        escapeHtml(label) + ' · ' + escapeHtml(ts) +
        '</div><div style="white-space:pre-wrap; word-wrap:break-word; color:#e0e0e0;">' +
        escapeHtml(t.content || "") + '</div>';
      convEl.appendChild(row);
    });
  }
}

function showFlagModal() { flagModal.style.display = "flex"; }

async function showLastFlag() {
  const status = document.getElementById("flag-status");
  status.textContent = "loading last flag...";
  status.style.color = "#8b949e";
  try {
    const r = await fetch("/api/timmy/feedback/last_flag");
    const j = await r.json();
    if (!j.available) {
      status.textContent = j.error
        ? ("last flag: " + j.error)
        : "no flags persisted yet — click 👍 or 👎 first";
      status.style.color = "#8b949e";
      setTimeout(() => { status.textContent = ""; }, 6000);
      return;
    }
    populateFlagModal({
      kind: j.kind,
      comment: j.comment,
      response: j.response,
      conversation_history: j.conversation_history || [],
      system_prompt: j.system_prompt || "",
      speaker: j.speaker,
      ts: j.ts,
    });
    showFlagModal();
    status.textContent = "";
  } catch (e) {
    status.textContent = "network error: " + e;
    status.style.color = "#f85149";
  }
}

document.getElementById("show-last-flag-btn").addEventListener("click", showLastFlag);
document.getElementById("flag-modal-close-btn").addEventListener("click", () => {
  flagModal.style.display = "none";
});
flagModal.addEventListener("click", (e) => {
  if (e.target === flagModal) flagModal.style.display = "none";
});

document.getElementById("reenroll-btn").addEventListener("click", async () => {
  const nameRaw = window.prompt("Name to re-enroll (leave blank for last speaker)", "");
  if (nameRaw === null) return;
  const status = document.getElementById("flag-status");
  status.textContent = "opening re-enrollment window...";
  status.style.color = "#8b949e";
  try {
    const body = {};
    if (nameRaw.trim()) body.name = nameRaw.trim();
    const r = await fetch("/api/timmy/speaker/reenroll", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)
    });
    const j = await r.json();
    if (j.ok) {
      status.textContent = "re-enrolling " + j.name + " for " + Math.round(j.duration_s) + "s — keep talking";
      status.style.color = "#58a6ff";
    } else {
      status.textContent = "failed: " + (j.error || "unknown");
      status.style.color = "#f85149";
    }
  } catch (e) {
    status.textContent = "network error: " + e;
    status.style.color = "#f85149";
  }
  setTimeout(() => { status.textContent = ""; status.style.color = "#8b949e"; }, 12000);
});

// Last-payload inspector — shows the most recent ephemeral system prompt and full message list.
const payloadModal = document.getElementById("payload-modal");
async function showLastPayload() {
  document.getElementById("payload-meta").textContent = "loading...";
  document.getElementById("payload-user").textContent = "";
  document.getElementById("payload-ephemeral").textContent = "";
  document.getElementById("payload-messages").textContent = "";
  // Bundle A 00:39: render the most recent assistant utterance in the new
  // header box so the operator can see WHAT Timmy said before scrolling
  // into the input payload that produced it.
  const lastAsstEl = document.getElementById("payload-last-asst");
  if (lastAsstEl) lastAsstEl.textContent = _lastAssistantTurn || "(no reply yet)";
  payloadModal.style.display = "flex";
  try {
    const r = await fetch("/api/timmy/last_payload");
    const j = await r.json();
    if (!j.available) {
      document.getElementById("payload-meta").textContent =
        j.error ? ("error: " + j.error) : "no payload captured yet — wait for one turn";
      return;
    }
    document.getElementById("payload-meta").textContent =
      "captured " + j.timestamp + " · " + j.history_turn_count + " history turns · "
      + (j.messages ? j.messages.length : 0) + " total messages";
    document.getElementById("payload-user").textContent = j.user_text || "(empty)";
    document.getElementById("payload-ephemeral").textContent = j.ephemeral_block || "(empty)";
    document.getElementById("payload-messages").textContent = JSON.stringify(j.messages, null, 2);
  } catch (e) {
    document.getElementById("payload-meta").textContent = "network error: " + e;
  }
}
document.getElementById("show-payload-btn").addEventListener("click", showLastPayload);
document.getElementById("payload-close-btn").addEventListener("click", () => {
  payloadModal.style.display = "none";
});
payloadModal.addEventListener("click", (e) => {
  if (e.target === payloadModal) payloadModal.style.display = "none";
});

document.getElementById("open-booth-btn").addEventListener("click", () => {
  const host = window.location.hostname || "localhost";
  const url = `https://${host}:8090/`;
  const status = document.getElementById("open-booth-status");
  const win = window.open(url, "lt_booth_display",
    "noopener,noreferrer,width=1920,height=1080,menubar=no,toolbar=no,location=no");
  if (win) {
    status.textContent = "opened " + url + " in a new window";
    status.style.color = "#3fb950";
  } else {
    status.innerHTML = 'popup blocked — open manually: <a href="' + url + '" target="_blank" style="color:#39c5cf;">' + url + '</a>';
    status.style.color = "#f0883e";
  }
});

loadConversationHistory();
loadSessionLog();
initLatencyChart();
const _latencyClearBtn = document.getElementById("latency-clear-btn");
if (_latencyClearBtn) _latencyClearBtn.addEventListener("click", clearLatencyChart);
pollMetrics();
metricsInterval = setInterval(pollMetrics, 30000);
pollFaceTracking();
setInterval(pollFaceTracking, 10000);
pollStreamerpiMain();
setInterval(pollStreamerpiMain, 10000);
pollLTToggles();
setInterval(pollLTToggles, 10000);

// --- P4 face-flap debounce knobs (Pi A1/A2 + LT B3/C5) ---------------------
const FT_FIELDS = {
  pi: { 'face_unknown_debounce_frames': 'ft-debounce',
        'face_identity_acquire_dist': 'ft-acquire',
        'face_identity_release_dist': 'ft-release' },
  lt: { 'people_novelty_min_persistence': 'ft-persistence',
        'enroll_candidate_min_span_s': 'ft-span',
        'enroll_candidate_min_dist': 'ft-enroll-dist' },
};

async function loadFaceTuning() {
  const st = document.getElementById('face-tuning-status');
  const errs = [];
  for (const [scope, url] of [['pi', '/api/timmy/face_id_tuning'],
                              ['lt', '/api/timmy/vision/tuning']]) {
    try {
      const d = await (await fetch(url)).json();
      if (d.error) { errs.push(scope + ': ' + d.error); continue; }
      for (const [key, elId] of Object.entries(FT_FIELDS[scope])) {
        const el = document.getElementById(elId);
        if (el && d[key] != null && document.activeElement !== el) el.value = d[key];
      }
    } catch (e) { errs.push(scope + ' unreachable'); }
  }
  if (st) {
    st.textContent = errs.length ? errs.join(' · ') : 'live (Pi A1/A2 + LT B3/C5)';
    st.style.color = errs.length ? '#f85149' : '#8b949e';
  }
}

async function commitFaceTuning(scope, key, value) {
  const st = document.getElementById('face-tuning-status');
  const url = scope === 'pi' ? '/api/timmy/face_id_tuning' : '/api/timmy/vision/tuning';
  const body = {}; body[key] = parseFloat(value);
  try {
    const r = await fetch(url, { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok || d.error || d.ok === false) {
      if (st) { st.textContent = 'rejected: ' + (d.error || r.status); st.style.color = '#f85149'; }
    } else {
      if (st) { st.textContent = key + ' saved'; st.style.color = '#3fb950'; }
    }
  } catch (e) {
    if (st) { st.textContent = 'unreachable'; st.style.color = '#f85149'; }
  }
  loadFaceTuning();  // re-sync (also restores a rejected field to server truth)
}

loadFaceTuning();
setInterval(loadFaceTuning, 15000);

// Slice A: situational-awareness regime select (live, read per-turn).
async function loadSituation() {
  const st = document.getElementById('situation-status');
  const sel = document.getElementById('situation-select');
  try {
    const d = await (await fetch('/api/timmy/situation')).json();
    if (d.error) {
      if (st) { st.textContent = d.error; st.style.color = '#f85149'; }
      return;
    }
    if (sel && document.activeElement !== sel) sel.value = d.situation_regime || '';
    if (st) {
      const on = (d.situation_regime || '') !== '';
      st.textContent = on ? ('active: ' + d.situation_regime) : 'off (no situation prior)';
      st.style.color = on ? '#3fb950' : '#8b949e';
    }
  } catch (e) {
    if (st) { st.textContent = 'unreachable'; st.style.color = '#f85149'; }
  }
}

async function commitSituation(value) {
  const st = document.getElementById('situation-status');
  try {
    const r = await fetch('/api/timmy/situation', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ situation_regime: value }) });
    const d = await r.json();
    if (!r.ok || d.error || d.ok === false) {
      if (st) { st.textContent = 'rejected: ' + (d.error || r.status); st.style.color = '#f85149'; }
    }
  } catch (e) {
    if (st) { st.textContent = 'unreachable'; st.style.color = '#f85149'; }
  }
  loadSituation();  // re-sync (also restores a rejected select to server truth)
  loadPartyMode();  // keep the PARTY banner in sync with the granular select
}

loadSituation();
setInterval(loadSituation, 15000);

// PARTY MODE banner — one-tap situation_regime=PARTY (the key party control).
function applyPartyBanner(regime) {
  const on = (regime || '').toUpperCase() === 'PARTY';
  const banner = document.getElementById('party-banner');
  const label = document.getElementById('party-label');
  const sub = document.getElementById('party-sub');
  const state = document.getElementById('party-state');
  if (banner) banner.classList.toggle('on', on);
  if (label) label.textContent = on ? '🎉 PARTY MODE — ON' : '🎉 Party Mode — Off';
  if (state) state.textContent = on ? 'ON' : 'OFF';
  if (sub) sub.textContent = on
    ? 'Assuming strangers · speaker auto-continuity DISABLED · prompt prior set'
    : 'Tap to assume strangers + disable speaker auto-continuity';
}

async function loadPartyMode() {
  try {
    const d = await (await fetch('/api/timmy/situation')).json();
    if (!d.error) applyPartyBanner(d.situation_regime);
  } catch (e) { /* keep last state */ }
}

async function togglePartyMode() {
  // Read current, flip PARTY <-> "" (off). Optimistic UI, then reconcile.
  let cur = '';
  try { cur = (await (await fetch('/api/timmy/situation')).json()).situation_regime || ''; }
  catch (e) {}
  const target = cur.toUpperCase() === 'PARTY' ? '' : 'PARTY';
  applyPartyBanner(target);
  try {
    await fetch('/api/timmy/situation', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ situation_regime: target }) });
  } catch (e) {}
  loadPartyMode();
  loadSituation();   // keep the granular regime <select> in sync
}

loadPartyMode();
setInterval(loadPartyMode, 15000);

// Mouth-mute (lab) toggle. Silences replies + fillers; announce still speaks.
async function loadTtsMute() {
  const detail = document.getElementById('tts-mute-detail');
  const toggle = document.querySelector('#tts-mute-toggle input');
  const card = document.getElementById('tts-mute-card');
  try {
    const d = await (await fetch('/api/timmy/tts_mute')).json();
    if (d.error) {
      if (detail) { detail.textContent = d.error; detail.style.color = '#f85149'; }
      return;
    }
    const muted = !!d.muted;
    if (toggle && document.activeElement !== toggle) toggle.checked = muted;
    if (card) card.style.borderLeftColor = muted ? '#d29922' : '#484f58';
    if (detail) {
      detail.textContent = muted
        ? 'MUTED — replies + fillers off; ears + matcher live; announce still speaks'
        : 'Speaking normally';
      detail.style.color = muted ? '#d29922' : '#8b949e';
    }
  } catch (e) {
    if (detail) { detail.textContent = 'unreachable'; detail.style.color = '#f85149'; }
  }
}

async function commitTtsMute(muted) {
  const detail = document.getElementById('tts-mute-detail');
  try {
    const r = await fetch('/api/timmy/tts_mute', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ muted: !!muted }) });
    const d = await r.json();
    if (!r.ok || d.error || d.ok === false) {
      if (detail) { detail.textContent = 'rejected: ' + (d.error || r.status); detail.style.color = '#f85149'; }
    }
  } catch (e) {
    if (detail) { detail.textContent = 'unreachable'; detail.style.color = '#f85149'; }
  }
  loadTtsMute();
}

loadTtsMute();
setInterval(loadTtsMute, 15000);

async function pollHostMetrics() {
  try {
    const r = await fetch('/api/host');
    const m = await r.json();
    if (!m.available) return;
    const cpu = document.getElementById('host-cpu');
    const ram = document.getElementById('host-ram');
    const disk = document.getElementById('host-disk');
    const load = document.getElementById('host-load');
    if (cpu) cpu.textContent = m.cpu_percent + '% · ' + m.cpu_count + ' cores';
    if (ram) ram.textContent = m.ram_used_gb + ' / ' + m.ram_total_gb + ' GB (' + m.ram_percent + '%)';
    if (disk) disk.textContent = m.disk_used_gb + ' / ' + m.disk_total_gb + ' GB (' + m.disk_percent + '%)';
    if (load) {
      const parts = [m.load_1m, m.load_5m, m.load_15m]
        .map(v => v == null ? '?' : v.toFixed(2));
      load.textContent = parts.join(' / ');
    }
    const vram = document.getElementById('host-vram');
    const gpu = document.getElementById('host-gpu');
    if (vram) {
      vram.textContent = (m.vram_used_gb != null && m.vram_total_gb != null)
        ? m.vram_used_gb + ' / ' + m.vram_total_gb + ' GB (' + m.vram_percent + '%)'
        : 'unavailable';
    }
    if (gpu) {
      gpu.textContent = (m.gpu_busy_percent != null)
        ? m.gpu_busy_percent + '%'
        : 'unavailable';
    }
  } catch(e) {}
}
pollHostMetrics();
setInterval(pollHostMetrics, 5000);

// --- Mic VU meter + energy floor -----------------------------------------
// last_peak / last_vad_prob / energy_floor come from LT's /api/audio/diag
// (peak = max-abs amplitude, 0..1 where 1.0 == clipping). Peak-hold is
// computed client-side: jump up instantly, decay slowly so the "most recent
// peak" stays visible. The orange floor marker is the onset energy gate:
// VAD-positive audio whose peak is left of it is ignored as background. Drag
// the slider to sit it between the room floor and your speaking peak.
const VU_POLL_MS = 120;
const VU_DECAY_PER_TICK = 0.035;     // ~0.29/sec hold decay -> ~3s visible peak
const VAD_SPEECH_THRESHOLD = 0.4;
let vuPeakHold = 0;
let vuFloor = 0;                     // last-known server floor (0..1)
let vuFloorHoldUntil = 0;           // suppress poll->slider sync briefly after a user edit
function vuColor(level) {
  if (level >= 0.85) return '#f85149';   // clip zone
  if (level >= 0.60) return '#f0883e';   // hot
  return '#3fb950';                       // nominal
}
function renderFloorMarker() {
  const fl = document.getElementById('vu-floor');
  if (fl) fl.style.left = (vuFloor * 100).toFixed(1) + '%';
  const val = document.getElementById('vu-floor-val');
  if (val) val.textContent = vuFloor.toFixed(3);
}
// Live drag: update label + marker only (no network until release).
function onFloorInput(v) {
  vuFloor = Math.max(0, Math.min(1, parseFloat(v) || 0));
  vuFloorHoldUntil = Date.now() + 2000;
  renderFloorMarker();
}
// Release: persist to LT (which applies it on the next audio chunk).
async function commitFloor(v) {
  const value = Math.max(0, Math.min(1, parseFloat(v) || 0));
  vuFloorHoldUntil = Date.now() + 2000;
  try {
    const r = await fetch('/api/timmy/capture/energy_floor', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({value: value})
    });
    const d = await r.json();
    if (d && d.energy_floor != null) { vuFloor = d.energy_floor; renderFloorMarker(); }
  } catch(e) {}
}
// Auto-calibrate: phase 1 samples the room (quiet), phase 2 samples the voice
// (talking), then sets the floor between them — same heuristic as the
// server-side hand calibration (between ambient max and speech median).
function vuMedian(a){ if(!a.length) return 0; const s=[...a].sort((x,y)=>x-y); return s[Math.floor(s.length/2)]; }
async function vuSample(durMs, statusFn) {
  const peaks=[], speech=[];
  const end = Date.now()+durMs;
  while (Date.now() < end) {
    try {
      const d = await (await fetch('/api/timmy/audio_diag')).json();
      if (d && d.last_peak != null) {
        peaks.push(d.last_peak);
        if ((d.last_vad_prob||0) >= VAD_SPEECH_THRESHOLD) speech.push(d.last_peak);
      }
    } catch(e) {}
    if (statusFn) statusFn(Math.ceil((end-Date.now())/1000));
    await new Promise(r=>setTimeout(r,100));
  }
  return {peaks, speech};
}
// Speak text out of Timmy's speaker (no "This is Claude" prefix — the tool
// just voices its own prompts).
async function ttsSay(text) {
  try {
    await fetch('/api/timmy/announce', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text: text, no_prefix: true})});
  } catch(e) {}
}
// tts.speak is non-blocking (queues PCM), so wait for playback to finish
// before sampling — else Timmy's own voice bleeds into the mic and inflates
// the measurement. /api/audio/diag.suppressed is True while TTS plays: wait
// for it to go true (started) then false (ended), capped at maxMs.
async function waitForPlayback(maxMs) {
  const start = Date.now(); let started=false;
  while (Date.now()-start < maxMs) {
    let sup=false;
    try { sup = !!(await (await fetch('/api/timmy/audio_diag')).json()).suppressed; } catch(e){}
    if (sup) started=true; else if (started) break;
    await new Promise(r=>setTimeout(r,120));
  }
  await new Promise(r=>setTimeout(r,250));   // brief settle after playback
}
let vuCalRunning=false;
let vuCalLastResult='';   // sticky: re-asserted by pollAudioMeter if the span blanks
async function autoCalibrateFloor() {
  if (vuCalRunning) return;
  vuCalRunning=true;
  const btn=document.getElementById('vu-cal-btn');
  const st=document.getElementById('vu-cal-status');
  const setSt=(t)=>{ if(st) st.textContent=t; };
  if (btn){ btn.disabled=true; btn.style.opacity=0.6; }
  try {
    // Phase 1 — room (Timmy voices the prompt, we wait for it to finish).
    setSt('Measuring room — stay quiet…');
    await ttsSay('Measuring the room. Please stay quiet for a few seconds.');
    await waitForPlayback(6000);
    const room = await vuSample(4000, s=>setSt('Measuring room — stay quiet ('+s+'s)'));
    const ambientMax = room.peaks.length ? Math.max(...room.peaks) : 0;
    // Phase 2 — voice.
    setSt('Get ready to talk…');
    await ttsSay('Now talk normally for about six seconds.');
    await waitForPlayback(6000);
    const voice = await vuSample(6000, s=>setSt('Talk normally now ('+s+'s)'));
    const speechMed = vuMedian(voice.speech);
    if (voice.speech.length < 5 || speechMed <= ambientMax) {
      const msg='Didn\'t hear enough speech — try again and talk a bit louder.';
      setSt(msg); vuCalLastResult=msg;
      await ttsSay('I did not hear enough speech. Please try again and talk a little louder.');
      return;
    }
    let rec = Math.max(ambientMax*1.5, ambientMax + 0.45*(speechMed-ambientMax));
    rec = Math.min(rec, speechMed*0.7);
    rec = Math.round(rec*1000)/1000;
    const sl=document.getElementById('vu-floor-slider');
    if (sl) sl.value = rec;
    await commitFloor(rec);
    const result='Set floor to '+rec.toFixed(3)+'  (room '+ambientMax.toFixed(3)+' · voice '+speechMed.toFixed(3)+')';
    setSt(result); vuCalLastResult=result;
    const pct=(rec*100).toFixed(1);
    await ttsSay('Onset floor set to '+pct+' percent. Your voice is well above it.');
  } finally {
    if (btn){ btn.disabled=false; btn.style.opacity=1; }
    vuCalRunning=false;
  }
}
async function pollAudioMeter() {
  const fill = document.getElementById('vu-fill');
  const peak = document.getElementById('vu-peak');
  const dot = document.getElementById('vu-speech-dot');
  const out = document.getElementById('vu-readout');
  if (!fill) return;
  // Sticky calibration result: if the status span is blank but we have a
  // result and aren't mid-run, restore it so it stays visible.
  const calSt=document.getElementById('vu-cal-status');
  if (calSt && !calSt.textContent && vuCalLastResult && !vuCalRunning) calSt.textContent=vuCalLastResult;
  try {
    const r = await fetch('/api/timmy/audio_diag');
    const d = await r.json();
    if (d.error || d.last_peak == null) {
      if (out) out.textContent = 'mic unreachable';
      fill.style.width = '0%';
      vuPeakHold = 0;
      return;
    }
    const cur = Math.max(0, Math.min(1, d.last_peak));
    const vad = d.last_vad_prob != null ? d.last_vad_prob : 0;
    // Sync the floor from the server unless the user just touched the slider.
    if (d.energy_floor != null && Date.now() > vuFloorHoldUntil) {
      vuFloor = d.energy_floor;
      const sl = document.getElementById('vu-floor-slider');
      if (sl && document.activeElement !== sl) sl.value = vuFloor;
      renderFloorMarker();
    }
    // peak-hold: instant attack, slow decay
    vuPeakHold = cur > vuPeakHold ? cur : Math.max(cur, vuPeakHold - VU_DECAY_PER_TICK);
    fill.style.width = (cur * 100).toFixed(1) + '%';
    fill.style.background = vuColor(cur);
    peak.style.left = (vuPeakHold * 100).toFixed(1) + '%';
    peak.style.background = vuColor(vuPeakHold);
    const speaking = vad >= VAD_SPEECH_THRESHOLD;
    // The dot reflects what capture will ACT on: VAD-positive AND above floor.
    const wouldFire = speaking && cur >= vuFloor;
    dot.style.background = wouldFire ? '#3fb950' : (speaking ? '#f0883e' : '#30363d');
    dot.style.boxShadow = wouldFire ? '0 0 4px #3fb950' : 'none';
    if (out) out.textContent =
      'cur ' + cur.toFixed(3) + ' · pk ' + vuPeakHold.toFixed(3) + ' · vad ' + vad.toFixed(2);
  } catch(e) {
    if (out) out.textContent = 'mic unreachable';
  }
}
pollAudioMeter();
setInterval(pollAudioMeter, VU_POLL_MS);

let recIsActive = false;
async function pollRecording() {
  try {
    const r = await fetch('/api/recording/status');
    const s = await r.json();
    recIsActive = !!s.active;
    const dot = document.getElementById('rec-dot');
    const state = document.getElementById('rec-state');
    const meta = document.getElementById('rec-meta');
    const btn = document.getElementById('rec-toggle-btn');
    if (!dot || !state || !meta || !btn) return;
    if (s.active) {
      dot.style.background = '#f85149';
      state.textContent = 'recording · ' + (s.duration_s || 0).toFixed(1) + 's';
      btn.textContent = 'Stop';
      btn.style.background = '#3a1f1f';
      btn.style.borderColor = '#f85149';
      btn.style.color = '#f85149';
      const kb = Math.round((s.bytes_received || 0) / 1024);
      meta.textContent = (s.session_id || '?') + ' · ' + (s.chunks_received || 0) + ' chunks · ' + kb + ' KB';
    } else {
      dot.style.background = '#484f58';
      state.textContent = 'idle';
      btn.textContent = 'Start';
      btn.style.background = '#1f6feb';
      btn.style.borderColor = '#1f6feb';
      btn.style.color = '#fff';
      if (s.session_id && s.duration_s) {
        const kb = Math.round((s.bytes_received || 0) / 1024);
        meta.textContent = 'last: ' + s.session_id + ' · ' + s.duration_s.toFixed(1) + 's · ' + (s.chunks_received || 0) + ' chunks · ' + kb + ' KB';
      } else {
        meta.textContent = 'No active recording';
      }
    }
  } catch (e) { /* booth-display down — leave UI as-is */ }
}
async function toggleRecording() {
  const btn = document.getElementById('rec-toggle-btn');
  if (btn) btn.disabled = true;
  const url = recIsActive ? '/api/recording/stop' : '/api/recording/start';
  try {
    await fetch(url, { method: 'POST' });
  } catch (e) { /* surface via next poll */ }
  await pollRecording();
  if (btn) btn.disabled = false;
}
pollRecording();
setInterval(pollRecording, 2000);

async function pollPresence() {
  try {
    const r = await fetch('/api/timmy/presence');
    const data = await r.json();
    const panel = document.getElementById('presence-panel');
    const meta = document.getElementById('presence-meta');
    if (!panel) return;
    if (!data || !data.enabled || !Array.isArray(data.present) || data.present.length === 0) {
      panel.innerHTML = '<div id="presence-empty" style="color:#8b949e; font-style:italic;">No one detected yet</div>';
      meta.textContent = data && data.unknown_voices_recent ? ('Unknown voices recent: ' + data.unknown_voices_recent) : '';
      return;
    }
    const fmtAge = (s) => {
      if (s == null) return '';
      if (s < 60) return Math.round(s) + 's';
      if (s < 3600) return Math.round(s / 60) + 'm';
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return m ? (h + 'h' + String(m).padStart(2, '0') + 'm') : (h + 'h');
    };
    // Bundle A 00:49: per-person face-ID + voice-print sub-rows. The
    // backend already returns last_seen_face_age_s and last_seen_voice_age_s
    // separately on each presence entry; previously the render collapsed
    // them into one fused timestamp (face preferred, voice as fallback),
    // which hid which modality was actually grounding the belief. Two
    // sub-rows make degraded-audio sessions (face fresh, voice drifting
    // stale) instantly visible.
    const rows = data.present.filter(p => {
      const n = (p.name || '').toLowerCase();
      return n && !n.startsWith('unknown');
    }).map(p => {
      const name = (p.name || '').replace(/^\w/, c => c.toUpperCase());
      const onCam = p.on_camera_now;
      const dot = onCam ? '●' : '○';
      const dotColour = onCam ? '#3fb950' : '#8b949e';
      const opacity = onCam ? '1' : '0.65';
      let poseStr = '';
      if (p.last_pose && p.last_pose.pan != null) {
        poseStr = ' pan ' + Math.round(p.last_pose.pan) + '° / tilt ' + Math.round(p.last_pose.tilt) + '°';
      }
      // Sub-row colour: green if signal fresh (<60s), amber if drifting
      // (60s-5m), grey if cold (>5m or never).
      const ageColour = (age) => {
        if (age == null) return '#484f58';
        if (age < 60) return '#3fb950';
        if (age < 300) return '#d29922';
        return '#8b949e';
      };
      const faceAge = p.last_seen_face_age_s;
      const voiceAge = p.last_seen_voice_age_s;
      const faceLabel = (faceAge != null) ? (onCam ? 'on camera' : ('seen ' + fmtAge(faceAge) + ' ago')) : 'never';
      const voiceLabel = (voiceAge != null) ? ('heard ' + fmtAge(voiceAge) + ' ago') : 'never';
      return ''
        + '<div style="opacity:' + opacity + '; padding:4px 0 4px 0; border-bottom:1px solid #161b22;">'
          + '<div>'
            + '<span style="color:' + dotColour + '; margin-right:6px;">' + dot + '</span>'
            + '<strong>' + name + '</strong>'
            + (poseStr ? '<span style="color:#484f58; font-size:10px; margin-left:8px;">' + poseStr + '</span>' : '')
          + '</div>'
          + '<div style="font-size:10px; padding-left:14px; line-height:1.5;">'
            + '<span style="color:#484f58;">face</span> '
            + '<span style="color:' + ageColour(faceAge) + ';">' + faceLabel + '</span>'
            + '<span style="color:#484f58; margin:0 6px;">·</span>'
            + '<span style="color:#484f58;">voice</span> '
            + '<span style="color:' + ageColour(voiceAge) + ';">' + voiceLabel + '</span>'
          + '</div>'
        + '</div>';
    }).join('');
    panel.innerHTML = rows;
    meta.textContent = data.unknown_voices_recent ? ('Unknown voices recent: ' + data.unknown_voices_recent) : '';
  } catch (e) {
    /* network burp; leave previous state */
  }
}
pollPresence();
setInterval(pollPresence, 3000);

// Suppress the next poll's override-UI sync for a beat after a user action so
// the checkbox/grid don't flicker against the in-flight server round-trip.
let moodOverrideBusyUntil = 0;

function applyMoodOverrideUI(isOverride) {
  const chk = document.getElementById('mood-override-chk');
  const grid = document.querySelector('.mood-grid');
  const status = document.getElementById('mood-override-status');
  if (chk && Date.now() > moodOverrideBusyUntil) chk.checked = !!isOverride;
  if (grid) grid.classList.toggle('override', !!isOverride);
  if (status) status.textContent = isOverride ? 'manual' : '';
}

async function postMoodOverride(body) {
  moodOverrideBusyUntil = Date.now() + 1500;
  try {
    const r = await fetch('/api/timmy/mood/override', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data && data.ok === false) { console.warn('mood override failed:', data.error); }
  } catch (e) { console.warn('mood override post error', e); }
  pollMood();
}

function setupMoodOverride() {
  const chk = document.getElementById('mood-override-chk');
  if (chk) {
    chk.addEventListener('change', () => {
      if (chk.checked) {
        // Enabling: pin to whatever cell is currently active (no jump).
        const active = document.querySelector('.mood-cell.active');
        const x = active ? Number(active.dataset.x) : 0;
        const y = active ? Number(active.dataset.y) : 0;
        applyMoodOverrideUI(true);
        postMoodOverride({enabled: true, x, y});
      } else {
        applyMoodOverrideUI(false);
        postMoodOverride({enabled: false});
      }
    });
  }
  document.querySelectorAll('.mood-cell').forEach(cell => {
    cell.addEventListener('click', () => {
      const chk = document.getElementById('mood-override-chk');
      if (!chk || !chk.checked) return;   // only clickable in override mode
      const x = Number(cell.dataset.x);
      const y = Number(cell.dataset.y);
      document.querySelectorAll('.mood-cell').forEach(c =>
        c.classList.toggle('active', c === cell));
      postMoodOverride({enabled: true, x, y});
    });
  });
}
setupMoodOverride();

async function pollMood() {
  try {
    const r = await fetch('/api/timmy/mood');
    if (!r.ok) return;
    const data = await r.json();
    if (!data || typeof data.x !== 'number' || typeof data.y !== 'number') return;
    applyMoodOverrideUI(!!data.override);
    document.querySelectorAll('.mood-cell').forEach(cell => {
      const cx = Number(cell.dataset.x);
      const cy = Number(cell.dataset.y);
      cell.classList.toggle('active', cx === data.x && cy === data.y);
    });
    const rendered = document.getElementById('mood-rendered');
    if (rendered) rendered.textContent = data.rendered || '--';
    const sig = document.getElementById('mood-signals');
    if (sig) {
      const xs = (typeof data.last_x_signal === 'number') ? data.last_x_signal.toFixed(2) : '--';
      const ys = (typeof data.last_y_signal === 'number') ? data.last_y_signal.toFixed(2) : '--';
      sig.textContent = 'X (engagement) ' + xs + '  ·  Y (tone, VADER inv) ' + ys;
    }
    const upd = document.getElementById('mood-updated');
    if (upd) {
      if (data.last_update_ts && data.last_update_ts > 0) {
        const age = Math.max(0, Date.now() / 1000 - data.last_update_ts);
        let ageStr;
        if (age < 60) ageStr = Math.round(age) + 's';
        else if (age < 3600) ageStr = Math.round(age / 60) + 'm';
        else ageStr = Math.round(age / 3600) + 'h';
        upd.textContent = 'updated ' + ageStr + ' ago';
      } else {
        upd.textContent = 'never updated';
      }
    }
  } catch (e) {
    /* network burp; leave previous state */
  }
}
pollMood();
setInterval(pollMood, 5000);


</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT, log_level="info")
