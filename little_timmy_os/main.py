"""Little Timmy OS — Standalone service manager and dashboard.

Runs independently of Little Timmy on port 8894.
Manages, monitors, and controls all Little Timmy stack services.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
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
                        if data.get("type") in ("turn", "metrics", "retrieval"):
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

@app.get("/api/face_tracking")
async def get_face_tracking():
    """Get face tracking status from streamerpi."""
    return await services.check_face_tracking_status()


@app.post("/api/face_tracking/toggle")
async def toggle_face_tracking(data: dict):
    """Toggle face tracking on streamerpi."""
    enabled = data.get("enabled", True)
    result = await services.toggle_face_tracking(enabled)
    return result


@app.get("/api/streamerpi/main")
async def get_streamerpi_main():
    """Probe streamerpi main codebase (little-timmy-motor.service)."""
    return await services.check_streamerpi_server_status()


@app.post("/api/streamerpi/main/toggle")
async def toggle_streamerpi_main(data: dict):
    """Start / stop little-timmy-motor.service on streamerpi via SSH+sudo."""
    enabled = data.get("enabled", True)
    return await services.toggle_streamerpi_server(enabled)




async def health_poll_loop():
    """Background task: poll service health and broadcast updates."""
    while True:
        await asyncio.sleep(config.HEALTH_POLL_INTERVAL)
        try:
            health = await services.check_all_health()
            await broadcast_event("health", {"services": health})
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
  grid-template-columns: 1fr 1.2fr 1fr;
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
</style>
</head>
<body>

<header>
  <h1>LITTLE TIMMY OS</h1>
  <span class="uptime" id="uptime">Connecting...</span>
</header>

<div class="main-content">
  <div>
    <details class="panel" open>
      <summary><h2>Services</h2></summary>
      <div id="services"></div>
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
        <div class="service-card" id="face-tracking-card" style="border-left:3px solid #484f58;">
          <div class="service-info">
            <div class="service-name">Face Tracking</div>
            <div class="service-detail" id="face-tracking-detail">Checking...</div>
          </div>
          <label class="toggle" id="face-tracking-toggle">
            <input type="checkbox" onchange="toggleFaceTracking(this.checked)">
            <span class="slider"></span>
          </label>
        </div>
      </div>
      <div class="model-selector" id="model-selector">
        <label>Conversation LLM Model</label>
        <select id="model-select" onchange="switchModel(this.value)" disabled>
          <option value="">Loading...</option>
        </select>
        <div class="model-status" id="model-status"></div>
      </div>
    </details>
  </div>
  <div>
    <details class="panel" open>
      <summary><h2>Body Behavior</h2></summary>
      <div id="behavior-panel" style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div id="behavior-mode" style="font-size:28px; font-weight:bold; color:#e94560;">--</div>
        <div style="flex:1; min-width:150px;">
          <div id="behavior-info" style="font-size:11px; margin-bottom:4px;"></div>
          <div id="behavior-stats" style="font-size:10px; color:#484f58;"></div>
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
    <details class="panel" open style="margin-top:16px; display:flex; flex-direction:column;">
      <summary><h2>Conversation</h2></summary>
      <div id="conversation" style="max-height:380px; overflow-y:auto;">
        <div id="conv-offline">Waiting for Little Timmy...</div>
      </div>
    </details>
    <details class="panel" open style="margin-top:16px;">
      <summary><h2>Mood</h2></summary>
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
    <details class="panel" open>
      <summary><h2>Latency</h2></summary>
      <div id="metrics">
        <div class="metric-row"><span class="label">STT</span><span class="value" id="m-stt">--</span></div>
        <div class="metric-row"><span class="label">Retrieval</span><span class="value" id="m-retrieval">--</span></div>
        <div class="metric-row"><span class="label">LLM 1st token</span><span class="value" id="m-llm-ft">--</span></div>
        <div class="metric-row"><span class="label">LLM total</span><span class="value" id="m-llm">--</span></div>
        <div class="metric-row"><span class="label">TTS</span><span class="value" id="m-tts">--</span></div>
        <div class="metric-row"><span class="label">End-to-end</span><span class="value" id="m-e2e">--</span></div>
        <div class="metric-row"><span class="label">Turns</span><span class="value" id="m-turns">--</span></div>
      </div>
      <div style="margin-top:12px; padding-top:10px; border-top:1px solid #21262d;">
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
          <span>Rolling history</span>
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
const SERVICE_ORDER = ["postgresql", "ollama", "qwen36", "qwen36_vision", "conversation_llm", "whisper", "little_timmy"];
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
    } else if (msg.type === "turn") {
      addTurn(msg.role, msg.content, msg.speaker);
    } else if (msg.type === "metrics") {
      updateMetricsFromWS(msg);
    } else if (msg.type === "retrieval") {
      renderRetrieval(msg);
    }
  };
}

function renderModelSelector(models, current) {
  currentModel = current;
  const sel = document.getElementById("model-select");
  sel.innerHTML = "";
  for (const [id, name] of Object.entries(models)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = name;
    if (id === current) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.disabled = modelSwitching;
  document.getElementById("model-status").textContent = modelSwitching ? "Switching..." : "";
}

function switchModel(modelId) {
  if (!modelId || modelId === currentModel || modelSwitching) return;
  modelSwitching = true;
  const sel = document.getElementById("model-select");
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
  }
  div.innerHTML =
    '<div class="role ' + speakerClass + '">' + label + '</div>' +
    '<div class="content">' + escapeHtml(content) + '</div>';
  conv.appendChild(div);
  conv.scrollTop = conv.scrollHeight;
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

function updateMetricsFromWS(msg) {
  document.getElementById("m-stt").textContent = msg.stt_ms != null ? msg.stt_ms + "ms" : "--";
  document.getElementById("m-retrieval").textContent = msg.retrieval_ms != null ? msg.retrieval_ms + "ms" : "--";
  document.getElementById("m-llm-ft").textContent = msg.llm_first_token_ms != null ? msg.llm_first_token_ms + "ms" : "--";
  document.getElementById("m-llm").textContent = msg.llm_total_ms != null ? msg.llm_total_ms + "ms" : "--";
  document.getElementById("m-tts").textContent = msg.tts_ms != null ? msg.tts_ms + "ms" : "--";
  document.getElementById("m-e2e").textContent = msg.e2e_ms != null ? msg.e2e_ms + "ms" : "--";
  document.getElementById("m-turns").textContent = msg.turns || "--";
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
    const port = svc.id === "conversation_llm" ? 8081 : getPort(sid);
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
  const ports = {postgresql:5432, ollama:11434, gptoss120b:8080, qwen36:8083, conversation_llm:8081, whisper:8891, little_timmy:8893};
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
      document.getElementById("m-retrieval").textContent = m.last_retrieval_ms != null ? m.last_retrieval_ms + "ms" : "--";
      document.getElementById("m-llm-ft").textContent = m.last_llm_first_token_ms != null ? m.last_llm_first_token_ms + "ms" : "--";
      document.getElementById("m-llm").textContent = m.last_llm_total_ms != null ? m.last_llm_total_ms + "ms" : "--";
      document.getElementById("m-tts").textContent = m.last_tts_ms != null ? m.last_tts_ms + "ms" : "--";
      document.getElementById("m-e2e").textContent = m.last_e2e_ms != null ? m.last_e2e_ms + "ms" : "--";
      document.getElementById("m-turns").textContent = m.turns || "--";
      pushLatencySample({
        stt: m.last_stt_ms, retrieval: m.last_retrieval_ms,
        llm_ft: m.last_llm_first_token_ms, llm: m.last_llm_total_ms,
        tts: m.last_tts_ms, e2e: m.last_e2e_ms,
      });
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

    const age = v.age_s != null ? v.age_s.toFixed(0) + 's ago' : '--';
    const chg = v.change_score != null ? v.change_score.toFixed(1) : '--';
    const ratio = v.stats.polled > 0 ? ((v.stats.analyzed / v.stats.polled) * 100).toFixed(0) : '0';
    stats.textContent = 'Age: ' + age + ' | Change: ' + chg + ' | VLM: ' + v.stats.analyzed + '/' + v.stats.polled + ' (' + ratio + '%)';

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
    const faceId = b.params && b.params.face_identity ? b.params.face_identity : null;
    const faceLabel = b.face_visible
      ? (faceId && faceId !== 'unknown'
        ? ' | Face: <span style="color:#58a6ff;font-weight:bold">' + faceId.toUpperCase() + '</span>'
        : ' | Face: YES')
      : '';
    const face = faceLabel;
    const prev = b.previous_mode ? ' | Prev: ' + b.previous_mode : '';
    infoEl.innerHTML =
      '<span style="color:#8b949e">Elapsed: ' + elapsed + 's' + timeout + '</span>' +
      '<span style="color:' + (b.face_visible ? '#3fb950' : '#484f58') + '">' + face + '</span>' +
      '<span style="color:#484f58">' + prev + '</span>';

    if (b.stats) {
      statsEl.textContent = 'Transitions: ' + b.stats.transitions +
        ' | Faces found: ' + b.stats.faces_found +
        ' | Faces lost: ' + b.stats.faces_lost;
    }
  } catch(e) {}
}

pollBehavior();
setInterval(pollBehavior, 2000);


// Face tracking toggle
let faceTrackingEnabled = false;
let faceTrackingBusy = false;

async function pollFaceTracking() {
  try {
    const r = await fetch('/api/face_tracking');
    const data = await r.json();
    faceTrackingEnabled = data.enabled || false;
    updateFaceTrackingUI();
  } catch(e) {}
}

function updateFaceTrackingUI() {
  const card = document.getElementById('face-tracking-card');
  const toggle = document.querySelector('#face-tracking-toggle input');
  const detail = document.getElementById('face-tracking-detail');
  const toggleLabel = document.getElementById('face-tracking-toggle');

  if (card) {
    card.style.borderLeftColor = faceTrackingEnabled ? '#3fb950' : '#484f58';
  }
  if (toggle) {
    toggle.checked = faceTrackingEnabled;
    toggle.disabled = faceTrackingBusy;
  }
  if (toggleLabel) {
    toggleLabel.classList.toggle('busy', faceTrackingBusy);
  }
  if (detail) {
    if (faceTrackingBusy) {
      detail.textContent = 'Toggling...';
    } else {
      detail.textContent = faceTrackingEnabled ? 'YuNet active (2Hz)' : 'Disabled';
    }
  }
}

function toggleFaceTracking(enabled) {
  faceTrackingBusy = true;
  updateFaceTrackingUI();
  ws.send(JSON.stringify({type: "toggle", service: "face_tracking", state: enabled}));
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
    const rows = data.present.filter(p => {
      const n = (p.name || '').toLowerCase();
      return n && !n.startsWith('unknown');
    }).map(p => {
      const name = (p.name || '').replace(/^\w/, c => c.toUpperCase());
      const onCam = p.on_camera_now;
      const dot = onCam ? '●' : '○';
      const colour = onCam ? '#3fb950' : '#8b949e';
      const opacity = onCam ? '1' : '0.65';
      const ageBits = [];
      if (p.last_seen_face_age_s != null) ageBits.push('seen ' + fmtAge(p.last_seen_face_age_s) + ' ago');
      else if (p.last_seen_voice_age_s != null) ageBits.push('heard ' + fmtAge(p.last_seen_voice_age_s) + ' ago');
      let poseStr = '';
      if (p.last_pose && p.last_pose.pan != null) {
        poseStr = ' · pan ' + Math.round(p.last_pose.pan) + '° / tilt ' + Math.round(p.last_pose.tilt) + '°';
      }
      const detail = onCam ? 'on camera' : ageBits.join(', ');
      return '<div style="opacity:' + opacity + '; padding:3px 0;">'
           + '<span style="color:' + colour + '; margin-right:6px;">' + dot + '</span>'
           + '<strong>' + name + '</strong>'
           + ' <span style="color:#8b949e;">' + detail + poseStr + '</span>'
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

async function pollMood() {
  try {
    const r = await fetch('/api/timmy/mood');
    if (!r.ok) return;
    const data = await r.json();
    if (!data || typeof data.x !== 'number' || typeof data.y !== 'number') return;
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
