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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

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
    <div class="panel">
      <h2>Services</h2>
      <div id="services"></div>
      <div id="streamerpi-controls" style="margin-top:12px; padding-top:10px; border-top:1px solid #21262d;">
        <div style="font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">streamerpi Controls</div>
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
    </div>
  </div>
  <div>
    <div class="panel">
      <h2>Body Behavior</h2>
      <div id="behavior-panel" style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div id="behavior-mode" style="font-size:28px; font-weight:bold; color:#e94560;">--</div>
        <div style="flex:1; min-width:150px;">
          <div id="behavior-info" style="font-size:11px; margin-bottom:4px;"></div>
          <div id="behavior-stats" style="font-size:10px; color:#484f58;"></div>
        </div>
      </div>
    </div>
    <div class="panel" style="margin-top:16px;">
      <h2>Who's in the Room</h2>
      <div id="presence-panel" style="font-size:12px; min-height:32px;">
        <div id="presence-empty" style="color:#8b949e; font-style:italic;">No one detected yet</div>
      </div>
      <div id="presence-meta" style="font-size:10px; color:#484f58; margin-top:6px;"></div>
    </div>
    <div class="panel" style="margin-top:16px; display:flex; flex-direction:column;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 style="margin:0;">Conversation</h2>
        <div style="display:flex; gap:6px;">
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
        </div>
      </div>
      <div id="flag-status" style="font-size:11px; color:#8b949e; margin-top:4px; min-height:14px;"></div>
      <div id="conversation" style="max-height:380px; overflow-y:auto; margin-top:8px;">
        <div id="conv-offline">Waiting for Little Timmy...</div>
      </div>
    </div>
    <div class="panel" style="margin-top:16px;">
      <h2>Retrieved Context</h2>
      <div id="retrieval" style="max-height:220px; overflow-y:auto; font-size:12px;">
        <div id="ret-empty" style="color:#8b949e; font-style:italic;">No retrievals yet</div>
      </div>
    </div>
  </div>
  <div>
    <div class="panel">
      <h2>Latency</h2>
      <div id="metrics">
        <div class="metric-row"><span class="label">STT</span><span class="value" id="m-stt">--</span></div>
        <div class="metric-row"><span class="label">Retrieval</span><span class="value" id="m-retrieval">--</span></div>
        <div class="metric-row"><span class="label">LLM 1st token</span><span class="value" id="m-llm-ft">--</span></div>
        <div class="metric-row"><span class="label">LLM total</span><span class="value" id="m-llm">--</span></div>
        <div class="metric-row"><span class="label">TTS</span><span class="value" id="m-tts">--</span></div>
        <div class="metric-row"><span class="label">End-to-end</span><span class="value" id="m-e2e">--</span></div>
        <div class="metric-row"><span class="label">Turns</span><span class="value" id="m-turns">--</span></div>
      </div>
    </div>
    <div class="panel" style="margin-top:16px">
      <h2>Session Log</h2>
      <div id="log-viewer" style="max-height:200px; overflow-y:auto; font-size:12px; color:#8b949e;">
        <em>No events yet</em>
      </div>
    </div>
    <div class="panel" style="margin-top:16px">
      <h2>Vision</h2>
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
    </div>

  </div>
</div>

<div id="status-bar"></div>

<script>
const SERVICE_ORDER = ["postgresql", "ollama", "gptoss120b", "qwen36", "conversation_llm", "whisper", "little_timmy"];
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
  div.innerHTML = '<div class="role">' + label + '</div><div class="content">' + escapeHtml(content) + '</div>';
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
  const bar = document.getElementById("status-bar");
  const div = document.createElement("div");
  div.className = "status-entry " + (msg.level || "info");
  const ts = msg.timestamp ? msg.timestamp.split("T")[1].split(".")[0] : "";
  div.innerHTML = '<span class="ts">' + ts + "</span>" + (msg.message || "");
  bar.prepend(div);
  while (bar.children.length > 50) bar.removeChild(bar.lastChild);
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

      const scoreItems = [
        '<span style="color:' + (r.novelty > 0.5 ? '#d29922' : '#484f58') + '">Novelty: ' + r.novelty.toFixed(2) + '</span>',
        '<span style="color:' + (r.humor_potential > 0.5 ? '#e94560' : '#484f58') + '">Humor: ' + r.humor_potential.toFixed(2) + '</span>',
        '<span style="color:' + (r.store_as_memory ? '#3fb950' : '#484f58') + '">Store: ' + (r.store_as_memory ? 'yes' : 'no') + '</span>',
        '<span style="color:' + (r.speak_now ? '#f85149' : '#484f58') + '">Speak: ' + (r.speak_now ? 'YES' : 'no') + '</span>',
      ];
      scores.innerHTML = scoreItems.join(' &middot; ');

      if (r.memory_tags && r.memory_tags.length) {
        scores.innerHTML += '<br><span style="color:#8b5cf6">Tags: ' + r.memory_tags.join(', ') + '</span>';
      }
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

loadConversationHistory();
pollMetrics();
metricsInterval = setInterval(pollMetrics, 30000);
pollFaceTracking();
setInterval(pollFaceTracking, 10000);
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


</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT, log_level="info")
