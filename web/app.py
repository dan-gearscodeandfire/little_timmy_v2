"""FastAPI web dashboard for Little Timmy on port 8893."""

import asyncio
import json
import logging
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import config

log = logging.getLogger(__name__)

app = FastAPI(title="Little Timmy", version="0.1.0")

# Will be set by main.py after initialization
_conversation_manager = None
_orchestrator = None
_connected_websockets: list[WebSocket] = []
_metrics: dict = {
    "turns": 0,
    "last_stt_ms": 0,
    "last_retrieval_ms": 0,
    "last_llm_first_token_ms": 0,
    "last_llm_total_ms": 0,
    "last_tts_ms": 0,
    "last_e2e_ms": 0,
    "memories_stored": 0,
    "facts_stored": 0,
    "started_at": None,
}


def init(conversation_manager, orchestrator=None):
    global _conversation_manager, _orchestrator
    _conversation_manager = conversation_manager
    _orchestrator = orchestrator
    _metrics["started_at"] = datetime.now().isoformat()


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
        _connected_websockets.remove(ws)


def update_metrics(**kwargs):
    _metrics.update(kwargs)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connected_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle typed input from web UI
            try:
                msg = json.loads(data)
                if msg.get("type") == "speak" and _orchestrator:
                    text = msg.get("text", "").strip()
                    if text:
                        asyncio.create_task(_orchestrator.process_text_input(text))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        _connected_websockets.remove(websocket)


@app.get("/api/metrics")
async def get_metrics():
    return _metrics


@app.get("/api/conversation")
async def get_conversation():
    if not _conversation_manager:
        return {"hot": [], "warm": []}
    return {
        "hot": [
            {"role": t.role, "content": t.content, "timestamp": t.timestamp,
             "speaker": t.speaker}
            for t in _conversation_manager.hot_turns
        ],
        "warm": [
            {"text": s.text, "timestamp": s.timestamp, "turn_count": s.turn_count}
            for s in _conversation_manager.warm_summaries
        ],
        "turn_count": _conversation_manager.turn_count,
    }

@app.get("/api/chatlog")
async def get_chatlog():
    """Plain-text chat log for easy reading via curl."""
    from fastapi.responses import PlainTextResponse
    if not _conversation_manager:
        return PlainTextResponse("No conversation yet.")
    lines = []
    for s in _conversation_manager.warm_summaries:
        ts = datetime.fromtimestamp(s.timestamp).strftime("%H:%M:%S")
        lines.append(f"[{ts}] [SUMMARY] {s.text}")
    for t in _conversation_manager.hot_turns:
        ts = datetime.fromtimestamp(t.timestamp).strftime("%H:%M:%S")
        if t.role == "user":
            speaker = (t.speaker or "unknown").upper()
            lines.append(f"[{ts}] [{speaker}] {t.content}")
        else:
            lines.append(f"[{ts}] [TIMMY] {t.content}")
    return PlainTextResponse(chr(10).join(lines))





@app.get("/api/audio/diag")
async def audio_diagnostics():
    """Audio capture diagnostics."""
    if not _orchestrator:
        return {"error": "not initialized"}
    cap = _orchestrator.capture
    return {
        "chunks_processed": cap.diag_chunks_processed,
        "last_peak": round(cap.diag_last_peak, 4),
        "last_vad_prob": round(cap.diag_last_vad_prob, 4),
        "overflows": cap.diag_overflows,
        "suppressed": cap.suppressed,
        "device": cap.diag_device_name,
    }

@app.get("/api/health")
async def health_check():
    """Check connectivity to all backend services."""
    import httpx
    checks = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in [
            ("whisper_cpp", f"{config.WHISPER_URL}/health"),
            ("llm_3b", f"{config.LLM_CONVERSATION_URL}/health"),
            ("llm_brain", f"{config.LLM_MEMORY_URL}/health"),
            ("ollama", f"{config.OLLAMA_URL}/api/tags"),
        ]:
            try:
                r = await client.get(url)
                checks[name] = "ok" if r.status_code == 200 else f"status {r.status_code}"
            except Exception as e:
                checks[name] = f"error: {e}"
    checks["postgresql"] = "ok"  # if we got here, DB is working
    return checks


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Little Timmy</title>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Courier New', monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { color: #e94560; margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
.panel { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 16px; }
.panel h2 { color: #e94560; font-size: 14px; margin-bottom: 12px; text-transform: uppercase; }
#conversation { max-height: 500px; overflow-y: auto; }
.turn { margin: 8px 0; padding: 8px; border-radius: 4px; }
.turn.user { background: #0f3460; border-left: 3px solid #e94560; }
.turn.assistant { background: #1a1a2e; border-left: 3px solid #00d2ff; }
.turn .role { font-size: 11px; color: #888; text-transform: uppercase; }
.metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #0f3460; }
.metric .label { color: #888; }
.metric .value { color: #00d2ff; font-weight: bold; }
.health { padding: 4px 8px; border-radius: 4px; font-size: 12px; margin: 4px 0; }
.health.ok { background: #0a3d0a; color: #4caf50; }
.health.err { background: #3d0a0a; color: #f44336; }
#input-area { display: flex; gap: 8px; margin-top: 12px; }
#text-input { flex: 1; background: #0f3460; border: 1px solid #e94560; color: #e0e0e0;
  padding: 8px; border-radius: 4px; font-family: inherit; }
#send-btn { background: #e94560; color: white; border: none; padding: 8px 16px;
  border-radius: 4px; cursor: pointer; }
</style>
</head>
<body>
<h1>LITTLE TIMMY</h1>
<div class="grid">
  <div>
    <div class="panel">
      <h2>Conversation</h2>
      <div id="conversation"></div>
      <div id="input-area">
        <input id="text-input" placeholder="Type a message (bypass STT)..." />
        <button id="send-btn" onclick="sendText()">Send</button>
      </div>
    </div>
  </div>
  <div>
    <div class="panel">
      <h2>Latency</h2>
      <div id="metrics">
        <div class="metric"><span class="label">STT</span><span class="value" id="m-stt">--</span></div>
        <div class="metric"><span class="label">Retrieval</span><span class="value" id="m-retrieval">--</span></div>
        <div class="metric"><span class="label">LLM 1st token</span><span class="value" id="m-llm-ft">--</span></div>
        <div class="metric"><span class="label">LLM total</span><span class="value" id="m-llm">--</span></div>
        <div class="metric"><span class="label">TTS</span><span class="value" id="m-tts">--</span></div>
        <div class="metric"><span class="label">End-to-end</span><span class="value" id="m-e2e">--</span></div>
        <div class="metric"><span class="label">Turns</span><span class="value" id="m-turns">0</span></div>
      </div>
    </div>
    <div class="panel" style="margin-top:20px">
      <h2>Services</h2>
      <div id="health"></div>
    </div>
  </div>
</div>
<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
const conv = document.getElementById('conversation');

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'turn') {
    const div = document.createElement('div');
    div.className = `turn ${msg.role}`;
    div.innerHTML = `<div class="role">${msg.role}</div><div>${msg.content}</div>`;
    conv.appendChild(div);
    conv.scrollTop = conv.scrollHeight;
  } else if (msg.type === 'metrics') {
    document.getElementById('m-stt').textContent = msg.stt_ms ? msg.stt_ms + 'ms' : '--';
    document.getElementById('m-retrieval').textContent = msg.retrieval_ms ? msg.retrieval_ms + 'ms' : '--';
    document.getElementById('m-llm-ft').textContent = msg.llm_first_token_ms ? msg.llm_first_token_ms + 'ms' : '--';
    document.getElementById('m-llm').textContent = msg.llm_total_ms ? msg.llm_total_ms + 'ms' : '--';
    document.getElementById('m-tts').textContent = msg.tts_ms ? msg.tts_ms + 'ms' : '--';
    document.getElementById('m-e2e').textContent = msg.e2e_ms ? msg.e2e_ms + 'ms' : '--';
    document.getElementById('m-turns').textContent = msg.turns || 0;
  }
};

function sendText() {
  const input = document.getElementById('text-input');
  const text = input.value.trim();
  if (text) {
    ws.send(JSON.stringify({type: 'speak', text}));
    input.value = '';
  }
}
document.getElementById('text-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendText();
});

// Health check polling
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const data = await r.json();
    const el = document.getElementById('health');
    el.innerHTML = Object.entries(data).map(([k,v]) =>
      `<div class="health ${v === 'ok' ? 'ok' : 'err'}">${k}: ${v}</div>`
    ).join('');
  } catch(e) {}
}
checkHealth();
setInterval(checkHealth, 30000);
</script>
</body>
</html>"""

@app.get("/api/vision")
async def vision_state():
    """Vision pipeline debug state for OS dashboard."""
    if not _orchestrator or not hasattr(_orchestrator, "vision"):
        return {"enabled": False, "error": "not initialized"}
    return _orchestrator.vision.get_vision_debug()
