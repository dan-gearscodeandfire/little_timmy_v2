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


@app.get("/api/last_payload")
async def get_last_payload_route():
    """Most recent (history + ephemeral system + user) payload sent to the LLM.
    Window into how build_ephemeral_block assembled context for the last turn.
    """
    from llm.prompt_builder import get_last_payload
    p = get_last_payload()
    if not p:
        return {"available": False}
    return {"available": True, **p}


@app.get("/api/mood")
async def get_mood():
    """Current 2-axis mood state.
    X: -1 BORED, 0 NEUTRAL, +1 SLIGHTLY_INTERESTED.
    Y: -1 MEAN, 0 NEUTRAL, +1 BEGRUDGINGLY_NICE.
    """
    from persona.state import get as _mood_get
    from persona.render import render as _render_mood
    s = _mood_get()
    return {
        "x": s.x, "y": s.y,
        "last_update_ts": s.last_update_ts,
        "last_x_signal": s.last_x_signal,
        "last_y_signal": s.last_y_signal,
        "x_signals": list(s.x_signals),
        "y_signals": list(s.y_signals),
        "rendered": _render_mood(s),
    }


@app.get("/api/feedback")
async def get_feedback(since: float | None = None, limit: int = 200):
    """Return meta-feedback events captured by feedback.detector.

    Query params:
      since: float unix-ts; only events with ts > since returned.
      limit: max events (default 200, newest last).
    """
    from feedback.storage import read_events
    try:
        events = read_events(since_ts=since, limit=limit)
    except Exception as e:
        log.warning('feedback read error: %s', e)
        events = []
    return {'count': len(events), 'events': events}


@app.post("/api/feedback/manual_flag")
async def manual_flag(payload: dict | None = None):
    """User clicked the "Flag last response" button (or curl-equivalent).
    Reads the orchestrator's snapshot of the most recent finalized turn
    and dual-writes:
      - persona_tuning/example_neg_<ts>.json   (LoRA negative example)
      - feedback_inbox.jsonl                   (vault-poller inbox)
    Both carry source="ui_button" so consumers can distinguish from the
    verbal_meta_feedback path.
    """
    import time as _time
    from feedback.storage import (append_event, write_persona_tuning_negative, write_persona_tuning_positive, append_flagged)
    if not _orchestrator or not getattr(_orchestrator, "_last_finalized_turn", None):
        return {"ok": False, "error": "no recent finalized turn to flag"}
    snap = _orchestrator._last_finalized_turn
    body = payload or {}
    reason = body.get("reason") or ""
    kind = body.get("kind", "bad")
    if kind not in ("good", "bad"):
        return {"ok": False, "error": f"invalid kind: {kind!r} (use good or bad)"}
    ts = _time.time()
    inbox_entry = {
        "ts": ts,
        "speaker": snap.get("speaker_name"),
        "feedback_text": reason or f"(ui_button {kind}; no reason text)",
        "current_assistant": "",
        "prev_user": snap.get("user_text", ""),
        "prev_assistant": snap.get("assistant_response", ""),
        "keyword_score": -1,
        "llm_confirmed": False,
        "source": "ui_button",
        "kind": kind,
        "system_prompt": snap.get("ephemeral", ""),
    }
    persona_entry = {
        "timestamp": ts,
        "penultimate_user": snap.get("user_text", ""),
        "system_prompt": snap.get("ephemeral", ""),
        "response": snap.get("assistant_response", ""),
        "source": "ui_button",
        "kind": kind,
    }
    if kind == "good":
        persona_entry["compliment"] = reason
    else:
        persona_entry["flag_reason"] = reason
    try:
        event_id = append_event(inbox_entry)
        if kind == "good":
            persona_path = write_persona_tuning_positive(persona_entry)
        else:
            persona_path = write_persona_tuning_negative(persona_entry)
        conversation_history = snap.get("conversation_history") or []
        append_flagged(kind, {
            "ts": ts,
            "source": "ui_button",
            "speaker": snap.get("speaker_name"),
            "user_prompt": snap.get("user_text", ""),
            "response": snap.get("assistant_response", ""),
            "comment": reason,
            "system_prompt": snap.get("ephemeral", ""),
            "conversation_history": conversation_history,
            "persona_tuning_file": persona_path.name,
        })
        log.info("[FEEDBACK] manual %s id=%s persona=%s reason=%r",
                 kind, event_id, persona_path.name, reason[:80])
        return {"ok": True, "kind": kind, "event_id": event_id,
                "persona_tuning_file": persona_path.name,
                "flagged_user": snap.get("user_text", ""),
                "flagged_assistant": snap.get("assistant_response", ""),
                "system_prompt": snap.get("ephemeral", ""),
                "conversation_history": conversation_history,
                "comment": reason,
                "speaker": snap.get("speaker_name")}
    except Exception as e:
        log.warning("manual %s persist error: %s", kind, e)
        return {"ok": False, "error": str(e)}


@app.get("/api/feedback/last_flag")
async def get_last_flag():
    """Return the most recent flagged.jsonl entry verbatim (or
    {available: False} if none yet). Backs the dashboard's "Last flag"
    review modal so the user can re-inspect an older flag without
    needing to re-click 👍/👎."""
    from feedback.storage import read_last_flagged
    try:
        entry = read_last_flagged()
    except Exception as e:
        log.warning("last_flag read error: %s", e)
        return {"available": False, "error": str(e)[:120]}
    if entry is None:
        return {"available": False}
    return {"available": True, **entry}


@app.post("/api/speaker/reenroll")
async def speaker_reenroll(payload: dict | None = None):
    """Open a re-enrollment window for a known speaker.

    Body may include {"name": "<speaker>", "duration_s": 60}. If name is
    omitted, falls back to whoever just spoke (the speaker_name field on
    the orchestrator's _last_finalized_turn snapshot). duration_s clamped
    to [10, 300] seconds.
    """
    if not _orchestrator or not getattr(_orchestrator, "speaker_id_module", None):
        return {"ok": False, "error": "orchestrator not ready"}
    sm = _orchestrator.speaker_id_module
    body = payload or {}
    name = (body.get("name") or "").strip().lower()
    if not name:
        snap = getattr(_orchestrator, "_last_finalized_turn", None)
        if snap:
            name = (snap.get("speaker_name") or "").strip().lower()
    if not name:
        return {"ok": False, "error": "no name and no recent speaker to fall back to"}
    duration = float(body.get("duration_s") or 60.0)
    duration = max(10.0, min(duration, 300.0))
    if not any(ks.name == name for ks in sm._known_speakers):
        return {"ok": False, "error": f"{name!r} is not a known speaker"}
    ok = sm.start_reenrollment(name, duration_s=duration)
    if not ok:
        return {"ok": False, "error": "start_reenrollment refused",
                "active_reenrollment": sm.get_active_reenrollment()}
    log.info("[T2] re-enrollment opened from API for %s (%.0fs)", name, duration)
    return {"ok": True, "name": name, "duration_s": duration,
            "active_reenrollment": sm.get_active_reenrollment()}


@app.get("/api/speaker/status")
async def speaker_status():
    """Speaker-ID state: known speakers + active re-enrollment + drift buffers."""
    if not _orchestrator or not getattr(_orchestrator, "speaker_id_module", None):
        return {"ok": False, "error": "orchestrator/speaker module not ready"}
    sm = _orchestrator.speaker_id_module
    return {
        "ok": True,
        "known": [
            {"name": ks.name, "speaker_id": ks.speaker_id}
            for ks in sm._known_speakers
        ],
        "unknown": [
            {"temp_id": us.temp_id, "name": us.name,
             "utterance_count": us.utterance_count}
            for us in sm._unknown_speakers
        ],
        "active_reenrollment": sm.get_active_reenrollment(),
        "drift_buffers": {n: len(buf) for n, buf in sm._drift_buffers.items()},
    }


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


@app.get("/api/presence")
async def presence_state():
    """Room ledger: who is present (visible or recently heard)."""
    if not _orchestrator or not hasattr(_orchestrator, "room_ledger") or _orchestrator.room_ledger is None:
        return {"now": None, "present": [], "unknown_voices_recent": 0, "enabled": False}
    state = _orchestrator.room_ledger.current_state()
    state["enabled"] = True
    return state


@app.get("/api/vision/auto_poll")
async def get_vision_auto_poll():
    """Read the periodic-poll-loop toggle. Event-driven trigger_capture
    calls (speech, visual question) are not affected by this flag."""
    if not _orchestrator or not hasattr(_orchestrator, "vision"):
        return {"enabled": False, "error": "not initialized"}
    return {"enabled": bool(_orchestrator.vision.is_auto_poll_enabled)}


@app.post("/api/vision/auto_poll")
async def set_vision_auto_poll(payload: dict | None = None):
    """Enable or disable the 1fps VLM poll loop."""
    if not _orchestrator or not hasattr(_orchestrator, "vision"):
        return {"ok": False, "error": "orchestrator not ready"}
    enabled = bool((payload or {}).get("enabled", True))
    _orchestrator.vision.set_auto_poll(enabled)
    return {"ok": True, "enabled": bool(_orchestrator.vision.is_auto_poll_enabled)}


@app.get("/api/hearing")
async def get_hearing():
    """Read the mic-mute toggle. whisper-server stays untouched; this only
    gates whether captured speech segments get enqueued for STT."""
    if not _orchestrator or not getattr(_orchestrator, "capture", None):
        return {"enabled": False, "muted": True, "error": "not initialized"}
    cap = _orchestrator.capture
    return {"enabled": bool(cap.is_hearing_enabled), "muted": bool(cap.hearing_muted)}


@app.post("/api/hearing")
async def set_hearing(payload: dict | None = None):
    """Mute or unmute Little Timmy's hearing."""
    if not _orchestrator or not getattr(_orchestrator, "capture", None):
        return {"ok": False, "error": "orchestrator not ready"}
    cap = _orchestrator.capture
    enabled = bool((payload or {}).get("enabled", True))
    cap.set_hearing(enabled)
    return {"ok": True, "enabled": bool(cap.is_hearing_enabled), "muted": bool(cap.hearing_muted)}
