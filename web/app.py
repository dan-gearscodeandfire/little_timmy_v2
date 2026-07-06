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
    # User-perceived reply latency (2026-06-25). reply_lag: from when the user
    # stopped talking (incl. the endpointing-silence delay) to Timmy's first
    # reply audio -- the "how responsive does it feel" number for the Booth.
    # speech_to_reply: full span from speech onset (incl. utterance duration).
    "last_reply_lag_ms": None,
    "last_speech_to_reply_ms": None,
    "memories_stored": 0,
    "facts_stored": 0,
    "started_at": None,
    # First-pass tool-call classifier surface (2026-06-18). last_tool_call_ts
    # advancing is the signal the Booth poll uses to flash the tool-fired badge.
    "last_tool_call": None,
    "last_tool_call_ts": 0,
    # Tier-1 route latency (ms) of the :8092 first-pass filter, per turn. Shown
    # as its own HUD row above payload->LLM on the Booth + LT-OS.
    "last_classifier_ms": None,
    # Coreference-resolution latency (ms) of the :8092 resolve call, set only on
    # turns where the deixis gate fired AND resolution is enabled. None until the
    # first resolved turn. Shown as its own HUD row on the LT-OS.
    "last_resolution_ms": None,
    # Classifier latency split (2026-06-20): Tier-1 route vs Tier-2 arg-extraction.
    # last_classifier_ms (above) stays = total classifier wall on the turn (route,
    # or route+args on a hit) for back-compat with the existing HUD row. These two
    # break it down so "why was the classifier slow" is answerable per turn.
    "last_classifier_route_ms": None,
    "last_classifier_args_ms": None,
    # E2E wall of a TOOL-routed turn (classifier+args+store+ack TTS, early-return).
    # The conversation-path e2e (last_e2e_ms) never covers these since the brain
    # pipeline is skipped; this is the only place a routed turn's total surfaces.
    "last_tool_turn_e2e_ms": None,
}

# --- Rolling latency stats (in-memory, reset on restart) ---------------------
# The _metrics dict above answers "what did the LAST turn cost?". These answer
# "what's the DISTRIBUTION under each condition?" -- the question point samples
# can't. Bounded ring buffers per series; percentiles computed on read. Series
# are keyed "<bucket>:<stage>" so the same stage can be sliced by turn class:
#   conversation:*  full-brain turns (banter / normal reply)
#   tool:*          classifier-routed early-return turns (e.g. store_fact)
#   stage:*         cross-cutting stages that run regardless of turn class
#                   (classifier_route every turn; classifier_args / resolution
#                    only when they fire -- so their series .n == fire count)
#   all:*           cross-class aggregate of a per-turn stage
from collections import deque, defaultdict

_STATS_WINDOW = 300  # samples retained per series (~ last 300 of each)
_stats: dict = defaultdict(lambda: deque(maxlen=_STATS_WINDOW))
_stats_counts: dict = defaultdict(int)  # turn/condition tallies for the "mix"


def _percentile(vals: list, p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def _series_summary(vals: list) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 1),
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
    }


def record_stage(series_key: str, ms) -> None:
    """Append one cross-cutting stage latency (e.g. classifier_route, resolution).
    Skips None / non-positive so a stage that didn't run never dilutes its own
    distribution. series_key is stored verbatim (caller adds the 'stage:' prefix)."""
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return
    if ms > 0:
        _stats[series_key].append(ms)


def record_turn_stats(turn_class: str, stages: dict, flags: dict | None = None) -> None:
    """Append one turn's per-stage latencies to its class bucket + the cross-class
    aggregate. turn_class: "conversation" | "tool". stages: {name: ms|None}.
    flags: truthy condition tags -> bumped in _stats_counts for the mix readout."""
    _stats_counts["turns"] += 1
    _stats_counts[f"class:{turn_class}"] += 1
    for name, ms in stages.items():
        try:
            ms = float(ms) if ms is not None else None
        except (TypeError, ValueError):
            continue
        if ms is None or ms <= 0:
            continue
        _stats[f"{turn_class}:{name}"].append(ms)
        _stats[f"all:{name}"].append(ms)
    for k, v in (flags or {}).items():
        if v:
            _stats_counts[f"flag:{k}"] += 1


def latency_stats_snapshot() -> dict:
    """Serializable summary of every series + the condition mix. Used by the
    /api/latency_stats endpoint and the periodic [PERF-AGG] log line."""
    return {
        "window": _STATS_WINDOW,
        "since": _metrics.get("started_at"),
        "counts": dict(_stats_counts),
        "series": {k: _series_summary(list(v)) for k, v in _stats.items() if v},
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
    # Merge the live classifier toggle so the Booth poll (which only reads
    # /api/metrics) can show the classifier-ON pip without a second request.
    from persistence import runtime_toggles
    return {**_metrics,
            "classifier_enabled": bool(runtime_toggles.get("classifier_enabled")),
            "query_resolution_enabled": bool(runtime_toggles.get("query_resolution_enabled")),
            "speculative_coref_enabled": bool(runtime_toggles.get("speculative_coref_enabled"))}


@app.get("/api/latency_stats")
async def get_latency_stats():
    """Rolling latency distributions (count/mean/p50/p95/min/max) per stage,
    sliced by turn class (conversation vs tool-routed) plus cross-cutting stages.
    Resets on restart; window = last _STATS_WINDOW samples per series."""
    return latency_stats_snapshot()


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


@app.post("/api/announce")
async def announce(payload: dict | None = None):
    """Speak operational text out of Timmy's speaker WITHOUT touching
    conversation history or STT. Used by Claude (the "supervisor"/"couple's
    therapist") to talk to Dan through Timmy so he isn't tied to the CC window.

    - Does NOT append a turn or broadcast a "turn" event, so it stays out of
      /api/chatlog, /api/conversation, and the supervisor WS feed.
    - tts.speak() queues PCM and sets capture.suppressed during playback, so
      Timmy never hears it via STT (no loopback turn) — regardless of voice.

    Voice (2026-06-28, Dan): the supervisor/couples-therapist channel speaks in
    its OWN voice (Piper en_US-kristin-medium, models/tts/personas/), audibly
    NOT Timmy and NOT Dan. That persona voice is the DEFAULT here, since this
    endpoint's purpose IS that channel. Mic-gating is unchanged (the engine
    suppresses capture during any playback), so Timmy still never hears it.

    Body: {"text": "...", "voice": "couples_therapist"|"timmy", "no_prefix": false}
    - voice: "timmy"/"skeletor" → Timmy's conversational voice (for tool voice,
      e.g. auto-calibrate prompts). Default/"couples_therapist"/"therapist" →
      the persona voice.
    - no_prefix/raw: skip the spoken self-identification prefix. Default false:
      therapist voice prefixes "This is your couples therapist."; timmy voice
      prefixes "This is Claude talking."
    """
    import os
    from fastapi.responses import JSONResponse
    if _orchestrator is None or getattr(_orchestrator, "tts", None) is None:
        return JSONResponse({"spoken": False, "error": "tts_unavailable"}, status_code=503)
    body = payload or {}
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"spoken": False, "error": "empty_text"}, status_code=400)

    voice = (body.get("voice") or "couples_therapist").strip().lower()
    use_timmy = voice in ("timmy", "skeletor", "default")
    if use_timmy:
        voice_model = None  # engine falls back to Timmy's conversational voice
        prefix, self_id = "This is Claude talking. ", "this is claude"
    else:
        voice_model = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "tts", "personas", "couples_therapist.onnx")
        prefix, self_id = "This is your couples therapist. ", "this is your couples therapist"

    no_prefix = bool(body.get("no_prefix") or body.get("raw"))
    if no_prefix or text.lower().startswith(self_id):
        spoken_text = text
    else:
        spoken_text = f"{prefix}{text}"
    # inject (2026-06-28, Dan): direct turn-injection. When true AND the persona
    # voice is used, the spoken text is ALSO injected into the conversation as a
    # turn from speaker 'couples_therapist' (id 6) so Timmy PROCESSES and responds
    # — bypassing the mic / VAD / close-talk addressee gate that otherwise drops
    # the (too-quiet, off-mic) room-speaker persona. This is the reliable way to
    # have the therapist address Timmy. Playback stays mic-gated (no acoustic
    # loopback) so Dan hears the persona while injection drives the processing.
    #
    # DEFAULT (no inject): the therapist is gated — spoken for Dan's ears, NOT
    # heard or processed by Timmy (Dan: "more just information for me").
    #
    # let_timmy_hear: legacy ACOUSTIC path (mic open); unreliable on this box
    # because of the close-talk gate. Prefer `inject`.
    inject = bool(body.get("inject")) and not use_timmy
    let_timmy_hear = bool(body.get("let_timmy_hear") or body.get("hear"))
    # force=True: the supervisor channel bypasses the mouth-mute (tts_muted) so
    # Claude can still talk to Dan while Timmy's conversational voice is muted.
    await _orchestrator.tts.speak(spoken_text, force=True, voice_model=voice_model,
                                  suppress_mic=not let_timmy_hear)
    if inject:
        import asyncio as _asyncio
        # Inject the bare text (no self-ID prefix) so Timmy sees the therapist's
        # actual words. Fire-and-forget: queues behind the gated playback.
        _asyncio.create_task(_orchestrator.inject_couples_therapist_turn(text))
    return {"spoken": True, "text": spoken_text,
            "voice": "timmy" if use_timmy else "couples_therapist",
            "heard_by_timmy": let_timmy_hear, "injected": inject}


@app.get("/api/tts_mute")
async def get_tts_mute():
    """Read the mouth-mute toggle (Timmy's conversational voice + fillers)."""
    from persistence import runtime_toggles
    return {"muted": bool(runtime_toggles.get("tts_muted"))}


@app.post("/api/tts_mute")
async def set_tts_mute(payload: dict | None = None):
    """Mute/unmute Timmy's mouth live. When muted, replies + fillers are
    silenced (mic stays open, matcher keeps running); /api/announce still
    speaks. Persisted by runtime_toggles, read live by the TTS engine — no
    restart."""
    from persistence import runtime_toggles
    muted = bool((payload or {}).get("muted", True))
    runtime_toggles.set("tts_muted", muted)
    return {"ok": True, "muted": bool(runtime_toggles.get("tts_muted"))}


@app.get("/api/guest_mode")
async def get_guest_mode():
    """Read the privacy/guest gate. When on, facts classified sensitive
    (memory/pii.py: contact, location, financial, health/credentials,
    family_minor) are withheld from prompt injection so Timmy can't speak them
    via TTS in front of guests."""
    from persistence import runtime_toggles
    return {"guest_mode": bool(runtime_toggles.get("guest_mode"))}


@app.post("/api/guest_mode")
async def set_guest_mode(payload: dict | None = None):
    """Flip the privacy/guest gate live (persisted by runtime_toggles, read per
    turn in conversation/turn.py — no restart). Manual toggle wins; a
    presence-driven auto layer may OR in later."""
    from persistence import runtime_toggles
    on = bool((payload or {}).get("guest_mode", True))
    runtime_toggles.set("guest_mode", on)
    return {"ok": True, "guest_mode": bool(runtime_toggles.get("guest_mode"))}


def _face_recognition_state() -> dict:
    """Current okDemerzel face-recognition runtime knobs (all live-tunable)."""
    from persistence import runtime_toggles
    from presence.face_thresholds import KNOWN_FACE_THRESHOLD
    authority = runtime_toggles.get("face_authority") or "pi"
    return {
        "authority": authority,                       # "pi" | "okdemerzel"
        "okdemerzel": authority == "okdemerzel",      # convenience bool for a switch
        "shadow": bool(runtime_toggles.get("face_shadow_enabled")),
        "threshold": float(runtime_toggles.get("face_threshold") or KNOWN_FACE_THRESHOLD),
        "frames": int(runtime_toggles.get("face_authority_frames") or 3),
    }


@app.get("/api/face_recognition")
async def get_face_recognition():
    """Read the okDemerzel face-recognition knobs (authority / shadow / accept
    threshold / frames-per-turn). All persisted by runtime_toggles and read live
    — no restart to change."""
    return _face_recognition_state()


@app.post("/api/face_recognition")
async def set_face_recognition(payload: dict | None = None):
    """Set any subset of the face-recognition knobs live. Accepts:
      okdemerzel: bool   -> face_authority "okdemerzel"/"pi"
      authority:  str    -> face_authority (explicit)
      shadow:     bool   -> face_shadow_enabled
      threshold:  float  -> face_threshold (clamped 0.30-0.70)
      frames:     int    -> face_authority_frames (clamped 1-5)
    """
    from persistence import runtime_toggles
    p = payload or {}
    if "okdemerzel" in p:
        runtime_toggles.set("face_authority", "okdemerzel" if p["okdemerzel"] else "pi")
    if "authority" in p and p["authority"] in ("pi", "okdemerzel"):
        runtime_toggles.set("face_authority", p["authority"])
    if "shadow" in p:
        runtime_toggles.set("face_shadow_enabled", bool(p["shadow"]))
    if "threshold" in p:
        try:
            t = max(0.30, min(0.70, float(p["threshold"])))
            runtime_toggles.set("face_threshold", t)
        except (TypeError, ValueError):
            pass
    if "frames" in p:
        try:
            f = max(1, min(5, int(p["frames"])))
            runtime_toggles.set("face_authority_frames", f)
        except (TypeError, ValueError):
            pass
    return {"ok": True, **_face_recognition_state()}


# --- Memory Inspector (read-only) -------------------------------------------
# Backs the LT-OS /memory page. LT-OS proxies these exactly like /api/timmy/*.
# Read-only by design: no write/supersede/delete endpoints. Sensitive facts ARE
# returned (this is a local admin tool); the UI badges them. guest_mode gating
# is a prompt-injection concern and intentionally does NOT apply here.

@app.get("/api/memory/stats")
async def get_memory_stats():
    """Summary counts for the inspector header (facts active/total/sensitive,
    episodes, speakers)."""
    from fastapi.responses import JSONResponse
    from memory import facts as _facts
    try:
        return await _facts.inspector_counts()
    except Exception as e:
        log.warning("memory_stats failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/memory/facts")
async def get_memory_facts(q: str | None = None, include_superseded: bool = False,
                           sort: str = "learned_at", limit: int = 500):
    """List facts for the inspector. q=substring (subject/predicate/value),
    include_superseded toggles the audit trail, sort in
    {learned_at,confidence,subject}. learned_at serialized to ISO."""
    from fastapi.responses import JSONResponse
    from memory import facts as _facts
    try:
        rows = await _facts.list_facts(q=q, include_superseded=include_superseded,
                                       sort=sort, limit=min(int(limit), 2000))
        for r in rows:
            la = r.get("learned_at")
            if la is not None and hasattr(la, "isoformat"):
                r["learned_at"] = la.isoformat()
        return {"facts": rows, "count": len(rows)}
    except Exception as e:
        log.warning("memory_facts failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/memory/episodes")
async def get_memory_episodes(start: str | None = None, end: str | None = None,
                              limit: int = 200):
    """List episodes for the inspector timeline. Optional start/end (ISO date or
    datetime) filter to the overlapping window; otherwise the whole timeline,
    newest-first. Timestamps serialized to ISO."""
    from fastapi.responses import JSONResponse
    from datetime import datetime, timezone
    from memory import manager as _mgr
    def _parse(s):
        if not s:
            return None
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    try:
        s_dt, e_dt = _parse(start), _parse(end)
        rows = await _mgr.list_episodes(start=s_dt, end=e_dt, limit=min(int(limit), 1000))
        for r in rows:
            for k in ("span_start", "span_end", "created_at", "accessed_at"):
                v = r.get(k)
                if v is not None and hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
        return {"episodes": rows, "count": len(rows)}
    except Exception as e:
        log.warning("memory_episodes failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/active")
async def get_active():
    """Whether Dan is actively conversing with Timmy right now: a stream is in
    flight, or it's been fewer than conversation_idle_gate_seconds since the
    last one. The shared signal background workers poll to back off the single
    :8083 brain slot -- consumed by the demerzel-mail ingest loop (defer email
    fetches while LT is talking) and mirrored in-process by the extraction /
    rollup idle gate. Read-only; tune the window via the
    conversation_idle_gate_seconds runtime toggle."""
    import time as _time
    from llm import client as _llm
    last = _llm._last_conversation_activity_ts
    idle = (_time.time() - last) if last > 0 else None
    return {
        "active": _llm.conversation_active(),
        "in_flight": _llm._conversation_in_flight.is_set(),
        "idle_seconds": round(idle, 2) if idle is not None else None,
        "gate_seconds": _llm._idle_gate_seconds(),
    }


@app.get("/api/conversation/idle_gate")
async def get_conversation_idle_gate():
    """Read the conversation_idle_gate_seconds runtime toggle (the 'conversation
    active' window the mail-ingest loop polls via /api/active to defer email
    while Dan is talking). Live-tunable via the LT-OS slider; 0 = only defer
    while a stream is literally in flight."""
    from persistence import runtime_toggles
    return {"conversation_idle_gate_seconds": float(
        runtime_toggles.get("conversation_idle_gate_seconds") or 0.0)}


@app.post("/api/conversation/idle_gate")
async def set_conversation_idle_gate(payload: dict | None = None):
    """Set conversation_idle_gate_seconds (clamped 0–300s, persisted by
    runtime_toggles, read live -- no restart). Float to match the default's
    type so the toggle-merge keeps it."""
    from persistence import runtime_toggles
    val = (payload or {}).get("conversation_idle_gate_seconds")
    if val is None:
        return {"ok": False, "error": "conversation_idle_gate_seconds required"}
    try:
        val = max(0.0, min(300.0, float(val)))
    except (TypeError, ValueError):
        return {"ok": False, "error": "conversation_idle_gate_seconds must be a number"}
    runtime_toggles.set("conversation_idle_gate_seconds", val)
    return {"ok": True, "conversation_idle_gate_seconds": float(
        runtime_toggles.get("conversation_idle_gate_seconds"))}


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
        "override": s.override,
        "last_update_ts": s.last_update_ts,
        "last_x_signal": s.last_x_signal,
        "last_y_signal": s.last_y_signal,
        "x_signals": list(s.x_signals),
        "y_signals": list(s.y_signals),
        "rendered": _render_mood(s),
    }


@app.post("/api/mood/override")
async def set_mood_override(payload: dict):
    """Manual mood override from the LT-OS dashboard.

    Body: {"enabled": true, "x": int, "y": int} to pin the mood, or
          {"enabled": false} to release and resume automatic drift.
    Returns the resulting {x, y, override, rendered}.
    """
    from persona import state as _mood_state
    from persona.render import render as _render_mood
    enabled = bool(payload.get("enabled", True))
    if enabled:
        try:
            x = int(payload["x"]); y = int(payload["y"])
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "enabled override requires integer x and y"}
        _mood_state.set_override(x, y)
    else:
        _mood_state.clear_override()
    s = _mood_state.get()
    return {"ok": True, "x": s.x, "y": s.y, "override": s.override,
            "rendered": _render_mood(s)}


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
        # 2026-05-15: positive feedback also saves the full LLM payload
        # for LoRA tuning. Pulled from the snapshot the orchestrator wrote
        # at finalize-time so it's still the exact prompt the LLM saw,
        # even if hot_turns have rolled since.
        if "messages" in snap:
            persona_entry["messages"] = snap["messages"]
        if "hyperparameters" in snap:
            persona_entry["hyperparameters"] = snap["hyperparameters"]
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
        "energy_floor": round(getattr(cap, "energy_floor", 0.0), 4),
        "overflows": cap.diag_overflows,
        "suppressed": cap.suppressed,
        "device": cap.diag_device_name,
    }


@app.get("/api/capture/energy_floor")
async def get_energy_floor():
    """Read the near-field onset energy floor (peak amplitude, 0.0..1.0).
    0.0 == disabled (VAD-only onset)."""
    if not _orchestrator or not getattr(_orchestrator, "capture", None):
        return {"energy_floor": 0.0, "error": "not initialized"}
    return {"energy_floor": round(getattr(_orchestrator.capture, "energy_floor", 0.0), 4)}


@app.post("/api/capture/energy_floor")
async def set_energy_floor(payload: dict | None = None):
    """Set the near-field onset energy floor. Takes effect on the next chunk
    and persists across restarts. Body: {"value": 0.06}."""
    if not _orchestrator or not getattr(_orchestrator, "capture", None):
        return {"ok": False, "error": "orchestrator not ready"}
    cap = _orchestrator.capture
    cap.set_energy_floor((payload or {}).get("value", 0.0))
    return {"ok": True, "energy_floor": round(cap.energy_floor, 4)}

@app.get("/api/health")
async def health_check():
    """Check connectivity to all backend services."""
    import httpx
    from llm import client as _llm
    checks = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in [
            ("whisper_cpp", f"{config.WHISPER_URL}/health"),
            ("llm_3b", f"{config.LLM_CONVERSATION_URL}/health"),
            # llm_brain = the live conversation brain (:8083 when qwen36 is
            # selected via override), NOT LLM_MEMORY_URL — extraction moved to
            # :8084 (2026-06-20), so check both separately or :8083 goes unseen.
            ("llm_brain", f"{_llm._current_conversation_url()}/health"),
            ("llm_memory", f"{config.LLM_MEMORY_URL}/health"),
            ("llm_classifier", f"{config.LLM_CLASSIFIER_URL}/health"),
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
    # Enrich each present person with a proper display name + creator flag so the
    # booth panel can render "William Osman" (not the "william_osman" slug) and
    # highlight recognized OpenSauce creators vs household.
    try:
        from presence.creators import display_name, is_creator
        for p in state.get("present", []):
            p["display_name"] = display_name(p.get("name", ""))
            p["is_creator"] = is_creator(p.get("name", ""))
    except Exception:
        pass
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


async def _classifier_up() -> bool:
    """Probe the :8092 classifier server's /health."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.LLM_CLASSIFIER_URL}/health")
            return r.status_code == 200
    except Exception:
        return False


@app.get("/api/classifier")
async def get_classifier():
    """Read the first-pass tool-call classifier gate + whether :8092 is reachable."""
    from persistence import runtime_toggles
    return {
        "enabled": bool(runtime_toggles.get("classifier_enabled")),
        "up": await _classifier_up(),
    }


@app.post("/api/classifier")
async def set_classifier(payload: dict | None = None):
    """Enable/disable the first-pass classifier. Persisted by runtime_toggles,
    read live per turn by conversation/tool_router.py -- no restart to flip."""
    from persistence import runtime_toggles
    enabled = bool((payload or {}).get("enabled", False))
    runtime_toggles.set("classifier_enabled", enabled)
    return {"ok": True, "enabled": bool(runtime_toggles.get("classifier_enabled")),
            "up": await _classifier_up()}


@app.get("/api/query_resolution")
async def get_query_resolution():
    """Read the elliptical-query coreference-resolution gate + whether the :8092
    server it shares with the classifier is reachable."""
    from persistence import runtime_toggles
    return {
        "enabled": bool(runtime_toggles.get("query_resolution_enabled")),
        "up": await _classifier_up(),  # same :8092 server as the classifier
    }


@app.post("/api/query_resolution")
async def set_query_resolution(payload: dict | None = None):
    """Enable/disable query resolution. Persisted by runtime_toggles, read live
    per retrieve() by memory/retrieval.py -- no restart to flip (once the code is
    loaded). When ON, a deictic/elliptical utterance is rewritten to a standalone
    query via :8092 before embedding; clean queries and a :8092 outage both fall
    back to the context blend, so flipping this ON is safe even if :8092 is down."""
    from persistence import runtime_toggles
    enabled = bool((payload or {}).get("enabled", False))
    runtime_toggles.set("query_resolution_enabled", enabled)
    return {"ok": True, "enabled": bool(runtime_toggles.get("query_resolution_enabled")),
            "up": await _classifier_up()}


@app.get("/api/speculative_coref")
async def get_speculative_coref():
    """Read the speculative-coref gate: when ON, the doorway resolves the query
    (:8093) in PARALLEL with the tool-call classifier (:8092) instead of inline
    after it. Gated internally by query_resolution_enabled -- this only changes
    WHEN the resolve runs, not WHETHER. Default ON (2026-06-22)."""
    from persistence import runtime_toggles
    return {"enabled": bool(runtime_toggles.get("speculative_coref_enabled"))}


@app.post("/api/speculative_coref")
async def set_speculative_coref(payload: dict | None = None):
    """Enable/disable speculative coref. Persisted by runtime_toggles, read live
    per turn by main.process_speech -- no restart to flip. OFF reverts to inline
    resolution inside retrieve() (byte-identical to the pre-2026-06-22 path), so
    flip OFF if the two Qwen3-4B servers contend on the single GPU and the overlap
    fails to hide the resolver latency. POST {"enabled": bool}."""
    from persistence import runtime_toggles
    enabled = bool((payload or {}).get("enabled", False))
    runtime_toggles.set("speculative_coref_enabled", enabled)
    return {"ok": True,
            "enabled": bool(runtime_toggles.get("speculative_coref_enabled"))}


# --- P4 face-flap debounce tuning (2026-06-11) -----------------------------
# The three LT-side knobs. Pi-side A1/A2 knobs (debounce frames, acquire/
# release dists) live on streamerpi's /face_id/config; LT-OS talks to that
# endpoint directly.
_VISION_TUNING_KEYS = {
    # knob -> (min, max). All consumed live via runtime_toggles.get() —
    # relevance.classify() per frame, the new-face trigger per decision.
    "people_novelty_min_persistence": (0.0, 1.0),
    "enroll_candidate_min_span_s": (0.0, 60.0),
    "enroll_candidate_min_dist": (0.05, 1.5),
}


@app.get("/api/vision/tuning")
async def get_vision_tuning():
    """Read the LT-side P4 debounce knobs (live runtime toggles)."""
    from persistence import runtime_toggles
    return {k: runtime_toggles.get(k) for k in _VISION_TUNING_KEYS}


@app.post("/api/vision/tuning")
async def set_vision_tuning(payload: dict | None = None):
    """Update any subset of the LT-side P4 debounce knobs. Values are floats,
    range-checked; persisted by runtime_toggles and read per-decision, so they
    apply live without a restart."""
    from persistence import runtime_toggles
    updates = {k: v for k, v in (payload or {}).items() if k in _VISION_TUNING_KEYS}
    for k, v in updates.items():
        lo, hi = _VISION_TUNING_KEYS[k]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"{k} must be numeric"}
        if not lo <= fv <= hi:
            return {"ok": False, "error": f"{k} must be {lo}-{hi}"}
        runtime_toggles.set(k, fv)
    return {"ok": True, **{k: runtime_toggles.get(k) for k in _VISION_TUNING_KEYS}}


# --- Slice A: manual situational-awareness regime (2026-06-12) --------------
# Live knob (LT-OS-set) injecting an NL [SITUATION] line into the ephemeral
# prompt. Empty string == OFF (no line). Enum-validated here; prompt_builder
# also fails safe on an unknown value.
# Binary since 2026-07-05 (Dan): the 5-value regime set (SOLO/GUEST/
# SMALL_GROUP/PARTY/EXPO) collapsed to Shop ('') vs Open Sauce ('EXPO').
# 'EXPO' stays the wire value so every downstream consumer (identifier
# continuity gate, identity_regime lockstep, prompt [SITUATION] text) works
# unchanged; the UI labels it "Open Sauce". Legacy values are rejected.
_SITUATION_REGIMES = {"", "EXPO"}


@app.get("/api/situation")
async def get_situation():
    """Read the live regime: '' == Shop (OFF), 'EXPO' == Open Sauce."""
    from persistence import runtime_toggles
    return {
        "situation_regime": runtime_toggles.get("situation_regime"),
        "options": sorted(_SITUATION_REGIMES),
    }


@app.post("/api/situation")
async def set_situation(payload: dict | None = None):
    """Set the regime. Binary whitelist; '' (Shop) disables (emits no
    [SITUATION] line). Persisted by runtime_toggles and read per-turn, so it
    applies live without a restart — and survives reboots (gameday flag)."""
    from persistence import runtime_toggles
    v = (payload or {}).get("situation_regime", "")
    if not isinstance(v, str):
        return {"ok": False, "error": "situation_regime must be a string"}
    v = v.strip().upper() if v.strip() else ""
    if v not in _SITUATION_REGIMES:
        return {"ok": False,
                "error": f"situation_regime must be one of {sorted(_SITUATION_REGIMES)}"}
    runtime_toggles.set("situation_regime", v)
    # Keep Slice B's identity_regime in lockstep: Open Sauce (EXPO) must also
    # disable the symmetric/temporal identity fusion (a wrong bind beats an
    # abstain in a crowd). One banner tap flips both; Shop restores 'normal'.
    runtime_toggles.set("identity_regime", "party" if v == "EXPO" else "normal")
    return {
        "ok": True,
        "situation_regime": runtime_toggles.get("situation_regime"),
        "identity_regime": runtime_toggles.get("identity_regime"),
    }


@app.get("/api/proactive")
async def get_proactive():
    """Read the proactive-speech runtime toggle. `enabled` is the live operator
    switch; `master` is the static config kill-switch (config.PROACTIVE_SPEECH_
    ENABLED). The feature only fires when BOTH are true."""
    from persistence import runtime_toggles
    return {
        "enabled": bool(runtime_toggles.get("proactive_speech_enabled")),
        "master": bool(config.PROACTIVE_SPEECH_ENABLED),
    }


@app.post("/api/proactive")
async def set_proactive(payload: dict | None = None):
    """Enable/disable proactive (unprompted) speech live. Persists the runtime
    toggle (read per-decision by maybe_speak_proactively, so it takes effect on
    the next visual event without a restart)."""
    from persistence import runtime_toggles
    enabled = bool((payload or {}).get("enabled", True))
    runtime_toggles.set("proactive_speech_enabled", enabled)
    return {
        "ok": True,
        "enabled": bool(runtime_toggles.get("proactive_speech_enabled")),
        "master": bool(config.PROACTIVE_SPEECH_ENABLED),
    }
