"""Booth display server on :8085.

Bridges Little Timmy (:8893) and streamerpi (https://192.168.1.110:8080)
to two browser kiosks:
  GET /visitor  -> theatrical visitor screen (camera + face boxes + mood + thinking)
  GET /operator -> operator dashboard (heartbeats, latency, overrides)
  WS  /ws       -> fanout of live events to all connected browsers

Background tasks:
  - lt_ws_subscriber: connects to ws://localhost:8893/ws as a client, forwards
    turn / retrieval / metrics / token events.
  - streamerpi_poller: polls /faces (2Hz) and /behavior/status (1Hz).
  - lt_state_poller: polls /api/mood, /api/vision, /api/presence (1Hz).

Each task is resilient: backoff on connection errors, never crash the server.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
# httpx logs every request at INFO; we poll 6/s, so this fills the disk fast.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
log = logging.getLogger("booth_display")

LT_HTTP = os.getenv("BOOTH_LT_HTTP", "http://127.0.0.1:8893")
LT_WS = os.getenv("BOOTH_LT_WS", "ws://127.0.0.1:8893/ws")
STREAMERPI = os.getenv("BOOTH_STREAMERPI", "https://192.168.1.110:8080")

STATIC_DIR = Path(__file__).parent / "static"

_browser_clients: set[WebSocket] = set()
_last_state: dict = {
    "faces": None,
    "behavior": None,
    "mood": None,
    "vision": None,
    "presence": None,
    "lt_ws_up": False,
    "streamerpi_up": False,
}

RECORDINGS_DIR = Path(
    os.getenv("BOOTH_RECORDINGS_DIR", os.path.expanduser("~/little_timmy/recordings"))
)
_recording: dict = {
    "active": False,
    "session_id": None,
    "started_at": None,
    "stopped_at": None,
    "chunks_received": 0,
    "bytes_received": 0,
    "last_chunk_at": None,
}
_recording_lock = asyncio.Lock()


async def broadcast(event_type: str, payload: dict) -> None:
    """Fan one event out to every connected browser."""
    if not _browser_clients:
        return
    msg = json.dumps({"type": event_type, "ts": time.time(), **payload})
    dead: list[WebSocket] = []
    for ws in _browser_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _browser_clients.discard(ws)


# ---------- background tasks ----------

async def lt_ws_subscriber() -> None:
    """Connect to LT /ws as a client; forward server-pushed events."""
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(LT_WS, ping_interval=20) as ws:
                _last_state["lt_ws_up"] = True
                log.info("[lt_ws] connected to %s", LT_WS)
                backoff = 1.0
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    etype = msg.get("type")
                    if not etype:
                        continue
                    # Forward all LT-side events with an "lt_" prefix so browsers
                    # can distinguish from booth-display-originated events.
                    await broadcast(f"lt_{etype}", {k: v for k, v in msg.items() if k != "type"})
        except Exception as e:
            _last_state["lt_ws_up"] = False
            log.warning("[lt_ws] disconnected: %s (retry in %.1fs)", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def streamerpi_poller() -> None:
    """Poll streamerpi /faces (2Hz) and /behavior/status (1Hz)."""
    async with httpx.AsyncClient(verify=False, timeout=2.0) as client:
        tick = 0
        while True:
            tick += 1
            t0 = time.time()
            # /faces every tick (2Hz when interval=0.5)
            try:
                r = await client.get(f"{STREAMERPI}/faces")
                if r.status_code == 200:
                    data = r.json()
                    _last_state["streamerpi_up"] = True
                    if data != _last_state["faces"]:
                        _last_state["faces"] = data
                        await broadcast("faces", {"data": data})
            except Exception as e:
                if _last_state["streamerpi_up"]:
                    log.warning("[streamerpi] /faces error: %s", e)
                _last_state["streamerpi_up"] = False
            # /behavior/status every other tick (1Hz)
            if tick % 2 == 0:
                try:
                    r = await client.get(f"{STREAMERPI}/behavior/status")
                    if r.status_code == 200:
                        data = r.json()
                        if data != _last_state["behavior"]:
                            _last_state["behavior"] = data
                            await broadcast("behavior", {"data": data})
                except Exception:
                    pass
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, 0.5 - elapsed))


def _strip_for_broadcast(key: str, data):
    """Drop fields that are wasteful or sensitive on the wire.

    /api/vision returns the live JPEG as frame_b64 — useless for booth display
    (video comes from streamerpi WebRTC) and ~5-15 KB per 1Hz tick.
    """
    if key == "vision" and isinstance(data, dict):
        return {k: v for k, v in data.items() if k != "frame_b64"}
    return data


async def lt_state_poller() -> None:
    """Poll LT /api/mood, /api/vision, /api/presence at 1Hz.

    Belt-and-suspenders alongside lt_ws_subscriber: catches state changes that
    LT broadcasts no event for, and keeps last_state warm for late-joining
    browsers.
    """
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            for key, path in (("mood", "/api/mood"), ("vision", "/api/vision"), ("presence", "/api/presence")):
                try:
                    r = await client.get(f"{LT_HTTP}{path}")
                    if r.status_code == 200:
                        data = r.json()
                        if data != _last_state[key]:
                            _last_state[key] = data
                            await broadcast(key, {"data": _strip_for_broadcast(key, data)})
                except Exception:
                    pass
            await asyncio.sleep(1.0)


# ---------- lifespan + FastAPI app ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(lt_ws_subscriber(), name="lt_ws_subscriber"),
        asyncio.create_task(streamerpi_poller(), name="streamerpi_poller"),
        asyncio.create_task(lt_state_poller(), name="lt_state_poller"),
    ]
    log.info("booth_display started; LT=%s streamerpi=%s", LT_HTTP, STREAMERPI)
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(title="booth_display", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root():
    return JSONResponse({
        "service": "booth_display",
        "pages": {"visitor": "/visitor", "operator": "/operator"},
        "ws": "/ws",
        "state_summary": {
            "lt_ws_up": _last_state["lt_ws_up"],
            "streamerpi_up": _last_state["streamerpi_up"],
            "have_faces": _last_state["faces"] is not None,
            "have_mood": _last_state["mood"] is not None,
        },
    })


@app.get("/health")
async def health():
    return _last_state


@app.get("/visitor")
async def visitor_page():
    return FileResponse(STATIC_DIR / "visitor.html")


@app.post("/streamerpi/offer")
async def streamerpi_offer_proxy(request: Request):
    """Forward a browser WebRTC SDP offer to streamerpi.

    Visitor page is HTTP on :8085; streamerpi is HTTPS on :8080 with a self-
    signed cert. A direct browser POST would be blocked (mixed-content) or
    rejected (untrusted cert). Proxying through here with verify=False lets the
    kiosk negotiate without a per-browser cert dance.
    """
    body = await request.json()
    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        r = await client.post(f"{STREAMERPI}/offer", json=body)
        # Forward upstream verbatim. r.json() blew up with JSONDecodeError
        # when streamerpi answered 500 with an aiohttp text body, masking the
        # actual aiortc traceback behind a useless proxy 500.
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )


@app.get("/operator")
async def operator_page():
    return FileResponse(STATIC_DIR / "operator.html")


# ---------- session recording (Supervisor M6) ----------
#
# Smallest variant: visitor.html runs MediaRecorder on its WebRTC <video>
# stream and POSTs 2s chunks to /api/recording/chunk. /start broadcasts
# `recording_start` over the existing browser_ws fanout; visitor.html
# instantiates the recorder. /stop broadcasts `recording_stop` and writes
# the chatlog + metadata sidecar files.
#
# Output: ~/little_timmy/recordings/<ts>.{webm,jsonl,json}
#
# Known limitation: the WebRTC stream is video-only (recvonly transceiver
# in visitor.html), so the .webm has no audio. LT's TTS is played through
# okdemerzel speakers, not routed back through the visitor stream. For a
# camera-side audio track we'd need either visitor mic capture or a server-
# side mix; both are deferred to the "better" variant.

def _recording_status_payload() -> dict:
    started = _recording["started_at"]
    if started is None:
        duration = 0.0
    elif _recording["active"]:
        duration = round(time.time() - started, 1)
    else:
        stopped = _recording["stopped_at"] or started
        duration = round(stopped - started, 1)
    return {
        "active": _recording["active"],
        "session_id": _recording["session_id"],
        "started_at": started,
        "stopped_at": _recording["stopped_at"],
        "duration_s": duration,
        "chunks_received": _recording["chunks_received"],
        "bytes_received": _recording["bytes_received"],
        "last_chunk_at": _recording["last_chunk_at"],
        "recordings_dir": str(RECORDINGS_DIR),
    }


@app.post("/api/recording/start")
async def recording_start():
    async with _recording_lock:
        if _recording["active"]:
            return JSONResponse(
                {"error": "already_recording", **_recording_status_payload()},
                status_code=409,
            )
        ts = time.strftime("%Y%m%d-%H%M%S")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        # Truncate target file in case of a stale .webm from a crashed prior run.
        (RECORDINGS_DIR / f"{ts}.webm").write_bytes(b"")
        _recording.update({
            "active": True,
            "session_id": ts,
            "started_at": time.time(),
            "stopped_at": None,
            "chunks_received": 0,
            "bytes_received": 0,
            "last_chunk_at": None,
        })
        await broadcast("recording_start", {"session_id": ts})
        log.info("[recording] start session=%s dir=%s", ts, RECORDINGS_DIR)
        return _recording_status_payload()


@app.post("/api/recording/chunk")
async def recording_chunk(request: Request):
    body = await request.body()
    async with _recording_lock:
        if not _recording["active"]:
            return JSONResponse({"error": "not_recording"}, status_code=409)
        ts = _recording["session_id"]
        path = RECORDINGS_DIR / f"{ts}.webm"
        with path.open("ab") as f:
            f.write(body)
        _recording["chunks_received"] += 1
        _recording["bytes_received"] += len(body)
        _recording["last_chunk_at"] = time.time()
    return {"ok": True, "bytes": len(body)}


@app.post("/api/recording/stop")
async def recording_stop():
    async with _recording_lock:
        if not _recording["active"]:
            return JSONResponse({"error": "not_recording"}, status_code=409)
        ts = _recording["session_id"]
        _recording["active"] = False
        _recording["stopped_at"] = time.time()
        await broadcast("recording_stop", {"session_id": ts})
        duration_s = round(_recording["stopped_at"] - (_recording["started_at"] or 0), 2)
        meta = {
            "session_id": ts,
            "started_at": _recording["started_at"],
            "stopped_at": _recording["stopped_at"],
            "duration_s": duration_s,
            "chunks_received": _recording["chunks_received"],
            "bytes_received": _recording["bytes_received"],
            "chatlog_fetched": False,
            "notes": "Audio absent — WebRTC stream is recvonly video.",
        }
        # Best-effort chatlog dump. LT may be down; don't fail the stop.
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{LT_HTTP}/api/chatlog")
            if r.status_code == 200:
                (RECORDINGS_DIR / f"{ts}.jsonl").write_text(r.text, encoding="utf-8")
                meta["chatlog_fetched"] = True
                meta["chatlog_bytes"] = len(r.text)
                log.info("[recording] chatlog dumped len=%d", len(r.text))
            else:
                log.warning("[recording] chatlog fetch status=%s", r.status_code)
        except Exception as e:
            log.warning("[recording] chatlog fetch failed: %s", e)
        (RECORDINGS_DIR / f"{ts}.json").write_text(json.dumps(meta, indent=2))
        log.info(
            "[recording] stop session=%s duration=%.1fs chunks=%d bytes=%d",
            ts, duration_s, meta["chunks_received"], meta["bytes_received"],
        )
        return _recording_status_payload()


@app.get("/api/recording/status")
async def recording_status():
    return _recording_status_payload()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.websocket("/ws")
async def browser_ws(ws: WebSocket):
    await ws.accept()
    _browser_clients.add(ws)
    log.info("[ws] browser connected (n=%d)", len(_browser_clients))
    # Send a snapshot so a late joiner doesn't have to wait for the next tick.
    snap = {
        "type": "snapshot",
        "ts": time.time(),
        "faces": _last_state["faces"],
        "behavior": _last_state["behavior"],
        "mood": _last_state["mood"],
        "vision": _strip_for_broadcast("vision", _last_state["vision"]),
        "presence": _last_state["presence"],
        "lt_ws_up": _last_state["lt_ws_up"],
        "streamerpi_up": _last_state["streamerpi_up"],
    }
    try:
        await ws.send_text(json.dumps(snap))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _browser_clients.discard(ws)
        log.info("[ws] browser disconnected (n=%d)", len(_browser_clients))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BOOTH_PORT", "8085")))
