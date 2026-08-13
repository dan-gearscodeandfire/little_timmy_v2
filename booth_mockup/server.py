"""
Booth-mockup server on :8090 (Concept B overlay preview).

Serves the local index.html (full-bleed WebRTC + flat overlay) and proxies
POST /streamerpi/offer to https://streamerpi.local:8080/offer so the same-
origin client can negotiate WebRTC without browser CORS/TLS friction.

Replaces booth_display visitor.html for testing the Concept B overlay
design. Stop booth-display.service before running this so the streamerpi
single-client lock isn't held by the old client.
"""
import asyncio
import logging
import ssl
from pathlib import Path

import aiohttp
from aiohttp import web

STREAMERPI_URL = "https://streamerpi.local:8080"
LT_URL = "http://127.0.0.1:8893"
LTOS_URL = "http://127.0.0.1:8894"  # LT-OS dashboard: host/GPU telemetry (/api/host)
PORT = 8090
HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"
CERT_FILE = HERE / "cert.pem"
KEY_FILE = HERE / "key.pem"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("booth-mockup")


async def index(_request: web.Request) -> web.FileResponse:
    # no-store (2026-08-11): FileResponse sends only Last-Modified, no
    # Cache-Control, so Chrome applies HEURISTIC freshness (~10% of the file's
    # age). index.html is edited in bursts and then sits for weeks, so a
    # month-old file bought itself a multi-day cache window — the kiosk kept
    # rendering the PREVIOUS overlay after a restart, revalidating nothing.
    # That's the "static -> needs a hard reload" behaviour; killed at the
    # source. One 500 KB page on a LAN, so there is nothing to conserve.
    return web.FileResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})


def _ssl_ctx_no_verify() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def proxy_offer(request: web.Request) -> web.Response:
    body = await request.read()
    headers = {"Content-Type": request.headers.get("Content-Type",
                                                   "application/json")}
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(
                f"{STREAMERPI_URL}/offer",
                data=body,
                headers=headers,
                ssl=_ssl_ctx_no_verify(),
            ) as upstream:
                payload = await upstream.read()
                log.info("offer proxied: streamerpi -> %d (%d bytes)",
                         upstream.status, len(payload))
                return web.Response(
                    body=payload,
                    status=upstream.status,
                    headers={"Content-Type": upstream.headers.get(
                        "Content-Type", "application/json")},
                )
        except Exception as e:
            log.warning("offer proxy failed: %s", e)
            return web.json_response(
                {"error": str(e)[:200]}, status=502)


async def proxy_get(request: web.Request) -> web.Response:
    """Generic GET proxy for streamerpi read-only endpoints (faces,
    face_pipeline/status, behavior/status, faces, etc.). The path segment
    after /streamerpi/ is forwarded as-is."""
    path = request.match_info["path"]
    timeout = aiohttp.ClientTimeout(total=4)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(
                f"{STREAMERPI_URL}/{path}",
                ssl=_ssl_ctx_no_verify(),
            ) as upstream:
                payload = await upstream.read()
                return web.Response(
                    body=payload,
                    status=upstream.status,
                    headers={"Content-Type": upstream.headers.get(
                        "Content-Type", "application/json")},
                )
        except Exception as e:
            return web.json_response(
                {"error": str(e)[:200], "path": path}, status=502)


async def proxy_lt_get(request: web.Request) -> web.Response:
    """GET proxy for LT (port 8893) read-only endpoints. Used by the visitor
    overlay to fetch /api/last_payload — the ephemeral_block in there is the
    [CONTEXT]…[/CONTEXT] body injected into the most-recent user prompt, and
    the booth scrolls it down the right edge so visitors can see what Timmy
    is being told about the room when he replies."""
    path = request.match_info["path"]
    timeout = aiohttp.ClientTimeout(total=3)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(f"{LT_URL}/{path}") as upstream:
                payload = await upstream.read()
                return web.Response(
                    body=payload,
                    status=upstream.status,
                    headers={"Content-Type": upstream.headers.get(
                        "Content-Type", "application/json")},
                )
        except Exception as e:
            return web.json_response(
                {"error": str(e)[:200], "path": path}, status=502)


async def proxy_lt_ws(request: web.Request) -> web.WebSocketResponse:
    """Relay LT's event websocket (:8893 /ws) to the booth client.

    Same-origin bridge, mirroring proxy_lt_get: the page is served over
    self-signed HTTPS, so a direct ws:// to :8893 would be blocked as mixed
    content and a wss:// would need its own cert. The booth consumes exactly
    two event types off this feed — `speaking` / `speech_end`, emitted by the
    TTS playback loop at true audible onset — to drive the VOX caption band.

    Read-only by design: client->upstream frames are NOT forwarded, so a booth
    page (or anything that can reach :8090) can never inject into LT.
    """
    ws_client = web.WebSocketResponse(heartbeat=30)
    await ws_client.prepare(request)
    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(f"{LT_URL}/ws", heartbeat=30) as upstream:

            async def pump_upstream() -> None:
                """Forward LT events down to the booth page."""
                async for msg in upstream:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await ws_client.send_str(msg.data)
                    elif msg.type in (aiohttp.WSMsgType.ERROR,
                                      aiohttp.WSMsgType.CLOSED):
                        break

            async def watch_client() -> None:
                """Drain the client socket purely to notice it going away.

                Frames are read and DISCARDED, never forwarded upstream -- the
                read-only invariant in the docstring is preserved. But we have
                to read: aiohttp only surfaces a client disconnect through this
                iterator, and without it a reloaded/closed booth page left this
                handler parked forever on `async for msg in upstream`.

                LT only emits on speaking/speech_end, so a parked handler could
                sit silent for hours still holding an upstream socket to :8893.
                Every booth reconnect leaked one more, LT's fds climbed until
                accept() hit EMFILE at 1024, and the resulting traceback-per-
                accept flood filled the root fs -- the 2026-08-12 ENOSPC outage.
                """
                async for _ in ws_client:
                    pass

            tasks = [asyncio.create_task(pump_upstream()),
                     asyncio.create_task(watch_client())]
            try:
                # Whichever side ends first tears down the other.
                _done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        # LT restarting / not listening. The client reconnects on its own
        # timer, so this is expected churn, not an error worth the log noise.
        log.debug("lt ws relay ended: %s", e)
    finally:
        await session.close()
        await ws_client.close()
    return ws_client


async def proxy_ltos_get(request: web.Request) -> web.Response:
    """GET proxy for the LT-OS dashboard (port 8894). Used by the visitor
    overlay to fetch /api/host — the host/GPU telemetry snapshot (cpu/ram/vram
    percentages, gpu_busy_percent, temp_c, power_w, sclk, disk, load) sourced
    from ops/gpu_sysfs.py. Drives the bottom-centre vitals donuts."""
    path = request.match_info["path"]
    timeout = aiohttp.ClientTimeout(total=3)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(f"{LTOS_URL}/{path}") as upstream:
                payload = await upstream.read()
                return web.Response(
                    body=payload,
                    status=upstream.status,
                    headers={"Content-Type": upstream.headers.get(
                        "Content-Type", "application/json")},
                )
        except Exception as e:
            return web.json_response(
                {"error": str(e)[:200], "path": path}, status=502)


def main() -> None:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/streamerpi/offer", proxy_offer)
    app.router.add_get("/streamerpi/{path:.+}", proxy_get)
    # /lt/ws must be registered BEFORE the /lt/{path} catch-all: a websocket
    # handshake is itself a GET, so the catch-all would otherwise swallow it
    # and try to proxy it as a plain request.
    app.router.add_get("/lt/ws", proxy_lt_ws)
    app.router.add_get("/lt/{path:.+}", proxy_lt_get)
    app.router.add_get("/ltos/{path:.+}", proxy_ltos_get)
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(CERT_FILE), str(KEY_FILE))
    log.info("booth-mockup serving on https://0.0.0.0:%d "
             "(proxying offer to %s)", PORT, STREAMERPI_URL)
    # access_log=None disables aiohttp's per-request access logging: the booth
    # client polls /streamerpi/faces (4 Hz) + /lt/api/presence (0.5 Hz) +
    # /lt/api/last_payload (0.25 Hz) continuously, which otherwise grew
    # server.log unbounded (~409 MB observed 2026-06-15). Warnings/errors
    # (offer-proxy failures etc.) still log via the module logger.
    web.run_app(app, host="0.0.0.0", port=PORT,
                ssl_context=server_ctx, print=None, access_log=None)


if __name__ == "__main__":
    main()
