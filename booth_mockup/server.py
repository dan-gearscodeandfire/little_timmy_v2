"""
Booth-mockup server on :8090 (Concept B overlay preview).

Serves the local index.html (full-bleed WebRTC + flat overlay) and proxies
POST /streamerpi/offer to https://streamerpi.local:8080/offer so the same-
origin client can negotiate WebRTC without browser CORS/TLS friction.

Replaces booth_display visitor.html for testing the Concept B overlay
design. Stop booth-display.service before running this so the streamerpi
single-client lock isn't held by the old client.
"""
import logging
import ssl
from pathlib import Path

import aiohttp
from aiohttp import web

STREAMERPI_URL = "https://streamerpi.local:8080"
LT_URL = "http://127.0.0.1:8893"
PORT = 8090
HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"
CERT_FILE = HERE / "cert.pem"
KEY_FILE = HERE / "key.pem"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("booth-mockup")


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(INDEX_HTML)


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


def main() -> None:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/streamerpi/offer", proxy_offer)
    app.router.add_get("/streamerpi/{path:.+}", proxy_get)
    app.router.add_get("/lt/{path:.+}", proxy_lt_get)
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(CERT_FILE), str(KEY_FILE))
    log.info("booth-mockup serving on https://0.0.0.0:%d "
             "(proxying offer to %s)", PORT, STREAMERPI_URL)
    web.run_app(app, host="0.0.0.0", port=PORT,
                ssl_context=server_ctx, print=None)


if __name__ == "__main__":
    main()
