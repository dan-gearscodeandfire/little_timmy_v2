"""Mock streamerpi for the enrollment stress-test rig (F1).

Serves the four streamerpi endpoints the brain reads over HTTP so the live
okDemerzel face path can run end-to-end WITHOUT the Pi or a live human. Point
the brain at it with:

    TIMMY_CAPTURE_URL=http://127.0.0.1:8899/capture
    TIMMY_FACES_URL=http://127.0.0.1:8899/faces
    TIMMY_BEHAVIOR_URL=http://127.0.0.1:8899/behavior/status
    TIMMY_SERVO_MOVE_URL=http://127.0.0.1:8899/servo/move

(plain HTTP is fine — the brain's httpx client is verify=False.)

Streamerpi surface:
  GET  /capture            -> synthesized JPEG (image/jpeg); honours ?w=&h=
  GET  /faces              -> {timestamp, age_s, image_size, faces:[...]}
  GET  /behavior/status    -> {mode, face_visible, elapsed_ms, ...}
  POST /servo/move         -> {} (recorded in the servo log for F10 asserts)

Rig control surface (the runner drives the scene):
  POST /rig/frame  {face, led, led_below_face, extra_faces, behavior}
       -> render the current scene; returns {image_size, face_bbox, led_xy}
  GET  /rig/servo_log      -> [{pan,tilt,speed,ts}, ...]
  POST /rig/servo_log/clear
  GET  /rig/health

A "face" is a path to a JPEG/PNG, or a bare name resolved against
tests/fixtures (e.g. "scene" -> tests/fixtures/scene.jpg). LED placement:
pass led=[x,y] explicitly, or led_below_face=true to auto-place an in-band
green disc just below the detected face (YuNet runs once per control call,
never per /capture GET). led=null / omitted -> no LED painted.

Run:  cd ~/little_timmy && .venv/bin/python -m ops.mock_streamerpi --port 8899
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

log = logging.getLogger("mock_streamerpi")

# Anchor-LED hue band is HSV 65-85 (OpenCV) — pure green (H=60) is BELOW band
# and would not detect. Paint mid-band. (project_lt_enroll_stress_rig.)
_LED_BGR = tuple(int(c) for c in cv2.cvtColor(
    np.uint8([[[75, 220, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])
_LED_RADIUS = 15  # area ~700 px^2, inside anchor_led area band [4, 1500]


def _resolve_face_path(face: str) -> str:
    if os.path.isabs(face) and os.path.exists(face):
        return face
    cand = os.path.join(REPO, face)
    if os.path.exists(cand):
        return cand
    fx = os.path.join(REPO, "tests", "fixtures", face)
    if os.path.exists(fx):
        return fx
    if not face.lower().endswith((".jpg", ".jpeg", ".png")):
        fxj = os.path.join(REPO, "tests", "fixtures", face + ".jpg")
        if os.path.exists(fxj):
            return fxj
    raise FileNotFoundError(f"face image not found: {face}")


def _detect_primary_bbox(frame_bgr):
    """Largest YuNet-aligned face bbox (x0,y0,x1,y1), or None. Import lazily so
    the module loads even outside the venv for --help."""
    from presence.face_detect import aligned_crops
    crops = aligned_crops(frame_bgr)
    if not crops:
        return None
    return max((bbox for _c, bbox, _f in crops),
               key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


class Scene:
    """The current rendered scene: a JPEG + the /faces payload + behavior."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._image_size = (0, 0)
        self._faces: list[dict] = []
        self._behavior = {"mode": "track", "face_visible": True,
                          "elapsed_ms": 0, "last_face_pan": None,
                          "last_face_tilt": None}
        self._face_bbox = None
        self._led_xy = None
        self.servo_log: list[dict] = []
        self._rendered_at = 0.0

    def set_frame(self, face=None, led=None, led_below_face=False,
                  extra_faces=None, behavior=None) -> dict:
        with self._lock:
            frame = None
            bbox = None
            if face is not None:
                frame = cv2.imread(_resolve_face_path(face))
                if frame is None:
                    raise ValueError(f"could not decode face image: {face}")
                bbox = _detect_primary_bbox(frame)
            elif self._jpeg is not None:
                frame = cv2.imdecode(np.frombuffer(self._jpeg, np.uint8),
                                     cv2.IMREAD_COLOR)
                bbox = self._face_bbox

            if frame is None:
                raise ValueError("no face set and no prior frame to reuse")

            h, w = frame.shape[:2]
            led_xy = None
            if led is not None:
                led_xy = (int(led[0]), int(led[1]))
            elif led_below_face and bbox is not None:
                x0, y0, x1, y1 = bbox
                fcx = int((x0 + x1) / 2)
                fcy = int(min(h - _LED_RADIUS - 2,
                              y1 + max(20, (y1 - y0) * 0.25)))
                led_xy = (fcx, fcy)
            if led_xy is not None:
                cv2.circle(frame, led_xy, _LED_RADIUS, _LED_BGR, -1)

            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("jpeg encode failed")

            self._jpeg = buf.tobytes()
            self._image_size = (w, h)
            self._face_bbox = bbox
            self._led_xy = led_xy
            self._rendered_at = time.time()

            faces = []
            if bbox is not None:
                faces.append({"name": "", "distance": 1.0, "confidence": 0.0,
                              "bbox": [int(v) for v in bbox]})
            for ef in (extra_faces or []):
                faces.append({"name": ef.get("name", ""),
                              "distance": float(ef.get("distance", 1.0)),
                              "confidence": float(ef.get("confidence", 0.0)),
                              "bbox": [int(v) for v in ef["bbox"]]})
            self._faces = faces
            if behavior:
                self._behavior.update(behavior)
            return {"image_size": [w, h],
                    "face_bbox": [int(v) for v in bbox] if bbox else None,
                    "led_xy": list(led_xy) if led_xy else None,
                    "faces": len(faces)}

    def capture_jpeg(self, w=None, h=None) -> bytes | None:
        with self._lock:
            if self._jpeg is None:
                return None
            if w and h:
                frame = cv2.imdecode(np.frombuffer(self._jpeg, np.uint8),
                                     cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, (int(w), int(h)))
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 92])
                return buf.tobytes() if ok else None
            return self._jpeg

    def faces_payload(self) -> dict:
        with self._lock:
            now = time.time()
            return {"timestamp": now,
                    "age_s": max(0.0, now - self._rendered_at),
                    "image_size": list(self._image_size),
                    "faces": list(self._faces)}

    def behavior_payload(self) -> dict:
        with self._lock:
            return dict(self._behavior)


SCENE = Scene()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def _path(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        return u.path, parse_qs(u.query)

    def do_GET(self):
        path, q = self._path()
        if path == "/capture":
            w = q.get("w", [None])[0]
            h = q.get("h", [None])[0]
            jpeg = SCENE.capture_jpeg(w, h)
            if jpeg is None:
                self._send_json({"error": "no frame set"}, 503)
            else:
                self._send(200, jpeg, "image/jpeg")
        elif path == "/faces":
            self._send_json(SCENE.faces_payload())
        elif path in ("/behavior/status", "/behavior"):
            self._send_json(SCENE.behavior_payload())
        elif path == "/rig/servo_log":
            self._send_json(SCENE.servo_log)
        elif path == "/rig/health":
            self._send_json({"ok": True, "has_frame": SCENE._jpeg is not None})
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def do_POST(self):
        path, _q = self._path()
        if path == "/servo/move":
            body = self._read_json()
            body["ts"] = time.time()
            SCENE.servo_log.append(body)
            self._send_json({"ok": True})
        elif path == "/rig/frame":
            body = self._read_json()
            try:
                res = SCENE.set_frame(
                    face=body.get("face"), led=body.get("led"),
                    led_below_face=bool(body.get("led_below_face", False)),
                    extra_faces=body.get("extra_faces"),
                    behavior=body.get("behavior"))
                self._send_json({"ok": True, **res})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
        elif path == "/rig/servo_log/clear":
            SCENE.servo_log.clear()
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found", "path": path}, 404)


def main():
    ap = argparse.ArgumentParser(description="Mock streamerpi for the enroll rig")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--face", help="initial face image to load at startup")
    ap.add_argument("--led-below-face", action="store_true",
                    help="auto-place an LED below the initial face")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    if args.face:
        res = SCENE.set_frame(face=args.face, led_below_face=args.led_below_face)
        log.info("initial frame: %s", res)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("mock streamerpi on http://%s:%d "
             "(/capture /faces /behavior/status /servo/move + /rig/*)",
             args.host, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
