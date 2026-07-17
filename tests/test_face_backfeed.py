"""Hermetic tests for the EdgeFace identity backfeed push (presence/
face_backfeed.py) — payload shape (corner->xywh bbox conversion), the
nothing-named skip, the rate-limit floor, and the never-raises contract.

Run:
    .venv/bin/pytest tests/test_face_backfeed.py -v
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence import face_backfeed
from presence.types import FacePrediction


class _FakeClient:
    def __init__(self, status_code=200, exc=None):
        self.status_code = status_code
        self.exc = exc
        self.calls = []

    async def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if self.exc:
            raise self.exc
        return types.SimpleNamespace(status_code=self.status_code)


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    face_backfeed._last_push_ts = 0.0
    face_backfeed._last_fail_log_ts = 0.0


def _pred(name="pat", bbox=(100, 50, 180, 150), conf=0.8):
    return FacePrediction(user_id=name, confidence=conf, bbox=bbox)


def test_payload_converts_corner_bbox_to_xywh():
    body = face_backfeed._payload((_pred(),), (640, 360), 123.0)
    assert body["identities"] == [
        {"name": "pat", "bbox": [100.0, 50.0, 80.0, 100.0], "confidence": 0.8}]
    assert body["image_size"] == [640, 360]
    assert body["captured_at"] == 123.0


def test_payload_none_when_nothing_named():
    assert face_backfeed._payload((), (640, 360), 1.0) is None


def test_push_posts_and_returns_true():
    client = _FakeClient()
    ok = asyncio.run(face_backfeed.push_identities(client, (_pred(),), (640, 360)))
    assert ok is True
    url, body = client.calls[0]
    assert url.endswith("/faces/backfeed")
    assert body["identities"][0]["name"] == "pat"


def test_push_rate_limited_second_call_skipped():
    client = _FakeClient()
    assert asyncio.run(
        face_backfeed.push_identities(client, (_pred(),), (640, 360))) is True
    assert asyncio.run(
        face_backfeed.push_identities(client, (_pred(),), (640, 360))) is False
    assert len(client.calls) == 1


def test_push_never_raises_on_network_error():
    client = _FakeClient(exc=ConnectionError("pi down"))
    ok = asyncio.run(face_backfeed.push_identities(client, (_pred(),), (640, 360)))
    assert ok is False


def test_push_never_raises_on_http_error():
    client = _FakeClient(status_code=500)
    ok = asyncio.run(face_backfeed.push_identities(client, (_pred(),), (640, 360)))
    assert ok is False
