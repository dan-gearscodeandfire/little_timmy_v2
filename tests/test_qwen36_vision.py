"""Verify Qwen3.6 vision path on :8083 with thinking=False.

Replaces the old Qwen2.5-VL on :8082. The user's framing:
"Qwen3.6 thinking=no image captioning" — the answer must come back in
`content` immediately, no thinking trace, with the structured JSON
keys the SceneRecord expects.

Run from repo root:
    .venv/bin/pytest tests/test_qwen36_vision.py -v
"""

import base64
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from vision import analyzer  # noqa: E402
from vision.analyzer import analyze_frame, STRUCTURED_PROMPT  # noqa: E402


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scene.jpg"


def _load_fixture() -> bytes:
    if not FIXTURE.exists():
        pytest.skip(f"missing fixture: {FIXTURE}")
    return FIXTURE.read_bytes()


@pytest.fixture(autouse=True)
def _reset_analyzer_client():
    """analyzer.py caches a module-level httpx.AsyncClient, but pytest-asyncio
    spins a fresh event loop per test — reuse across loops raises
    'Event loop is closed' during connection cleanup. Reset before each test."""
    analyzer._client = None
    yield
    analyzer._client = None


# ---------- Raw probe: shape of the /v1/chat/completions response ----------

@pytest.mark.asyncio
async def test_thinking_off_keeps_reasoning_empty():
    """With chat_template_kwargs:{enable_thinking:false} the vision response
    should put JSON in `content` and leave `reasoning_content` empty."""
    jpeg = _load_fixture()
    b64 = base64.b64encode(jpeg).decode("ascii")
    body = {
        "model": config.LLM_BRAIN_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": STRUCTURED_PROMPT},
            ],
        }],
        "max_tokens": 300,
        "temperature": 0.2,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{config.LLM_VISION_URL}/v1/chat/completions",
                              json=body)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]

    assert (msg.get("content") or "").strip()
    assert not (msg.get("reasoning_content") or "").strip(), (
        f"reasoning_content leaked under thinking=False: "
        f"{(msg.get('reasoning_content') or '')[:200]!r}"
    )
    assert "<think>" not in (msg.get("content") or "")


# ---------- End-to-end through analyze_frame() ----------

@pytest.mark.asyncio
async def test_analyze_frame_returns_scene_record():
    """analyze_frame() should parse Qwen3.6's JSON response into a
    SceneRecord with the four core fields populated. Latency budget
    is permissive (cold first call can be 1-2s; warm should be sub-second)."""
    jpeg = _load_fixture()

    t0 = time.monotonic()
    record = await analyze_frame(jpeg)
    elapsed = time.monotonic() - t0

    assert record is not None, "analyze_frame returned None — check :8083 logs"
    # Core SceneRecord fields exist (lists may be empty for an unfamiliar image,
    # but the attributes themselves must be present).
    assert hasattr(record, "people")
    assert hasattr(record, "objects")
    assert hasattr(record, "actions")
    assert hasattr(record, "scene_state")
    assert isinstance(record.novelty, float)
    assert 0.0 <= record.novelty <= 1.0

    # Reasonable latency upper bound — generous to absorb cold-cache + CLIP encode.
    # qwen36-vision-on-okdemerzel notes: cold think-OFF 0.6-1.3s, but a fresh
    # pytest run with mmproj cold-cache can stretch this. Keep the assertion
    # loose; the regression we care about is "didn't accidentally turn thinking
    # back on", which would push this past 5s.
    assert elapsed < 5.0, f"vision call took {elapsed:.2f}s (expected < 5s)"


@pytest.mark.asyncio
async def test_analyze_frame_warm_call_is_fast():
    """Second call (warm cache) should land inside the documented budget."""
    jpeg = _load_fixture()
    # Prime
    await analyze_frame(jpeg)
    # Measure
    t0 = time.monotonic()
    record = await analyze_frame(jpeg)
    elapsed = time.monotonic() - t0

    assert record is not None
    # Production target is sub-second warm (cached httpx client + model warm),
    # but the autouse fixture above resets the analyzer's client per test so
    # each call eats fresh TCP + first-request overhead. 2.5s is a regression
    # ceiling, not a perf target — the production orchestrator keeps one
    # client for the process lifetime and routinely sees sub-second warm calls.
    assert elapsed < 2.5, f"warm vision call took {elapsed:.2f}s (expected < 2.5s)"
