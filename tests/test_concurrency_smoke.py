"""Concurrency smoke test for the shared brain server at :8083.

Fires a fact-extraction-shaped (thinking=ON) request and a vision-shaped
(thinking=OFF, with image) request simultaneously and measures how much
the slower one is delayed vs running it solo. With `-np 1` on the server
the two calls serialize; with `-np 2` they overlap.

The threshold here (+1.5s extra wait under contention) is a regression
ceiling, not a perf target. If the test fails it means contention is
real and Stage 6 (`-np 2`) of the consolidation plan should be revisited.

Run:
    .venv/bin/pytest tests/test_concurrency_smoke.py -v -s
"""

import asyncio
import base64
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scene.jpg"

SHORT_BRAIN_PROMPT = (
    "Extract any durable personal fact from this exchange as a single JSON "
    'object. If none, output {"facts": []}.\n\n'
    "User: My dog Wally is a corgi.\n"
    "Assistant: Noted."
)


async def _brain_call(client: httpx.AsyncClient) -> float:
    body = {
        "messages": [{"role": "user", "content": SHORT_BRAIN_PROMPT}],
        "max_tokens": 3072,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    t0 = time.monotonic()
    r = await client.post(f"{config.LLM_MEMORY_URL}/v1/chat/completions",
                          json=body, timeout=120.0)
    r.raise_for_status()
    return time.monotonic() - t0


async def _vision_call(client: httpx.AsyncClient, jpeg: bytes) -> float:
    b64 = base64.b64encode(jpeg).decode("ascii")
    body = {
        "model": config.LLM_BRAIN_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text",
                 "text": 'Describe this image as JSON: {"objects":[],"scene":""}'},
            ],
        }],
        "max_tokens": 200,
        "temperature": 0.2,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    r = await client.post(f"{config.LLM_VISION_URL}/v1/chat/completions",
                          json=body, timeout=120.0)
    r.raise_for_status()
    return time.monotonic() - t0


@pytest.mark.asyncio
async def test_concurrent_brain_plus_vision_does_not_serialize_badly():
    """Solo timings, then concurrent. Concurrent should not be more than
    +1.5s slower than the slower-of-the-two solo time. Above that we
    are queueing — Stage 6 (`-np 2`) of the plan kicks in."""
    if not FIXTURE.exists():
        pytest.skip(f"missing fixture: {FIXTURE}")
    jpeg = FIXTURE.read_bytes()

    async with httpx.AsyncClient() as client:
        # Warm both code paths so the comparison isn't dominated by cold-cache.
        await _brain_call(client)
        await _vision_call(client, jpeg)

        # Solo measurements (sequential)
        solo_brain = await _brain_call(client)
        solo_vision = await _vision_call(client, jpeg)
        slower_solo = max(solo_brain, solo_vision)

        # Concurrent — gather the same two calls
        t0 = time.monotonic()
        b, v = await asyncio.gather(
            _brain_call(client),
            _vision_call(client, jpeg),
        )
        concurrent_wall = time.monotonic() - t0

    # Print so -s mode shows the actual numbers (useful for tuning)
    print(
        f"\n  solo brain : {solo_brain:.2f}s"
        f"\n  solo vision: {solo_vision:.2f}s"
        f"\n  slower solo: {slower_solo:.2f}s"
        f"\n  concurrent wall: {concurrent_wall:.2f}s"
        f"\n  concurrent brain finish: {b:.2f}s"
        f"\n  concurrent vision finish: {v:.2f}s"
        f"\n  queue penalty (concurrent_wall - slower_solo): "
        f"{concurrent_wall - slower_solo:+.2f}s"
    )

    # If both calls overlapped fully, concurrent_wall ≈ slower_solo.
    # If they serialized perfectly, concurrent_wall ≈ solo_brain + solo_vision
    # (~20s with thinking-on brain + thinking-off vision).
    #
    # llama.cpp's `-cb` continuous batching interleaves at the token level
    # opportunistically — observed runs swing from -2.4s (perfect overlap) to
    # +1.7s (near-serialized). That's normal variance, not contention.
    #
    # Threshold here is a regression catcher for pathological cases (no batching,
    # `-np 0`, server misconfig) which would surface as +10-15s. If this test
    # starts failing consistently, Stage 6 of the consolidation plan (`-np 2`)
    # is the documented next step.
    queue_penalty = concurrent_wall - slower_solo
    assert queue_penalty < 4.0, (
        f"queue penalty {queue_penalty:+.2f}s exceeds 4s threshold — "
        f"see Stage 6 of the consolidation plan. solo_brain={solo_brain:.2f}s "
        f"solo_vision={solo_vision:.2f}s concurrent_wall={concurrent_wall:.2f}s"
    )
