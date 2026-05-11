"""LLM client for llama.cpp servers (conversation on 8081, memory on 8080)."""

import json
import logging
from typing import AsyncIterator
import httpx
import config

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None

# Optional callback set by main.py: fn(source: str, reasoning: str, content: str)
# fires after a thinking-on response if reasoning_content is non-empty. Used by
# the booth_display visitor "ghost reasoning" panel to surface Qwen3.6's
# actual thought process from real production work (memory extraction +
# rollup) instead of running a dedicated narrator server.
_reasoning_tap = None


def set_reasoning_tap(fn) -> None:
    """Register an async (source, reasoning, content) callback fired after
    each thinking-on completion. Pass None to clear."""
    global _reasoning_tap
    _reasoning_tap = fn


async def _maybe_emit_reasoning(source: str, data: dict, content: str) -> None:
    if _reasoning_tap is None:
        return
    try:
        msg = data["choices"][0]["message"]
        reasoning = (msg.get("reasoning_content") or "").strip()
    except (KeyError, IndexError, TypeError):
        return
    if not reasoning:
        return
    try:
        await _reasoning_tap(source, reasoning, content)
    except Exception:
        log.debug("reasoning tap callback failed", exc_info=True)


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    return _client


async def stream_conversation(
    messages: list[dict],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Stream tokens from Llama 3.2 3B on port 8081 via /v1/chat/completions SSE."""
    client = await _get_client()
    payload = {
        "messages": messages,
        "max_tokens": max_tokens or config.CONVERSATION_MAX_TOKENS,
        "temperature": config.CONVERSATION_TEMPERATURE,
        "stream": True,
    }

    async with client.stream(
        "POST",
        f"{config.LLM_CONVERSATION_URL}/v1/chat/completions",
        json=payload,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    yield delta["content"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


async def generate_memory(prompt: str, thinking: bool | None = None) -> str:
    """Send a prompt to the brain LLM (LLM_MEMORY_URL) for memory extraction. Non-streaming.

    `thinking`: when True/False, sends `chat_template_kwargs:{enable_thinking:bool}`
    so a Qwen3.6-style server gates the thinking trace per-request. None preserves
    legacy behavior (no kwarg, server default applies).
    """
    client = await _get_client()
    payload: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.MEMORY_MAX_TOKENS,
        "temperature": config.MEMORY_TEMPERATURE,
        "stream": False,
    }
    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    resp = await client.post(
        f"{config.LLM_MEMORY_URL}/v1/chat/completions",
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content", "")
    # Defensive fallback: legacy GPT-OSS-120B blended thinking into reasoning_content
    # when content was empty. With Qwen3.6 + --reasoning-format deepseek this should
    # rarely fire, but keep the fallback for any edge-case server config.
    if not content:
        content = data["choices"][0]["message"].get("reasoning_content", "")
    await _maybe_emit_reasoning("memory", data, content)
    return content


async def generate_summary(turns_text: str) -> str:
    """Summarize conversation turns using Qwen3.6 thinking-on at LLM_MEMORY_URL."""
    client = await _get_client()
    payload = {
        "messages": [
            {"role": "user", "content": (
                "Summarize this conversation segment in 2-3 concise sentences. "
                "Preserve key facts, names, and decisions.\n\n"
                f"{turns_text}"
            )},
        ],
        "max_tokens": 800,
        "temperature": 0.3,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    resp = await client.post(
        f"{config.LLM_MEMORY_URL}/v1/chat/completions",
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content", "").strip()
    await _maybe_emit_reasoning("summary", data, content)
    return content
