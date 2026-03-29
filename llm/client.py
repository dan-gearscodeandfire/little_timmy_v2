"""LLM client for llama.cpp servers (conversation on 8081, memory on 8080)."""

import json
import logging
from typing import AsyncIterator
import httpx
import config

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


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


async def generate_memory(prompt: str) -> str:
    """Send a prompt to GPT-OSS-120B on port 8080 for memory extraction. Non-streaming."""
    client = await _get_client()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.MEMORY_MAX_TOKENS,
        "temperature": config.MEMORY_TEMPERATURE,
        "stream": False,
    }
    resp = await client.post(
        f"{config.LLM_MEMORY_URL}/v1/chat/completions",
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content", "")
    # GPT-OSS-120B may put actual content in reasoning_content if content is empty
    if not content:
        content = data["choices"][0]["message"].get("reasoning_content", "")
    return content


async def generate_summary(turns_text: str) -> str:
    """Summarize conversation turns using the conversation LLM (fast 3B)."""
    client = await _get_client()
    payload = {
        "messages": [
            {"role": "user", "content": (
                "Summarize this conversation segment in 2-3 concise sentences. "
                "Preserve key facts, names, and decisions.\n\n"
                f"{turns_text}"
            )},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
        "stream": False,
    }
    resp = await client.post(
        f"{config.LLM_CONVERSATION_URL}/v1/chat/completions",
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"].get("content", "").strip()
