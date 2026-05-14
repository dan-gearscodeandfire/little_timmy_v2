"""LLM client for llama.cpp servers (conversation on 8081, memory on 8080)."""

import asyncio
import json
import logging
from typing import AsyncIterator
import httpx
import config

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


# --- Conversation-priority gate (Phase 1 of the Qwen3.6 conversation-tier
# option, 2026-05-14). Only matters when the conversation tier and the
# memory/rollup tier share a server -- i.e. LLM_CONVERSATION_URL ==
# LLM_MEMORY_URL. With a single-slot llama.cpp server (-np 1) the requests
# would otherwise FIFO-serialize and a user-facing reply could wait
# 15-45 s behind a thinking-on memory extraction. Dan: 'conversational
# call always takes preference over summarization.'
_conversation_in_flight = asyncio.Event()
_slow_call_tasks: set[asyncio.Task] = set()


def _conversation_shares_brain() -> bool:
    """True when conversation routes to the same server as memory."""
    return config.LLM_CONVERSATION_URL == config.LLM_MEMORY_URL


async def _wait_for_conversation_idle() -> None:
    """Block until no conversation stream is in flight. No-op when
    conversation and memory live on separate servers."""
    if not _conversation_shares_brain():
        return
    while _conversation_in_flight.is_set():
        await asyncio.sleep(0.1)


def _register_slow_call() -> asyncio.Task | None:
    """Mark the current task as a slow call so stream_conversation can
    cancel it if a user-facing reply needs the brain. Returns the task
    so the caller can deregister in a finally block. No-op on separate
    servers."""
    if not _conversation_shares_brain():
        return None
    task = asyncio.current_task()
    if task is not None:
        _slow_call_tasks.add(task)
    return task


def _deregister_slow_call(task: asyncio.Task | None) -> None:
    if task is not None:
        _slow_call_tasks.discard(task)


def _cancel_in_flight_slow_calls() -> None:
    """Called at conversation-start to free the brain for low-latency
    reply. Cancels any in-flight extract_and_store / rollup summary that
    registered itself via _register_slow_call. The cancelled task's
    httpx request raises CancelledError; calling code's try/finally
    blocks fire to release locks / flags. No-op on separate servers."""
    if not _conversation_shares_brain():
        return
    cancelled = 0
    for task in list(_slow_call_tasks):
        if not task.done():
            task.cancel()
            cancelled += 1
    if cancelled:
        log.info(
            "[CONV-PRIORITY] cancelled %d in-flight slow call(s) to free Qwen3.6",
            cancelled,
        )
    _slow_call_tasks.clear()

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
    """Stream tokens from the conversation tier via /v1/chat/completions SSE.

    Routes to config.LLM_CONVERSATION_URL (Llama 3.2 3B on :8081 by default;
    Qwen3.6 brain on :8083 when the LT-OS dropdown is set to qwen36).
    Phase 1 priority gate: if conversation and memory share a server, cancel
    any in-flight slow call before starting and mark the conversation as
    in-flight so new slow calls wait.
    """
    _cancel_in_flight_slow_calls()
    _conversation_in_flight.set()
    client = await _get_client()
    payload = {
        "messages": messages,
        "max_tokens": max_tokens or config.CONVERSATION_MAX_TOKENS,
        "temperature": config.CONVERSATION_TEMPERATURE,
        "stream": True,
    }
    # When routed to the Qwen3.6 brain, suppress thinking on conversation
    # turns -- thinking-on adds 15-45 s of chain-of-thought before the first
    # token. Llama 3B ignores the kwarg.
    if _conversation_shares_brain():
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
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
    finally:
        _conversation_in_flight.clear()


async def generate_memory(prompt: str, thinking: bool | None = None) -> str:
    """Send a prompt to the brain LLM (LLM_MEMORY_URL) for memory extraction. Non-streaming.

    `thinking`: when True/False, sends `chat_template_kwargs:{enable_thinking:bool}`
    so a Qwen3.6-style server gates the thinking trace per-request. None preserves
    legacy behavior (no kwarg, server default applies).

    Phase 1 priority gate: when conversation and memory share a server, this
    call blocks until conversation is idle and registers itself so a future
    conversation request can cancel it.
    """
    await _wait_for_conversation_idle()
    task = _register_slow_call()
    try:
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
    finally:
        _deregister_slow_call(task)


async def generate_summary(turns_text: str) -> str:
    """Summarize conversation turns using Qwen3.6 thinking-on at LLM_MEMORY_URL.

    Phase 1 priority gate: when conversation and memory share a server, this
    call blocks until conversation is idle and registers itself so a future
    conversation request can cancel it.
    """
    await _wait_for_conversation_idle()
    task = _register_slow_call()
    try:
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
    finally:
        _deregister_slow_call(task)
