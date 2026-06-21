"""LLM client for llama.cpp servers (conversation on 8081, memory on 8080)."""

import asyncio
import contextlib
import json
import logging
import time
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
# Vision-priority gate (2026-06-20, Dan): memory/extraction now shares the
# :8084 VISION server (own KV cache, off the :8083 conversation slot). On that
# shared server a VLM scene call takes priority over a background extraction/
# rollup the same way conversation does on :8083 — set while a vision call is
# in flight so new slow calls wait, and a starting vision call cancels any
# in-flight slow call. Only meaningful when LLM_MEMORY_URL == LLM_VISION_URL.
_vision_in_flight = asyncio.Event()
# Wall-clock of the last conversation-stream activity (stamped at stream start
# AND end). Lets "idle" mean "the user has actually been quiet for a beat",
# not just "no stream is in flight this microsecond" -- so background slow
# calls defer through the gaps between rapid turns instead of firing into them
# and getting cancelled mid-decode. 0.0 == no conversation has happened yet.
_last_conversation_activity_ts = 0.0


def _idle_gate_seconds() -> float:
    """Quiet-gap (seconds) a conversation slow call waits for before issuing,
    ON TOP OF the in-flight check. Read live from runtime_toggles so Dan can
    tune the 'wait until I'm not actively using you' window without a restart.
    0 == legacy behavior (defer only while a stream is literally in flight)."""
    try:
        from persistence import runtime_toggles
        v = runtime_toggles.get("conversation_idle_gate_seconds")
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def conversation_active(window: float | None = None) -> bool:
    """The shared 'is Dan actively conversing with Timmy right now' signal:
    True if a stream is in flight, or if fewer than `window` seconds have
    elapsed since the last conversation activity (defaults to the live
    conversation_idle_gate_seconds toggle). Consumed by /api/active -- and
    through it the demerzel-mail ingest loop, which polls it to defer email
    fetches while LT is talking -- and mirrors the in-process slow-call idle
    gate below. Pure activity signal, independent of brain topology."""
    if _conversation_in_flight.is_set():
        return True
    w = _idle_gate_seconds() if window is None else float(window)
    if w <= 0.0 or _last_conversation_activity_ts <= 0.0:
        return False
    return (time.time() - _last_conversation_activity_ts) < w


def _current_conversation_url() -> str:
    """Return the URL to use for the next conversation request. Prefers
    the runtime override (set by LT-OS when the dropdown picks an
    external-service model like qwen36); falls back to the static
    config.LLM_CONVERSATION_URL otherwise."""
    try:
        from persistence import runtime_toggles
        override = runtime_toggles.get("conversation_url_override")
    except Exception:
        override = None
    if override:
        return str(override)
    return config.LLM_CONVERSATION_URL


def _conversation_shares_brain() -> bool:
    """True when conversation routes to the same server as memory."""
    return _current_conversation_url() == config.LLM_MEMORY_URL


def _memory_shares_vision() -> bool:
    """True when memory/extraction routes to the same server as vision, so a
    VLM scene call and a background extraction/rollup contend for one slot +
    KV cache (the :8084 co-location, 2026-06-20). Vision wins on that slot."""
    return config.LLM_MEMORY_URL == config.LLM_VISION_URL


def _memory_contended() -> bool:
    """True when some priority holder (conversation OR vision) shares the
    memory server, so a slow call must register itself as cancellable and wait
    its turn. False == memory has its own server; slow calls run unimpeded."""
    return _conversation_shares_brain() or _memory_shares_vision()


async def _wait_for_conversation_idle() -> None:
    """Block until the conversation brain is free for background work: no
    stream in flight AND at least conversation_idle_gate_seconds elapsed since
    the last conversation activity. No-op when conversation and memory live on
    separate servers. The widened window (vs. the bare in-flight check) keeps
    memory extraction / rollup from firing into the micro-gaps between rapid
    turns and then getting cancelled mid-decode -- which left the single shared
    :8083 slot server-side busy and stalled the next reply (the 43s hang Dan
    hit live 2026-06-20). Cancellable: callers run as registered slow tasks."""
    if not _conversation_shares_brain():
        return
    window = _idle_gate_seconds()
    while True:
        if _conversation_in_flight.is_set():
            await asyncio.sleep(0.1)
            continue
        if window > 0.0:
            elapsed = time.time() - _last_conversation_activity_ts
            if elapsed < window:
                await asyncio.sleep(min(0.5, window - elapsed))
                continue
        return


async def _wait_for_vision_idle() -> None:
    """Block while a vision (VLM) call holds the shared memory server, so a
    background extraction/rollup yields the :8084 slot to scene analysis.
    No-op unless memory and vision share a server. Cancellable: callers run as
    registered slow tasks, so a vision call mid-wait simply keeps us parked."""
    if not _memory_shares_vision():
        return
    while _vision_in_flight.is_set():
        await asyncio.sleep(0.05)


@contextlib.asynccontextmanager
async def vision_priority():
    """Wrap a vision (VLM) call so it takes priority over background memory work
    on the shared :8084 server: cancel any in-flight extraction/rollup to free
    the slot, mark vision in-flight (new slow calls wait via _wait_for_vision_
    idle), and clear on exit. No-op unless memory and vision share a server
    (otherwise vision owns the server and there's nothing to arbitrate)."""
    if not _memory_shares_vision():
        yield
        return
    _cancel_in_flight_slow_calls("VISION-PRIORITY")
    _vision_in_flight.set()
    try:
        yield
    finally:
        _vision_in_flight.clear()


def _register_slow_call() -> asyncio.Task | None:
    """Mark the current task as a slow call so a priority holder (a user-facing
    conversation reply on a shared brain, or a vision scene call on the shared
    :8084 server) can cancel it. Returns the task so the caller can deregister
    in a finally block. No-op when memory has its own uncontended server."""
    if not _memory_contended():
        return None
    task = asyncio.current_task()
    if task is not None:
        _slow_call_tasks.add(task)
    return task


def _deregister_slow_call(task: asyncio.Task | None) -> None:
    if task is not None:
        _slow_call_tasks.discard(task)


def _cancel_in_flight_slow_calls(reason: str = "CONV-PRIORITY") -> None:
    """Cancel any in-flight extract_and_store / rollup summary that registered
    itself via _register_slow_call, to free the shared slot for a higher-
    priority caller. The cancelled task's httpx request raises CancelledError;
    calling code's try/finally blocks fire to release locks / flags. The CALLER
    guards on whether it actually shares the slot (conversation guards on
    _conversation_shares_brain; vision on _memory_shares_vision) — this is
    unconditional so it can serve both priority holders. `reason` only labels
    the log line."""
    cancelled = 0
    for task in list(_slow_call_tasks):
        if not task.done():
            task.cancel()
            cancelled += 1
    if cancelled:
        log.info(
            "[%s] cancelled %d in-flight slow call(s) to free Qwen3.6",
            reason, cancelled,
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
    global _last_conversation_activity_ts
    # Only cancel background slow calls when conversation actually shares the
    # memory server (the :8083 case). With memory on its own :8084 slot, an
    # extraction is NOT on the conversation slot, so a turn must NOT cancel it
    # (that's vision's job now, via vision_priority()).
    if _conversation_shares_brain():
        _cancel_in_flight_slow_calls("CONV-PRIORITY")
    _conversation_in_flight.set()
    _last_conversation_activity_ts = time.time()
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

    conv_url = _current_conversation_url()
    try:
        async with client.stream(
            "POST",
            f"{conv_url}/v1/chat/completions",
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
        # Re-stamp on END so the idle window measures quiet-since-last-reply,
        # not quiet-since-stream-start (a long reply would otherwise let the
        # window expire before the user even finishes hearing it).
        _last_conversation_activity_ts = time.time()


async def generate_memory(prompt: str, thinking: bool | None = None) -> str:
    """Send a prompt to the brain LLM (LLM_MEMORY_URL) for memory extraction. Non-streaming.

    `thinking`: when True/False, sends `chat_template_kwargs:{enable_thinking:bool}`
    so a Qwen3.6-style server gates the thinking trace per-request. None preserves
    legacy behavior (no kwarg, server default applies).

    Priority gate: when conversation and memory share a server, this call
    blocks until conversation is idle; when memory and vision share the :8084
    server, it also blocks while a VLM scene call is in flight (vision wins).
    Registers itself so the relevant priority holder can cancel it.
    """
    await _wait_for_conversation_idle()
    await _wait_for_vision_idle()
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


async def classify_constrained(
    messages: list[dict],
    grammar: str,
    max_tokens: int | None = None,
) -> str | None:
    """GBNF-constrained, thinking-OFF call to the first-pass classifier server
    (config.LLM_CLASSIFIER_URL, :8092 by default). Returns the raw assistant
    content (grammar-guaranteed valid for the caller to json.loads), or None on
    ANY failure (timeout, connection error, non-200, empty content).

    Deliberately does NOT touch the conversation-priority gate: the classifier
    runs on its OWN server (separate KV cache), inline before every turn, so
    there is no shared-brain contention to arbitrate. Returning None on every
    error path is the graceful-degradation contract — the caller falls through
    to the normal conversation pipeline, so a classifier outage never drops a
    turn.
    """
    try:
        client = await _get_client()
        payload = {
            "messages": messages,
            "grammar": grammar,
            "temperature": config.CLASSIFIER_TEMPERATURE,
            "max_tokens": max_tokens or config.CLASSIFIER_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        resp = await client.post(
            f"{config.LLM_CLASSIFIER_URL}/v1/chat/completions",
            json=payload,
            timeout=config.CLASSIFIER_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        return content or None
    except Exception as e:
        log.warning("[CLASSIFIER] call failed (%s); falling through to normal pipeline", e)
        return None


_RESOLVE_SYS = (
    "Rewrite the user's LAST message as a single standalone search query. Replace "
    "every pronoun or vague reference (it, its, that, there, them, they, he, him, "
    "she, her) with the specific thing it refers to, taken from the conversation. "
    "Keep it a short question. If nothing needs resolving, repeat the message "
    "unchanged. Output ONLY the rewritten query, nothing else."
)


async def resolve_query(utterance: str, context_text: str) -> str | None:
    """Coreference-resolve an elliptical utterance into a standalone search query
    via the :8092 classifier server (UNCONSTRAINED, thinking-OFF). `context_text`
    is the recent conversation transcript supplying antecedents. Returns the
    rewritten query, or None on ANY failure / empty output (callers fall back to
    the bare/blended query -- graceful degradation, same contract as
    classify_constrained). Extractive by design: the model resolves the REFERENCE
    from context, it does not invent the answer (cf. the rejected HyDE arm)."""
    try:
        client = await _get_client()
        convo = f"{context_text}\nUser: {utterance}" if context_text else f"User: {utterance}"
        resp = await client.post(
            f"{config.LLM_CLASSIFIER_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "system", "content": _RESOLVE_SYS},
                             {"role": "user", "content": convo}],
                "temperature": config.CLASSIFIER_TEMPERATURE,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            },
            timeout=config.CLASSIFIER_TIMEOUT_S,
        )
        resp.raise_for_status()
        content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        if "</think>" in content:  # strip any stray thinking block
            content = content.split("</think>")[-1].strip()
        return content.strip().strip('"') or None
    except Exception as e:
        log.warning("[RESOLVE] query resolution failed (%s); using bare/blended query", e)
        return None


async def generate_summary(turns_text: str) -> str:
    """Summarize conversation turns using Qwen3.6 thinking-OFF at LLM_MEMORY_URL.

    Thinking must stay OFF here: llama.cpp ignores thinking_budget for Qwen3,
    so with thinking on the model burns the entire max_tokens budget in
    reasoning_content and content comes back empty (finish_reason=length) —
    every rollup was a silent no-op that still occupied the shared :8083 slot.

    Priority gate: when conversation and memory share a server, this call
    blocks until conversation is idle; when memory and vision share the :8084
    server, it also blocks while a VLM scene call is in flight (vision wins).
    Registers itself so the relevant priority holder can cancel it.
    """
    await _wait_for_conversation_idle()
    await _wait_for_vision_idle()
    task = _register_slow_call()
    try:
        client = await _get_client()
        payload = {
            "messages": [
                {"role": "user", "content": (
                    "Summarize this conversation segment so it can be searched "
                    "and recalled later by date. Write for a reader who wasn't "
                    "there.\n\n"
                    "Start with a one-line topic header. Then, in about 4-6 "
                    "sentences, record the specifics: who said or did what, the "
                    "exact proper nouns (people, places, products, projects), any "
                    "dates, times, or numbers mentioned, and any decisions, plans, "
                    "or open questions. Prefer concrete detail over generality — "
                    "name names instead of saying 'someone', quote the actual "
                    "figure instead of 'a number'. Do not add information that "
                    "isn't in the segment, and do not editorialize.\n\n"
                    f"{turns_text}"
                )},
            ],
            "max_tokens": 800,
            "temperature": 0.3,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await client.post(
            f"{config.LLM_MEMORY_URL}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if not content:
            log.warning(
                "Summary call returned empty content (finish_reason=%s, usage=%s)",
                data["choices"][0].get("finish_reason"),
                data.get("usage"),
            )
        await _maybe_emit_reasoning("summary", data, content)
        return content
    finally:
        _deregister_slow_call(task)
