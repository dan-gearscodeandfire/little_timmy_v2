"""Async memory formation via the brain LLM at config.LLM_MEMORY_URL (fire-and-forget).

Two-pass design: a thinking-OFF yes/no classifier first decides if there is
anything durable in the exchange; only on yes do we make the slow thinking-ON
extraction call. Saves substantial GPU on the common case where most exchanges
produce empty arrays.
"""

import asyncio
import json
import logging
import time
from collections import deque
from memory.manager import store_memory
from memory.facts import store_fact
from llm.client import generate_memory
import config

log = logging.getLogger(__name__)

# Two-stage pipeline:
#
#   extract_and_store()  -->  _pending (debounce buffer)
#                                   |  debounce / max-hold timer fires
#                                   v
#                             _do_flush(): coalesce by speaker
#                                   |
#                                   v
#                             _queue (serialized executor)  -->  _pump --> _do_extraction
#
# Stage 1 (_pending + debounce timer, 2026-06-06): each conversation turn is
# buffered, NOT extracted immediately. A new turn re-arms the timer, so during a
# lively back-and-forth no extraction ever STARTS -- which is the whole point.
# The earlier per-turn design started a fresh extraction every turn and let the
# priority gate cancel it the instant the user spoke again; task.cancel() only
# drops the httpx connection while llama.cpp keeps computing the abandoned
# generation server-side, so a burst stacked abandoned-but-running generations on
# the single-slot (-np 1) Strix Halo brain -> amdgpu hard-wedge (okDemerzel freeze
# 2026-05-12, 2026-06-06). Debouncing means extraction only fires after
# EXTRACTION_DEBOUNCE_SECONDS of quiet (or EXTRACTION_MAX_HOLD_SECONDS of unbroken
# chatter), when the conversation is idle and nothing will be cancelled.
#
# Stage 2 (_queue + _pump, the proven 2026-06-03 executor): at flush time the
# buffer is coalesced -- grouped by speaker, all of one speaker's turns joined
# into ONE classifier+extraction pass instead of one-per-turn -- and the coalesced
# items feed the existing single-flight queue. The cancel -> _requeue safety net
# stays underneath: if the rare lull-time extraction IS cancelled by a resuming
# conversation, it re-enqueues as one coalesced item (never a per-turn explosion)
# and parks on generate_memory's _wait_for_conversation_idle until the next lull.
_pending: deque[dict] = deque()   # stage 1: raw per-turn exchanges awaiting debounce
_queue: deque[dict] = deque()     # stage 2: coalesced items awaiting serialized extraction
# True while a _do_extraction task is active. Check+set is atomic in
# single-threaded asyncio (no await between check in _pump and the set).
_extraction_running = False
# Pending debounce timer + when the current buffer started filling (loop.time()).
_flush_handle: "asyncio.TimerHandle | None" = None
_buffer_started: float | None = None


# L7 2026-05-14: prompts are externalized so they version-control cleanly
# and can be tweaked without code edits. Lives in
# ~/little_timmy/prompts/{classify_durable,extract_facts}.txt. Loaded once
# at module import; if a file is missing or unreadable LT fails loudly at
# startup rather than silently misbehaving on the first turn.
from pathlib import Path as _Path
_PROMPTS_DIR = _Path(__file__).resolve().parent.parent / "prompts"
_CLASSIFIER_PROMPT = (_PROMPTS_DIR / "classify_durable.txt").read_text(encoding="utf-8")
_EXTRACTION_PROMPT = (_PROMPTS_DIR / "extract_facts.txt").read_text(encoding="utf-8")


_SELF_REFERENCE_SUBJECTS = frozenset({"user", "i", "me", "myself"})


def _normalize_subject(subj: str, speaker_name: str | None) -> str:
    """Map LLM self-reference subjects to the canonical speaker name.

    No-op if speaker_name is missing or the subject is already a proper
    name / non-self-reference. Case-insensitive matching; result is
    always lowercase (matches memory.facts.store_fact normalization).
    """
    if not speaker_name:
        return subj
    name = speaker_name.strip().lower()
    if not name or name.startswith("unknown"):
        return subj
    lowered = subj.lower().strip()
    if lowered in _SELF_REFERENCE_SUBJECTS:
        return name
    # Handle possessive self-references like "user's wife", "my wife".
    # Preserve the relation suffix, only swap the leading possessor.
    for prefix in ("user's ", "user’s ", "user' s ", "my "):
        if lowered.startswith(prefix):
            return f"{name}'s {lowered[len(prefix):].strip()}"
    return subj


async def extract_and_store(
    user_text: str,
    assistant_text: str,
    speaker_id: int | None = None,
    speaker_name: str | None = None,
):
    """Buffer a conversation exchange for debounced, coalesced extraction.

    Returns fast (just buffers + arms a timer) -- the actual two-pass
    classifier+extraction runs later, only after the conversation has been
    quiet for config.EXTRACTION_DEBOUNCE_SECONDS (or buffering has hit
    config.EXTRACTION_MAX_HOLD_SECONDS). See the module-level pipeline note.

    speaker_name (when known) is used to normalize self-reference subjects
    in the LLM output: "user" / "i" / "me" become the canonical name,
    and "user\'s X" becomes "<name>\'s X". Pairs with the retrieval-side
    alias in memory.facts.get_facts_about_speaker so the write path lands
    canonical going forward while the historical "user" rows stay
    surfaceable via the alias.
    """
    # Skip extraction for very short/empty user messages (likely STT hallucinations)
    stripped = user_text.strip()
    if len(stripped) < 15 and not any(c.isupper() for c in stripped[1:]):
        log.debug("Skipping extraction - user text too short: %r", stripped)
        return

    # Buffer (bounded). On overflow, drop the OLDEST pending exchange with a
    # WARN -- never silently truncate (memory-hygiene "no silent caps").
    if len(_pending) >= config.EXTRACTION_QUEUE_MAX:
        dropped = _pending.popleft()
        log.warning(
            "Extraction buffer full (%d); dropping oldest pending exchange: %r",
            config.EXTRACTION_QUEUE_MAX, dropped["user_text"][:60],
        )
    _pending.append({
        "user_text": user_text,
        "assistant_text": assistant_text,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "ts": time.time(),   # source-turn wall clock, for store_fact recency gate
        "retries": 0,
    })
    _arm_flush()


def _arm_flush() -> None:
    """(Re)arm the debounce timer after a turn is buffered. A new turn pushes
    the flush out by EXTRACTION_DEBOUNCE_SECONDS, so a lively back-and-forth
    keeps deferring extraction until it lulls -- bounded by
    EXTRACTION_MAX_HOLD_SECONDS measured from when the buffer started filling,
    so an unbroken monologue still flushes."""
    global _flush_handle, _buffer_started
    loop = asyncio.get_running_loop()
    now = loop.time()
    if _buffer_started is None:
        _buffer_started = now
    # Cap by remaining hold budget so a non-stop talker can't defer forever.
    remaining_hold = config.EXTRACTION_MAX_HOLD_SECONDS - (now - _buffer_started)
    if remaining_hold <= 0:
        _do_flush()
        return
    delay = min(config.EXTRACTION_DEBOUNCE_SECONDS, remaining_hold)
    if _flush_handle is not None:
        _flush_handle.cancel()
    _flush_handle = loop.call_later(delay, _do_flush)


def _do_flush() -> None:
    """Debounce fired: drain the buffer, coalesce by speaker into the executor
    queue, and start draining. Runs in the event loop (call_later), so it only
    builds items + calls _pump -- no awaits."""
    global _flush_handle, _buffer_started
    if _flush_handle is not None:
        _flush_handle.cancel()
    _flush_handle = None
    _buffer_started = None
    if not _pending:
        return
    items = list(_pending)
    _pending.clear()
    coalesced = _coalesce_by_speaker(items)
    log.info("Extraction flush: %d buffered turn(s) -> %d coalesced pass(es)",
             len(items), len(coalesced))
    _queue.extend(coalesced)
    _pump()


def _coalesce_by_speaker(items: list[dict]) -> list[dict]:
    """Group buffered turns by speaker (preserving first-seen order) and join
    each group's turns into one extraction item. Coalescing per-speaker keeps
    store_fact's speaker_id / _normalize_subject(speaker_name) attribution
    correct when two people talked during the same lull (usually it's one).
    User/assistant texts are newline-joined; the extractor only mines USER
    statements, so a block of them reads naturally under the prompt's
    'User:' framing."""
    groups: dict = {}
    order: list = []
    for it in items:
        key = (it["speaker_id"], it["speaker_name"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(it)
    out: list[dict] = []
    for key in order:
        group = groups[key]
        speaker_id, speaker_name = key
        out.append({
            "user_text": "\n".join(g["user_text"] for g in group),
            "assistant_text": "\n".join(g["assistant_text"] for g in group),
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            # OLDEST turn in the coalesced group gates the recency check. A
            # buffer can STRADDLE an explicit correction (pre-correction stale
            # mentions + a newer turn). Gating on the oldest turn means the
            # extractor may overwrite a tool-written fact only if the ENTIRE
            # evidence window post-dates the correction -- otherwise the stale
            # mention is held back (AMBIGUITY: prior value retained, self-heals
            # on the next clean pass) instead of clobbering it (FALSE). See
            # ops/test_coalesced_correction_race.py and the 2026-06-21 race note.
            "ts": min(g["ts"] for g in group),
            "retries": 0,
        })
    return out


def _pump() -> None:
    """Start the next queued extraction if none is running. Called from the
    flush and from each _do_extraction's finally so the queue drains serially."""
    global _extraction_running
    if _extraction_running or not _queue:
        return
    item = _queue.popleft()
    _extraction_running = True
    # Own task (not awaited by the pump) so the priority gate cancels the
    # extraction child, never the caller -- and so the finally can re-pump.
    asyncio.create_task(_do_extraction(item))


def _requeue(item: dict) -> None:
    """Push a cancelled exchange back for another attempt, bounded by
    EXTRACTION_MAX_RETRIES."""
    item["retries"] += 1
    if item["retries"] > config.EXTRACTION_MAX_RETRIES:
        log.warning(
            "Extraction dropped after %d cancellations: %r",
            item["retries"] - 1, item["user_text"][:60],
        )
        return
    _queue.appendleft(item)  # front of line: retry before newer exchanges


async def _do_extraction(item: dict):
    global _extraction_running
    user_text = item["user_text"]
    assistant_text = item["assistant_text"]
    speaker_id = item["speaker_id"]
    speaker_name = item["speaker_name"]
    try:
        # Pass 1: cheap thinking-OFF classifier
        classify_prompt = _CLASSIFIER_PROMPT.format(
            user_text=user_text, assistant_text=assistant_text
        )
        verdict = await generate_memory(classify_prompt, thinking=False)
        verdict_clean = (verdict or "").strip().lower()
        first_word = verdict_clean.split()[0].rstrip(".,!?:;") if verdict_clean else ""
        if first_word != "yes":
            log.debug("Memory classifier verdict: %r (skipping extraction)",
                      first_word or verdict_clean[:30])
            return
        log.info("Memory classifier said yes - running full extraction")

        # Pass 2: thinking-OFF structured extraction. Prompt loaded from
        # prompts/extract_facts.txt at module import; formatted with the
        # turn-specific user/assistant text here. Was thinking=True until
        # 2026-06-06: the ~1436-token CoT per attempt was the per-call
        # amplifier in the okDemerzel hard-freeze (cancel-churn of abandoned
        # thinking-ON generations on the single-slot -np1 Vulkan brain). This
        # is a structured-JSON tool, so thinking OFF matches the standing
        # thinking-OFF-for-structured-tools rule. (A/B extraction quality.)
        prompt = _EXTRACTION_PROMPT.format(
            user_text=user_text, assistant_text=assistant_text,
        )

        result = await generate_memory(prompt, thinking=False)
        if not result:
            return

        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(result)

        for fact_data in parsed.get("facts", []):
            subj = str(fact_data.get("subject", "")).strip()
            pred = str(fact_data.get("predicate", "")).strip()
            val = str(fact_data.get("value", "")).strip()
            if subj and pred and val:
                # Bundle B option b: rewrite self-reference subjects to the
                # canonical speaker_name so new rows don\'t accumulate under
                # the generic "user" bucket. The LLM prompt biases toward
                # subject="user" because the prompt frames the input as
                # "the user said X"; normalizing at store time is more
                # robust than tightening the prompt.
                subj_canonical = _normalize_subject(subj, speaker_name)
                await store_fact(subj_canonical, pred, val, speaker_id=speaker_id,
                                 source="extraction", turn_ts=item.get("ts"))

        # Vectorized memory creation is gated by config.PERSIST_EXTRACTED_MEMORIES
        # (default False since 2026-06-18, per Dan): the structured `facts` writer
        # above still runs; only the embedded episodic/semantic `memories` rows
        # are suppressed. See config.PERSIST_EXTRACTED_MEMORIES.
        mems = parsed.get("memories", [])
        stored_mems = 0
        if getattr(config, "PERSIST_EXTRACTED_MEMORIES", False):
            for mem_data in mems:
                mem_type = str(mem_data.get("type", "episodic")).strip()
                content = str(mem_data.get("content", "")).strip()
                if content and mem_type in ("episodic", "semantic", "procedural"):
                    await store_memory(mem_type, content, speaker_id=speaker_id)
                    stored_mems += 1
        elif mems:
            log.info("Skipped %d extracted memory(ies) "
                     "(PERSIST_EXTRACTED_MEMORIES=False)", len(mems))

        log.info("Memory extraction complete: %d facts, %d memories",
                 len(parsed.get("facts", [])), stored_mems)

    except asyncio.CancelledError:
        # Priority gate cancelled us mid-extraction so a user reply can use the
        # brain. Re-enqueue (bounded) so the work isn't lost; the retry parks on
        # generate_memory's idle-gate until the conversation lulls.
        _requeue(item)
        raise
    except json.JSONDecodeError as e:
        log.warning("Memory extraction JSON parse failed: %s", e)
    except Exception as e:
        log.error("Memory extraction error: %s", e)
    finally:
        _extraction_running = False
        _pump()  # drain the next queued exchange (or the just-requeued retry)
