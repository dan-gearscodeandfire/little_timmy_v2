"""First-pass tool-call router (2026-06-18).

Runs BEFORE the conversation brain. Each utterance is classified by a small,
dedicated, GBNF-constrained Qwen3-4B server (config.LLM_CLASSIFIER_URL, :8092).
A recognized intent routes the turn to a tool instead of the :8083 reply; any
non-tool utterance falls through to the normal pipeline.

First slice ships ONE tool: `store_fact`. Two-tier, consequence-tiered design:
  Tier 1 (route)  -- cheap, every enabled turn: store_fact | none
  Tier 2 (args)   -- only on a store_fact route: subject/predicate/value
store_fact is side-effecting (writes/supersedes a DB row), so the Tier-2
deliberation pass only pays its cost on a confirmed hit.

Key invariant: this path NEVER calls ConversationTurn.respond()/_memory.save()
(the sole trigger of background fact extraction). The triggering user turn is
already injected by the caller (main.process_speech/process_text_input add it +
broadcast it before branching here); the router only injects the assistant ACK
turn. So the fact is written exactly once -- no extraction double-write.

The whole first-pass is gated by runtime_toggles "classifier_enabled" (default
OFF). Every classifier failure path returns falsy -> caller falls through, so a
classifier-server outage degrades gracefully and never drops a turn.
"""

import json
import logging
import random
from pathlib import Path

from llm import client
from memory.facts import store_fact
from memory.extraction import _normalize_subject
from persistence import runtime_toggles

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_ROUTE_PROMPT = (_PROMPTS_DIR / "classify_route.txt").read_text(encoding="utf-8")
_ARGS_PROMPT = (_PROMPTS_DIR / "store_fact_args.txt").read_text(encoding="utf-8")


def _load_acks() -> list[str]:
    """Re-read the canned ACK lines each call so they can be edited live."""
    try:
        raw = (_PROMPTS_DIR / "store_fact_acks.txt").read_text(encoding="utf-8")
    except Exception:
        return ["Noted."]
    acks = [ln.strip() for ln in raw.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    return acks or ["Noted."]


# GBNF grammars -- the server emits ONLY these shapes (mirrors the benchmarked
# tool_router_bench_constrained.py). char excludes the JSON string delimiters so
# the emitted object is always parseable.
_ROUTE_GRAMMAR = r'''
root        ::= store-call | none-call
store-call  ::= "{\"tool\":\"store_fact\"}"
none-call   ::= "{\"tool\":\"none\"}"
'''

_ARGS_GRAMMAR = r'''
root  ::= "{\"subject\":\"" str "\",\"predicate\":\"" str "\",\"value\":\"" str "\"}"
str   ::= char+
char  ::= [^"\\]
'''


async def classify_intent(user_text: str) -> str | None:
    """Tier-1 route. Returns 'store_fact' | 'none', or None on any classifier error."""
    content = await client.classify_constrained(
        messages=[{"role": "system", "content": _ROUTE_PROMPT},
                  {"role": "user", "content": user_text}],
        grammar=_ROUTE_GRAMMAR,
        max_tokens=16,
    )
    if not content:
        return None
    try:
        return json.loads(content).get("tool")
    except (json.JSONDecodeError, AttributeError):
        log.warning("[CLASSIFIER] unparseable route output: %r", content)
        return None


async def extract_store_fact_args(user_text: str) -> dict | None:
    """Tier-2 args. Returns {'subject','predicate','value'} or None on error."""
    content = await client.classify_constrained(
        messages=[{"role": "system", "content": _ARGS_PROMPT},
                  {"role": "user", "content": user_text}],
        grammar=_ARGS_GRAMMAR,
    )
    if not content:
        return None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        log.warning("[CLASSIFIER] unparseable args output: %r", content)
        return None
    if not isinstance(obj, dict):
        return None
    return obj


async def maybe_handle_tool_call(
    user_text: str,
    speaker_name: str | None,
    speaker_db_id: int | None,
    conversation,
    tts,
) -> bool:
    """Classify `user_text`; if it routes to a tool, execute it and return True
    (the caller must then early-return, skipping the normal LLM pipeline).
    Returns False to fall through to normal conversation.

    The caller has ALREADY added + broadcast the user turn before calling this,
    so on a tool hit we only inject the assistant ACK turn.
    """
    if not runtime_toggles.get("classifier_enabled"):
        return False

    route = await classify_intent(user_text)
    if route != "store_fact":
        return False  # 'none', None (error), or unknown tool -> normal pipeline

    args = await extract_store_fact_args(user_text)
    if not args or not args.get("subject") or not args.get("value"):
        # Grammar guarantees shape, not sense -- a failed/empty extraction
        # degrades to a normal conversational reply rather than a junk fact.
        log.info("[TOOL store_fact] extraction empty/failed (%r); falling through", args)
        return False

    subject = _normalize_subject(args["subject"], speaker_name)
    predicate = (args.get("predicate") or "fact").strip()
    value = args["value"].strip()
    try:
        await store_fact(subject, predicate, value, speaker_id=speaker_db_id)
    except Exception:
        log.exception("[TOOL store_fact] store_fact failed; falling through to normal reply")
        return False

    ack = random.choice(_load_acks())
    # Inject the assistant ACK turn ONLY (user turn already added by caller).
    # No respond()/save() -> background extraction never fires on this turn.
    await conversation.add_assistant_turn(ack)
    # Plain speak (NOT force): respects tts_muted and sets capture suppression so
    # the spoken ACK isn't transcribed back as a new user utterance.
    await tts.speak(ack)

    # Publish to both UI surfaces (LT-OS via WS relay, Booth via /api/metrics poll).
    try:
        from web.app import broadcast_event, update_metrics
        import time
        await broadcast_event("turn", {"role": "assistant", "content": ack})
        await broadcast_event("tool_call", {
            "name": "store_fact",
            "subject": subject, "predicate": predicate, "value": value,
        })
        update_metrics(last_tool_call="store_fact", last_tool_call_ts=time.time())
    except Exception:
        log.debug("[TOOL store_fact] UI publish failed (non-fatal)", exc_info=True)

    log.info("[TOOL store_fact] %s.%s = %s (speaker=%s) -> %r",
             subject, predicate, value, speaker_name, ack)
    return True
