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
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config
from llm import client
from memory.facts import store_fact
from memory.extraction import _normalize_subject
from persistence import runtime_toggles

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_ROUTE_PROMPT = (_PROMPTS_DIR / "classify_route.txt").read_text(encoding="utf-8")
# Extended route prompt (adds recall_semantic) — used ONLY when
# config.RECALL_SEMANTIC_ENABLED, so the default classifier stays byte-identical.
_ROUTE_PROMPT_SEM = (_PROMPTS_DIR / "classify_route_semantic.txt").read_text(encoding="utf-8")
_ARGS_PROMPT = (_PROMPTS_DIR / "store_fact_args.txt").read_text(encoding="utf-8")
_RECALL_ARGS_PROMPT = (_PROMPTS_DIR / "recall_temporal_args.txt").read_text(encoding="utf-8")

# Max episode summaries injected into one [WHAT WE TALKED ABOUT] block.
_RECALL_EPISODE_LIMIT = 12


@dataclass(frozen=True)
class ToolOutcome:
    """Result of the first-pass router.

    `handled=True`  -> the router OWNS the turn (terminal tool, e.g. store_fact);
                       the caller early-returns, skipping the brain.
    `handled=False` -> fall through to the normal conversation pipeline. If
                       `recall_block` is set (recall_temporal), the caller passes
                       it into _generate_response so the brain answers grounded
                       in the recalled episodes -- this is retrieval AUGMENTATION,
                       not a terminal action.
    """
    handled: bool = False
    recall_block: str | None = None


# Falsy-by-convention sentinel for the common "do nothing, fall through" path.
_FALLTHROUGH = ToolOutcome(handled=False, recall_block=None)


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
root         ::= store-call | recall-call | none-call
store-call   ::= "{\"tool\":\"store_fact\"}"
recall-call  ::= "{\"tool\":\"recall_temporal\"}"
none-call    ::= "{\"tool\":\"none\"}"
'''

# Extended grammar adding recall_semantic — selected ONLY when
# config.RECALL_SEMANTIC_ENABLED. Default OFF => _ROUTE_GRAMMAR above, unchanged.
_ROUTE_GRAMMAR_SEM = r'''
root          ::= store-call | recall-call | semantic-call | none-call
store-call    ::= "{\"tool\":\"store_fact\"}"
recall-call   ::= "{\"tool\":\"recall_temporal\"}"
semantic-call ::= "{\"tool\":\"recall_semantic\"}"
none-call     ::= "{\"tool\":\"none\"}"
'''

_ARGS_GRAMMAR = r'''
root  ::= "{\"subject\":\"" str "\",\"predicate\":\"" str "\",\"value\":\"" str "\"}"
str   ::= char+
char  ::= [^"\\]
'''

# Tier-2 for recall_temporal: extract the natural-language time phrase only
# ("last Saturday", "yesterday", "a couple weeks ago"). resolve_date_range
# turns it into a concrete window downstream.
_RECALL_ARGS_GRAMMAR = r'''
root  ::= "{\"phrase\":\"" str "\"}"
str   ::= char+
char  ::= [^"\\]
'''


async def classify_intent(user_text: str) -> str | None:
    """Tier-1 route. Returns 'store_fact' | 'recall_temporal' | 'none' (or also
    'recall_semantic' when config.RECALL_SEMANTIC_ENABLED), or None on any
    classifier error. The extended prompt/grammar is selected ONLY when the
    semantic flag is on, so the default routing stays byte-identical."""
    if getattr(config, "RECALL_SEMANTIC_ENABLED", False):
        prompt, grammar = _ROUTE_PROMPT_SEM, _ROUTE_GRAMMAR_SEM
    else:
        prompt, grammar = _ROUTE_PROMPT, _ROUTE_GRAMMAR
    content = await client.classify_constrained(
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": user_text}],
        grammar=grammar,
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


_VALUE_STOPWORDS = {
    "the", "a", "an", "my", "is", "are", "was", "named", "name", "called",
    "of", "to", "and", "his", "her", "their", "your", "i", "me",
}


def _value_grounded_in_utterance(utterance: str, value: str) -> bool:
    """True if the store_fact VALUE actually appears in the user's words.

    Tier-2 sometimes fabricates/recalls a value the user never said — e.g. the
    QUESTION "name my iguana" gets misrouted to store_fact and the extractor
    fills the answer "Nacho" from injected memory, then writes it as a NEW fact
    (data-corruption risk on a read-intent turn, 2026-06-20). Require at least
    one significant value token (len>=2, not a stopword) to appear in the
    utterance. Pure fabrications have zero overlap and fall through to the
    normal pipeline — whose background extractor stores genuine facts WITH full
    context, so nothing is lost (graceful: a false block just routes the store
    through the slower, more careful path). Lenient by design: normalized
    values (e.g. "June 3rd" from "june third") still share the "june" token."""
    if not utterance or not value:
        return False
    u = utterance.lower()
    toks = [t for t in re.findall(r"[a-z0-9]+", value.lower())
            if len(t) >= 2 and t not in _VALUE_STOPWORDS]
    if not toks:
        return True  # nothing checkable (value all stopwords) -> don't block
    return any(t in u for t in toks)


_SELF_NAMING_RE = re.compile(r"\b(my name|i am|i'm|call me|name's|name is)\b", re.I)
_POSSESSED_ENTITY_RE = re.compile(r"\bmy ([a-z]+)(?:'s)?\b", re.I)


def _speaker_name_overwrite_collapse(
    utterance: str, subject: str, predicate: str, speaker_name: str | None
) -> bool:
    """Guard the SPEAKER'S OWN name against the subject-collapse corruption.

    The Tier-2 args model sometimes drops a possessed entity and collapses the
    subject onto the speaker: "remember my robot is named Sparky" -> it emits
    subject="user", predicate="name", value="Sparky". After _normalize_subject
    that becomes <speaker>.name = Sparky, silently OVERWRITING the speaker's real
    name (store_fact's in-place ON CONFLICT upsert keeps no history -- root-caused
    2026-06-20, it nuked dan.name -> "Sparky"). The value IS grounded (Sparky is
    in the utterance) so _value_grounded_in_utterance can't catch it -- the
    SUBJECT is what's wrong.

    Block iff ALL hold: (a) the triple sets the speaker's own name
    (subject==speaker, predicate in name/is/is called); (b) the utterance is NOT
    actually self-naming ("my name is...", "call me...", "I'm..."); (c) it
    references a possessed entity ("my <noun>", noun != name). Falls through to
    the normal pipeline whose full-context extractor still stores the real fact,
    so nothing legitimate is lost. Self-naming ("remember my name is Dan") is
    unaffected -- (b) short-circuits."""
    if (predicate or "").strip().lower() not in ("name", "is", "is called"):
        return False
    spk = (speaker_name or "").strip().lower()
    if not spk or spk.startswith("unknown"):
        return False  # no enrolled name to protect (guest)
    if (subject or "").strip().lower() != spk:
        return False  # attributed to an entity, not the speaker -> legitimate
    u = utterance or ""
    if _SELF_NAMING_RE.search(u):
        return False  # genuine self-naming -> allow
    entities = [w.lower() for w in _POSSESSED_ENTITY_RE.findall(u)
                if w.lower() != "name"]
    return bool(entities)


async def extract_recall_phrase(user_text: str) -> str | None:
    """Tier-2 for recall_temporal. Returns the time phrase, or None on error."""
    content = await client.classify_constrained(
        messages=[{"role": "system", "content": _RECALL_ARGS_PROMPT},
                  {"role": "user", "content": user_text}],
        grammar=_RECALL_ARGS_GRAMMAR,
    )
    if not content:
        return None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        log.warning("[CLASSIFIER] unparseable recall-args output: %r", content)
        return None
    if not isinstance(obj, dict):
        return None
    return (obj.get("phrase") or "").strip()


def _fmt_local(dt: datetime) -> datetime:
    """Render a (tz-aware, UTC-from-DB) datetime in the host's local tz."""
    return dt.astimezone()


def _fmt_episode_line(span_start: datetime, span_end: datetime, text: str) -> str:
    s = _fmt_local(span_start)
    e = _fmt_local(span_end)
    if s.date() == e.date():
        when = f"{s.strftime('%a %b %d, %-I:%M')}–{e.strftime('%-I:%M %p')}"
    else:
        when = f"{s.strftime('%a %b %d %-I:%M %p')} – {e.strftime('%a %b %d %-I:%M %p')}"
    return f"- ({when}) {text.strip()}"


def _fmt_window_label(phrase: str, start: datetime, end: datetime) -> str:
    """Human label for the resolved window, e.g. 'Saturday, June 13' or
    'June 8-15'. Prefers the user's own phrase when present."""
    s = _fmt_local(start)
    e = _fmt_local(end)
    # Day-granular window (<= ~26h) -> name the single day.
    span_days = (e - s).total_seconds() / 86400.0
    if span_days <= 1.1:
        label = s.strftime("%A, %B %-d")
    else:
        label = f"{s.strftime('%B %-d')}–{(e).strftime('%B %-d')}"
    if phrase:
        return f"{phrase} ({label})"
    return label


def _build_recall_block(phrase: str, start: datetime, end: datetime,
                        episodes: list[dict]) -> str:
    """Format the [WHAT WE TALKED ABOUT] context block (hit or empty-marker)."""
    label = _fmt_window_label(phrase, start, end)
    if not episodes:
        return (
            f"[WHAT WE TALKED ABOUT] You have NO recorded conversation summaries "
            f"from {label}. Tell the user honestly that you don't have anything "
            f"saved from then; do NOT invent or guess what was discussed."
        )
    lines = [
        f"[WHAT WE TALKED ABOUT] Your recorded conversation summaries from "
        f"{label}. Answer the user's question using ONLY these; if they don't "
        f"cover what was asked, say so rather than guessing:"
    ]
    for ep in episodes:
        lines.append(_fmt_episode_line(ep["span_start"], ep["span_end"], ep["text"]))
    return "\n".join(lines)


async def _resolve_recall_block(user_text: str) -> str | None:
    """Run the recall_temporal tool: phrase -> date range -> episodes -> block.

    Returns the pre-formatted [WHAT WE TALKED ABOUT] block (including the
    empty-window 'nothing from then' marker), or None when no date range can be
    resolved at all (-> caller falls through to the normal pipeline unchanged)."""
    from memory.temporal import resolve_date_range
    from memory.manager import query_episodes_by_range

    phrase = await extract_recall_phrase(user_text)
    if phrase is None:
        phrase = ""  # extractor error -> fall back to scanning the raw utterance
    now = datetime.now().astimezone()
    # Prefer the extracted phrase; fall back to the whole utterance (the resolver
    # scans for a time expression anywhere in the string).
    rng = resolve_date_range(phrase, now) or resolve_date_range(user_text, now)
    if not rng:
        log.info("[TOOL recall_temporal] no date range from %r / %r; falling through",
                 phrase, user_text[:60])
        return None
    start, end = rng
    try:
        episodes = await query_episodes_by_range(start, end, limit=_RECALL_EPISODE_LIMIT)
    except Exception:
        log.exception("[TOOL recall_temporal] episode query failed; falling through")
        return None
    log.info("[TOOL recall_temporal] phrase=%r range=%s..%s -> %d episode(s)",
             phrase or f"(from utterance) {user_text[:40]}", start, end, len(episodes))
    return _build_recall_block(phrase, start, end, episodes)


def _build_semantic_block(episodes: list[dict]) -> str:
    """Format the [WHAT WE TALKED ABOUT] block for a topic (recall_semantic) hit,
    most-relevant first. Only called with a non-empty list (an empty semantic
    search falls through rather than asserting "nothing saved" — absence of an
    episode match doesn't prove we never discussed it; it may live in facts)."""
    lines = [
        "[WHAT WE TALKED ABOUT] Your recorded conversation summaries most "
        "relevant to the user's question, most relevant first. Answer using "
        "ONLY these; if they don't actually cover it, say so rather than guessing:"
    ]
    for ep in episodes:
        lines.append(_fmt_episode_line(ep["span_start"], ep["span_end"], ep["text"]))
    return "\n".join(lines)


async def _resolve_semantic_block(user_text: str) -> str | None:
    """Run recall_semantic: similarity search over episodes (recency-decayed) ->
    [WHAT WE TALKED ABOUT] block. Returns None (fall through) on error OR when
    nothing matches — see _build_semantic_block for why empty falls through."""
    from memory.episodic_search import search_episodes
    now = datetime.now().astimezone()
    try:
        episodes = await search_episodes(user_text, now)
    except Exception:
        log.exception("[TOOL recall_semantic] episode search failed; falling through")
        return None
    if not episodes:
        log.info("[TOOL recall_semantic] no episode match for %r; falling through",
                 user_text[:50])
        return None
    log.info("[TOOL recall_semantic] %r -> %d episode(s)", user_text[:50], len(episodes))
    return _build_semantic_block(episodes)


async def maybe_handle_tool_call(
    user_text: str,
    speaker_name: str | None,
    speaker_db_id: int | None,
    conversation,
    tts,
    t_start: float | None = None,
) -> ToolOutcome:
    """Classify `user_text` and route it. Returns a ToolOutcome:

    - terminal tool hit (store_fact): executes the tool, speaks the ACK, injects
      the assistant turn, returns ToolOutcome(handled=True) -> caller early-returns.
    - recall_temporal: returns ToolOutcome(handled=False, recall_block=...) ->
      caller passes the block into the normal pipeline (augmentation, not terminal).
    - everything else / any error: ToolOutcome(handled=False) -> fall through.

    The caller has ALREADY added + broadcast the user turn before calling this,
    so on a terminal tool hit we only inject the assistant ACK turn.
    """
    if not runtime_toggles.get("classifier_enabled"):
        return _FALLTHROUGH

    # Time the Tier-1 route -- the "first-pass tool-call filter" latency that every
    # utterance pays when the classifier is on. Published on EVERY turn (hit or
    # fall-through) so the HUDs show the per-turn cost, not just on tool hits.
    import time
    t0 = time.perf_counter()
    route = await classify_intent(user_text)
    classifier_ms = int((time.perf_counter() - t0) * 1000)
    try:
        from web.app import broadcast_event, update_metrics, record_stage
        update_metrics(last_classifier_ms=classifier_ms,
                       last_classifier_route_ms=classifier_ms,
                       last_classifier_args_ms=None)
        # Tier-1 route runs on EVERY turn the classifier is on -> this series'
        # sample count is the total classified-turn count since restart.
        record_stage("stage:classifier_route", classifier_ms)
        await broadcast_event("classifier_metric", {"ms": classifier_ms, "route": route})
    except Exception:
        log.debug("[CLASSIFIER] latency publish failed (non-fatal)", exc_info=True)

    # recall_temporal: retrieval AUGMENTATION, not a terminal tool. Gated
    # additionally by RECALL_TEMPORAL_ENABLED; when off, a recall route just
    # falls through to the normal pipeline (the brain answers from vector recall
    # as before). The user turn is NOT consumed here — we hand a context block
    # back and let _generate_response run the brain grounded in it.
    if route == "recall_temporal":
        if not getattr(config, "RECALL_TEMPORAL_ENABLED", False):
            return _FALLTHROUGH
        t_args = time.perf_counter()
        block = await _resolve_recall_block(user_text)
        args_ms = int((time.perf_counter() - t_args) * 1000)
        try:
            from web.app import update_metrics, record_stage, broadcast_event
            update_metrics(last_classifier_args_ms=args_ms,
                           last_classifier_ms=classifier_ms + args_ms)
            record_stage("stage:classifier_args", args_ms)
            if block is not None:
                update_metrics(last_tool_call="recall_temporal",
                               last_tool_call_ts=time.time())
                await broadcast_event("tool_call", {"name": "recall_temporal"})
        except Exception:
            log.debug("[TOOL recall_temporal] publish failed (non-fatal)", exc_info=True)
        # block is None when no date range could be resolved -> fall through with
        # no augmentation (recall_block=None is a no-op in build_ephemeral_block).
        return ToolOutcome(handled=False, recall_block=block)

    # recall_semantic: topic (no time) similarity recall over episodes, recency-
    # decayed. Same AUGMENTATION shape as recall_temporal. Gated by
    # RECALL_SEMANTIC_ENABLED; the route can only be emitted when the flag is on
    # (it selects the extended grammar), but we re-check defensively.
    if route == "recall_semantic":
        if not getattr(config, "RECALL_SEMANTIC_ENABLED", False):
            return _FALLTHROUGH
        t_args = time.perf_counter()
        block = await _resolve_semantic_block(user_text)
        args_ms = int((time.perf_counter() - t_args) * 1000)
        try:
            from web.app import update_metrics, record_stage, broadcast_event
            update_metrics(last_classifier_args_ms=args_ms,
                           last_classifier_ms=classifier_ms + args_ms)
            record_stage("stage:classifier_args", args_ms)
            if block is not None:
                update_metrics(last_tool_call="recall_semantic",
                               last_tool_call_ts=time.time())
                await broadcast_event("tool_call", {"name": "recall_semantic"})
        except Exception:
            log.debug("[TOOL recall_semantic] publish failed (non-fatal)", exc_info=True)
        return ToolOutcome(handled=False, recall_block=block)

    if route != "store_fact":
        return _FALLTHROUGH  # 'none', None (error), or unknown tool -> pipeline

    t_args = time.perf_counter()
    args = await extract_store_fact_args(user_text)
    args_ms = int((time.perf_counter() - t_args) * 1000)
    try:
        from web.app import update_metrics, record_stage
        # On a hit the turn pays route + args; surface the split and let the HUD
        # "Tool filter" row show the full classifier wall for this routed turn.
        update_metrics(last_classifier_args_ms=args_ms,
                       last_classifier_ms=classifier_ms + args_ms)
        record_stage("stage:classifier_args", args_ms)
    except Exception:
        log.debug("[CLASSIFIER] args-latency publish failed (non-fatal)", exc_info=True)
    if not args or not args.get("subject") or not args.get("value"):
        # Grammar guarantees shape, not sense -- a failed/empty extraction
        # degrades to a normal conversational reply rather than a junk fact.
        log.info("[TOOL store_fact] extraction empty/failed (%r); falling through", args)
        return _FALLTHROUGH

    if not _value_grounded_in_utterance(user_text, args["value"]):
        # The extracted value isn't in what the user actually said -- almost
        # always a recall QUESTION ("name my iguana") misrouted to store_fact
        # with the answer hallucinated from memory. Never persist it; fall
        # through so the normal pipeline answers (and its background extractor
        # handles any genuine fact with full context).
        log.warning(
            "[TOOL store_fact] value %r not grounded in utterance %r; falling "
            "through (likely a recall question misrouted to store)",
            args["value"], user_text[:80],
        )
        return _FALLTHROUGH

    subject = _normalize_subject(args["subject"], speaker_name)
    predicate = (args.get("predicate") or "fact").strip()
    value = args["value"].strip()

    if _speaker_name_overwrite_collapse(user_text, subject, predicate, speaker_name):
        # Subject-collapse corruption: the args model dropped a possessed entity
        # and aimed the speaker's own .name at a value from "my <thing> is named
        # X". Refuse -- never let this overwrite the speaker's real name. The
        # background extractor (full context) still records the real entity fact.
        log.warning(
            "[TOOL store_fact] BLOCKED speaker-name overwrite collapse: "
            "%s.%s=%r from utterance %r; falling through",
            subject, predicate, value, user_text[:80],
        )
        return _FALLTHROUGH

    try:
        await store_fact(subject, predicate, value, speaker_id=speaker_db_id,
                         source="tool")
    except Exception:
        log.exception("[TOOL store_fact] store_fact failed; falling through to normal reply")
        return _FALLTHROUGH

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
        # Tool-routed turn total wall (classifier+args+store+ack TTS). The brain
        # pipeline is skipped, so this is the ONLY place a routed turn's e2e and
        # its per-stage distribution get recorded.
        from web.app import record_turn_stats
        tool_e2e_ms = int((time.time() - t_start) * 1000) if t_start else None
        update_metrics(last_tool_turn_e2e_ms=tool_e2e_ms)
        record_turn_stats("tool", {
            "classifier_route": classifier_ms,
            "classifier_args": args_ms,
            "e2e": tool_e2e_ms,
        })
    except Exception:
        log.debug("[TOOL store_fact] UI publish failed (non-fatal)", exc_info=True)

    log.info("[TOOL store_fact] %s.%s = %s (speaker=%s) -> %r",
             subject, predicate, value, speaker_name, ack)
    return ToolOutcome(handled=True)
