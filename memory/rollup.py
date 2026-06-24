"""Sliding window conversation rollup: hot -> warm -> cold."""

import logging
import re
import time
from llm.client import generate_summary
from memory.manager import store_memory, store_episode
import config

log = logging.getLogger(__name__)

# Strip leading "Here is/Here's a summary..." preambles the conversation LLM
# tends to emit before the actual summary content. Otherwise it eats the
# 200-char retrieval truncation budget in build_ephemeral_block and bloats
# the in-context warm tier.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:\s+is|'s)\s+(?:a|the)\s+summary[^\n:]*[:.]?\s*)+",
    re.IGNORECASE,
)

# Skip persistence entirely when the LLM refused or had nothing to summarize.
_REFUSAL_RE = re.compile(
    r"^\s*there\s+is\s+no\s+conversation",
    re.IGNORECASE,
)

# Belt-and-suspenders for the summarizer's old habit of leading with a bold
# title line — sometimes stamping a FABRICATED date ("**Summary of Conversation
# on 2024-05-22: ...**"). The generate_summary prompt now forbids both a header
# and any unspoken date, but strip a leading title/header line defensively so a
# stray one never bloats the warm tier or, worse, misdates an episode that
# inherits this text. Only fires when a body follows (trailing \n+), so a
# legitimate single-line summary is never eaten.
_TITLE_RE = re.compile(
    r"^\s*\*{0,2}(?:summary\b|conversation\s+segment\b|discussion\s+regarding\b|"
    r"topic\s*:)[^\n]*\n+",
    re.IGNORECASE,
)


def _clean_summary(text: str) -> str | None:
    """Strip preamble + any leading title/header line; return None if the
    summary is a refusal or empty."""
    if not text:
        return None
    cleaned = _PREAMBLE_RE.sub("", text).strip()
    cleaned = _TITLE_RE.sub("", cleaned).strip()
    if not cleaned:
        return None
    if _REFUSAL_RE.match(cleaned):
        return None
    return cleaned


# The summarizer leads with a SUBSTANTIVE/BANTER verdict word on its own line
# (generate_summary prompt). Tolerant of optional **markdown**, [brackets], and a
# trailing ":"/"-" before the summary body (same line or next line).
_VERDICT_RE = re.compile(
    r"^\s*\*{0,2}\[?(SUBSTANTIVE|BANTER)\]?\*{0,2}\s*[:\-]?\s*\n?",
    re.IGNORECASE,
)


def _parse_summary(raw: str) -> tuple[bool, str | None]:
    """Split the summarizer output into (is_substantive, clean_summary).

    The first line is a SUBSTANTIVE/BANTER usefulness verdict that gates durable
    episode persistence (BANTER = don't embed; the 2026-06-18 low-density
    pollution guard). The verdict is stripped before the summary reaches the warm
    tier. Fail-safe toward persistence: a missing/garbled verdict is treated as
    SUBSTANTIVE and its (real) first line is preserved as summary content, so an
    LLM hiccup never silently drops useful gist."""
    if not raw:
        return True, None
    m = _VERDICT_RE.match(raw)
    is_substantive = True
    body = raw
    if m:
        is_substantive = m.group(1).upper() == "SUBSTANTIVE"
        body = raw[m.end():]
    return is_substantive, _clean_summary(body)


async def maybe_rollup(conversation) -> bool:
    """Check if hot buffer needs rolling up. Returns True if rollup occurred."""
    if not conversation.hot_turns:
        return False

    hot_tokens = sum(t.token_count for t in conversation.hot_turns)
    oldest_age = time.time() - conversation.hot_turns[0].timestamp

    if hot_tokens <= config.HOT_MAX_TOKENS and oldest_age <= config.ROLLUP_AGE_SECONDS:
        return False

    # Take oldest half of hot turns
    split = max(1, len(conversation.hot_turns) // 2)
    to_summarize = conversation.hot_turns[:split]

    # Format turns for summarization. Label each line with the real participant
    # name — the identified speaker for user turns (Turn.speaker, e.g. "Dan" or a
    # named guest), "Timmy" for the assistant — instead of the generic
    # "user:"/"assistant:" roles. This is what makes the summary read "Dan asked
    # ... Timmy refused ..." rather than "The user ... The assistant ..."; the
    # generate_summary prompt is told to carry these names through.
    def _label(t):
        if t.role == "assistant":
            return "Timmy"
        return (t.speaker or "User").strip().title() or "User"
    text = "\n".join(f"{_label(t)}: {t.content}" for t in to_summarize)
    raw_summary = await generate_summary(text)

    is_substantive, summary = _parse_summary(raw_summary)
    if not summary:
        log.warning(
            "Rollup summarization unusable (preamble-only or refusal): %r — skipping",
            (raw_summary or "")[:80],
        )
        return False

    # Move to warm tier. Capture the real wall-clock span of the summarized
    # turns (not the rollup time) so episodic memory is queryable by date range.
    span_start = min(t.timestamp for t in to_summarize)
    span_end = max(t.timestamp for t in to_summarize)
    from conversation.models import WarmSummary
    conversation.warm_summaries.append(WarmSummary(
        text=summary,
        timestamp=time.time(),
        turn_count=len(to_summarize),
        span_start=span_start,
        span_end=span_end,
    ))
    conversation.hot_turns = conversation.hot_turns[split:]
    log.info("Rolled up %d turns -> warm summary (%d chars, %s)",
             split, len(summary), "substantive" if is_substantive else "banter")

    # WRITE-THROUGH at MINT (not eviction): persist the summary to the durable
    # `episodes` tier the moment it is formed, so a summary that survives in the
    # warm tier until the conversation ends (warm never overflowed) is still
    # preserved — closing the loss-gap where un-evicted gist died in-memory on
    # restart. Only SUBSTANTIVE summaries become durable episodes; pure banter
    # advances the rolling window (warm-append + hot-trim above) but is NOT
    # embedded, the 2026-06-18 low-density-pollution guard. store_episode is
    # idempotent on an exact content-hash, so a re-minted identical summary is a
    # no-op. With config.EMBED_EPISODES on, the episode is vector-embedded at
    # write for cosine recall (recall_semantic); recall_temporal queries it by
    # date range regardless of embedding.
    if is_substantive and getattr(config, "PERSIST_EPISODES", False):
        await store_episode(
            span_start,
            span_end,
            summary,
            token_count=max(1, len(summary) // 4),
            source={"trigger": "warm_mint", "turn_count": len(to_summarize)},
        )

    # If the warm tier is too large, evict the oldest from the IN-PROMPT tier.
    # Episodes are now written at mint (above), so eviction no longer persists
    # anything to the episode tier — a surviving summary is already durable and
    # an evicted one was too. The legacy `memories`-tier write stays gated by
    # PERSIST_COLD_SUMMARIES (False since 2026-06-18; kept only for rollback).
    # The hard-ceiling backstop placeholder carries no content and is just
    # dropped.
    if len(conversation.warm_summaries) > config.WARM_MAX_SUMMARIES:
        cold = conversation.warm_summaries.pop(0)
        placeholder = getattr(config, "HARD_CEILING_PLACEHOLDER",
                              "[earlier turns omitted under load]")
        if cold.text != placeholder and getattr(config, "PERSIST_COLD_SUMMARIES", False):
            await store_memory("conversation_summary", cold.text)
            log.info("Promoted warm summary to legacy cold storage")

    return True
