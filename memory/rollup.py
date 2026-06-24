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

    # Format turns for summarization
    text = "\n".join(f"{t.role}: {t.content}" for t in to_summarize)
    raw_summary = await generate_summary(text)

    summary = _clean_summary(raw_summary)
    if not summary:
        log.warning(
            "Rollup summarization unusable (preamble-only or refusal): %r — skipping",
            (raw_summary or "")[:80],
        )
        return False

    # Move to warm tier. Capture the real wall-clock span of the summarized
    # turns (not the rollup time) so episodic memory is queryable by date range
    # on cold eviction.
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
    log.info("Rolled up %d turns -> warm summary (%d chars)", split, len(summary))

    # If warm tier is too large, evict the oldest. Whether the evicted summary
    # is persisted to the embedded memories table is gated by
    # config.PERSIST_COLD_SUMMARIES (default False since 2026-06-18, per Dan):
    # rollups summarize live hot/warm context for the in-prompt session, but the
    # stalest one is DROPPED on overflow rather than auto-embedded as a durable
    # conversation_summary memory. See config.PERSIST_COLD_SUMMARIES.
    # The hard-ceiling backstop placeholder is never persisted regardless — it
    # carries no content, just a marker that turns were dropped under load.
    if len(conversation.warm_summaries) > config.WARM_MAX_SUMMARIES:
        cold = conversation.warm_summaries.pop(0)
        if cold.text == getattr(config, "HARD_CEILING_PLACEHOLDER",
                                "[earlier turns omitted under load]"):
            # The backstop placeholder carries no content and no real span —
            # never persisted to either tier.
            log.info("Dropped backstop placeholder from warm tier (not persisted)")
        else:
            # Two independent persistence channels for an evicted summary:
            #   - episodes: date-range-queryable episodic memory (S1+), gated
            #     by PERSIST_EPISODES. NOT vectorized (embedding NULL until S5).
            #   - memories: legacy vector tier, gated by PERSIST_COLD_SUMMARIES
            #     (False since 2026-06-18). Kept separate so episodic memory is
            #     unaffected by the frozen-vector-writer decision.
            persisted = False
            if getattr(config, "PERSIST_EPISODES", False):
                # Real turn span; fall back to creation time if a legacy summary
                # lacks spans (shouldn't happen for summaries minted by S1 code).
                span_start = cold.span_start if cold.span_start is not None else cold.timestamp
                span_end = cold.span_end if cold.span_end is not None else cold.timestamp
                await store_episode(
                    span_start,
                    span_end,
                    cold.text,
                    token_count=max(1, len(cold.text) // 4),
                    source={"trigger": "warm_eviction", "turn_count": cold.turn_count},
                )
                persisted = True
            if getattr(config, "PERSIST_COLD_SUMMARIES", False):
                await store_memory("conversation_summary", cold.text)
                log.info("Promoted warm summary to cold storage")
                persisted = True
            if not persisted:
                log.info("Evicted stalest warm summary without persisting "
                         "(PERSIST_EPISODES=False, PERSIST_COLD_SUMMARIES=False)")

    return True
