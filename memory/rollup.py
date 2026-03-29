"""Sliding window conversation rollup: hot -> warm -> cold."""

import logging
import time
from llm.client import generate_summary
from memory.manager import store_memory
import config

log = logging.getLogger(__name__)


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
    summary = await generate_summary(text)

    if not summary:
        log.warning("Rollup summarization returned empty, skipping")
        return False

    # Move to warm tier
    from conversation.models import WarmSummary
    conversation.warm_summaries.append(WarmSummary(
        text=summary,
        timestamp=time.time(),
        turn_count=len(to_summarize),
    ))
    conversation.hot_turns = conversation.hot_turns[split:]
    log.info("Rolled up %d turns -> warm summary (%d chars)", split, len(summary))

    # If warm tier is too large, promote oldest to cold (stored in DB)
    if len(conversation.warm_summaries) > config.WARM_MAX_SUMMARIES:
        cold = conversation.warm_summaries.pop(0)
        await store_memory("conversation_summary", cold.text)
        log.info("Promoted warm summary to cold storage")

    return True
