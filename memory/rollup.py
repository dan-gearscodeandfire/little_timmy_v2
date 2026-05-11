"""Sliding window conversation rollup: hot -> warm -> cold."""

import logging
import re
import time
from llm.client import generate_summary
from memory.manager import store_memory
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


def _clean_summary(text: str) -> str | None:
    """Strip preamble; return None if the summary is a refusal or empty."""
    if not text:
        return None
    cleaned = _PREAMBLE_RE.sub("", text).strip()
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
