"""Conversation data models."""

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    token_count: int
    speaker: str | None = None  # speaker name for user turns


@dataclass
class WarmSummary:
    text: str
    timestamp: float
    turn_count: int
    # Real wall-clock span of the turns this summary covers (epoch seconds).
    # Threaded through to `episodes.span_start/span_end` on cold eviction so
    # episodic memory is queryable by date range. None for the hard-ceiling
    # placeholder (which is never persisted) and any legacy/restored summary.
    span_start: float | None = None
    span_end: float | None = None


@dataclass
class ConversationState:
    hot_turns: list[Turn] = field(default_factory=list)
    warm_summaries: list[WarmSummary] = field(default_factory=list)
    turn_count: int = 0
