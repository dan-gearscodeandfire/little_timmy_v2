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


@dataclass
class ConversationState:
    hot_turns: list[Turn] = field(default_factory=list)
    warm_summaries: list[WarmSummary] = field(default_factory=list)
    turn_count: int = 0
