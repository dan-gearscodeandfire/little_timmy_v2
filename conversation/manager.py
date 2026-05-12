"""Conversation turn tracking and history assembly."""

import asyncio
import time
import logging
from conversation.models import Turn, ConversationState
from memory.rollup import maybe_rollup

log = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


class ConversationManager:
    def __init__(self):
        self.state = ConversationState()
        # Serializes background rollups so concurrent triggers don't race
        # on hot_turns mutation. maybe_rollup truncates self.state.hot_turns
        # AFTER awaiting generate_summary; without the lock a second turn
        # could append in that gap and get sliced off too.
        self._rollup_lock = asyncio.Lock()

    @property
    def hot_turns(self):
        return self.state.hot_turns

    @property
    def warm_summaries(self):
        return self.state.warm_summaries

    @property
    def turn_count(self):
        return self.state.turn_count

    async def _rollup_in_background(self) -> None:
        """Run maybe_rollup off the response hot path. Errors are logged
        and swallowed so a rollup failure can't break the next turn."""
        async with self._rollup_lock:
            try:
                await maybe_rollup(self.state)
            except Exception:
                log.exception("background rollup failed")

    async def add_user_turn(self, text: str, speaker: str | None = None):
        turn = Turn(
            role="user",
            content=text,
            timestamp=time.time(),
            token_count=estimate_tokens(text),
            speaker=speaker,
        )
        self.state.hot_turns.append(turn)
        self.state.turn_count += 1
        # Detached: rollup uses Qwen3.6 thinking-on which can take 15-45 s.
        # Awaiting it here used to stall the next response by that long.
        asyncio.create_task(self._rollup_in_background())

    async def add_assistant_turn(self, text: str):
        turn = Turn(
            role="assistant",
            content=text,
            timestamp=time.time(),
            token_count=estimate_tokens(text),
        )
        self.state.hot_turns.append(turn)
        asyncio.create_task(self._rollup_in_background())

    def build_history_messages(self) -> list[dict]:
        """Build conversation history for prompt prefix (KV-cacheable)."""
        messages = []
        # Warm summaries first (compressed older context)
        for summary in self.state.warm_summaries:
            messages.append({
                "role": "system",
                "content": f"[Earlier conversation summary: {summary.text}]",
            })
        # Hot turns verbatim
        for turn in self.state.hot_turns:
            content = turn.content
            if turn.speaker and turn.role == "user":
                content = f"[{turn.speaker.title()}]: {content}"
            messages.append({"role": turn.role, "content": content})
        return messages

    def get_last_exchange(self) -> tuple[str, str] | None:
        """Get the most recent user/assistant pair for memory extraction."""
        user_text = None
        assistant_text = None
        for turn in reversed(self.state.hot_turns):
            if turn.role == "assistant" and assistant_text is None:
                assistant_text = turn.content
            elif turn.role == "user" and user_text is None:
                user_text = turn.content
            if user_text and assistant_text:
                return (user_text, assistant_text)
        return None

    def reset(self):
        """Clear all conversation state."""
        self.state = ConversationState()
