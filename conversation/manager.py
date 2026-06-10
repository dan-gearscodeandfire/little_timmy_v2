"""Conversation turn tracking and history assembly."""

import asyncio
import time
import logging
from conversation.models import Turn, ConversationState
from memory.rollup import maybe_rollup
import config

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
        # Idle-window rollup scheduling (2026-05-15): on every turn we
        # (re)arm a delayed task that fires maybe_rollup only after
        # ROLLUP_IDLE_DELAY_SECONDS of no further turns. This avoids the
        # starvation pattern where the conversation-priority gate in
        # llm/client.py cancels every rollup mid-flight when the user is
        # actively talking. Cancellation of this task is cheap (just
        # interrupts the asyncio.sleep), so rearming on every turn is fine.
        self._idle_rollup_task: asyncio.Task | None = None

    @property
    def hot_turns(self):
        return self.state.hot_turns

    @property
    def warm_summaries(self):
        return self.state.warm_summaries

    @property
    def turn_count(self):
        return self.state.turn_count

    async def _idle_then_rollup(self, delay_s: float) -> None:
        """Sleep delay_s; if not cancelled by another turn, run maybe_rollup.

        Cancellation during the sleep is the common case (next turn arrives
        before the idle window closes). Once we pass the sleep, the rollup
        is in-flight and the conversation-priority gate in llm/client.py
        becomes the only remaining cancellation source — and that one is
        deliberate.
        """
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        async with self._rollup_lock:
            try:
                await maybe_rollup(self.state)
            except asyncio.CancelledError:
                # Priority gate cancelled us mid-generate_summary; expected.
                raise
            except Exception:
                log.exception("background rollup failed")

    def _schedule_rollup_after_idle(self) -> None:
        """Arm the idle-window rollup. Cancels any prior pending arm so
        the timer always counts from the latest turn."""
        if self._idle_rollup_task is not None and not self._idle_rollup_task.done():
            self._idle_rollup_task.cancel()
        delay = getattr(config, "ROLLUP_IDLE_DELAY_SECONDS", 20)
        self._idle_rollup_task = asyncio.create_task(self._idle_then_rollup(delay))

    def _enforce_hard_ceiling(self) -> None:
        """Synchronous, non-LLM backstop for the runaway-active-burst case.

        The idle-window rollup (_schedule_rollup_after_idle) is the QUALITY
        path, but it only fires after ROLLUP_IDLE_DELAY_SECONDS of silence.
        During a rapid back-and-forth (turns arriving faster than that delay)
        it is perpetually re-armed and never runs, so hot_turns grows
        unbounded — even though token totals stay well under ctx. Precision
        decays from turn DEPTH, not token count (Qwen3.6 ~3B active params is
        weak at many-turn needle-tracking; handoff 2026-06-10).

        This drops the oldest HALF of the hot buffer the instant it blows past
        HOT_HARD_CEILING_TOKENS, replacing it with a cheap non-LLM placeholder
        warm summary. Chunky (half at a time, mirroring maybe_rollup) so the
        KV-cache prefix shifts only once per drop instead of every turn — the
        whole reason we don't sliding-window-clip in build_history_messages.
        Runs entirely within one event-loop tick (no await), so it cannot
        interleave with the async idle rollup mid-slice.

        Accepts minor content loss (placeholder, not a real summary) ONLY in
        this rare active-burst the idle gate structurally cannot serve. The
        idle LLM rollup remains the normal path.
        """
        ceiling = getattr(config, "HOT_HARD_CEILING_TOKENS", 4000)
        hot_tokens = sum(t.token_count for t in self.state.hot_turns)
        if hot_tokens <= ceiling or len(self.state.hot_turns) < 2:
            return
        # Don't race the idle LLM rollup: if it holds the lock it is already
        # collapsing the buffer with a REAL summary — let that win.
        if self._rollup_lock.locked():
            return

        from conversation.models import WarmSummary
        split = max(1, len(self.state.hot_turns) // 2)
        dropped = self.state.hot_turns[:split]
        self.state.hot_turns = self.state.hot_turns[split:]

        # Coalesce consecutive placeholders so a long burst doesn't fill the
        # warm tier with markers (and so the rendered prefix stays stable).
        marker = getattr(config, "HARD_CEILING_PLACEHOLDER",
                         "[earlier turns omitted under load]")
        if (self.state.warm_summaries
                and self.state.warm_summaries[-1].text == marker):
            last = self.state.warm_summaries[-1]
            last.turn_count += len(dropped)
            last.timestamp = time.time()
        else:
            self.state.warm_summaries.append(WarmSummary(
                text=marker,
                timestamp=time.time(),
                turn_count=len(dropped),
            ))
        log.warning(
            "Hard-ceiling backstop: dropped %d hot turns (%d tok > %d ceiling) "
            "— idle rollup starved by active burst",
            len(dropped), hot_tokens, ceiling,
        )

    async def add_user_turn(self, text: str, speaker: str | None = None):
        # Strip-on-store discipline: the [CONTEXT]/[UTTERANCE] wrap is a
        # render-time decoration in llm/prompt_builder.wrap_user_message().
        # If it leaks into persisted history, we hit the failure mode from
        # feedback_ephemeral_prompt_kv_cache_misconception: per-turn context
        # accumulating in past turns, model treating it as user speech.
        if "[CONTEXT]" in text or "[UTTERANCE]" in text:
            raise ValueError(
                "add_user_turn refusing wrapped user text (markers present). "
                "Pass raw utterance only. "
                f"Got: {text[:120]!r}"
            )
        turn = Turn(
            role="user",
            content=text,
            timestamp=time.time(),
            token_count=estimate_tokens(text),
            speaker=speaker,
        )
        self.state.hot_turns.append(turn)
        self.state.turn_count += 1
        # Detached + idle-windowed: rollup uses Qwen3.6 thinking-on which
        # can take 15-45 s. Awaiting it here used to stall the next
        # response. Firing immediately starved the rollup because the
        # priority gate cancels in-flight rollup on every new conversation
        # turn (see llm/client._cancel_in_flight_slow_calls). Idle-window
        # scheduling defers the actual generate_summary call until N
        # seconds of conversation silence (config.ROLLUP_IDLE_DELAY_SECONDS).
        self._schedule_rollup_after_idle()
        # Synchronous backstop for the burst case the idle gate can't serve.
        self._enforce_hard_ceiling()

    async def add_assistant_turn(self, text: str):
        turn = Turn(
            role="assistant",
            content=text,
            timestamp=time.time(),
            token_count=estimate_tokens(text),
        )
        self.state.hot_turns.append(turn)
        self._schedule_rollup_after_idle()
        self._enforce_hard_ceiling()

    def build_history_messages(self) -> list[dict]:
        """Build conversation history for prompt prefix (KV-cacheable).

        Warm summaries (rollup output) are emitted as ONE synthetic
        user/assistant pair right after the static system prompt, not as
        multiple `system` messages. Qwen 3.6's chat template assumes a single
        system message at chat-start; reserving the persona system slot keeps
        the template hygienic and the static-prefix KV cache intact.
        """
        messages: list[dict] = []

        if self.state.warm_summaries:
            summary_lines = ["[SESSION SUMMARY] Earlier in this conversation:"]
            for summary in self.state.warm_summaries:
                summary_lines.append(f"- {summary.text}")
            messages.append({"role": "user", "content": "\n".join(summary_lines)})
            messages.append({"role": "assistant", "content": "Acknowledged."})

        # Hot turns verbatim, with speaker prefix on user turns. Prefix is
        # tolerated by prompt_builder._matches_current_user_turn() dedup.
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

    def recent_turns_excluding_current(self, n: int) -> list[Turn]:
        """Return the last `n` hot turns EXCLUDING the current utterance.

        Used to build a coreference-aware retrieval query. add_user_turn runs
        before retrieval, so hot_turns[-1] is the current utterance; we drop it
        and take the preceding `n` (oldest-first). Returns [] if there's no
        prior context yet.
        """
        if n <= 0 or len(self.state.hot_turns) <= 1:
            return []
        return self.state.hot_turns[:-1][-n:]

    def reset(self):
        """Clear all conversation state."""
        self.state = ConversationState()
