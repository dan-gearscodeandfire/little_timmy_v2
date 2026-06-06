"""ConversationTurn — the deep module that owns one conversation turn.

See CONTEXT.md for the design. In one line: everything that happens between
"someone said something" (or "Timmy decides to speak") and "Timmy has spoken
and saved what he learned" lives behind this module's two front doors:

    respond(words, who)            -> reactive turn (voice or text)
    speak_proactively(situation)   -> unprompted turn (same engine)

The pipeline (retrieve -> build prompt -> stream from LLM -> filter -> speak
sentence by sentence -> save) is private. Three collaborators are injected at
construction so a full turn can be driven with fakes (no mic / GPU / speaker /
DB):

    speaker : speaks a finished sentence aloud      (Speaker)
    llm     : streams reply tokens for a prompt      (TokenStreamer)
    memory  : gathers context, then saves the turn   (Memory)

plus `history` (the ConversationManager) and `settings` (config captured once,
not re-read per line). Identity resolution and presence side-effects (head
turn, voice learning) happen at the doorway, OUTSIDE this module; the resolved
`who` is handed in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Optional, Protocol

from llm.prompt_builder import (
    build_ephemeral_block,
    build_messages,
    build_proactive_messages,
)
from conversation.reply_filter import (
    filtered_assistant_stream,
    user_invites_longer_reply,
    _REPLY_LONGER_SENTENCES,
)


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeakerIdentity:
    """Who is talking, resolved at the doorway before the turn runs."""
    name: str = "dan"
    db_id: Optional[int] = 1


@dataclass(frozen=True)
class Retrieved:
    """What the memory seam found for this turn's prompt."""
    memories: list = field(default_factory=list)
    facts: list = field(default_factory=list)


@dataclass(frozen=True)
class TurnResult:
    """What a turn produced. `text` is what actually reached TTS (post
    narration-veto / sentence-cap), which can differ from a raw model reply.
    The timing/token fields are returned (not reported here) so the doorway
    can assemble the metrics broadcast — reporting stays outside the turn."""
    text: str
    timings: dict                       # first_token_ms, first_tts_ms, total_ms
    retrieval_ms: int = 0
    est_prompt_tokens: int = 0
    est_completion_tokens: int = 0
    # The assembled prompt is returned so the doorway's after-chores
    # (compliment log, feedback capture, finalized-turn snapshot) can see the
    # exact (ephemeral, messages) the LLM saw — without the turn owning them.
    messages: list = field(default_factory=list)
    ephemeral: str = ""


@dataclass
class TurnContext:
    """Per-turn inputs resolved at the doorway and handed into the turn.

    Presence/vision are computed BEFORE the turn (identity fusion, head-move
    and voice-learning are doorway side-effects, not the turn's job — see
    CONTEXT.md). The hooks let the doorway layer eye-LED / supervisor cues onto
    the speak phase without the turn knowing about them."""
    stt_ms: int = 0
    spk_ms: int = 0
    t_start: Optional[float] = None
    vision_description: Optional[str] = None
    visual_question: bool = False
    presence_state: Optional[dict] = None
    fusion_source: Optional[str] = None
    face_hint_name: Optional[str] = None
    on_first_token: Optional[Callable[[], Awaitable[None]]] = None
    on_first_sentence: Optional[Callable[[], Awaitable[None]]] = None


@dataclass
class TurnSettings:
    """Config captured once at construction (was 37 inline config reads)."""
    retrieval_top_k: int = 5
    context_turns: int = 4
    coreference_enabled: bool = True
    proactive_max_sentences: int = 2

    @classmethod
    def from_config(cls) -> "TurnSettings":
        import config
        return cls(
            retrieval_top_k=config.RETRIEVAL_TOP_K,
            context_turns=config.CONTEXT_TURNS,
            coreference_enabled=config.COREFERENCE_ENABLED,
            proactive_max_sentences=config.PROACTIVE_MAX_SENTENCES,
        )


# --------------------------------------------------------------------------
# Seams (Protocols) — production adapters and test fakes both satisfy these
# --------------------------------------------------------------------------

class Speaker(Protocol):
    async def speak(self, text: str) -> None: ...


class TokenStreamer(Protocol):
    def stream(self, messages: list[dict],
               max_tokens: int | None = None) -> AsyncIterator[str]: ...


class Memory(Protocol):
    async def gather(self, *, user_text: str, speaker_name: str,
                     speaker_db_id: int | None, subjects: list[str],
                     context_turns) -> Retrieved: ...

    async def save(self, *, user_text: str, response: str,
                   speaker_id: int | None, speaker_name: str) -> None: ...


# Optional broadcast hook: (event_type, payload) -> awaitable. None = no-op.
EventHook = Optional[Callable[[str, dict], Awaitable[None]]]


# --------------------------------------------------------------------------
# The deep module
# --------------------------------------------------------------------------

class ConversationTurn:
    def __init__(self, *, speaker: Speaker, llm: TokenStreamer, memory: Memory,
                 history, settings: TurnSettings | None = None,
                 on_event: EventHook = None):
        self._speaker = speaker
        self._llm = llm
        self._memory = memory
        self._history = history
        self._settings = settings or TurnSettings()
        self._on_event = on_event

    # -- front door 1: reactive (voice or text) ----------------------------
    async def respond(self, words: str, who: SpeakerIdentity,
                      ctx: "TurnContext | None" = None) -> TurnResult:
        """Produce and speak Timmy's reply to `words` from `who`, then save
        what was learned. `ctx` carries doorway-resolved vision/presence and
        the speak-phase hooks; it defaults to an empty context (used by tests
        and the text path)."""
        ctx = ctx or TurnContext()
        # The caller (doorway) must have already recorded the user turn via
        # history.add_user_turn — it does so for every heard utterance, incl.
        # the enroll / name-ask paths that never reach a turn. We rely on it
        # being the history tail: coreference excludes it, and build_messages
        # strips it before re-adding the wrapped form. (Matches the old
        # add_user_turn-in-process_speech, _generate_response-reads-history
        # ordering.)
        t_retrieval = time.time()
        retrieved = await self._gather(words, who)
        retrieval_ms = int((time.time() - t_retrieval) * 1000)
        await self._emit("retrieval", {
            "memories": [
                {"type": m.type, "content": m.content[:200], "score": round(m.score, 3)}
                for m in retrieved.memories
            ],
            "facts": [
                {"subject": f.subject, "predicate": f.predicate, "value": f.value}
                for f in retrieved.facts
            ],
        })

        ephemeral = build_ephemeral_block(
            memories=retrieved.memories,
            facts=retrieved.facts,
            speaker_name=who.name,
            vision_description=ctx.vision_description,
            visual_question=ctx.visual_question,
            presence_state=ctx.presence_state,
            fusion_source=ctx.fusion_source,
            face_hint_name=ctx.face_hint_name,
        )
        messages = build_messages(self._history.build_history_messages(),
                                  ephemeral, words)

        cap = _REPLY_LONGER_SENTENCES if user_invites_longer_reply(words) else None
        result = await self._stream_and_speak(
            messages, max_sentences=cap,
            on_first_token=ctx.on_first_token,
            on_first_sentence=ctx.on_first_sentence,
        )

        await self._emit("turn", {"role": "assistant", "content": result.text})
        await self._history.add_assistant_turn(result.text)

        # Final real step of the turn: save what was learned. This is also the
        # slow call the priority gate (Candidate 2) pre-empts on the next turn.
        await self._memory.save(user_text=words, response=result.text,
                                speaker_id=who.db_id, speaker_name=who.name)

        return self._finalize(result, messages, retrieval_ms, ephemeral)

    # -- front door 2: proactive (unprompted) ------------------------------
    async def speak_proactively(self, ephemeral_block: str) -> TurnResult:
        """Timmy initiates with no incoming utterance. Shares the same engine
        as respond(); the synthetic prompt is render-time only and is never
        stored in history (build_proactive_messages handles that). Identity /
        situation assembly into `ephemeral_block` happens at the doorway.

        The doorway builds `ephemeral_block` from the current vision/presence
        snapshot and wraps the call with its own gates + eye-LED/supervisor
        cues; this owns the LLM->TTS engine plus broadcast/persistence (only
        when something was actually said).
        """
        messages = build_proactive_messages(
            self._history.build_history_messages(), ephemeral_block)
        result = await self._stream_and_speak(
            messages, max_sentences=self._settings.proactive_max_sentences)
        if result.text and result.text.strip():
            await self._emit("turn", {"role": "assistant",
                                      "content": result.text, "proactive": True})
            await self._history.add_assistant_turn(result.text)
        return result

    # -- a minimal prompted utterance (sub-dialogs, e.g. Introductions) -----
    async def say(self, prompt_text: str) -> TurnResult:
        """Speak a single synthetic-prompted line — no memory retrieval and no
        save. Used by sub-dialogs (asking/confirming a name) that need Timmy to
        say something in-character but aren't a full conversation turn. Builds a
        minimal prompt (empty context block), streams, speaks, broadcasts the
        assistant turn and persists it."""
        ephemeral = build_ephemeral_block(memories=[], facts=[])
        messages = build_messages(
            self._history.build_history_messages(), ephemeral, prompt_text)
        result = await self._stream_and_speak(messages, max_sentences=None)
        await self._emit("turn", {"role": "assistant", "content": result.text})
        await self._history.add_assistant_turn(result.text)
        return result

    # -- private engine: stream -> filter -> per-sentence TTS --------------
    async def _stream_and_speak(self, messages: list[dict], *,
                                max_sentences: int | None,
                                on_first_token=None,
                                on_first_sentence=None) -> TurnResult:
        """Ported verbatim from main.Orchestrator._stream_to_tts, with the LLM
        and TTS as injected seams and broadcast via the event hook."""
        t_start = time.time()
        full_response = ""
        sentence_buffer = ""
        first_token_time: float | None = None
        first_tts_time: float | None = None

        async for token in filtered_assistant_stream(
            self._llm.stream(messages), max_sentences=max_sentences,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                if on_first_token is not None:
                    await on_first_token()
            full_response += token
            sentence_buffer += token
            await self._emit("token", {"content": token})
            stripped = sentence_buffer.rstrip()
            if stripped and stripped[-1] in ".?!;:":
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                if sentence:
                    if first_tts_time is None:
                        first_tts_time = time.time()
                        if on_first_sentence is not None:
                            await on_first_sentence()
                    await self._speaker.speak(sentence)

        if sentence_buffer.strip():
            if first_tts_time is None:
                first_tts_time = time.time()
                if on_first_sentence is not None:
                    await on_first_sentence()
            await self._speaker.speak(sentence_buffer.strip())

        end_time = time.time()
        return TurnResult(full_response, {
            "first_token_ms": int((first_token_time - t_start) * 1000) if first_token_time else None,
            "first_tts_ms":   int((first_tts_time - t_start) * 1000) if first_tts_time else None,
            "total_ms":       int((end_time - t_start) * 1000),
        })

    # -- helpers -----------------------------------------------------------
    async def _gather(self, words: str, who: SpeakerIdentity) -> Retrieved:
        subjects = _extract_my_subjects(words)
        ctx_turns = (
            self._history.recent_turns_excluding_current(self._settings.context_turns)
            if self._settings.coreference_enabled else None
        )
        return await self._memory.gather(
            user_text=words, speaker_name=who.name, speaker_db_id=who.db_id,
            subjects=subjects, context_turns=ctx_turns,
        )

    async def _emit(self, event_type: str, payload: dict) -> None:
        if self._on_event is not None:
            await self._on_event(event_type, payload)

    @staticmethod
    def _finalize(result: TurnResult, messages: list[dict],
                  retrieval_ms: int, ephemeral: str) -> TurnResult:
        """Attach retrieval timing + ~4-chars/token estimates (matches
        conversation.manager.estimate_tokens) + the assembled prompt for the
        doorway's metrics and after-chores."""
        prompt_chars = sum(len(m.get("content", "") or "") for m in messages)
        return TurnResult(
            text=result.text,
            timings=result.timings,
            retrieval_ms=retrieval_ms,
            est_prompt_tokens=max(1, prompt_chars // 4),
            est_completion_tokens=max(0, len(result.text) // 4),
            messages=messages,
            ephemeral=ephemeral,
        )


def _extract_my_subjects(user_text: str) -> list[str]:
    """Pull "my X" phrases out of the utterance for subject-scoped fact
    retrieval. Mirrors the inline parse in the old _generate_response."""
    words = user_text.lower().split()
    return [f"my {words[i+1]}" for i, w in enumerate(words)
            if w == "my" and i + 1 < len(words)]


# --------------------------------------------------------------------------
# Production adapters — heavy imports deferred so importing this module for a
# unit test stays light (no DB pool, no httpx clients).
# --------------------------------------------------------------------------

class LiveLLM:
    """Streams via llm.client.stream_conversation, which also owns the
    priority-gate (cancel in-flight slow calls) behaviour."""
    def stream(self, messages, max_tokens=None):
        from llm.client import stream_conversation
        return stream_conversation(messages, max_tokens=max_tokens)


class LiveMemory:
    """Retrieval + fact fusion + extraction against the real stores. Note:
    face/presence fetch is deliberately NOT here — that is a doorway concern
    (see CONTEXT.md, presence handoff)."""
    def __init__(self, top_k: int = 5):
        self._top_k = top_k

    async def gather(self, *, user_text, speaker_name, speaker_db_id,
                     subjects, context_turns) -> Retrieved:
        import asyncio
        from memory.retrieval import retrieve
        from memory.facts import get_all_facts_for_prompt, get_facts_about_speaker

        async def _empty():
            return []

        speaker_for_facts = speaker_name if speaker_name != "timmy" else "dan"
        gathered = await asyncio.gather(
            retrieve(user_text, top_k=self._top_k, context_turns=context_turns),
            get_all_facts_for_prompt(subjects, limit=5) if subjects else _empty(),
            get_facts_about_speaker(speaker_for_facts, speaker_db_id, limit=5),
        )
        memories, non_speaker_facts, speaker_facts = gathered
        # Dedupe by fact id, speaker-side first (already learned_at-ordered).
        seen, resolved = set(), []
        for f in (*speaker_facts, *non_speaker_facts):
            if f.id in seen:
                continue
            seen.add(f.id)
            resolved.append(f)
        return Retrieved(memories=memories, facts=resolved)

    async def save(self, *, user_text, response, speaker_id, speaker_name):
        from memory.extraction import extract_and_store
        await extract_and_store(user_text, response,
                                speaker_id=speaker_id, speaker_name=speaker_name)
