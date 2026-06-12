"""Offline tests for ConversationTurn — the whole turn driven with fakes, no
mic / GPU / speaker / DB. This is the test surface the refactor exists to
create (CONTEXT.md). Run:

    .venv/bin/pytest tests/test_conversation_turn.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation.turn import ConversationTurn, SpeakerIdentity, Retrieved, TurnSettings


# --- fakes ----------------------------------------------------------------

class FakeSpeaker:
    """Records the sentences that reached TTS, in order."""
    def __init__(self):
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeLLM:
    """Yields canned tokens; ignores the prompt. Records the messages it saw."""
    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self.calls: list[list[dict]] = []

    def stream(self, messages, max_tokens=None):
        self.calls.append(messages)
        return self._gen()

    async def _gen(self):
        for t in self._tokens:
            yield t


class FakeMemory:
    """Canned retrieval; records save() calls."""
    def __init__(self, memories=None, facts=None):
        self._retrieved = Retrieved(memories or [], facts or [])
        self.saves: list[dict] = []

    async def gather(self, **kwargs) -> Retrieved:
        return self._retrieved

    async def save(self, **kwargs) -> None:
        self.saves.append(kwargs)


class FakeHistory:
    """Minimal ConversationManager stand-in."""
    def __init__(self):
        self.user_turns: list[tuple[str, str | None]] = []
        self.assistant_turns: list[str] = []

    async def add_user_turn(self, text, speaker=None):
        self.user_turns.append((text, speaker))

    async def add_assistant_turn(self, text):
        self.assistant_turns.append(text)

    def build_history_messages(self):
        return []

    def recent_turns_excluding_current(self, n):
        return []


def _make_turn(tokens):
    speaker, llm, memory, history = FakeSpeaker(), FakeLLM(tokens), FakeMemory(), FakeHistory()
    turn = ConversationTurn(
        speaker=speaker, llm=llm, memory=memory, history=history,
        settings=TurnSettings(),
    )
    return turn, speaker, llm, memory, history


async def _respond(turn, history, words, who):
    """Drive a turn the way the doorway does: record the heard utterance,
    then run the turn (which reads it back out of history)."""
    await history.add_user_turn(words, who.name)
    return await turn.respond(words, who)


# --- tests ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_respond_speaks_each_sentence_and_saves_the_turn():
    # A reply long enough (>30 chars) to flow through the streaming path,
    # not just the end-of-stream flush, so per-sentence speaking is exercised.
    tokens = ["I'm ", "doing ", "really ", "well ", "today, ", "thanks. ",
              "How ", "about ", "you?"]
    turn, speaker, llm, memory, history = _make_turn(tokens)

    result = await _respond(turn, history, "how are you?", SpeakerIdentity("dan", 1))

    # Each finished sentence reached TTS, in order.
    assert speaker.spoken == ["I'm doing really well today, thanks.", "How about you?"]
    # The full spoken text is returned.
    assert result.text == "I'm doing really well today, thanks. How about you?"
    # The user turn was recorded before the prompt was built.
    assert history.user_turns == [("how are you?", "dan")]
    assert history.assistant_turns == [result.text]
    # The turn's final step saved what was learned, with the right attribution.
    assert len(memory.saves) == 1
    assert memory.saves[0]["user_text"] == "how are you?"
    assert memory.saves[0]["response"] == result.text
    assert memory.saves[0]["speaker_name"] == "dan"


@pytest.mark.asyncio
async def test_respond_vetoes_narration_through_the_filter():
    # A reply that opens with a known narration prefix must be swallowed and
    # replaced by the terse fallback before TTS sees any of it.
    tokens = ["the room is full of glowing monitors and code scrolling everywhere you look"]
    turn, speaker, llm, memory, history = _make_turn(tokens)

    result = await _respond(turn, history, "what do you see?", SpeakerIdentity("dan", 1))

    assert speaker.spoken == ["Sure."]
    assert result.text == "Sure."
    assert "monitors" not in result.text
    # A vetoed reply is still a turn: it is saved.
    assert memory.saves[0]["response"] == "Sure."


@pytest.mark.asyncio
async def test_sentence_cap_truncates_a_long_reply():
    # Three sentences in, default cap is 2 -> only the first two are spoken.
    tokens = ["First thing here is true. ", "Second thing follows on. ",
              "Third thing should be dropped entirely."]
    turn, speaker, llm, memory, history = _make_turn(tokens)

    result = await _respond(turn, history, "tell me three things", SpeakerIdentity("dan", 1))

    assert "Third thing" not in result.text
    assert result.text.count(".") <= 2


@pytest.mark.asyncio
async def test_proactive_speaks_then_suppresses_verbatim_repeat():
    # Same canned line twice: the first speaks + persists, the second is
    # dropped BEFORE any TTS (2026-06-12 "lost puppy" repeat). Guard compares
    # normalized text, so punctuation/case wobble doesn't evade it.
    tokens = ["You've got ", "company, Dan."]
    turn, speaker, llm, memory, history = _make_turn(tokens)

    first = await turn.speak_proactively("[SITUATION] someone arrived")
    assert first.text == "You've got company, Dan."
    assert speaker.spoken == ["You've got company, Dan."]
    assert history.assistant_turns == [first.text]

    second = await turn.speak_proactively("[SITUATION] someone arrived")
    assert second.text == ""                      # suppressed
    assert speaker.spoken == ["You've got company, Dan."]  # nothing new reached TTS
    assert history.assistant_turns == [first.text]         # not persisted twice

    # Case/punctuation variants still count as the same remark.
    llm._tokens = ["you've got company, dan"]
    third = await turn.speak_proactively("[SITUATION] someone arrived")
    assert third.text == ""
    assert speaker.spoken == ["You've got company, Dan."]


@pytest.mark.asyncio
async def test_proactive_distinct_lines_still_speak():
    # The guard only blocks repeats — a different remark goes through, and
    # multi-sentence proactive lines keep per-sentence TTS chunking.
    tokens = ["You've got company, Dan."]
    turn, speaker, llm, memory, history = _make_turn(tokens)

    await turn.speak_proactively("[SITUATION] someone arrived")
    llm._tokens = ["New face at the door. ", "Should I be worried?"]
    result = await turn.speak_proactively("[SITUATION] someone else arrived")

    assert result.text == "New face at the door. Should I be worried?"
    assert speaker.spoken == ["You've got company, Dan.",
                              "New face at the door.", "Should I be worried?"]
