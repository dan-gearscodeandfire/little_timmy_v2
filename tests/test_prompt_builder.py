"""Unit tests for llm.prompt_builder.build_messages dedup logic.

The dedup at build_messages strips the trailing user turn from `history`
if it matches the current `user_text` (because the orchestrator's
add_user_turn already appended the current turn to hot_turns before
prompt-building runs — so without the strip, the user message would
appear twice).

A regression caught 2026-05-12 by inspection of /api/last_payload: the
strip's content-equality check failed against `[Dan]: hey timmy`-style
speaker-prefixed content from ConversationManager.build_history_messages,
so duplication came back silently.

PROMPT SHAPE (post-2026-05-28 Qwen refactor — see module docstring of
llm/prompt_builder.py and Zettel little-timmy-conversation-tier-qwen36-
shipped-2026-05-28): the per-turn dynamic context is NO LONGER a separate
system message at the tail. The layout is now:

    [0]      system = static persona + protocol clause (KV-cached forever)
    [1..M-1] history (dedup'd)
    [M]      user   = [CONTEXT]\\n<ephemeral>\\n[/CONTEXT]\\n
                      [UTTERANCE]\\n<user_text>\\n[/UTTERANCE]

So the ephemeral block lives INSIDE the wrapped final user message, and
the raw user_text is wrapped, not bare. These tests pin the dedup to the
speaker-prefixed AND bare cases under that shape, plus the guard warning
if duplication slips through anyway.

Run:
    .venv/bin/pytest tests/test_prompt_builder.py -v
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.prompt_builder import (
    build_messages,
    wrap_user_message,
    build_static_persona_system,
)


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _wrapped_tail(ephemeral: str, utterance: str) -> dict:
    """The final user message as build_messages assembles it: the ephemeral
    context block + raw utterance wrapped in [CONTEXT]/[UTTERANCE]."""
    return {"role": "user", "content": wrap_user_message(ephemeral, utterance)}


def _assert_static_system_head(messages: list[dict]) -> None:
    """system[0] is the stable persona+protocol, NOT the per-turn ephemeral."""
    assert messages[0] == {
        "role": "system",
        "content": build_static_persona_system(),
    }


def _assert_no_adjacent_duplicate_users(messages: list[dict]) -> None:
    """Cross-check the test invariant directly. If this passes, the prompt
    payload is well-formed wrt the dedup property the regression broke."""
    for i in range(1, len(messages)):
        prev, cur = messages[i - 1], messages[i]
        assert not (
            prev.get("role") == "user"
            and cur.get("role") == "user"
            and prev.get("content") == cur.get("content")
        ), (
            f"adjacent duplicate user messages at positions {i-1}, {i}: "
            f"{prev.get('content')!r}"
        )


def test_dedup_strips_bare_user_turn_at_tail():
    """Text-input path: hot_turns has no speaker prefix on the trailing
    user turn. Dedup matches by bare equality."""
    history = [
        _user("first"),
        _assistant("ok"),
        _user("hey timmy"),  # current turn, already appended
    ]
    msgs = build_messages(history, "EPHEMERAL", "hey timmy")
    _assert_static_system_head(msgs)
    # Tail is the wrapped current turn — NOT a second bare user turn.
    assert msgs[-1] == _wrapped_tail("EPHEMERAL", "hey timmy")
    # No earlier bare "hey timmy" user turn remaining (stripped).
    assert all(m.get("content") != "hey timmy" for m in msgs[:-1])
    _assert_no_adjacent_duplicate_users(msgs)


def test_dedup_strips_speaker_prefixed_user_turn_at_tail():
    """REGRESSION CASE: voice path. ConversationManager prefixes user
    turns with `[Dan]: ` when speaker is set. Pre-fix code did bare-equality
    against `user_text` and missed the strip → duplication."""
    history = [
        {"role": "user", "content": "[Dan]: first"},
        _assistant("ok"),
        {"role": "user", "content": "[Dan]: hey timmy"},  # current turn
    ]
    msgs = build_messages(history, "EPHEMERAL", "hey timmy")
    _assert_static_system_head(msgs)
    assert msgs[-1] == _wrapped_tail("EPHEMERAL", "hey timmy")
    # The [Dan]:-prefixed trailing turn must be gone.
    assert all(m.get("content") != "[Dan]: hey timmy" for m in msgs[:-1])
    _assert_no_adjacent_duplicate_users(msgs)


def test_dedup_handles_unknown_speaker_prefix():
    """Speaker tagged as unknown_N still produces a `[Unknown_3]: ...`-style
    prefix per build_history_messages. The strip must tolerate it."""
    history = [
        {"role": "user", "content": "[Unknown_3]: who am I"},
    ]
    msgs = build_messages(history, "EPHEMERAL", "who am I")
    assert msgs[-1] == _wrapped_tail("EPHEMERAL", "who am I")
    # The prefixed trailing turn was the only history entry and was stripped.
    assert all(m.get("content") != "[Unknown_3]: who am I" for m in msgs[:-1])
    _assert_no_adjacent_duplicate_users(msgs)


def test_dedup_does_not_strip_unrelated_trailing_user_turn():
    """If somehow hot_turns' last user turn is for a DIFFERENT utterance
    (shouldn't happen in production but defend against weird states),
    we should NOT strip it. The dedup is content-aware, not blind."""
    history = [
        _user("different earlier question"),
    ]
    msgs = build_messages(history, "EPHEMERAL", "current question")
    # Earlier question should still be in history (no false strip).
    assert any(m.get("content") == "different earlier question" for m in msgs)
    assert msgs[-1] == _wrapped_tail("EPHEMERAL", "current question")
    _assert_no_adjacent_duplicate_users(msgs)


def test_empty_history():
    """First turn — no history. Just static system[0] + wrapped user tail."""
    msgs = build_messages([], "EPHEMERAL", "hello")
    assert msgs == [
        {"role": "system", "content": build_static_persona_system()},
        _wrapped_tail("EPHEMERAL", "hello"),
    ]
    _assert_no_adjacent_duplicate_users(msgs)


def test_history_ends_in_assistant_turn():
    """If history ends in an assistant turn (rare — would mean add_user_turn
    hasn't been called yet), no strip should happen."""
    history = [
        _user("[Dan]: q1"),
        _assistant("a1"),
    ]
    msgs = build_messages(history, "EPHEMERAL", "q2")
    _assert_static_system_head(msgs)
    # Assistant turn retained, wrapped new user appended after it.
    assert msgs[-2] == _assistant("a1")
    assert msgs[-1] == _wrapped_tail("EPHEMERAL", "q2")
    _assert_no_adjacent_duplicate_users(msgs)


def test_guard_warns_on_actual_duplication(caplog):
    """Under normal use the dedup guard should NOT warn — dedup handled the
    trailing duplicate. (The warning path fires only if dedup regresses and
    two adjacent identical user turns reach the assembled payload.)"""
    caplog.set_level(logging.WARNING, logger="llm.prompt_builder")
    history = [
        _user("[Dan]: hello"),
    ]
    msgs = build_messages(history, "EPHEMERAL", "hello")
    assert "adjacent duplicate user messages" not in caplog.text
    _assert_no_adjacent_duplicate_users(msgs)
