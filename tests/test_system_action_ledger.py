"""Theme B — system-action ledger.

SYSTEM-initiated spoken lines (the enroll FSM's consent ask / name ask / pose
cues, spoken via `_safe_speak` -> fixed TTS) must land in conversation history
so the LLM knows what it said and stops contradicting its own enroll requests
(the Voss "I did not ask you to move your head" confabulation, 2026-06-14).

- Unit:  ConversationManager.add_system_action_turn -> assistant turn in history.
- Glue:  a REAL FaceEnroller wired to a REAL ConversationManager via the
         `record_action` seam records its fixed lines end-to-end. A unit test of
         either side alone would not catch the wiring gap that let pose cues
         bypass hot_turns (cf. the 8e38f0a "passed unit, crashed glue" lesson).
"""
import asyncio
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation.manager import ConversationManager
from presence.face_enroller import FaceEnroller, EnrollerConfig, State


@pytest.mark.asyncio
async def test_add_system_action_turn_is_assistant_in_history():
    mgr = ConversationManager()
    await mgr.add_system_action_turn("Now turn your head slowly to the left.")
    assert {
        "role": "assistant",
        "content": "Now turn your head slowly to the left.",
    } in mgr.build_history_messages()


def _box(cx, W=640, H=360, h=90):
    w = int(h * 0.8)
    return [int(cx - w / 2), int(H / 2 - h / 2), w, h]


def _name_extractor(text):
    m = re.search(r"(?:i'm|i am|my name is|it's|call me)\s+(\w+)", text.lower())
    if m:
        return m.group(1)
    m = re.search(r"^(\w+)$", text.strip().lower())
    if m and m.group(1) not in ("yes", "no", "yeah", "sure", "nope"):
        return m.group(1)
    return None


@pytest.mark.asyncio
async def test_enroller_fixed_lines_reach_history_via_record_action():
    mgr = ConversationManager()

    async def speak(text):  # fixed-TTS sink (the path that used to bypass history)
        return None

    async def say(prompt):  # _safe_say path; unused before capture here
        return type("R", (), {"text": prompt})()

    async def enroll_stream(name, count, interval_s, mode):
        yield ("complete", {"saved": True, "name": name})

    async def verify_faces():
        return []

    cfg = EnrollerConfig(
        enabled=True, engagement_window_s=12.0, response_timeout_s=25.0,
        cooldown_s=90.0, capture_count=6, capture_interval_s=0.0,
        verify_polls=1, verify_interval_s=0.0, hold_enabled=False,
    )

    clock = [1000.0]
    enroller = FaceEnroller(
        say=say, speak=speak, enroll_stream=enroll_stream,
        verify_faces=verify_faces, turn_lock=asyncio.Lock(),
        now=lambda: clock[0], cfg=cfg, name_extractor=_name_extractor,
        record_action=mgr.add_system_action_turn,
    )

    # Drive a true-stranger /faces stream until the consent offer fires.
    W, H = 640, 360
    for _ in range(40):
        faces = [{"name": "unknown", "distance": 0.75, "confidence": "low",
                  "bbox": _box(W / 2)}]
        await enroller.observe_faces(faces, (W, H), engaged=True)
        clock[0] += 0.25
        if enroller.state == State.OFFERING:
            break
    assert enroller.state == State.OFFERING, "FSM never reached OFFERING"

    # Consent (no name) -> ASK_NAME; then a name -> CONFIRM_NAME. Each line is
    # spoken via _safe_speak and must now be recorded.
    await enroller.handle("yes, you have my consent", "unknown_1")
    await enroller.handle("my name is Sarah", "unknown_1")

    history = " ".join(
        m["content"] for m in mgr.build_history_messages()
        if m["role"] == "assistant"
    ).lower()

    assert "recognise your face" in history    # the consent offer
    assert "what's your name" in history        # the name ask
    assert "sarah" in history                   # the name confirmation
