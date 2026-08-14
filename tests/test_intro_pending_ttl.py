"""F3 regression: a pending name exchange must expire on wall-clock silence."""
import sys, time, types
from pathlib import Path
import pytest
sys.path.insert(0, str(Path("/home/gearscodeandfire/little_timmy")))
from conversation.introductions import Introductions


class _Spk:
    _known_speakers = []


class _Turn:
    def __init__(self): self.said = []
    async def say(self, text):
        self.said.append(text)
        return types.SimpleNamespace(text="ok")


def _intro():
    return Introductions(speaker_id_module=_Spk(), turn=_Turn())


@pytest.mark.asyncio
async def test_pending_confirm_expires_after_ttl(monkeypatch):
    i = _intro()
    await i.offer_confirm("unknown_1", "Marcus")
    assert i.awaiting, "confirm should be armed immediately after offer_confirm"

    # 7m40s of silence — the exact gap that hijacked a fresh conversation live.
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base + 460.0)
    assert not i.awaiting, "a 7m40s-stale confirm must not still be armed"


@pytest.mark.asyncio
async def test_pending_confirm_survives_inside_the_ttl(monkeypatch):
    i = _intro()
    await i.offer_confirm("unknown_1", "Marcus")
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base + 30.0)
    assert i.awaiting, "a 30s-old confirm is a live conversation, keep it"


@pytest.mark.asyncio
async def test_ask_name_arms_and_expires(monkeypatch):
    i = _intro()
    await i.ask_name(types.SimpleNamespace(temp_id="unknown_1", last_text="hello"))
    assert i.awaiting
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base + 200.0)
    assert not i.awaiting


@pytest.mark.asyncio
async def test_ask_name_prompt_carries_no_roster():
    # It used to interpolate every enrolled name and assert they were present.
    _Spk._known_speakers = [types.SimpleNamespace(name=f"person{n}") for n in range(43)]
    t = _Turn()
    i = Introductions(speaker_id_module=_Spk(), turn=t)
    await i.ask_name(types.SimpleNamespace(temp_id="unknown_1", last_text="hello"))
    prompt = t.said[0]
    assert "person0" not in prompt and "person42" not in prompt, "no roster in the prompt"
    assert "is here" not in prompt, "must not assert who is present"
    assert len(prompt) < 400, f"prompt should stay small, got {len(prompt)} chars"


@pytest.mark.asyncio
async def test_expiry_is_idempotent_and_quiet_when_nothing_pending():
    i = _intro()
    for _ in range(3):
        assert not i.awaiting
