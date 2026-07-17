"""Hermetic tests for the name-tell three-way link (2026-07-06).

A confirmed introductions name-tell now also commits the co-sampled face
crops through commit_identity (name<->voiceprint<->faceprint) when the
intro_face_commit_enabled toggle is on — closing the voice-only gap in the
introductions path. The commit must never block or break the name promotion:
toggle off, empty buffer, a refused assign_name (tombstone), a declining
CommitResult, and a raising committer all leave the promotion intact.

Fakes per tests/test_introductions.py; runtime_toggles redirected to a tmp
file; the committer is injected (never touches presence.identity_commit).

Run:
    .venv/bin/pytest tests/test_intro_face_commit.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import runtime_toggles
from conversation.introductions import Introductions
from presence.cosample import CoSampleBuffer


@pytest.fixture
def toggles(monkeypatch, tmp_path):
    """Isolated runtime_toggles: state lives in a per-test tmp file."""
    monkeypatch.setattr(runtime_toggles, "STATE_PATH",
                        tmp_path / "lt_runtime_toggles.json")
    return runtime_toggles


class FakeTurn:
    def __init__(self):
        self.said: list[str] = []

    async def say(self, prompt_text: str):
        self.said.append(prompt_text)
        return SimpleNamespace(text=f"[said: {prompt_text[:24]}...]")


class FakeSpeakerID:
    def __init__(self, known=("dan", "timmy"), unknown_temp_ids=(),
                 assign_ok=True):
        self._known_speakers = [SimpleNamespace(name=n) for n in known]
        self._unknown_speakers = [
            SimpleNamespace(temp_id=t, name_asked=True) for t in unknown_temp_ids
        ]
        self.assigned: list[tuple[str, str]] = []
        self._assign_ok = assign_ok

    def assign_name(self, temp_id, name, **kwargs):
        self.assigned.append((temp_id, name))
        if self._assign_ok:
            # Mirror the real T1 promotion: a session-local id that may
            # differ from the id-map's authoritative allocation.
            self._known_speakers.append(
                SimpleNamespace(name=name, speaker_id=99))
            return name  # final canonical (may be an auto-suffixed fork)
        return None


class FakeCommitter:
    def __init__(self, face_committed=True, status="committed", raises=False):
        self.calls: list[dict] = []
        self._res = SimpleNamespace(face_committed=face_committed,
                                    voice_committed=False,
                                    status=status, speaker_id=42)
        self._raises = raises

    async def __call__(self, name, **kwargs):
        self.calls.append({"name": name, **kwargs})
        if self._raises:
            raise RuntimeError("boom")
        return self._res


def _crop(v):
    return np.full((112, 112, 3), v, dtype=np.uint8)


def _make(*, assign_ok=True, committer=None, crops_key=None, n_crops=2):
    spk = FakeSpeakerID(unknown_temp_ids=("unknown_1",), assign_ok=assign_ok)
    turn = FakeTurn()
    cos = CoSampleBuffer()
    if crops_key:
        cos.add(crops_key, [_crop(i) for i in range(n_crops)])
    intro = Introductions(speaker_id_module=spk, turn=turn,
                          cosample=cos, committer=committer)
    return intro, spk, turn, cos


async def _confirm_yes(intro):
    """Drive the FSM to a confirmed 'bob'."""
    await intro.ask_name(SimpleNamespace(temp_id="unknown_1", last_text="hi"))
    await intro.handle("I'm Bob", "unknown_1")   # -> pending confirm
    return await intro.handle("yes", "unknown_1")


# --- the happy triple --------------------------------------------------------

@pytest.mark.asyncio
async def test_confirm_yes_commits_buffered_crops(toggles):
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    intro, spk, _, cos = _make(committer=committer, crops_key="unknown_1")

    out = await _confirm_yes(intro)
    assert out.handled is False and out.speaker_name == "bob"
    assert spk.assigned == [("unknown_1", "bob")]
    assert len(committer.calls) == 1
    call = committer.calls[0]
    assert call["name"] == "bob"
    assert len(call["face_crops"]) == 2
    assert call["speaker_identifier"] is spk


@pytest.mark.asyncio
async def test_speaker_id_synced_from_commit(toggles):
    """The commit's authoritative id-map id replaces assign_name's
    session-local _next_known_id() on the in-memory KnownSpeaker (facts
    written this session key off it)."""
    toggles.set("intro_face_commit_enabled", True)
    intro, spk, _, _ = _make(committer=FakeCommitter(), crops_key="unknown_1")
    await _confirm_yes(intro)
    ks = next(k for k in spk._known_speakers if k.name == "bob")
    assert ks.speaker_id == 42  # FakeCommitter's res.speaker_id, not 99


@pytest.mark.asyncio
async def test_buffer_cleared_after_face_commit(toggles):
    toggles.set("intro_face_commit_enabled", True)
    intro, _, _, cos = _make(committer=FakeCommitter(), crops_key="unknown_1")
    await _confirm_yes(intro)
    assert cos.crops_for("unknown_1") == []
    assert cos.crops_for("bob") == []


@pytest.mark.asyncio
async def test_crops_fall_back_to_name_key(toggles):
    """Crops buffered under the NAME (post-promotion turns) still bind."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    intro, _, _, cos = _make(committer=committer, crops_key="bob")
    await _confirm_yes(intro)
    assert len(committer.calls) == 1
    assert len(committer.calls[0]["face_crops"]) == 2
    assert cos.crops_for("bob") == []


# --- inertness / never-break-promotion ---------------------------------------

@pytest.mark.asyncio
async def test_toggle_off_no_commit(toggles):
    committer = FakeCommitter()
    intro, spk, _, cos = _make(committer=committer, crops_key="unknown_1")
    out = await _confirm_yes(intro)
    assert out.speaker_name == "bob"            # promotion unchanged
    assert committer.calls == []                # default toggle False
    assert len(cos.crops_for("unknown_1")) == 2  # buffer untouched


@pytest.mark.asyncio
async def test_no_crops_no_commit(toggles):
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    intro, _, _, _ = _make(committer=committer, crops_key=None)
    out = await _confirm_yes(intro)
    assert out.speaker_name == "bob"
    assert committer.calls == []


@pytest.mark.asyncio
async def test_assign_refused_no_commit(toggles):
    """Tombstoned/reserved name: assign_name False -> NO face commit under it."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    intro, spk, _, cos = _make(assign_ok=False, committer=committer,
                               crops_key="unknown_1")
    await _confirm_yes(intro)
    assert committer.calls == []
    assert len(cos.crops_for("unknown_1")) == 2


@pytest.mark.asyncio
async def test_committer_exception_promotion_survives(toggles):
    toggles.set("intro_face_commit_enabled", True)
    intro, spk, _, cos = _make(committer=FakeCommitter(raises=True),
                               crops_key="unknown_1")
    out = await _confirm_yes(intro)
    assert out.handled is False and out.speaker_name == "bob"
    assert spk.assigned == [("unknown_1", "bob")]
    # Failed commit leaves the buffer for a retry path (e.g. "enroll me").
    assert len(cos.crops_for("unknown_1")) == 2


@pytest.mark.asyncio
async def test_commit_declined_keeps_buffer(toggles):
    """commit_identity's guards (mismatch/lookalike/retired) decline ->
    voice-only promotion stands, crops kept."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter(face_committed=False, status="lookalike")
    intro, _, _, cos = _make(committer=committer, crops_key="unknown_1")
    out = await _confirm_yes(intro)
    assert out.speaker_name == "bob"
    assert len(committer.calls) == 1
    assert len(cos.crops_for("unknown_1")) == 2


@pytest.mark.asyncio
async def test_forked_name_face_commits_under_fork(toggles):
    """Duplicate display name (expo 2026-07-16): assign_name returns the
    auto-suffixed fork; the face commit and the turn's speaker_name must use
    the FORKED canonical, never the other person's name."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()

    class ForkingSpeakerID(FakeSpeakerID):
        def assign_name(self, temp_id, name, **kwargs):
            FakeSpeakerID.assign_name(self, temp_id, name, **kwargs)
            forked = f"{name}_2"
            self._known_speakers[-1].name = forked
            return forked

    spk = ForkingSpeakerID(unknown_temp_ids=("unknown_1",))
    cos = CoSampleBuffer()
    cos.add("unknown_1", [_crop(0), _crop(1)])
    intro = Introductions(speaker_id_module=spk, turn=FakeTurn(),
                          cosample=cos, committer=committer)
    out = await _confirm_yes(intro)
    assert out.speaker_name == "bob_2"
    assert committer.calls[0]["name"] == "bob_2"
    # id sync targets the forked KnownSpeaker.
    ks = next(k for k in spk._known_speakers if k.name == "bob_2")
    assert ks.speaker_id == 42


@pytest.mark.asyncio
async def test_no_cosample_injected_is_inert(toggles):
    """Constructed without a buffer (legacy call sites) -> nothing fires."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    spk = FakeSpeakerID(unknown_temp_ids=("unknown_1",))
    intro = Introductions(speaker_id_module=spk, turn=FakeTurn(),
                          committer=committer)
    await intro.ask_name(SimpleNamespace(temp_id="unknown_1", last_text="hi"))
    await intro.handle("I'm Bob", "unknown_1")
    out = await intro.handle("yes", "unknown_1")
    assert out.speaker_name == "bob"
    assert committer.calls == []


# --- live-grab fallback (2026-07-16: anchored=[] all session -> buffer never
# --- filled -> passive name-tell degraded to voice-only; sampler closes it) ---

class FakeSampler:
    """Stands in for orchestrator._gather_enroll_samples(name, "face")."""
    def __init__(self, crops=None, raises=False):
        self.calls: list[str] = []
        self._crops = crops if crops is not None else []
        self._raises = raises

    async def __call__(self, name):
        self.calls.append(name)
        if self._raises:
            raise RuntimeError("camera down")
        return list(self._crops), None


def _make_with_sampler(*, sampler, committer=None, crops_key=None):
    spk = FakeSpeakerID(unknown_temp_ids=("unknown_1",))
    cos = CoSampleBuffer()
    if crops_key:
        cos.add(crops_key, [_crop(i) for i in range(2)])
    intro = Introductions(speaker_id_module=spk, turn=FakeTurn(),
                          cosample=cos, committer=committer,
                          face_sampler=sampler)
    return intro, spk, cos


@pytest.mark.asyncio
async def test_empty_buffer_falls_back_to_live_grab(toggles):
    """No co-sampled crops -> the sampler's live grab feeds the commit."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    sampler = FakeSampler(crops=[_crop(1), _crop(2), _crop(3)])
    intro, _, _ = _make_with_sampler(sampler=sampler, committer=committer)
    out = await _confirm_yes(intro)
    assert out.speaker_name == "bob"
    assert sampler.calls == ["bob"]           # grabbed under the FINAL name
    assert len(committer.calls) == 1
    assert len(committer.calls[0]["face_crops"]) == 3


@pytest.mark.asyncio
async def test_buffered_crops_skip_live_grab(toggles):
    """Buffer hit -> no camera round-trip."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    sampler = FakeSampler(crops=[_crop(9)])
    intro, _, _ = _make_with_sampler(sampler=sampler, committer=committer,
                                     crops_key="unknown_1")
    await _confirm_yes(intro)
    assert sampler.calls == []
    assert len(committer.calls[0]["face_crops"]) == 2  # the buffered pair


@pytest.mark.asyncio
async def test_live_grab_empty_stays_voice_only(toggles):
    """Sampler abstains (crowd/no camera) -> voice-only promotion, no commit."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    intro, _, _ = _make_with_sampler(sampler=FakeSampler(crops=[]),
                                     committer=committer)
    out = await _confirm_yes(intro)
    assert out.speaker_name == "bob"          # promotion intact
    assert committer.calls == []


@pytest.mark.asyncio
async def test_live_grab_exception_promotion_survives(toggles):
    """A raising sampler must never break the name promotion."""
    toggles.set("intro_face_commit_enabled", True)
    committer = FakeCommitter()
    intro, spk, _ = _make_with_sampler(sampler=FakeSampler(raises=True),
                                       committer=committer)
    out = await _confirm_yes(intro)
    assert out.handled is False and out.speaker_name == "bob"
    assert spk.assigned == [("unknown_1", "bob")]
    assert committer.calls == []
