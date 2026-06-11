"""Unit tests for the B3 people-persistence gate in vision/relevance.py
(P4 fix 2026-06-11: single-frame face-ID flap must not fire proactive).

Pure-logic tests - no network, no LT services. Run:
    .venv/bin/pytest tests/test_relevance_people_gate.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.analyzer import SceneRecord
from vision import relevance
from vision.relevance import _confirmed_people, classify


def rec(people, actions=None, objects=None, scene="workshop, lights on",
        novelty=0.0, speak_now=False):
    return SceneRecord(
        people=list(people),
        actions=list(actions or ["working at bench"]),
        objects=list(objects or ["bench", "tools"]),
        scene_state=scene,
        change_from_prior="none",
        novelty=novelty,
        speak_now=speak_now,
    )


@pytest.fixture
def gate_on(monkeypatch):
    """Pin the knob so tests don't depend on the on-disk toggle file."""
    monkeypatch.setattr(
        relevance.runtime_toggles, "get",
        lambda key: 0.4 if key == "people_novelty_min_persistence" else None,
    )


@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.setattr(
        relevance.runtime_toggles, "get",
        lambda key: 0.0 if key == "people_novelty_min_persistence" else None,
    )


# ---------------------------------------------------------------------------
# _confirmed_people window math
# ---------------------------------------------------------------------------


class TestConfirmedPeople:
    def test_requires_two_of_five_at_default(self):
        # 0.4 * lookback 5 -> required 2
        records = [rec(["dan"])] * 4 + [rec(["unidentified person"])]
        confirmed = _confirmed_people(records, 0.4)
        assert "dan" in confirmed
        assert "unidentified person" not in confirmed

    def test_confirmed_survives_one_frame_dropout(self):
        # dan in 4 of last 5, absent from the newest record
        records = [rec(["dan"])] * 4 + [rec([])]
        assert "dan" in _confirmed_people(records, 0.4)

    def test_cold_start_confirms_nobody(self):
        # required is computed against the lookback constant, not len(records)
        assert _confirmed_people([rec(["dan"])], 0.4) == {}

    def test_casing_preserved(self):
        records = [rec(["Dan"]), rec(["Dan"])]
        assert _confirmed_people(records, 0.4) == {"dan": "Dan"}

    def test_dupes_within_record_count_once(self):
        records = [rec(["dan", "Dan"])]
        confirmed = _confirmed_people(records, 0.2)  # required = 1
        assert list(confirmed) == ["dan"]


# ---------------------------------------------------------------------------
# classify: the P4 flap scenario
# ---------------------------------------------------------------------------


class TestFlapScenario:
    def test_single_frame_flap_is_not_a_new_person(self, gate_on):
        """Dan in profile -> one 'unidentified person' frame. The exact P4
        misfire: must contribute no people-novelty and no new_people."""
        history = [rec(["dan"]) for _ in range(5)]
        flap = rec(["unidentified person"])
        result = classify(flap, history)
        assert result.new_people == []
        assert "unidentified person" not in result.confirmed_people
        assert "dan" in result.confirmed_people  # presence survives the flap
        # people term fully overlapped -> novelty stays low
        assert result.novelty_score < 0.3

    def test_flap_recovery_does_not_refire(self, gate_on):
        """Frame after the flap (dan back) must not look novel either."""
        history = [rec(["dan"]) for _ in range(4)] + [rec(["unidentified person"])]
        result = classify(rec(["dan"]), history)
        assert result.new_people == []
        assert result.novelty_score < 0.3

    def test_all_faces_miss_does_not_empty_presence(self, gate_on):
        """Sibling bug: one frame with no faces must not read 'everyone left'."""
        history = [rec(["dan"]) for _ in range(5)]
        result = classify(rec([]), history)
        assert "dan" in result.confirmed_people

    def test_real_guest_confirms_and_fires_once(self, gate_on):
        """A sustained genuine arrival crosses the threshold exactly once."""
        history = [rec(["dan"]) for _ in range(5)]
        # Frame 1 of guest: unconfirmed (1 of 5 < 2)
        r1 = classify(rec(["dan", "devon"]), history)
        assert "devon" not in r1.confirmed_people
        assert r1.new_people == []
        # Frame 2: 2 of 5 -> confirmed, rising edge fires
        history = history[1:] + [rec(["dan", "devon"])]
        r2 = classify(rec(["dan", "devon"]), history)
        assert "devon" in r2.confirmed_people
        assert r2.new_people == ["devon"]
        assert r2.novelty_score > 0.0
        # Frame 3: still confirmed, edge does NOT re-fire
        history = history[1:] + [rec(["dan", "devon"])]
        r3 = classify(rec(["dan", "devon"]), history)
        assert "devon" in r3.confirmed_people
        assert r3.new_people == []

    def test_flap_with_vlm_novelty_spike_still_no_new_people(self, gate_on):
        """Even if the VLM flags high novelty on the flap frame, the rising
        edge (what proactive keys on) stays quiet."""
        history = [rec(["dan"]) for _ in range(5)]
        result = classify(rec(["unidentified person"], novelty=0.96), history)
        assert result.new_people == []
        assert "unidentified person" not in result.confirmed_people


# ---------------------------------------------------------------------------
# gate off -> legacy behavior
# ---------------------------------------------------------------------------


class TestGateOff:
    def test_flap_counts_as_new_person_when_disabled(self, gate_off):
        history = [rec(["dan"]) for _ in range(5)]
        result = classify(rec(["unidentified person"]), history)
        assert result.new_people == ["unidentified person"]
        assert result.confirmed_people == ["unidentified person"]

    def test_first_record_is_novel(self, gate_off):
        result = classify(rec(["dan"]), [])
        assert result.novelty_score == 1.0


class TestGateOnEmptyHistory:
    def test_first_record_still_novel_overall(self, gate_on):
        # score_novelty returns 1.0 on empty history regardless of the gate
        result = classify(rec(["dan"]), [])
        assert result.novelty_score == 1.0
        # but nobody is confirmed yet (cold start earns frames first)
        assert result.confirmed_people == []
        assert result.new_people == []
