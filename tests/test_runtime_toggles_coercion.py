"""Hermetic tests for runtime_toggles type-checked merge + numeric coercion
(F10, review 7-07) and the mtime parse cache. Run:

    .venv/bin/pytest tests/test_runtime_toggles_coercion.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import runtime_toggles


@pytest.fixture
def toggles(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_toggles, "STATE_PATH",
                        tmp_path / "lt_runtime_toggles.json")
    monkeypatch.setattr(runtime_toggles, "_cache_stamp", None)
    monkeypatch.setattr(runtime_toggles, "_cache_state", None)
    return runtime_toggles


def _write(toggles, obj):
    toggles.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    toggles.STATE_PATH.write_text(json.dumps(obj))


def test_int_for_float_knob_coerces(toggles):
    # The booth-tuning trap: a hand-edited int for a float knob used to be
    # silently dropped back to the default (ttl 60 -> 30.0).
    _write(toggles, {"anchor_ttl_s": 60})
    assert toggles.get("anchor_ttl_s") == 60.0
    assert isinstance(toggles.get("anchor_ttl_s"), float)


def test_float_for_int_knob_coerces_when_integral(toggles):
    _write(toggles, {"anchor_led_v_min": 180.0})
    assert toggles.get("anchor_led_v_min") == 180
    assert isinstance(toggles.get("anchor_led_v_min"), int)


def test_non_integral_float_for_int_knob_drops_to_default(toggles):
    _write(toggles, {"anchor_led_v_min": 180.5})
    assert toggles.get("anchor_led_v_min") == 200   # design default


def test_bool_never_coerces_to_number(toggles):
    _write(toggles, {"anchor_ttl_s": True, "anchor_led_v_min": True})
    assert toggles.get("anchor_ttl_s") == 30.0
    assert toggles.get("anchor_led_v_min") == 200


def test_number_never_coerces_to_bool(toggles):
    _write(toggles, {"anchor_enabled": 1})
    assert toggles.get("anchor_enabled") is False


def test_wrong_type_string_drops_to_default(toggles):
    _write(toggles, {"anchor_ttl_s": "60"})
    assert toggles.get("anchor_ttl_s") == 30.0


def test_cache_invalidates_on_write_through_set(toggles):
    _write(toggles, {"anchor_ttl_s": 45.0})
    assert toggles.get("anchor_ttl_s") == 45.0
    toggles.set("anchor_ttl_s", 12.5)
    assert toggles.get("anchor_ttl_s") == 12.5


def test_cache_invalidates_on_manual_file_edit(toggles):
    _write(toggles, {"anchor_ttl_s": 45.0})
    assert toggles.get("anchor_ttl_s") == 45.0
    # Manual edit (different content -> different size/mtime) applies live.
    _write(toggles, {"anchor_ttl_s": 99.0})
    assert toggles.get("anchor_ttl_s") == 99.0
