"""Theme D — occupancy gate for proactive speech.

Proactive remarks previously fired off VLM person-detection flapping
(is_new_arrival) even when nobody was present — 89 lines into an empty workshop
overnight (2026-06-13). The room ledger already computes a TTL-windowed presence
snapshot; `anyone_present()` interprets it so main.maybe_speak_proactively can
stay silent in an empty room.

- Unit: anyone_present truth table.
- Glue: tie the interpreter to the REAL RoomLedger.current_state() output, so a
        rename of the `present` / `unknown_voices_recent` keys can't silently
        defeat the gate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.ledger import RoomLedger, anyone_present


def test_anyone_present_truth_table():
    assert anyone_present(None) is False
    assert anyone_present({}) is False
    assert anyone_present({"present": [], "unknown_voices_recent": 0}) is False
    assert anyone_present({"present": [{"name": "dan"}], "unknown_voices_recent": 0}) is True
    # An unknown voice with nobody on the named/face list still counts as occupied.
    assert anyone_present({"present": [], "unknown_voices_recent": 2}) is True


def test_empty_real_ledger_reads_as_unoccupied():
    led = RoomLedger()  # save_path=None -> no disk
    assert anyone_present(led.current_state()) is False


def test_named_voice_makes_real_ledger_occupied():
    led = RoomLedger()
    led.update_from_voice("dan")
    state = led.current_state()
    assert state["present"], "expected dan on the present list"
    assert anyone_present(state) is True


def test_unknown_voice_makes_real_ledger_occupied():
    led = RoomLedger()
    led.update_from_voice("unknown_3")
    state = led.current_state()
    # unknown_N voices are not on `present` but are counted separately.
    assert state["unknown_voices_recent"] >= 1
    assert anyone_present(state) is True


def test_stale_ledger_ages_out_to_unoccupied():
    led = RoomLedger(presence_ttl_sec=900.0)
    led.update_from_voice("dan", ts=1000.0)
    # Query far past the TTL -> nobody present -> gate suppresses (the overnight
    # empty-room case: ledger had aged Dan out while the VLM kept flapping).
    assert anyone_present(led.current_state(now_ts=1000.0 + 100_000)) is False
