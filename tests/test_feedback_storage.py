"""Unit tests for feedback.storage.read_last_flagged. Backs the dashboard's
'Last flag' review modal. Pure I/O — uses tmp_path + monkeypatch."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feedback import storage as fb_storage


def _set_flagged_path(monkeypatch, path: Path):
    monkeypatch.setattr(fb_storage, "FLAGGED_PATH", path)


def test_returns_none_when_file_missing(tmp_path, monkeypatch):
    _set_flagged_path(monkeypatch, tmp_path / "nope.jsonl")
    assert fb_storage.read_last_flagged() is None


def test_returns_none_when_file_empty(tmp_path, monkeypatch):
    p = tmp_path / "flagged.jsonl"
    p.write_text("", encoding="utf-8")
    _set_flagged_path(monkeypatch, p)
    assert fb_storage.read_last_flagged() is None


def test_returns_single_entry(tmp_path, monkeypatch):
    p = tmp_path / "flagged.jsonl"
    entry = {"ts": 1700000000.0, "kind": "bad", "response": "hi"}
    p.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    _set_flagged_path(monkeypatch, p)
    got = fb_storage.read_last_flagged()
    assert got == entry


def test_returns_last_of_multiple(tmp_path, monkeypatch):
    p = tmp_path / "flagged.jsonl"
    e1 = {"ts": 1.0, "kind": "good", "response": "first"}
    e2 = {"ts": 2.0, "kind": "bad", "response": "second"}
    e3 = {"ts": 3.0, "kind": "bad", "response": "third"}
    p.write_text(
        json.dumps(e1) + "\n" + json.dumps(e2) + "\n" + json.dumps(e3) + "\n",
        encoding="utf-8",
    )
    _set_flagged_path(monkeypatch, p)
    got = fb_storage.read_last_flagged()
    assert got == e3


def test_skips_corrupt_lines_returns_last_valid(tmp_path, monkeypatch):
    # A truncated power-loss write or a tee mistake shouldn't break read.
    p = tmp_path / "flagged.jsonl"
    e1 = {"ts": 1.0, "kind": "good"}
    e2 = {"ts": 2.0, "kind": "bad"}
    p.write_text(
        json.dumps(e1) + "\n"
        + json.dumps(e2) + "\n"
        + "{not valid json\n",
        encoding="utf-8",
    )
    _set_flagged_path(monkeypatch, p)
    got = fb_storage.read_last_flagged()
    assert got == e2


def test_returns_none_when_all_lines_corrupt(tmp_path, monkeypatch):
    p = tmp_path / "flagged.jsonl"
    p.write_text("garbage\nmore garbage\n", encoding="utf-8")
    _set_flagged_path(monkeypatch, p)
    assert fb_storage.read_last_flagged() is None


def test_preserves_conversation_history_roundtrip(tmp_path, monkeypatch):
    """The point of the new field. Make sure nested lists survive
    append_flagged -> read_last_flagged with no data loss."""
    p = tmp_path / "flagged.jsonl"
    monkeypatch.setattr(fb_storage, "FLAGGED_PATH", p)
    monkeypatch.setattr(fb_storage, "PERSONA_TUNING_DIR", tmp_path)
    history = [
        {"role": "user", "content": "hey", "speaker": "dan", "timestamp": 1.0},
        {"role": "assistant", "content": "yo", "speaker": None, "timestamp": 2.0},
    ]
    fb_storage.append_flagged("bad", {
        "ts": 100.0,
        "source": "ui_button",
        "speaker": "dan",
        "user_prompt": "hey",
        "response": "yo",
        "comment": "test",
        "system_prompt": "(sys)",
        "conversation_history": history,
    })
    got = fb_storage.read_last_flagged()
    assert got is not None
    assert got["conversation_history"] == history
    assert got["kind"] == "bad"
    assert got["iso_ts"]  # added by append_flagged
