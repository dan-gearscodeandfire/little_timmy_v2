"""Unit tests for presence/display.py — meta-driven display names.

The invariant under test: an auto-suffixed fork (``mike_2``) renders as its
base display name ("Mike") ONLY via an id-map ``_meta`` hit — never by
string-parsing the suffix — so genuine underscored names survive intact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import presence.display as display  # noqa: E402
from presence.prototype_base import IdMap  # noqa: E402


def _idmap(tmp_path, monkeypatch) -> IdMap:
    """Point presence.display at a tmp id-map and reset its cache."""
    p = tmp_path / "_id_map.json"
    monkeypatch.setattr(display, "_SHARED_ID_MAP", p)
    monkeypatch.setattr(display, "_cache", None)
    return IdMap(p, reserved_ids={"dan": 1, "timmy": 2}, first_free_id=3)


def test_auto_suffix_fork_renders_as_base(tmp_path, monkeypatch):
    m = _idmap(tmp_path, monkeypatch)
    m.allocate("mike")
    m.allocate("mike_2")
    m.mark_auto_suffixed("mike_2", "mike")
    assert display.display_name("mike_2") == "Mike"
    assert display.display_base("mike_2") == "mike"
    assert display.display_name("mike") == "Mike"


def test_genuine_underscore_names_never_stripped(tmp_path, monkeypatch):
    m = _idmap(tmp_path, monkeypatch)
    m.allocate("dan_the_barbarian")
    m.allocate("mary_2")  # unlikely via voice, but must not be mangled
    assert display.display_name("dan_the_barbarian") == "Dan The Barbarian"
    assert display.display_name("mary_2") == "Mary 2"
    assert display.display_base("mary_2") == "mary_2"


def test_creators_registry_wins(tmp_path, monkeypatch):
    _idmap(tmp_path, monkeypatch)
    assert display.display_name("william_osman") == "William Osman"


def test_missing_id_map_falls_back_to_deslug(tmp_path, monkeypatch):
    monkeypatch.setattr(display, "_SHARED_ID_MAP", tmp_path / "absent.json")
    monkeypatch.setattr(display, "_cache", None)
    assert display.display_name("mike_2") == "Mike 2"
    assert display.display_base("mike_2") == "mike_2"


def test_cache_refreshes_on_file_change(tmp_path, monkeypatch):
    m = _idmap(tmp_path, monkeypatch)
    m.allocate("mike")
    m.allocate("mike_2")
    assert display.display_name("mike_2") == "Mike 2"   # no marker yet
    m.mark_auto_suffixed("mike_2", "mike")               # mtime moves
    assert display.display_name("mike_2") == "Mike"
