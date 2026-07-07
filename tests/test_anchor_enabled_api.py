"""Hermetic tests for GET/POST /api/anchor_enabled (webui button, 2026-07-07).

The anchor master toggle had no API surface — flipping it meant a shell on
okdemerzel (`rt.set('anchor_enabled', ...)`), and Dan has no Claude Code at
the show. The endpoint follows the standard toggle-POST contract
({"enabled": bool}, like /api/identity_dialogs) and returns `active` so a
flip's effect is visible in one read. Disabling also CLEARS any live anchor
(the bench teardown recipe): the gate goes dark via the disjunct either way,
but a residual armed anchor would keep GET /api/anchor reporting active=true
until TTL expiry — lying status on the booth screen.

runtime_toggles redirected to a tmp file (+ mtime parse cache reset — the
cache is module-global, a stale stamp would leak live state across the
monkeypatched STATE_PATH).

Run:
    .venv/bin/pytest tests/test_anchor_enabled_api.py -v
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import runtime_toggles
from presence import anchor
from web import app as web_app


@pytest.fixture
def toggles(monkeypatch, tmp_path):
    """Isolated runtime_toggles: tmp state file + parse cache reset."""
    monkeypatch.setattr(runtime_toggles, "STATE_PATH",
                        tmp_path / "lt_runtime_toggles.json")
    monkeypatch.setattr(runtime_toggles, "_cache_stamp", None)
    monkeypatch.setattr(runtime_toggles, "_cache_state", None)
    return runtime_toggles


@pytest.fixture(autouse=True)
def clean_anchor():
    anchor.clear_anchor()
    yield
    anchor.clear_anchor()


def _get():
    return asyncio.run(web_app.get_anchor_enabled())


def _post(payload):
    return asyncio.run(web_app.set_anchor_enabled(payload))


# --- GET --------------------------------------------------------------------

def test_get_default_off(toggles):
    d = _get()
    assert d == {"enabled": False, "active": False}


def test_get_reflects_toggle_and_anchor(toggles):
    toggles.set("anchor_enabled", True)
    anchor.set_anchor((100.0, 200.0), source="stub")
    d = _get()
    assert d["enabled"] is True
    assert d["active"] is True


# --- POST -------------------------------------------------------------------

def test_post_enable_persists(toggles):
    d = _post({"enabled": True})
    assert d["ok"] is True
    assert d["enabled"] is True
    assert toggles.get("anchor_enabled") is True


def test_post_enable_does_not_arm(toggles):
    # Enabling is arming the FEATURE, not declaring an anchor — active stays
    # false until the poll loop (or a stub POST) actually anchors someone.
    d = _post({"enabled": True})
    assert d["active"] is False
    assert anchor.get_anchor() is None


def test_post_enable_preserves_live_anchor(toggles):
    # Re-enabling with a stub already declared must not wipe it (F4 spirit:
    # only an explicit clear or the disable teardown touches live state).
    toggles.set("anchor_enabled", True)
    anchor.set_anchor((100.0, 200.0), source="stub")
    d = _post({"enabled": True})
    assert d["active"] is True
    assert anchor.get_anchor() is not None


def test_post_disable_clears_anchor(toggles):
    # The teardown recipe in one flip: toggle off AND clear, so GET /api/anchor
    # can't report a residual active=true for up to ttl_s after the operator
    # turned the feature off.
    toggles.set("anchor_enabled", True)
    anchor.set_anchor((100.0, 200.0), source="stub")
    d = _post({"enabled": False})
    assert d["ok"] is True
    assert d["enabled"] is False
    assert d["active"] is False
    assert toggles.get("anchor_enabled") is False
    assert anchor.get_anchor() is None


def test_post_empty_payload_means_off(toggles):
    # Same defaulting as /api/identity_dialogs: absent field -> False.
    toggles.set("anchor_enabled", True)
    d = _post(None)
    assert d["enabled"] is False
    assert toggles.get("anchor_enabled") is False
