"""Hermetic tests for the introductions short-reply window freshness
(_dialog_owns_turn's intro_fresh term) — the gate that decides whether a
bare "Yes."/"No." survives the STT hallucination filter mid-dialog.

Regression for the 7-16 Pat enroll session: the window was stamped only at
the INITIAL ask, so a meandering confirm dialog (re-asks at 19:12:17 and
19:12:27) outlived its own 30s window and Pat's bare "Yes" at 19:12:31
(30.2s after the first ask, 4s after the last re-ask) was silently eaten.
Fix: main restamps _introductions_asked_ts on every handled introductions
exchange while still awaiting — same 067840b semantics as the enroll latch
(TTL measures silence since the LAST exchange).

Run:
    .venv/bin/pytest tests/test_intro_reply_window.py -v
"""

import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def _stub(intro_awaiting=True, asked_ago=0.0):
    """Minimal Orchestrator stand-in for the unbound _dialog_owns_turn."""
    stub = types.SimpleNamespace(
        _pending_enroll=None,
        _pending_enroll_confirm=None,
        _enroll_latch_ts=0.0,
        ENROLL_LATCH_TTL_SEC=main.Orchestrator.ENROLL_LATCH_TTL_SEC,
        _introductions=types.SimpleNamespace(awaiting=intro_awaiting),
        _introductions_asked_ts=time.time() - asked_ago,
        _face_enroller=types.SimpleNamespace(awaiting=False),
    )
    stub._enroll_latch_pending = (
        lambda: main.Orchestrator._enroll_latch_pending(stub))
    return stub


def test_intro_fresh_within_ttl_owns_turn():
    assert main.Orchestrator._dialog_owns_turn(_stub(asked_ago=5.0)) is True


def test_intro_stale_past_ttl_releases_turn():
    # 30.2s after the last stamp — exactly the Pat failure geometry. A bare
    # "Yes" here goes through the STANDING hallucination filter and dies.
    assert main.Orchestrator._dialog_owns_turn(_stub(asked_ago=30.2)) is False


def test_intro_not_awaiting_never_owns_turn():
    assert main.Orchestrator._dialog_owns_turn(
        _stub(intro_awaiting=False, asked_ago=1.0)) is False


def test_restamp_semantics_revive_the_window():
    """The fix contract: after a handled exchange restamps asked_ts, the
    window must be fresh again even if the FIRST ask is long past."""
    stub = _stub(asked_ago=45.0)  # first ask long stale
    assert main.Orchestrator._dialog_owns_turn(stub) is False
    stub._introductions_asked_ts = time.time()  # what main does per exchange
    assert main.Orchestrator._dialog_owns_turn(stub) is True
