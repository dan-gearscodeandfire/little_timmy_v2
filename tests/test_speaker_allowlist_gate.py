"""Unit tests for the speaker-allowlist reply-gate predicate
(main.speaker_allowlist_drop). Pure predicate — no audio, no toggles I/O.

The call site in process_speech is SHELVED (commented out) as of 2026-06-10 —
speaker-ID isn't reliable enough to gate replies on yet. These tests keep the
predicate honest for when it's re-enabled.

Run:
    .venv/bin/pytest tests/test_speaker_allowlist_gate.py -v
"""

import pytest

from main import speaker_allowlist_drop


# (name, allowlist, expect_drop)
CASES = [
    # Allowlisted speaker passes
    ("dan", ["dan"], False),
    ("Dan", ["dan"], False),            # case-insensitive name
    ("dan", ["Dan"], False),            # case-insensitive allowlist
    ("robin", ["dan", "robin"], False),
    # Non-allowlisted enrolled speaker drops
    ("erin", ["dan"], True),
    # Unknown speakers always drop when an allowlist is set
    ("unknown_1", ["dan"], True),
    ("unknown_12", ["dan", "robin"], True),
    # Empty/None allowlist == allow all (energy floor only)
    ("dan", [], False),
    ("unknown_1", [], False),
    ("erin", None, False),
]


@pytest.mark.parametrize("name,allowlist,expect_drop", CASES)
def test_speaker_allowlist_drop(name, allowlist, expect_drop):
    assert speaker_allowlist_drop(name, allowlist) is expect_drop


def test_unknown_never_allowlistable():
    # Even if someone fat-fingers an unknown_* name INTO the allowlist,
    # unknowns still drop — unknown identities are unstable cluster ids,
    # never a grant.
    assert speaker_allowlist_drop("unknown_1", ["dan", "unknown_1"]) is True
