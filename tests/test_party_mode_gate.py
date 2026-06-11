"""Unit tests for the party-mode near-field Phase 2 allowlist gate
(main.party_mode_drop). Pure predicate — no audio, no toggles I/O.
Run:
    .venv/bin/pytest tests/test_party_mode_gate.py -v
"""

import pytest

from main import party_mode_drop


# (name, allowlist, expect_drop)
CASES = [
    # Allowlisted speaker passes
    ("dan", ["dan"], False),
    ("Dan", ["dan"], False),            # case-insensitive name
    ("dan", ["Dan"], False),            # case-insensitive allowlist
    ("devon", ["dan", "devon"], False),
    # Non-allowlisted enrolled speaker drops
    ("erin", ["dan"], True),
    # Unknown speakers always drop when an allowlist is set
    ("unknown_1", ["dan"], True),
    ("unknown_12", ["dan", "devon"], True),
    # Empty/None allowlist == allow all (energy floor only)
    ("dan", [], False),
    ("unknown_1", [], False),
    ("erin", None, False),
]


@pytest.mark.parametrize("name,allowlist,expect_drop", CASES)
def test_party_mode_drop(name, allowlist, expect_drop):
    assert party_mode_drop(name, allowlist) is expect_drop


def test_unknown_never_allowlistable():
    # Even if someone fat-fingers an unknown_* name INTO the allowlist,
    # unknowns still drop — unknown identities are unstable cluster ids,
    # never a grant.
    assert party_mode_drop("unknown_1", ["dan", "unknown_1"]) is True
