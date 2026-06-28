"""Tests for near-homophone name de-duplication (presence/identity.py).

Pure, hermetic — no DB, no network, no live services. Safe to run anytime.
Covers the Devon/Devin split (2026-06-24): a face name minted as the homophone
"devin" must snap onto the already-known voice identity "devon", while
genuinely distinct enrolled names must never cross-match.
"""

from presence.identity import resolve_alias, soundex

# The actual enrolled set the resolver must keep distinct.
KNOWN = ["dan", "timmy", "thea", "devon", "erin"]


class TestSoundex:
    def test_homophones_share_a_code(self):
        assert soundex("devon") == soundex("devin")

    def test_distinct_names_differ(self):
        # Every enrolled name has a distinct Soundex from every other.
        codes = {n: soundex(n) for n in KNOWN}
        assert len(set(codes.values())) == len(KNOWN), codes

    def test_empty_and_nonalpha(self):
        assert soundex("") == ""
        assert soundex("123") == ""
        assert soundex(None) == ""

    def test_case_insensitive(self):
        assert soundex("Devon") == soundex("DEVON") == soundex("devon")


class TestResolveAlias:
    def test_devin_snaps_to_devon(self):
        assert resolve_alias("devin", KNOWN) == "devon"

    def test_devin_snaps_regardless_of_case(self):
        assert resolve_alias("Devin", KNOWN) == "devon"
        assert resolve_alias("  DEVIN ", KNOWN) == "devon"

    def test_exact_known_name_is_noop(self):
        # An already-canonical known name returns itself, not None.
        assert resolve_alias("devon", KNOWN) == "devon"
        assert resolve_alias("dan", KNOWN) == "dan"

    def test_genuinely_new_name_does_not_snap(self):
        # A real new person must NOT be swallowed into an existing identity.
        assert resolve_alias("marcus", KNOWN) is None
        assert resolve_alias("priya", KNOWN) is None

    def test_distinct_enrolled_names_never_cross_match(self):
        # Snapping any known name only ever returns itself (no false merges,
        # e.g. dan must never resolve to devon).
        for n in KNOWN:
            assert resolve_alias(n, KNOWN) == n

    def test_unknown_prefixed_names_never_match(self):
        assert resolve_alias("unknown_1", KNOWN) is None
        assert resolve_alias("unknown_face_abc123", KNOWN) is None

    def test_unknown_records_in_known_set_are_ignored(self):
        # A synthetic unknown record must not be a snap target.
        assert resolve_alias("devin", ["unknown_face_x", "devon"]) == "devon"

    def test_empty_inputs(self):
        assert resolve_alias("", KNOWN) is None
        assert resolve_alias(None, KNOWN) is None
        assert resolve_alias("devin", []) is None

    def test_threshold_blocks_low_similarity_same_soundex(self):
        # Same Soundex but low string overlap should not snap when the ratio is
        # below the floor. "d" and "devon" share Soundex bucketing risk only if
        # codes match; assert the guard holds for a deliberately strict ratio.
        assert resolve_alias("devin", KNOWN, min_ratio=0.95) is None
        # ...but the default floor DOES accept the genuine near-homophone.
        assert resolve_alias("devin", KNOWN) == "devon"
