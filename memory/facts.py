"""Fact store: structured entity-attribute-value with provenance."""

import logging
from dataclasses import dataclass
from db.connection import get_pool

log = logging.getLogger(__name__)

# Identity-class predicates for the EXPO facts gate (Dan ruling 2026-07-07):
# any predicate that RENAMES its subject. The 'name' substring deliberately
# over-matches (name, nickname, preferred_name, first_name, has_a_robot_named
# ...) — the gate only runs while the identity dialogs are dark, where a
# false block costs one booth-chatter fact and a false pass rewrites who
# somebody IS (observed live 7-07: dan.name overwritten to a visitor's
# self-intro, twice).
_IDENTITY_PREDICATE_ALIASES = frozenset(
    {"goes by", "goes_by", "alias", "aka", "called", "known as", "known_as"})


def _is_identity_predicate(predicate: str) -> bool:
    return "name" in predicate or predicate in _IDENTITY_PREDICATE_ALIASES


@dataclass
class Fact:
    id: int
    subject: str
    predicate: str
    value: str
    learned_at: object
    confidence: float
    sensitive: bool = False


async def store_fact(
    subject: str,
    predicate: str,
    value: str,
    source_memory_id: int | None = None,
    speaker_id: int | None = None,
    confidence: float = 1.0,
    source: str = "extraction",
    turn_ts: float | None = None,
) -> int:
    """Upsert a fact. If (subject, predicate) exists and isn't superseded, supersede it.

    source: which writer is calling -- "tool" = explicit store_fact route (a
    user-directed correction), "extraction" = async background extractor.
    turn_ts: epoch seconds of the source turn (extraction only). Used for the
    recency-gated precedence below: the extractor must not clobber an explicit
    tool-written correction with a STALE earlier mention. See
    lt-store-fact-correction-clobbered-by-extractor-race-2026-06-21.
    """
    pool = await get_pool()
    subject = subject.strip().lower()
    predicate = predicate.strip().lower()

    # Redaction: never persist a fact containing a blocked term (e.g. Dan's last
    # name) -- it does not belong in any stored memory. Terms are loaded from a
    # gitignored file (config.REDACT_TERMS) so they stay out of source. Drop the
    # whole fact (word-boundary, case-insensitive) and return a sentinel id.
    import re as _re
    import config as _cfg
    _terms = getattr(_cfg, "REDACT_TERMS", ())
    if _terms:
        _hay = f"{subject} {predicate} {value}".lower()
        for _t in _terms:
            if _re.search(rf"\b{_re.escape(_t.lower())}\b", _hay):
                log.warning("[REDACT] dropped fact containing blocked term %r: %s.%s",
                            _t, subject, predicate)
                return -1

    # EXPO facts gate (Dan ruling 2026-07-07). While the identity dialogs are
    # dark (crowd regime, no override), the dialog interceptors that normally
    # catch name-claims BEFORE the LLM are off — so store_fact heard "my name
    # is Zorbo" from a misattributed visitor and overwrote dan.name, twice,
    # live. Both auto-writers (tool router + background extractor) come
    # through this chokepoint, so gate here:
    #   (a) identity-class predicates are blocked for ANY subject — renames
    #       flow only through the sanctioned dialog path (assign_name /
    #       commit_identity), never through booth chatter;
    #   (b) subjects that are not ENROLLED speakers are blocked entirely —
    #       no facts about (or keyed to) strangers. A visitor who enrolls via
    #       the anchored mic gets a speakers row and facts flow again.
    # Deliberately keyed on the PURE regime+override predicate, NOT the
    # LED-anchor disjunct: the anchor un-darks the identity DIALOGS for the
    # mic-holder, but fact-writing stays gated for the whole show (the
    # anchor's TTL window is exactly when a misattributed bystander turn can
    # fire — the observed leak). Same -1 sentinel contract as redaction.
    from persistence import runtime_toggles
    if not runtime_toggles.identity_dialogs_allowed():
        if _is_identity_predicate(predicate):
            log.warning("[FACT-GATE] blocked identity-key write while dialogs "
                        "dark: %s.%s = %r", subject, predicate, value)
            return -1
        enrolled = await pool.fetchrow(
            "SELECT 1 FROM speakers WHERE lower(name) = $1 AND retired_at IS NULL",
            subject,
        )
        if enrolled is None:
            log.warning("[FACT-GATE] blocked fact about unenrolled subject "
                        "while dialogs dark: %s.%s = %r",
                        subject, predicate, value)
            return -1

    # Classify sensitivity at creation (PII gating). Both fact writers -- the
    # extraction pipeline and the :8092 tool-call classifier -- pass through
    # here, so this is the single chokepoint. Recomputed on every upsert so a
    # changed value re-evaluates (e.g. a value that newly contains a phone#).
    from memory.pii import classify_sensitivity
    sensitive, pii_category = classify_sensitivity(subject, predicate, value)

    # Cross-predicate dedup. The (subject,predicate) unique index only collapses
    # EXACT duplicates; the background extractor and the tool-call writer record
    # the same fact under different free-text predicates ("has_robot" vs "has a
    # robot named" vs "name"), which escapes the index (observed 2026-06-20:
    # user/has_robot/Sparky + user/has_a_robot_named/Sparky). If writing this
    # triple would INSERT a brand-new (subject,predicate) row AND an active row
    # already states this exact VALUE about this SUBJECT under a DIFFERENT
    # predicate, treat it as a duplicate phrasing and return the existing row.
    #
    # GUARD ON THE GUARD (2026-06-20, found live): only dedup an INSERT, never an
    # UPDATE. If this exact (subject,predicate) already exists, the write is a
    # correction to THAT attribute -- never a new duplicate -- so it must go
    # through. Without this, a legit correction whose value coincides with a
    # DIFFERENT predicate's value gets silently dropped (it skipped restoring
    # dan.name="Dan" because dan."preferred name"="Dan" already existed).
    target = await pool.fetchrow(
        """SELECT id, source, learned_at FROM facts
           WHERE subject = $1 AND predicate = $2 AND superseded_by IS NULL""",
        subject, predicate,
    )

    # Recency-gated source precedence (2026-06-21, found live under acoustic
    # multi-turn load): the async extractor coalesces a debounce buffer and can
    # flush a STALE earlier mention AFTER an explicit store_fact correction has
    # already landed, overwriting the newer value via this same upsert key
    # (observed: robot Rusty -> Sparky, then recall served the stale Sparky).
    # An extraction write may overwrite a 'tool'-written (explicit) fact ONLY if
    # its source turn is newer than when the tool wrote. Tool writes always pass
    # (an explicit correction is the current user intent); extraction-over-
    # extraction and tool-over-* are unaffected. turn_ts=None from extraction is
    # treated as not-newer -> the explicit correction is protected.
    if target is not None and source != "tool" and target["source"] == "tool":
        la = target["learned_at"]
        la_epoch = la.timestamp() if la is not None else 0.0
        if turn_ts is None or turn_ts <= la_epoch:
            log.info(
                "Skip extraction overwrite of tool-written fact #%d (%s.%s = %s); "
                "source turn_ts=%s not newer than tool learned_at=%.0f",
                target["id"], subject, predicate, value, turn_ts, la_epoch,
            )
            return target["id"]

    if target is None:
        dup = await pool.fetchrow(
            """SELECT id, predicate FROM facts
               WHERE subject = $1 AND lower(value) = lower($2)
                 AND predicate <> $3 AND superseded_by IS NULL
               ORDER BY id LIMIT 1""",
            subject, value, predicate,
        )
        if dup is not None:
            log.info("Dedup fact: %s.%s = %s already stored as .%s (#%d); skipping",
                     subject, predicate, value, dup["predicate"], dup["id"])
            return dup["id"]

    row = await pool.fetchrow(
        """INSERT INTO facts (subject, predicate, value, source_memory_id, speaker_id, confidence, sensitive, pii_category, source)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
           ON CONFLICT (subject, predicate) WHERE superseded_by IS NULL
           DO UPDATE SET value = EXCLUDED.value,
                         learned_at = now(),
                         confidence = EXCLUDED.confidence,
                         source_memory_id = EXCLUDED.source_memory_id,
                         speaker_id = EXCLUDED.speaker_id,
                         sensitive = EXCLUDED.sensitive,
                         pii_category = EXCLUDED.pii_category,
                         source = EXCLUDED.source
           RETURNING id, (xmax = 0) AS inserted""",
        subject, predicate, value, source_memory_id, speaker_id, confidence,
        sensitive, pii_category, source,
    )
    new_id = row["id"]
    if row["inserted"]:
        log.info("Stored fact #%d: %s.%s = %s", new_id, subject, predicate, value)
    else:
        log.info("Updated fact #%d: %s.%s = %s", new_id, subject, predicate, value)
    return new_id


async def resolve_entity(name: str) -> str | None:
    """Resolve an entity reference. E.g., 'my wife' -> 'Erin'."""
    pool = await get_pool()
    name_lower = name.strip().lower()

    # Try exact subject match first
    row = await pool.fetchrow(
        """SELECT value FROM facts
           WHERE subject = $1 AND predicate IN ('is', 'name', 'is called')
           AND superseded_by IS NULL
           ORDER BY confidence DESC, learned_at DESC LIMIT 1""",
        name_lower,
    )
    if row:
        return row["value"]

    # Try trigram fuzzy match
    row = await pool.fetchrow(
        """SELECT value FROM facts
           WHERE subject % $1 AND predicate IN ('is', 'name', 'is called')
           AND superseded_by IS NULL
           ORDER BY similarity(subject, $1) DESC, confidence DESC LIMIT 1""",
        name_lower,
    )
    return row["value"] if row else None


async def get_facts_about(subject: str, limit: int = 10) -> list[Fact]:
    """Get all active facts about a subject."""
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT id, subject, predicate, value, learned_at, confidence, sensitive
           FROM facts
           WHERE (subject = $1 OR subject % $1)
           AND superseded_by IS NULL
           ORDER BY similarity(subject, $1) DESC, confidence DESC
           LIMIT $2""",
        subject.strip().lower(),
        limit,
    )
    return [Fact(**dict(r)) for r in rows]


async def get_all_facts_for_prompt(subjects: list[str], limit: int = 10) -> list[Fact]:
    """Get facts about multiple subjects for prompt injection."""
    all_facts = []
    seen = set()
    for subj in subjects:
        facts = await get_facts_about(subj, limit=5)
        for f in facts:
            if f.id not in seen:
                seen.add(f.id)
                all_facts.append(f)
                if len(all_facts) >= limit:
                    return all_facts
    return all_facts


# Generic first-person subjects the extractor has used to record speaker
# self-disclosure. Treat them as aliases of the canonical speaker name when
# retrieving ground-truths so the 2026-03-29 subject-normalization regression
# (which rerouted Dan's self-disclosures from subject='dan' to subject='user')
# doesn't keep ground-truths frozen at March data.
_SELF_REFERENCE_ALIASES = ("user", "i", "me")


async def get_speaker_id_by_name(name: str) -> int | None:
    """Resolve an enrolled speaker's row id from their canonical name.

    Used by the PARTY-2 face-trust path (conversation/turn.py): when the voice
    is unknown but a face is confidently recognized, fact retrieval keys on the
    face's name AND needs the face person's own speaker_id so id-tagged fact
    rows (not just NULL-speaker_id name matches) come back. Returns None for an
    unknown/retired/blank name — get_facts_about_speaker then falls back to the
    strict canonical-name match, so a miss degrades gracefully.
    """
    canon = (name or "").strip().lower()
    if not canon or canon.startswith("unknown"):
        return None
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id FROM speakers WHERE lower(name) = $1 AND retired_at IS NULL",
        canon,
    )
    return row["id"] if row is not None else None


async def get_facts_about_speaker(
    speaker_name: str,
    speaker_id: int | None,
    limit: int = 10,
) -> list[Fact]:
    """Retrieve facts authored by a given speaker across all the subjects the
    extractor may have written them under: the speaker's canonical name AND
    the generic self-reference aliases (user / i / me).

    speaker_id is the authoritative filter when populated (rows < ~2026-03
    have it NULL). For NULL-speaker_id rows we fall back to a strict
    canonical-name match so we do not surface other speakers' self-disclosure
    from the shared 'user'/'i' subject buckets.
    """
    pool = await get_pool()
    name = speaker_name.strip().lower()
    aliases = (name, *_SELF_REFERENCE_ALIASES)
    rows = await pool.fetch(
        """SELECT id, subject, predicate, value, learned_at, confidence, sensitive
           FROM facts
           WHERE subject = ANY($1::text[])
           AND superseded_by IS NULL
           AND (
               speaker_id = $2
               OR (speaker_id IS NULL AND subject = $3)
           )
           ORDER BY learned_at DESC, confidence DESC
           LIMIT $4""",
        list(aliases),
        speaker_id,
        name,
        limit,
    )
    return [Fact(**dict(r)) for r in rows]


# Sortable columns for the read-only Memory Inspector. Whitelisted to keep the
# ORDER BY clause off user input.
_INSPECT_FACT_SORT = {
    "learned_at": "learned_at DESC NULLS LAST",
    "confidence": "confidence DESC NULLS LAST, learned_at DESC",
    "subject": "subject ASC, predicate ASC",
}


async def list_facts(
    q: str | None = None,
    include_superseded: bool = False,
    sort: str = "learned_at",
    limit: int = 500,
) -> list[dict]:
    """Read-only listing for the Memory Inspector UI. Returns rich rows
    (provenance + speaker name + supersession state), NOT the lean prompt-facing
    Fact dataclass. Active-only by default (`superseded_by IS NULL`); pass
    include_superseded=True to see the full audit trail. `q` does a
    case-insensitive substring match across subject/predicate/value.
    """
    pool = await get_pool()
    order = _INSPECT_FACT_SORT.get(sort, _INSPECT_FACT_SORT["learned_at"])
    where = []
    params: list = []
    if not include_superseded:
        where.append("f.superseded_by IS NULL")
    if q and q.strip():
        params.append(f"%{q.strip()}%")
        where.append(
            f"(f.subject ILIKE ${len(params)} OR f.predicate ILIKE ${len(params)}"
            f" OR f.value ILIKE ${len(params)})"
        )
    params.append(int(limit))
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = await pool.fetch(
        f"""SELECT f.id, f.subject, f.predicate, f.value, f.learned_at,
                   f.confidence, f.sensitive, f.pii_category, f.source,
                   f.superseded_by, s.name AS speaker
            FROM facts f
            LEFT JOIN speakers s ON s.id = f.speaker_id
            {clause}
            ORDER BY {order}
            LIMIT ${len(params)}""",
        *params,
    )
    return [dict(r) for r in rows]


async def inspector_counts() -> dict:
    """Summary counts for the inspector header bar."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT
             (SELECT count(*) FROM facts) AS facts_total,
             (SELECT count(*) FROM facts WHERE superseded_by IS NULL) AS facts_active,
             (SELECT count(*) FROM facts WHERE superseded_by IS NULL AND sensitive) AS facts_sensitive,
             (SELECT count(*) FROM episodes) AS episodes,
             (SELECT count(*) FROM speakers) AS speakers"""
    )
    return dict(row)
