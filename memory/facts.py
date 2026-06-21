"""Fact store: structured entity-attribute-value with provenance."""

import logging
from dataclasses import dataclass
from db.connection import get_pool

log = logging.getLogger(__name__)


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
) -> int:
    """Upsert a fact. If (subject, predicate) exists and isn't superseded, supersede it."""
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
    target_exists = await pool.fetchval(
        """SELECT 1 FROM facts
           WHERE subject = $1 AND predicate = $2 AND superseded_by IS NULL""",
        subject, predicate,
    )
    if not target_exists:
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
        """INSERT INTO facts (subject, predicate, value, source_memory_id, speaker_id, confidence, sensitive, pii_category)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
           ON CONFLICT (subject, predicate) WHERE superseded_by IS NULL
           DO UPDATE SET value = EXCLUDED.value,
                         learned_at = now(),
                         confidence = EXCLUDED.confidence,
                         source_memory_id = EXCLUDED.source_memory_id,
                         speaker_id = EXCLUDED.speaker_id,
                         sensitive = EXCLUDED.sensitive,
                         pii_category = EXCLUDED.pii_category
           RETURNING id, (xmax = 0) AS inserted""",
        subject, predicate, value, source_memory_id, speaker_id, confidence,
        sensitive, pii_category,
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
