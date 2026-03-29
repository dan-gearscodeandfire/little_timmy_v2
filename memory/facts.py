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

    row = await pool.fetchrow(
        """INSERT INTO facts (subject, predicate, value, source_memory_id, speaker_id, confidence)
           VALUES ($1, $2, $3, $4, $5, $6)
           ON CONFLICT (subject, predicate) WHERE superseded_by IS NULL
           DO UPDATE SET value = EXCLUDED.value,
                         learned_at = now(),
                         confidence = EXCLUDED.confidence,
                         source_memory_id = EXCLUDED.source_memory_id,
                         speaker_id = EXCLUDED.speaker_id
           RETURNING id, (xmax = 0) AS inserted""",
        subject, predicate, value, source_memory_id, speaker_id, confidence,
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
        """SELECT id, subject, predicate, value, learned_at, confidence
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
