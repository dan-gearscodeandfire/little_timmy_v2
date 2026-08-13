"""Fact store: structured entity-attribute-value with provenance."""

import logging
import re
from dataclasses import dataclass

import asyncpg

from db.connection import get_pool
import config

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

    # --- Write-side hygiene (2026-08-13) -------------------------------------
    # The Open Sauce audit found 30 facts written across three days with ZERO
    # superseded, including a clock reading kept as a durable fact
    # ("dan time -> 5.50 p.m.", confidence 0.25) and a 0.10-confidence
    # mishearing ("flynn high_school -> science work"). Under the old
    # recency-slice retrieval most junk was unreachable; relevance ranking
    # makes it reachable in exactly the wrong moment -- an ephemeral "time"
    # fact now surfaces precisely when someone asks the time, and is asserted
    # as GROUND TRUTH. Both gates apply to the EXTRACTOR only: an explicit
    # user-directed correction (source="tool") is the user's word and is never
    # second-guessed.
    if source == "extraction":
        if _EPHEMERAL_PRED_RE.search(predicate):
            log.info("[FACTS] rejected ephemeral predicate %s.%s (=%r) -- true "
                     "only at the moment it was said", subject, predicate, value[:40])
            return -1
        if confidence < config.FACT_MIN_WRITE_CONFIDENCE:
            log.info("[FACTS] rejected low-confidence extraction %s.%s (=%r, "
                     "conf %.2f < %.2f)", subject, predicate, value[:40],
                     confidence, config.FACT_MIN_WRITE_CONFIDENCE)
            return -1

    # Embedded at write time so the turn can rank facts by relevance to what was
    # actually asked (see get_relevant_facts_about_speaker). Failure is
    # non-fatal: a NULL embedding just makes the row invisible to the vector
    # path, still reachable by the recency path.
    _fact_embedding = None
    try:
        from memory.manager import embed as _embed
        _fact_embedding = await _embed(f"{subject} {predicate} {value}")
    except Exception:
        log.warning("[FACTS] embed failed for %s.%s; storing without", subject, predicate)

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

    async def _upsert():
        return await pool.fetchrow(
            """INSERT INTO facts (subject, predicate, value, source_memory_id, speaker_id, confidence, sensitive, pii_category, source, embedding)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
               ON CONFLICT (subject, predicate) WHERE superseded_by IS NULL
               DO UPDATE SET value = EXCLUDED.value,
                             learned_at = now(),
                             confidence = EXCLUDED.confidence,
                             source_memory_id = EXCLUDED.source_memory_id,
                             speaker_id = EXCLUDED.speaker_id,
                             sensitive = EXCLUDED.sensitive,
                             pii_category = EXCLUDED.pii_category,
                             source = EXCLUDED.source,
                             embedding = EXCLUDED.embedding
               RETURNING id, (xmax = 0) AS inserted""",
            subject, predicate, value, source_memory_id, speaker_id, confidence,
            sensitive, pii_category, source, _fact_embedding,
        )

    try:
        row = await _upsert()
    except asyncpg.ForeignKeyViolationError:
        # Same-turn race (review 7-15): a voice promotion's eager speakers-DB
        # flush is only SCHEDULED (create_task in assign_name), so a fact
        # FK-ing the brand-new speaker_id can reach Postgres first — and the
        # extraction worker's blanket except would drop the fact for good
        # (startup sync reconciles the row, not the lost fact). Reconcile
        # inline from the id-map (idempotent) and retry once; a genuinely
        # unknown speaker_id still raises to the caller.
        from db.speakers import sync_speakers_from_id_map
        log.warning("[FACT] speaker FK miss for speaker_id=%s (%s.%s) — "
                    "syncing speakers from id-map and retrying",
                    speaker_id, subject, predicate)
        await sync_speakers_from_id_map()
        row = await _upsert()
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


async def get_facts_about(subject: str, limit: int = 10,
                          speaker_id: int | None = None,
                          speaker_name: str | None = None) -> list[Fact]:
    """Get all active facts about a subject.

    speaker_id / speaker_name SCOPE a possessive lookup to the person who
    actually said it (2026-08-13). This path is reached from "my X" phrases via
    _extract_my_subjects, and it matched subjects by trigram similarity with NO
    speaker filter whatsoever -- so ANY guest asking about "my wife" retrieved
    `dan's wife -> Erin` (similarity 0.357, the only row in the corpus that
    matches). Flynn discussed his wife at the booth on 7-19; nothing would have
    stopped that leak. When a speaker is supplied we keep only rows that are
    either theirs by speaker_id or whose subject actually names them; passing
    neither preserves the old unscoped behaviour for callers that want it."""
    pool = await get_pool()
    subject = subject.strip().lower()
    rows = await pool.fetch(
        """SELECT id, subject, predicate, value, learned_at, confidence, sensitive,
                  speaker_id
           FROM facts
           WHERE (subject = $1 OR subject % $1)
           AND superseded_by IS NULL
           ORDER BY similarity(subject, $1) DESC, confidence DESC
           LIMIT $2""",
        subject,
        limit * 4 if (speaker_id or speaker_name) else limit,
    )
    out = []
    name = (speaker_name or "").strip().lower()
    for r in rows:
        d = dict(r)
        rid = d.pop("speaker_id", None)
        if speaker_id is not None or name:
            owned = (rid is not None and rid == speaker_id)
            # "dan's wife" names dan; "my wife" written under a bare possessive
            # with a matching speaker_id is covered by `owned` above.
            named = bool(name) and name in d["subject"]
            if not (owned or named):
                continue
        out.append(Fact(**d))
        if len(out) >= limit:
            break
    return out


async def get_all_facts_for_prompt(subjects: list[str], limit: int = 10,
                                   speaker_id: int | None = None,
                                   speaker_name: str | None = None) -> list[Fact]:
    """Get facts about multiple subjects for prompt injection.

    speaker_id/speaker_name scope possessive subjects to the speaker -- see
    get_facts_about for the cross-speaker leak this closes."""
    all_facts = []
    seen = set()
    for subj in subjects:
        facts = await get_facts_about(subj, limit=5, speaker_id=speaker_id,
                                      speaker_name=speaker_name)
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


# Predicates whose value is true only at the instant it was spoken. Storing one
# as a durable fact is a category error -- "dan time -> 5.50 p.m." was still
# being injected a month later. Matched on the normalized predicate.
_EPHEMERAL_PRED_RE = re.compile(
    r"^(?:current_|currently_)|"
    r"^(?:time|date|weather|today|now|right_now|current|mood|"
    r"current_location|current_activity|current_time|current_date)$",
    re.IGNORECASE,
)


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


async def embed_fact_text(subject: str, predicate: str, value: str) -> str:
    """The string a fact is embedded as. One place so the write path and any
    backfill cannot drift into embedding different text for the same row."""
    return f"{subject} {predicate} {value}"


async def get_relevant_facts_about_speaker(
    speaker_name: str,
    speaker_id: int | None,
    query: str,
    limit: int = 5,
) -> list[Fact]:
    """Facts about `speaker_name` ranked by relevance to `query`.

    Replaces the recency slice for prompt injection. get_facts_about_speaker is
    ORDER BY learned_at DESC with no query term, so out of 167 live facts the
    turn always received the same 5 newest ones -- during a conversation about
    Christopher Nolan those were "dan occupation documentary filmmaker / dan
    name Dan / dan typical_clothing_color black", carried under a
    never-contradict directive.

    An IDENTITY CORE is always included regardless of the query: a handful of
    predicates that are relevant to any turn because they are who the person
    is, not what they last mentioned. Everything else must earn its slot by
    similarity, and rows below the distance floor are dropped entirely -- an
    empty fact block is a valid, and often correct, outcome.

    Falls back to the recency ordering when nothing is embedded yet, so the
    behaviour degrades rather than going blank on an unbackfilled corpus."""
    import config
    from memory.manager import embed

    pool = await get_pool()
    name = speaker_name.strip().lower()
    aliases = (name, *_SELF_REFERENCE_ALIASES)

    core = await pool.fetch(
        """SELECT id, subject, predicate, value, learned_at, confidence, sensitive
           FROM facts
           WHERE subject = ANY($1::text[]) AND superseded_by IS NULL
             AND predicate = ANY($2::text[])
             AND (speaker_id = $3 OR (speaker_id IS NULL AND subject = $4))
           ORDER BY confidence DESC, learned_at DESC""",
        list(aliases), list(config.FACT_IDENTITY_CORE_PREDICATES), speaker_id, name,
    )

    q_emb = None
    try:
        q_emb = await embed(query)
    except Exception:
        log.warning("[FACTS] query embed failed; falling back to recency", exc_info=True)

    ranked = []
    if q_emb is not None:
        ranked = await pool.fetch(
            """SELECT id, subject, predicate, value, learned_at, confidence, sensitive,
                      embedding <=> $5 AS distance
               FROM facts
               WHERE subject = ANY($1::text[]) AND superseded_by IS NULL
                 AND (speaker_id = $2 OR (speaker_id IS NULL AND subject = $3))
                 AND embedding IS NOT NULL
                 AND embedding <=> $5 < $6
               ORDER BY embedding <=> $5
               LIMIT $4""",
            list(aliases), speaker_id, name, limit, q_emb,
            config.FACT_SEMANTIC_DISTANCE_MAX,
        )

    out, seen = [], set()
    for r in (*core, *ranked):
        d = dict(r)
        d.pop("distance", None)
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        out.append(Fact(**d))
        if len(out) >= limit:
            break

    if not out and not ranked:
        # Nothing embedded yet (or all beyond the floor) -> old behaviour.
        return await get_facts_about_speaker(speaker_name, speaker_id, limit=limit)
    return out


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


async def get_fact_row(fact_id: int) -> dict | None:
    """One rich inspector row (same shape as list_facts). Used by the fact
    editor to echo back what actually stored after a write — store_fact may
    have absorbed the write into a different row (cross-predicate dedup) or
    re-classified sensitivity, so the caller must not assume its input."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT f.id, f.subject, f.predicate, f.value, f.learned_at,
                  f.confidence, f.sensitive, f.pii_category, f.source,
                  f.superseded_by, s.name AS speaker
           FROM facts f
           LEFT JOIN speakers s ON s.id = f.speaker_id
           WHERE f.id = $1""",
        fact_id,
    )
    return dict(row) if row is not None else None


async def delete_fact(fact_id: int) -> bool:
    """Hard-delete one fact (Memory Inspector editor). Clears inbound
    superseded_by references first (self-FK) — same unlink-then-delete shape
    as the persona-retire purge in presence/identity_commit.py. Returns True
    if a row was actually deleted."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE facts SET superseded_by = NULL WHERE superseded_by = $1",
                fact_id,
            )
            res = await conn.execute("DELETE FROM facts WHERE id = $1", fact_id)
    deleted = res.endswith("1")
    if deleted:
        log.info("Deleted fact #%d via inspector editor", fact_id)
    return deleted
