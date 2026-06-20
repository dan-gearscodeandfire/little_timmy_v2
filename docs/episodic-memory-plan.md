# Episodic + Semantic Memory — Multi-Session Implementation Plan

**Created:** 2026-06-20 · **Status:** proposed · **Owner:** Dan
**Goal:** Give Little Timmy timestamped, time-range-queryable episodic memory (built on the rollup summaries), keep `facts` as the durable semantic channel, and restore vector embedding — fixed — only once the corpus earns it.

---

## Decisions locked (Dan, 2026-06-20)

1. **Episodic memory = the rollup summaries** (coarse multi-turn spans), NOT per-exchange extracted episodes. The episodic write trigger is warm→cold eviction in `memory/rollup.py`.
2. **The rollup summarizer must be more specific.** Current prompt (`llm/client.py:327`) asks for "2-3 concise sentences" — too vague to query later. Rewrite it to preserve concrete who/what/when/numbers.
3. **`facts` table stays the semantic channel** — already deduped (`(subject,predicate)` upsert), recency-ordered (`confidence DESC, learned_at DESC`), PII/guest-gated. Untouched by this plan except as a link target.
4. **Vector embedding is deferred to the last phase** and only re-enabled after the two root-cause bugs are fixed (dedup-free writer + recency-blind ranker).

## Design model

| | Semantic (facts) | Episodic (episodes) |
|---|---|---|
| Answers | "What is true?" | "What happened, and when?" |
| Table | `facts` (exists) | **`episodes`** (new; replaces frozen `memories` for this role) |
| Key time field | `learned_at` | **`span_start` / `span_end`** (real turn timestamps) |
| Retrieval | recency lookup + (Phase 5) vector | **date-range overlap** + (Phase 5) vector |
| "both" overlap | a fact row… | …optionally `source_episode_id` FK to its event |

**Why a new `episodes` table, not reuse `memories`:** leaves the frozen `memories` tier legacy/untouched (no migration risk), and a clean table lets us add `span_*` + a nullable `embedding` column without overloading the old schema. (Open: confirm in Session 0 — alternative is to add columns to `memories` and filter by `type`.)

## Verified ground truth (code, 2026-06-20)

- Turns carry timestamps: `conversation/manager.py:122,126,150,171` (`timestamp=time.time()`) → span math is free.
- Summary prompt is inline: `llm/client.py:307 generate_summary()`, prompt at line 326-328.
- Cold eviction (episodic write point): `memory/rollup.py:86-95`, gated by `config.PERSIST_COLD_SUMMARIES` (currently False → dropped).
- Extraction `memories[]` branch: `extraction.py:300`, gated by `config.PERSIST_EXTRACTED_MEMORIES` (False). `facts` write still runs.
- Retrieval is unconditional: `conversation/turn.py:468` (`_gather` → `retrieve()`), every turn. No `recall_semantic` gate exists.
- Recency-blind fuse: `memory/retrieval.py:93 _fuse` (RRF + cosine, no time term).
- Router harness: `conversation/tool_router.py` (`store_fact` shipped `cfff000`), Qwen3-4B on `:8092`, gated by `runtime_toggles.classifier_enabled`. Two-tier GBNF.

## Cross-cutting conventions (every session)

- Branch per session; offline tests + **live validation** before calling it done.
- Guard every behavior change behind a **config flag or runtime toggle** (rollback = flip it).
- Deploy LT via `sudo -n systemctl restart little-timmy{,-os}.service` — POST :8894 restart is a no-op. Verify `started_at` moved.
- Router/structured calls run on `:8092` (separate from the `:8083` conversation slot) and **thinking OFF**.
- Update `~/little_timmy/CONTEXT.md` + cross-post a Zettel to Obsidian per session; update `MEMORY.md` pointer.

---

## Session 0 — Schema foundation (no behavior change) ✅ DONE 2026-06-20

**Goal:** `episodes` table exists; nothing reads/writes it yet.

**Implemented (2026-06-20):** Dedicated `episodes` table chosen (frozen `memories` left untouched). Added to `db/schema.sql` (idempotent, applied via `python -m db.migrate`): columns `id, span_start, span_end, created_at, text, token_count, embedding vector(768) NULL, source jsonb` + btree indexes `idx_episodes_span_start/_end`. FTS/trigram/HNSW deferred to S5. Config flags `PERSIST_EPISODES=False`, `RECALL_TEMPORAL_ENABLED=False` added (`config.py`). Verified: table created, 0 rows, migration idempotent on re-run. No runtime path touches it. NOT yet committed to git.

- Confirm table decision (new `episodes` vs columns-on-`memories`).
- Migration: create `episodes(id, span_start timestamptz, span_end timestamptz, created_at timestamptz default now(), text, token_count int, embedding vector NULL, source jsonb)`. B-tree indexes on `span_start`, `span_end`. Leave `embedding` nullable (filled Phase 5).
- Add config flags (all default-safe): `PERSIST_EPISODES=False`, `RECALL_TEMPORAL_ENABLED=False`.
- **Exit:** migration applies + reverts cleanly; no runtime path touches the table.

## Session 1 — Specific summaries + episodic dual-write ✅ DONE + LIVE 2026-06-20

**Goal:** episodes accumulate with real time spans and *specific* text; quarantined from semantic retrieval by construction (different table).

**Implemented (2026-06-20, branch `feat/episodic-memory-s1-summaries`, commits `dceac7c` + go-live `d8d6201`):** `generate_summary()` prompt rewritten (`llm/client.py`) — one-line topic header + ~4-6 sentences preserving proper nouns/dates/numbers/decisions/who-did-what; thinking OFF, temp 0.3. A/B on a synthetic transcript confirmed the new prompt retains every name/figure/date the old "2-3 sentences" prompt dropped. `WarmSummary` gained `span_start/span_end` (epoch secs), set from min/max turn timestamps in `maybe_rollup`. New `store_episode()` (`memory/manager.py`) inserts to `episodes` via `to_timestamp()`, embedding left NULL. Eviction block (`memory/rollup.py`) writes an evicted summary to `episodes` when `PERSIST_EPISODES`, independent of `PERSIST_COLD_SUMMARIES`; placeholder still never persisted; never written to `memories` (quarantined from `retrieve()`, which only reads `FROM memories`). Validated: store_episode roundtrip (540s span, NULL embedding, jsonb source) + real `maybe_rollup` eviction test (480s span, evicted-not-fresh text, memories count unchanged, placeholder skipped). **Live: `PERSIST_EPISODES=True` committed + both services restarted 16:54 EDT, flag confirmed loaded.**

- **Rewrite `generate_summary()` prompt** (`llm/client.py:326`) for specificity: preserve proper nouns, dates/times mentioned, numbers, decisions, and **who said/did what**; drop the "2-3 sentences" brevity cap (allow ~4-6); lead with a one-line topic. Keep thinking OFF, temp ~0.3. A/B a few real transcripts before/after.
- **Write episodes on cold eviction** (`memory/rollup.py:86`): when `PERSIST_EPISODES`, insert into `episodes` with `span_start/span_end` derived from the timestamps of the turns the evicted summary covers (thread the min/max turn `timestamp` through the rollup). Still NOT written to `memories`.
- **Exit:** trigger a real rollup live; verify an episode row with a sane span + specific text; confirm it does **not** appear in `retrieve()` output.

## Session 2 — Temporal resolver + date-range query (pure, offline-testable) ✅ DONE 2026-06-20

**Goal:** deterministic "phrase → date range → matching episodes", no router yet.

**Implemented (2026-06-20, branch `feat/episodic-memory-s1-summaries`):** `memory/temporal.py` — pure `resolve_date_range(phrase, now) -> (start, end) | None`, half-open windows, tz-aware (operates in `now`'s tz; caller passes local `datetime.now().astimezone()`). Covers: today / yesterday / day-before-yesterday / earlier-today; dayparts (this/yesterday morning·afternoon·evening, last night); bare + "last" weekdays; this/last week, this/last weekend, this/last month; "N days/weeks ago" (digit or word); fuzzy windows ("a couple/few weeks ago", "recently", "the last few days"); None for non-temporal input. `memory/manager.py:query_episodes_by_range(start, end, limit)` — overlap `span_start < end AND span_end >= start`, `ORDER BY span_start`, untruncated text. `tests/test_temporal_resolver.py` — 31 tests (resolver edge cases at a fixed Wed `now` + DB overlap query over seeded episodes, self-cleaning). All green; no conversation wiring (that's S3).

- `resolve_date_range(phrase, now)` — pure Python: "yesterday", "last Saturday", "last week", "this morning", "a couple weeks ago". Return `(start, end)` or None. Heavily unit-tested (this is the deterministic core that sidesteps the recency-blind ranker).
- Query: episodes whose `[span_start, span_end]` **overlaps** `[start, end]`, `ORDER BY span_start`. Cap + untruncated text (these don't go through the 200-char guillotine).
- **Exit:** unit tests for resolver edge cases + overlap query over seeded episodes. No conversation wiring.

## Session 3 — `recall_temporal` router tool (the original goal, realized) ✅ DONE + LIVE 2026-06-20

**Goal:** "what did we talk about last Saturday" returns a grounded answer.

**Implemented (2026-06-20, branch `feat/episodic-memory-s1-summaries`, commits `2747136` + go-live `<flag>`):** Tier-1 route grammar + `classify_route.txt` gained `recall_temporal` ("last week" example flipped from `none`); Tier-2 `recall_temporal_args.txt` extracts the time phrase. `tool_router.maybe_handle_tool_call` now returns `ToolOutcome(handled, recall_block)` instead of `bool` (both `main.py` call sites updated). `recall_temporal` is retrieval-AUGMENTATION, not terminal: `_resolve_recall_block()` runs `extract_recall_phrase` → `resolve_date_range` → `query_episodes_by_range` → formats a `[WHAT WE TALKED ABOUT]` block (untruncated episodes, or an honest "NO recorded summaries from <window>" marker on empty), and hands it back; the caller falls through to `_generate_response`. `recall_block` threads `_generate_response` → `TurnContext` → `build_ephemeral_block` (after vector memories, inside `[CONTEXT]`). Gated by `RECALL_TEMPORAL_ENABLED` (now **True**) + `classifier_enabled` (ON). Graceful fall-through on unresolved phrase / query error / parse failure.

**Validation:** 8 offline router tests + 31 S2 resolver tests green. Live routing battery on `:8092`: **15/15 intents, 4/4 phrases** (above the store_fact bar). Live end-to-end turns (services restarted 17:53 EDT): HIT ("yesterday" → 1 episode → grounded answer), EMPTY ("three days ago" → 0 episodes → honest "I have no memory of it", no confabulation), and fall-through ("back in March" → unresolved → normal pipeline). All `[TOOL recall_temporal]` log lines confirm correct ranges + episode counts.

**Known follow-up (not blocking):** `resolve_date_range` does NOT handle named-month phrases ("back in March", "in April") — they fall through harmlessly but don't recall. Add named-month + "in <month> <year>" support to `memory/temporal.py` as a quick enhancement.

- Add intent to `conversation/tool_router.py`: Tier-1 route gains `recall_temporal`; Tier-2 emits the date phrase. Reuse the `store_fact` GBNF harness.
- **Key difference from `store_fact`:** this is a *retrieval-augmentation* intent, not a terminal action — it does **not** early-return with an ACK. On hit: `resolve_date_range` → query episodes → inject as a dedicated untruncated `[WHAT WE TALKED ABOUT]` block → fall through to `_generate_response` so the brain answers from it. On empty range → graceful "I don't have anything from then."
- Gate: `classifier_enabled` + `RECALL_TEMPORAL_ENABLED`. Dead-port/parse failure → normal pipeline.
- **Exit:** live test several time phrases; routing accuracy ≥ store_fact bar; answers grounded in real episodes, not hallucinated.

## Session 4 — Gate the read path (`needs_retrieval`)

**Goal:** stop running vector `retrieve()` on every banter turn over the empty store.

- Add a `needs_retrieval` decision (router flag or cheap heuristic): banter → skip `retrieve()` at `conversation/turn.py:468`, inject only facts-about-speaker. Recall/question turns → retrieve.
- Independent of the episodic work; pure latency/token win (~202 tok + a DB round-trip per skipped turn).
- **Exit:** banter turns show `retrieval_ms≈0` / no memories block; recall turns unchanged.

## Session 5 — Restore vector embedding on episodes, *fixed* (scale phase)

**Goal:** semantic "find the time I mentioned something like X" — only once the corpus is big enough that embedding beats date-range + facts.

- Fill `episodes.embedding` at write. **Prerequisite fixes (or it re-rots):**
  - **Dedup at write:** content-hash or similarity threshold before insert (no blind INSERT).
  - **Recency in rank:** add decay to `_fuse` — `score = similarity × halflife_decay(now − span_end)`. `access_count` is already written and unused — free usage signal.
  - **Index:** pgvector HNSW (or IVFFlat) once row count warrants.
- Add `recall_semantic` router intent: vector + FTS over `episodes` (with decay). Distinct from `recall_temporal` (which stays pure date-range).
- **Exit:** A/B retrieval quality vs date-range-only; decay half-life tuned; index latency acceptable.

---

## Sequencing rationale

Sessions 0-3 deliver the thing Dan actually asked for — timestamped, time-range-queryable episodic recall — entirely via the **deterministic** date-range path that sidesteps every known retrieval bug. Vector restoration (the hard part) is deferred to Session 5, after the bugs are fixed and the dataset is large enough for embedding to pay off. Each session is independently shippable and toggle-guarded.
