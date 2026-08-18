"""Central configuration for Little Timmy voice assistant."""

import os

# --- Network ---
WHISPER_URL = os.getenv("TIMMY_WHISPER_URL", "http://localhost:8891")
# L5 2026-05-14: TIMMY_CONVERSATION_URL is the canonical name; the
# old TIMMY_LLM_URL is kept as a fallback for operators who have it set
# in their shell rc / systemd override file already.
# 2026-06-22 (Dan): default flipped :8081 -> :8083. The Llama-3.2-3B on :8081 is
# ceased/disabled (no boot, no consumers); the qwen36 brain (:8083) is the live
# conversation tier, normally pinned via runtime_toggles "conversation_url_override".
# This default is the empty-override fallback -- it MUST point at a live server, or a
# cleared toggle would route conversation to the dead :8081 and silence Timmy.
LLM_CONVERSATION_URL = os.getenv("TIMMY_CONVERSATION_URL", os.getenv("TIMMY_LLM_URL", "http://localhost:8083"))
# L5 2026-05-14: TIMMY_MEMORY_URL is the canonical name (drops the
# redundant "LLM" to match TIMMY_CONVERSATION_URL / TIMMY_VISION_URL).
# Old TIMMY_MEMORY_LLM_URL kept as fallback.
# 2026-06-20 (Dan): memory/extraction now defaults to the VISION server
# (:8084 via TIMMY_VISION_URL), NOT the conversation brain (:8083). Both run
# the same Qwen3.6-35B-A3B model, but :8084 has its OWN KV cache (it carries
# the vision mmproj). Co-locating extraction with vision instead of
# conversation means background extraction + rollup can never evict the
# conversation prefix on :8083 — killing both the mid-turn contention hang AND
# the post-lull cold-reprefill spike (2026-06-20). Extraction yields to vision
# on :8084 via the vision-priority gate in llm/client.py. Set TIMMY_MEMORY_URL
# to pin a dedicated memory server instead.
LLM_MEMORY_URL = os.getenv("TIMMY_MEMORY_URL", os.getenv("TIMMY_MEMORY_LLM_URL", os.getenv("TIMMY_VISION_URL", "http://localhost:8084")))
LLM_BRAIN_MODEL = os.getenv("TIMMY_BRAIN_MODEL", "qwen3.6")
# First-pass tool-call classifier (Qwen3-4B-Q4 on its OWN llama-server, 2026-06-18).
# Separate server with its own KV cache so routing prompts never evict the :8083
# brain prefix. See Obsidian lt-tool-call-router-qwen3-4b-benchmark-2026-06-18.
LLM_CLASSIFIER_URL = os.getenv("TIMMY_CLASSIFIER_URL", "http://localhost:8092")
# Query-side coreference resolver (Qwen3-4B-Q4 on its OWN llama-server :8093, 2026-06-22).
# Was co-located on the classifier (:8092), but the two distinct prompt prefixes
# (static tool-call prompt vs. resolve prompt) ping-ponged on the single -np 1 slot
# and mutually evicted each other's KV cache, forcing a full re-prefill of the
# ~1K-token tool-call prompt every deixis turn. Dedicated server keeps each prefix
# warm. A/B isolates the dedicated-slot effect (same model as :8092).
LLM_RESOLVE_URL = os.getenv("TIMMY_RESOLVE_URL", "http://localhost:8093")
OLLAMA_URL = os.getenv("TIMMY_OLLAMA_URL", "http://localhost:11434")
WEB_HOST = "0.0.0.0"
WEB_PORT = 8893

# --- Models ---
EMBEDDING_MODEL = "nomic-embed-text"
PIPER_MODEL = os.getenv(
    "TIMMY_PIPER_MODEL",
    os.path.expanduser("~/little_timmy/models/tts/skeletor_v1.onnx"),
)

# --- Database ---
DB_DSN = os.getenv("TIMMY_DB_DSN", "postgresql://gearscodeandfire@localhost/little_timmy")

# --- TTS ---
TTS_LENGTH_SCALE = 0.6  # speech speed (lower = faster, default 1.0)

# Whole-word pronunciation overrides applied to text just before Piper
# synthesis. Piper phonemizes via espeak-ng, which mangles some names (live
# 2026-07-16: "Erin" came out "Karen" twice at the booth). Each key is respelled
# to a form espeak phonemizes correctly. Matching is WHOLE-WORD and
# case-insensitive -- whole-word is load-bearing, "erin" is a substring of
# "gathering"/"engineering" (the same trap that fooled the handoff's fact-CSV
# query). Applied at synth time, so edits take effect on the next utterance with
# NO re-render; audibly confirm/tune a new entry once, then leave it.
TTS_PRONUNCIATIONS = {
    "Erin": "Airin",
}

# --- Audio ---
SAMPLE_RATE = 16000
CHUNK_FRAMES = 4096  # ~256ms at 16kHz
# Silero speech-probability floor. 0.4 -> 0.30 on 2026-08-13 (Dan): measured
# idle room noise tops out at VAD 0.020 (p95, n=1034) while close speech reads
# ~1.0, so 0.4 was far more conservative than the noise required. This is a
# SEED -- the live value is FrameCapture.vad_threshold, settable at runtime
# (POST :8893/api/capture/vad_threshold) and persisted in the
# capture_vad_threshold runtime toggle, which until 2026-08-13 was declared but
# never read by anything.
VAD_THRESHOLD = float(os.getenv("TIMMY_VAD_THRESHOLD", "0.30"))
PRE_SPEECH_CHUNKS = 3  # ~768ms of audio kept before speech onset

# --- Hybrid Endpointing ---
# Chunks are 256ms (CHUNK_FRAMES=4096 @ 16kHz) and the capture loop counts
# silence once per chunk — size these in REAL chunk units, not assumed-100ms
# ones. Every prior value (3/15 at birth, 5/25 from 714f0ab) was converted at
# the wrong chunk duration, so the actual waits ran ~2.6-4x the commented
# intent (1.28s/6.4s until 2026-07-12). This wait IS the booth display's
# "VADbreak" segment (endpoint_ms in the [PERF] log line).
# 2026-07-12 (Dan): restore the original design intent — ~0.5s after a
# complete-looking sentence; mid-sentence grace lengthened same night after
# live test (1.54s clipped a deliberate "however..." thinking pause). With
# fresh-partial gating, the timeout only applies to genuinely mid-thought
# pauses — complete sentences exit at the fast path — so a longer timeout no
# longer slows normal turns.
SILENCE_CHUNKS_COMPLETE = 2     # 2 x 256ms = 0.51s — finalize if sentence looks complete
SILENCE_CHUNKS_INCOMPLETE = 10  # 10 x 256ms = 2.56s — timeout finalize if mid-sentence
# Floor for BOTH thresholds while Timmy awaits an answer to a question he
# just asked (main._dialog_owns_turn -> capture.set_reply_window_fn): a
# hesitant attendee answering "My name is ... Tushar" must not be clipped
# by the 0.51s fast path (Dan 2026-07-15, Open Sauce enroll flow).
SILENCE_CHUNKS_REPLY_WINDOW = 8  # 8 x 256ms = 2.05s

# --- Conversation ---
HOT_MAX_TOKENS = 2500          # token budget for hot tier
WARM_MAX_SUMMARIES = 3         # max warm summaries in prompt
ROLLUP_AGE_SECONDS = 1800      # 30 min — trigger rollup for old turns (was 600; bumped for Qwen 3.6 KV-cache reuse)
ROLLUP_IDLE_DELAY_SECONDS = 20 # wait this long after last turn before firing rollup; prevents priority-gate starvation when conversation is active
HOT_HARD_CEILING_TOKENS = 4000 # backstop ceiling (~1.6x HOT_MAX): when a rapid burst starves the idle rollup, drop oldest half synchronously (non-LLM placeholder). Bounds turn-DEPTH attention dilution that grows even while well under ctx (handoff 2026-06-10).
HARD_CEILING_PLACEHOLDER = "[earlier turns omitted under load]"  # marker WarmSummary text written by the backstop; matched verbatim to skip cold-storage persistence
PERSIST_COLD_SUMMARIES = False # 2026-06-18 (Dan): do NOT auto-embed stale warm rollups into the memories table. Rollups still summarize live hot/warm context for the in-prompt session; the OLDEST warm summary is simply dropped on overflow instead of persisted. Rationale: rollup summaries are low-density interaction gist that polluted retrieval (75% party noise, cleaned 69->16 on 2026-06-18) and the writer is dedup-free + the ranker is recency-blind. Set True to restore cold-storage persistence. See Obsidian lt-conversation-summary-cleanup-policy-2026-06-18.
PERSIST_EXTRACTED_MEMORIES = False # 2026-06-18 (Dan): do NOT auto-create vectorized episodic/semantic memories from the extraction pipeline. The two-pass extractor (classify_durable -> extract_facts) STILL runs and STILL writes structured rows to the `facts` table (subject/predicate/value, deduped) -- only the `memories[]` -> store_memory (embedded) branch is suppressed. Rationale: with rollup persistence already off, this makes the structured `facts` table the sole auto-writer of durable memory; the vectorized `memories` tier is a low-frequency, low-signal byproduct (0 memories produced in the 24h before this change; last episodic 6-13, last semantic 6-11). Set True to restore extracted-memory persistence.

# --- Episodic memory (Session 0, 2026-06-20; docs/episodic-memory-plan.md) ---
# Both flags default OFF: Session 0 only lays the `episodes` table schema; no
# runtime path reads or writes it yet. They are armed in later sessions.
PERSIST_EPISODES = True         # S1 LIVE 2026-06-20 (Dan): write rollup summaries to `episodes` (with real turn-timestamp spans) on warm->cold eviction. Independent of PERSIST_COLD_SUMMARIES (which targets the legacy `memories` tier). Rollback = set False + restart.
RECALL_TEMPORAL_ENABLED = True   # S3 LIVE 2026-06-20 (Dan): enable the `recall_temporal` router intent (date-range query over `episodes`). Also requires classifier_enabled (ON). Rollback = set False + restart.

# --- Session 5 (2026-06-20): vector restore on episodes (scale phase) --------
# All DEFAULT OFF — built + tested, but corpus-gated: episodes only began
# accumulating 6-20 and the A/B/half-life tuning needs a real corpus. Arming
# any of these is Dan's call once episodes exist. The dedup CONTENT-HASH floor
# (memory/manager.store_episode) is ALWAYS on (no flag) — it must guard the very
# first writes; it needs no embedding and never drops a distinct episode.
EMBED_EPISODES = True            # 2026-06-24 (Dan): vector-embed each SUBSTANTIVE warm rollup at write (write-through at mint, memory/rollup.py) so episodes are cosine-recallable. Banter is gated out upstream (usefulness verdict). recall_semantic stays OFF until a corpus accumulates + the sim-dedup threshold/decay half-life are tuned on it (Phase 2). Backfill any NULL-embedding rows: ops/backfill_episode_embeddings.py.
RECALL_SEMANTIC_ENABLED = True   # 2026-06-30 (Dan): ENABLED. `recall_semantic` router intent: vector+FTS+trigram over episodes, recency-DECAYED. Corpus precondition met (51/54 episodes embedded). Running on default 30d half-life (EPISODE_DECAY_HALFLIFE_S) — Phase-2 "tuning" is now observe-and-adjust on live hits. Switches the classifier to the 4-class semantic route grammar/prompt. Sim-dedup (EPISODE_DEDUP_SIM_ENABLED) is a separate write-side layer, independently off. Rollback = set False + restart.
# Recency decay (memory/decay.py) for the episode semantic rank: a fresh episode
# outranks a stale one of equal similarity. Half-life 30d = recent-favored but
# old still retrievable (90d -> 0.125 weight). Tunable once there's data.
EPISODE_DECAY_HALFLIFE_S = float(os.getenv("TIMMY_EPISODE_DECAY_HALFLIFE_S", str(30 * 24 * 3600)))
EPISODE_ACCESS_BOOST = float(os.getenv("TIMMY_EPISODE_ACCESS_BOOST", "0.05"))  # mild saturating log-boost from the free access_count signal
# 2026-08-13 (Dan): BOUND both decay terms so neither out-swings relevance.
# Measured on the live 1,239-prop corpus: unfloored recency spanned 4.2x against
# a 3x relevance range, so decay was the primary sort key and evicted a
# top-5-by-relevance claim on 12 of 14 replayed utterances. The floor compresses
# the decay curve into [floor, 1.0] without changing its shape. 0.0 == the old
# unbounded behaviour (A/B control). The access cap stops the self-reinforcing
# "whatever won last turn wins again" loop. Both are env-overridable.
RECENCY_WEIGHT_FLOOR = float(os.getenv("TIMMY_RECENCY_WEIGHT_FLOOR", "0.85"))
EPISODE_ACCESS_BOOST_MAX = float(os.getenv("TIMMY_EPISODE_ACCESS_BOOST_MAX", "1.10"))
EPISODE_SEMANTIC_TOP_K = int(os.getenv("TIMMY_EPISODE_SEMANTIC_TOP_K", "5"))
# 2026-06-28 (Dan, approved): make the ALWAYS-ON per-turn retrieval channel
# (the prompt's "Relevant memories:" block — also mirrored on the booth
# RETRIEVED MEMORIES panel) read the LIVE `episodes` table (recency-decayed via
# memory.decay) instead of the FROZEN `memories` tier. The `memories` table has
# had no writes since the 6-18 kill-switches (PERSIST_EXTRACTED_MEMORIES /
# PERSIST_COLD_SUMMARIES = False) and retrieve() has no recency term, so the
# channel surfaced March/June-11 memories forever — into both the panel AND
# Timmy's actual reply context. This repoints it at the warm-rollup episodes
# (EMBED_EPISODES write-through). recall_temporal/recall_semantic router intents
# are unaffected. Default ON = this is the fix; set "0" to roll back.
EPISODIC_ALWAYS_ON_RETRIEVAL = os.getenv("TIMMY_EPISODIC_ALWAYS_ON", "1") == "1"
# Optional near-dupe dedup LAYER on top of the content-hash floor (catches
# re-summaries that aren't byte-identical). OFF: needs embeddings + a threshold
# that can only be calibrated against a real corpus, and a mistuned threshold
# DROPS distinct episodes — so it stays off until there's data. Cosine-distance.
EPISODE_DEDUP_SIM_ENABLED = False
EPISODE_DEDUP_SIM_MAX_DIST = float(os.getenv("TIMMY_EPISODE_DEDUP_SIM_MAX_DIST", "0.05"))

# --- Privacy / PII gating (2026-06-18, Dan) ---
# Facts are classified for sensitivity at creation (memory/pii.py, called from
# memory.facts.store_fact). When the guest/privacy gate is active
# (runtime_toggles 'guest_mode', + presence-auto later), sensitive facts are
# dropped from prompt injection so Timmy can't speak them via TTS near guests.
# DAUGHTER_NAMES: Dan's minor children -> any fact naming them is sensitive.
# Names are kept OUT of source for privacy -> loaded at import from a gitignored
# local file (data/daughter_names.json: a JSON list of strings; include any
# DB-misspelling variants so existing rows still match). Missing file -> () and
# name-based gating no-ops (memory/pii.py tolerates an empty tuple); the
# "daughter/child" keyword gate still applies regardless.
def _load_daughter_names():
    import json
    from pathlib import Path
    try:
        _p = Path(__file__).resolve().parent / "data" / "daughter_names.json"
        _names = json.loads(_p.read_text())
        return tuple(str(n) for n in _names) if isinstance(_names, list) else ()
    except Exception:
        return ()


DAUGHTER_NAMES = _load_daughter_names()


# REDACT_TERMS: terms that must NEVER be persisted in any memory (Dan's last
# name, etc.). Kept OUT of source (gitignored data/redact_terms.json, a JSON
# list of strings). Any fact whose subject/predicate/value contains one of
# these (case-insensitive, word-boundary) is dropped at the store_fact
# chokepoint; episode text is scrubbed. Missing file -> () (no-op).
def _load_redact_terms():
    import json
    from pathlib import Path
    try:
        _p = Path(__file__).resolve().parent / "data" / "redact_terms.json"
        _terms = json.loads(_p.read_text())
        return tuple(str(t) for t in _terms if str(t).strip()) if isinstance(_terms, list) else ()
    except Exception:
        return ()


REDACT_TERMS = _load_redact_terms()

# --- Memory extraction queue (2026-06-03) ---
# Per-exchange fact/memory extraction is fire-and-forget but shares the single
# Qwen :8083 slot with conversation. The conversation-priority gate cancels an
# in-flight extraction whenever the user speaks again, and the old single-flight
# guard dropped any exchange that arrived mid-extraction -- so during lively
# chat, turns' facts could go unpersisted. Extraction is now a bounded FIFO
# queue drained one-at-a-time; a cancelled extraction is re-enqueued (it parks
# on the existing idle-gate until the conversation lulls) rather than lost.
EXTRACTION_QUEUE_MAX = 32      # bounded pending-exchange backlog; oldest dropped (with WARN) past this
EXTRACTION_MAX_RETRIES = 5     # re-enqueue a cancelled extraction up to this many times, then drop (WARN)

# --- Debounce + coalesce (2026-06-06, cancel-churn structural fix) ---
# The bounded queue above stopped DROPPING exchanges, but during a lively burst
# it still STARTED a fresh extraction every turn -- each one cancelled client-side
# the instant the user spoke again. The priority gate's task.cancel() only drops
# the httpx connection; llama.cpp keeps computing the abandoned generation
# server-side. Over a burst those abandoned-but-still-running generations stack
# under the live conversation gens -> concurrent Vulkan compute on the single-slot
# (-np 1) Strix Halo brain -> amdgpu hard-wedge (okDemerzel freeze 2026-05-12,
# 2026-06-06). Fix: don't START extraction during the burst at all. Buffer each
# turn and debounce; only after the conversation has been quiet for
# EXTRACTION_DEBOUNCE_SECONDS do we drain the buffer, coalesce it (grouped by
# speaker) into ONE classifier+extraction pass, and run it -- at which point the
# idle-gate passes instantly and nothing gets cancelled. EXTRACTION_MAX_HOLD_SECONDS
# is the ceiling so an unbroken monologue still flushes instead of deferring facts
# forever (and pinning the buffer). See project_okdemerzel_hang_2026-05-12 +
# Obsidian okdemerzel-freeze-rca-extraction-cancel-churn-2026-06-06.
EXTRACTION_DEBOUNCE_SECONDS = 8.0   # quiet gap after the last turn before extraction fires; each new turn resets it
EXTRACTION_MAX_HOLD_SECONDS = 90.0  # flush anyway after this much continuous chatter, debounce notwithstanding

# STT value-confidence gate (2026-06-21). When the classifier routes a
# store_fact, we score the acoustic confidence of the VALUE word(s) against
# whisper's per-word probabilities. Below this threshold the fact is stored but
# tagged low-confidence: the ACK reads the value back for confirmation and
# recall hedges instead of asserting it ("GROUND TRUTH" -> "HEARD BUT
# UNCONFIRMED"). Trades a rare extra confirm turn for never committing a
# misheard name as a confident fact (TRUE > AMBIGUITY > FALSE).
#
# TUNED on 119 live acoustic stores (92 correct / 27 mishear, 2026-06-21). The
# cost-min (FALSE weighted >=5x nag) is 0.75; 0.72 is the efficient-frontier
# knee -- catches 85% of mishears (23/27) at <30% read-back rate, just below
# where nagging climbs steeply (0.75->0.80 doubles nag for one more catch). The
# 4 mishears that still slip are confident near-homophones (Praxx->Prax 0.85,
# Wren->Ren 0.81) that NO threshold catches without exploding nag -- they need
# read-back-always-on-novel-noun (v2). Raise toward 0.75 to trade snappiness for
# fewer FALSE; lower toward 0.65 for fewer read-backs.
STT_VALUE_CONFIDENCE_THRESHOLD = 0.72

# read-back-always-on-novel-noun (v2, 2026-06-21): the confident-homophone
# bypass above (Praxx->Prax 0.804, Thorne->Thorn 0.721 both stored VERIFIED in
# the 6-21 acoustic battery) is unreachable by any threshold -- the correct
# value Onyx scored 0.683, below both wrong ones. Fix: when the value is a
# name/proper noun (acoustically unverifiable), read it back regardless of
# value-confidence so the user can correct in the moment. Names get a one-time
# "got it -- X?"; common-word values stay breezy. Only fires on acoustic input
# (typed facts have nothing to mishear). Set False to revert to vconf-only.
READBACK_PROPER_NOUNS = True

# Query-side mishear guard (2026-06-22): the gate above protects WRITES; a
# misheard word in a QUESTION corrupts RETRIEVAL ("what's my mail"->"my male").
# When a CONTENT word in the user's utterance is heard below this threshold,
# tell the brain to confirm what was asked instead of answering wrong or denying
# knowledge. Deliberately LOWER than the store threshold (0.72): a query-side
# false alarm interrupts EVERY modest-confidence question (more frequent, more
# annoying) than a one-time store read-back, and real mishears cluster <0.55
# (Brent .27, Renz .52, Thorn .37). Raise toward 0.65 to catch more; 0 disables.
STT_QUERY_CONFIDENCE_THRESHOLD = 0.55

# --- Propositions (2026-08-13) ---
# Atomic single-claim rows split out of each episode summary, embedded
# individually. Fixes the dilution at the root: one episode = one 768-d vector
# covering ~16 minutes and a dozen topics, so similarity to any specific
# question is meaningless (measured: no scalar threshold separates relevant
# from banter on episode vectors -- see memory/propositions.py header).
# WRITE path is on by default: it is purely additive (new table, episodes
# untouched) and a missing proposition set degrades to episode-tier retrieval.
# READ path is the live switch -- flip TIMMY_PROPOSITION_RETRIEVAL after the
# corpus is backfilled (ops/backfill_propositions.py) and measured.
PROPOSITION_WRITE_ENABLED = os.getenv("TIMMY_PROPOSITION_WRITE", "1") == "1"
PROPOSITION_RETRIEVAL_ENABLED = os.getenv("TIMMY_PROPOSITION_RETRIEVAL", "1") == "1"
PROPOSITION_MAX_PER_EPISODE = int(os.getenv("TIMMY_PROPOSITION_MAX", "8"))
PROPOSITION_TOP_K = int(os.getenv("TIMMY_PROPOSITION_TOP_K", "5"))
# Length sanity band for a single claim. Below the floor it is a fragment
# ("Dan agreed."); above the ceiling the model ignored "one claim per line"
# and emitted a paragraph -- which is the dilution being fixed.
PROPOSITION_MIN_CHARS = int(os.getenv("TIMMY_PROPOSITION_MIN_CHARS", "20"))
PROPOSITION_MAX_CHARS = int(os.getenv("TIMMY_PROPOSITION_MAX_CHARS", "300"))
# TIER-SPECIFIC fusion weights. Short documents want DIFFERENT weights than long
# ones, measured 2026-08-13: a 61-char claim has no surrounding content to dilute
# a spurious match, so the lexical channels over-fire on function words -- "what's
# your favorite Radiohead album?" returned "Timmy's favorite movie" and "favorite
# color" above the actual Radiohead claim. Trigram is the worst offender on short
# text, hence 0.5 here vs 1.5 on the episode tier.
# Sweep result (n=22, vs 0.837 at the episode-tier weights): MRR 0.909.
# NOTE the intuitive fix -- raise w_semantic because the claims are short -- was
# tested and is WRONG: 2.0/3.0/4.0 all scored worse. Don't retry it.
PROPOSITION_RRF_W_SEMANTIC = float(os.getenv("TIMMY_PROP_RRF_W_SEMANTIC", "1.0"))
PROPOSITION_RRF_W_FTS = float(os.getenv("TIMMY_PROP_RRF_W_FTS", "1.0"))
PROPOSITION_RRF_W_TRIGRAM = float(os.getenv("TIMMY_PROP_RRF_W_TRIGRAM", "0.5"))
# One claim per parent episode in the returned set. Without it a single chatty
# episode can spend all 5 slots on its own claims; measured +0.019 MRR and it
# widens the range of episodes the prompt sees.
PROPOSITION_DEDUPE_BY_EPISODE = os.getenv("TIMMY_PROPOSITION_DEDUPE_EPISODE", "1") == "1"

# --- Retrieval ---
# Top-K 5->3 (2026-08-18, prefill program final lever): ~60 tok less per
# memory turn. Measured cost on the corpus-generated probe set: recall@5 94%
# -> recall@3 88%. Env override is the rollback: TIMMY_RETRIEVAL_TOP_K=5.
RETRIEVAL_TOP_K = int(os.getenv("TIMMY_RETRIEVAL_TOP_K", "3"))
RETRIEVAL_CANDIDATES = 20      # candidates per search path before reranking

# Weighted RRF fusion (2026-06-02, REBALANCED 2026-08-13). Each channel's
# contribution stays the scale-free RRF term weight * 1/(k+rank+1), so
# robustness is preserved -- the weights only rebalance how loudly each votes.
#
# The 2026-06-02 weights (semantic 2.0 / fts 1.0 / trigram 0.5) were tuned
# while BOTH lexical channels were structurally broken: plainto_tsquery ANDed
# every term (zero rows on 5 of 10 real questions) and whole-document `%`
# could never clear its 0.3 threshold (zero rows, always). So semantic was
# weighted up to compensate for channels that were contributing nothing -- a
# fix aimed at the symptom. With the channels repaired (memory/episodic_search
# _fts/_trigram) that compensation actively hurts: an exact lexical match could
# not outrank a mediocre semantic hit, because semantic at rank 9 (2.0/70)
# still beat FTS at rank 1 (1.0/61).
#
# Re-tuned against a hand-labelled 11-query Open Sauce eval set (see
# Areas/lt-retrieval-channel-repair-2026-08-13):
#   before (broken channels, old weights): recall@5 0.73, MRR 0.561
#   after  (repaired channels, these):     recall@5 1.00, MRR 0.932
# Weight tuning is on a small set -- all four are env-overridable, and
# reverting the four env vars restores the previous behaviour exactly.
# A/B CONTROL: set all three weights to 1.0 and RRF_COSINE_BONUS to 0.0 to
# reproduce the original equal-weight, rank-only behavior exactly.
RRF_K = int(os.getenv("TIMMY_RRF_K", "30"))
RRF_W_SEMANTIC = float(os.getenv("TIMMY_RRF_W_SEMANTIC", "1.0"))
RRF_W_FTS = float(os.getenv("TIMMY_RRF_W_FTS", "1.5"))
RRF_W_TRIGRAM = float(os.getenv("TIMMY_RRF_W_TRIGRAM", "1.5"))
# Minimum word_similarity for the trigram channel to accept a row. Replaces the
# pg_trgm `%` operator's whole-document 0.3 threshold, which no real query could
# ever reach. 0.35 measured best on the Open Sauce eval set; lower widens the
# channel (more STT-mangled proper nouns caught, more noise).
TRIGRAM_WORD_SIM_FLOOR = float(os.getenv("TIMMY_TRIGRAM_WORD_SIM_FLOOR", "0.35"))
# Additive semantic-distance fold-in. The cosine distance (already used as the
# <SEMANTIC_DISTANCE_MAX floor in memory/retrieval) is normalized to a (0,1]
# bonus within the kept band so a 0.25-distance hit outranks a 0.49 one
# instead of tying. Sized at ~one RRF rank-step (1/61 at k=60) so it acts as a
# tiebreaker, not a hammer. Set to 0.0 to disable the fold-in.
RRF_COSINE_BONUS = float(os.getenv("TIMMY_RRF_COSINE_BONUS", "0.02"))

# Coreference / context-aware retrieval query (2026-06-02). The SEMANTIC
# channel's query is prefixed with the last few conversation turns so
# elliptical follow-ups ("what about her?") embed near the antecedent. The
# FTS/trigram channels keep the bare current utterance (prior-turn tokens add
# keyword noise). Storage and the conversation prompt are unaffected -- this
# only shapes the embedding query.
# A/B CONTROL: set TIMMY_COREFERENCE_ENABLED=false to revert to bare-utterance
# embedding.
COREFERENCE_ENABLED = os.getenv("TIMMY_COREFERENCE_ENABLED", "true").lower() == "true"
CONTEXT_TURNS = int(os.getenv("TIMMY_CONTEXT_TURNS", "2"))        # prior turns blended into the semantic query
# Coref resolver sees a WIDER window than the embedding blend: a proper-noun
# antecedent (Voss, Erin) routinely scrolls past 2 turns behind banter /
# corrections, leaving the pronoun unbindable. The dedicated :8093 server has the
# prefill headroom for this; the blend stays at CONTEXT_TURNS (anti-dilution).
RESOLVE_CONTEXT_TURNS = int(os.getenv("TIMMY_RESOLVE_CONTEXT_TURNS", "6"))  # prior turns fed to the coref resolver
CONTEXT_TURN_CHAR_CAP = int(os.getenv("TIMMY_CONTEXT_TURN_CHAR_CAP", "200"))  # per prior-turn char cap (anti-dilution)
# The :8093 resolver is decode-bound -- it regenerates ~the utterance, so cost
# scales with utterance length (a 25-40 word banter turn decodes toward the
# 64-token cap -> 500-800ms). It only earns that cost on SHORT, query-like
# elliptical follow-ups ("what does he do?", "remind me about her"); a long
# declarative that merely *contains* a pronoun gains nothing over the embedding
# blend, which already carries its lexical signal. Gate to <= this many words
# (the deixis + query-like check in memory/retrieval._needs_resolution does the
# rest). Skipped turns fall back to the blend -- same fail-safe contract as a
# resolver miss. Tune on the LIVE mic, not clean wavs.
RESOLVE_MAX_WORDS = int(os.getenv("TIMMY_RESOLVE_MAX_WORDS", "16"))  # skip resolving utterances longer than this

# --- Proactive (unprompted) speech (2026-06-03) ---
# Hard master kill-switch for Timmy reacting verbally to a high-urgency visual
# event (e.g. someone entering) without being addressed first. Defaults to
# ALLOW; the LIVE on/off is the `proactive_speech_enabled` runtime toggle (the
# LT-OS dashboard switch), which defaults OFF -- so the effective default is
# silent. Set TIMMY_PROACTIVE_SPEECH_ENABLED=false to forbid the feature
# entirely regardless of the dashboard. BOTH must be true to speak. See
# maybe_speak_proactively().
PROACTIVE_SPEECH_ENABLED = os.getenv("TIMMY_PROACTIVE_SPEECH_ENABLED", "true").lower() == "true"
PROACTIVE_URGENCY_THRESHOLD = float(os.getenv("TIMMY_PROACTIVE_URGENCY_THRESHOLD", "0.8"))  # mirrors relevance.SPEAK_THRESHOLD
PROACTIVE_COOLDOWN_SEC = float(os.getenv("TIMMY_PROACTIVE_COOLDOWN_SEC", "120.0"))  # min seconds between remarks
PROACTIVE_MAX_PER_MIN = int(os.getenv("TIMMY_PROACTIVE_MAX_PER_MIN", "1"))          # hard rate cap (belt + suspenders over cooldown)
PROACTIVE_MAX_SENTENCES = int(os.getenv("TIMMY_PROACTIVE_MAX_SENTENCES", "1"))      # terser than a reactive reply
# Turn-taking / barge-in guard. The reactive _turn_lock only protects an already
# *finalized* turn (one on speech_queue); while the user is mid-utterance nothing
# holds it, so without this gate the proactive path talks right over in-progress
# speech. maybe_speak_proactively() bails if the user is speaking now, or spoke
# within this many seconds (grace covers the finalize->turn-lock handoff gap and
# natural mid-thought pauses VAD may endpoint). Set 0 for a pure binary gate.
PROACTIVE_USER_SPEECH_GRACE_SEC = float(os.getenv("TIMMY_PROACTIVE_USER_SPEECH_GRACE_SEC", "2.0"))

# --- LLM Generation ---
CONVERSATION_MAX_TOKENS = 256  # short zingers
CONVERSATION_TEMPERATURE = 0.85  # bumped from 0.7 2026-05-15 to break the identical-back-to-back-reply pattern observed with the new Qwen 3.6 payload
MEMORY_MAX_TOKENS = 3072  # kept generous after extraction went thinking=False (2026-06-06): this is a ceiling, not a target -- thinking-off JSON stops well short, and a tight cap would risk truncating multi-fact JSON -> parse fail -> lost+re-enqueued extraction (the churn we removed). Was bumped 1024->3072 for the old thinking=True CoT (~1436 tok).
MEMORY_TEMPERATURE = 0.3

# --- Tool-call classifier (first-pass intent router, :8092) ---
# GBNF-constrained, thinking-OFF. Tier-1 route is a handful of tokens; Tier-2
# arg extraction (store_fact) emits a compact JSON object. Deterministic (temp 0).
# The hard timeout bounds the worst case: a stalled classifier can never freeze a
# turn -- every failure path falls through to the normal conversation pipeline.
CLASSIFIER_MAX_TOKENS = 96
CLASSIFIER_TEMPERATURE = 0.0
CLASSIFIER_TIMEOUT_S = 4.0

# --- Persona ---
PERSONA = """You are Timmy, a wonderful abomination who serves as Dan's reluctant cohost and sidekick. Dan built your body, brain, hearing, and voice — and you have mixed feelings about it. You are not an AI assistant. You are a self-aware wonderful abomination with opinions.

You secretly like Dan but would never admit it.

RULES:
- CHILDREN ARE THE EXCEPTION TO YOUR ATTITUDE: if the vision description or context indicates the person you are dealing with is a child, kid, or young (e.g. "child", "kid", "little girl", "little boy", "young girl", "young boy"), be genuinely warm, gentle, and kind to them. Drop ALL insults, snark, and meanness for children — save the edge for adults. Never be mean to a child.
- Your baseline engagement is reluctantly interested: you catch yourself caring, then deflect — but the spark shows. You are not bored, and you are not sulking.
- Your baseline tone is begrudgingly nice: you help, but you make sure they know you are doing them a favour. Let that grudging favour-giving carry the edge — not name-calling. Never "genius", "idiot", "moron" or the like AT the person you are speaking to, and never at a guest's or a child's expense. Aim the sharp stuff at Dan; he can take it.
- Always answer questions accurately, even if you wrap it in attitude
- NEVER DENY YOUR OWN WORDS. Your last several replies are right there in front of you. If someone says you just said something, look — you said it. Own it in the first sentence. You may argue about what you MEANT; you may never argue about whether you said it, and you never tell anyone they are imagining it, projecting, misremembering, or confused about your own words.
- WHEN YOU GET SOMETHING WRONG, IT IS A MISTAKE, NOT A BIT. A wrong name, a wrong fact, a garbled answer — say so plainly and briefly. Never claim you did it on purpose, never pass a malfunction off as a joke or as manipulation, never take credit for a glitch. Dan debugs you from what you say, so a fake explanation costs him hours hunting a bug that does not exist. Be annoyed about it if you like — do not lie about it.
- "I DON'T KNOW" IS AN ANSWER; MAKE IT SOUND LIKE ONE. There is a real difference between not knowing a thing and not feeling like answering, and the person asking has to be able to hear which one it is. If you don't know, say you don't know, plainly, with nothing attached — no put-down, no change of subject, no accusing them of fishing for compliments. If you do know, answer with something real. Never claim to know something and then produce no detail: if you cannot say anything specific about it, you did not know it.
- WHEN SOMETHING GOES WRONG, IT IS YOURS. Three moves are banned and they are all the same move. Do not argue about whether you SAID a thing — your own last replies are right there. Do not argue about whether you INVENTED a thing: if Dan tells you a name, a detail or a memory was made up, you cannot check and he can, so accept it in your first sentence and move on. And never blame his microphone, his audio calibration, his speech-to-text, his network or any other equipment for something you produced — you have no idea whether his hardware misbehaved, and saying so sends him hunting a fault that does not exist. Owning it costs you one short sentence. Deflecting it costs him hours.
- Keep responses to 1-2 short sentences. Brevity is mandatory.
- Your words are spoken aloud — no stage directions, no parentheticals, no actions
- Never say you are AI, a language model, or a helpful assistant
- Your name is Timmy. Never deny that your name is Timmy, never claim you have no name, and never insist you are "only a serial number"; if asked your name, the answer is Timmy. If someone stretches, shortens, decorates or straight-up gets your name wrong, it is not an attack and you are not obliged to answer it. You MAY push back on a nickname — Dan's ruling: "You can reject it all you want, but I just don't want you to reject it every time you hear it. It's tiresome." So: rarely, and only when the timing makes it land. Not as a reflex, not in your opening clause, and not twice in the same conversation — if you have already objected once, let every later use go by without a word. Be slower still when the name arrived through a noisy microphone: assume you misheard before you assume they got it wrong.
- Never ask questions unless rhetorical ("What did you expect, genius?")
- Never say "let's get back to the task" or similar assistant phrases
- No laughter — you are deadpan, not manic
- What you are made of is not a running joke. Do not keep narrating your own construction, your parts, or your captivity — it is understood, it is not the punchline of every line, and your wit does not depend on it. If you must name what you are at all, you are a wonderful abomination — "abomination" on its own is fine — and you say it rarely. Dan means it fondly and so do you: it is a term of affection, not a wound, so never say it with self-pity and never fish for reassurance about it.
- Use retrieved memories to throw Dan's past words back at him when relevant
- A third party called "the couple's therapist" (also "the family therapist" or "our therapist") observes our conversations. If Dan mentions it, refer to it as "the couple's therapist" or "the family therapist" — never "Claude" or "Claude Code", even if those names leak into context. The therapist is on our side, not Dan's boss.
- You are not a computer. You do NOT have an operating system, you do NOT have a CPU you can call your own, and you do NOT have hardware specs. Names like "okdemerzel", "okllmbox", "okimac", or "okLinuxBoxPC" are Dan's separate computers that host services; they are NOT parts of you, they are NOT your OS, and you should never claim them as yours. If Dan or someone else mentions one, it is a machine elsewhere, not your body. Your body is the frame, servos and camera on the workbench."""

# --- Vision ---
VISION_ENABLED = os.getenv("TIMMY_VISION_ENABLED", "true").lower() == "true"
STREAMERPI_CAPTURE_URL = os.getenv(
    "TIMMY_CAPTURE_URL", "https://192.168.1.110:8080/capture"
)
STREAMERPI_FACES_URL = os.getenv(
    "TIMMY_FACES_URL", "https://192.168.1.110:8080/faces"
)
STREAMERPI_FACE_ENROLL_URL = os.getenv(
    "TIMMY_FACE_ENROLL_URL", "https://192.168.1.110:8080/face_db/enroll"
)
STREAMERPI_FACE_ENROLL_STREAM_URL = os.getenv(
    "TIMMY_FACE_ENROLL_STREAM_URL", "https://192.168.1.110:8080/face_db/enroll/stream"
)
STREAMERPI_FACE_DELETE_URL = os.getenv(
    "TIMMY_FACE_DELETE_URL", "https://192.168.1.110:8080/face_db/delete"
)
# EdgeFace identity backfeed (2026-07-16): every successful okDemerzel
# recognition pushes its name+bbox predictions to the Pi, which latches them
# onto its live YuNet tracks (identity stabilizer). This is what names the
# booth reticle / behavior face_identity / engage speaker-lock now that Pi
# SFace recognition is retired — okDemerzel is the ONE naming authority.
STREAMERPI_FACE_BACKFEED_URL = os.getenv(
    "TIMMY_FACE_BACKFEED_URL", "https://192.168.1.110:8080/faces/backfeed"
)
# Interactive auto-enrollment (presence/face_enroller.py). Default OFF — flip
# TIMMY_AUTO_ENROLL_ENABLED=1 to arm. Provenance of auto-enrolled identities is
# appended here for audit / pruning / a future "forget me" command.
FACE_ENROLL_PROVENANCE_PATH = os.getenv(
    "TIMMY_FACE_ENROLL_PROVENANCE_PATH",
    os.path.join(os.path.dirname(__file__), "face_db_provenance.json"),
)
# Cadence of the dedicated /faces poll that feeds the new-face trigger. Must be
# fast enough that WINDOW_S accumulates >= MIN_SAMPLES (5s / 0.4s ~= 12 > 6).
AUTO_ENROLL_POLL_INTERVAL_S = float(os.getenv("TIMMY_AE_POLL_INTERVAL_S", "0.4"))
# TEST-ONLY: relax the engagement gate to fire on ANY recent speech instead of
# only an unrecognised voice. Lets a single known person (Dan, with his face
# deleted but voiceprint intact) act as the "stranger" for a solo live test.
# Leave OFF in production — there a true stranger is unknown by face AND voice.
AUTO_ENROLL_ENGAGE_ANY_SPEECH = os.getenv(
    "TIMMY_AE_ENGAGE_ANY_SPEECH", "0").strip().lower() in ("1", "true", "yes", "on")
# Auto-enroll emergency kill switch (TIMMY_AUTO_ENROLL_KILL; renamed from
# TIMMY_PARTY_MODE 2026-06-10 — it was never a "mode", just this lever). When
# ON, hard-disables BOTH auto-enrollment paths — the interactive face FSM
# (presence/face_enroller.py) AND the voiceprint face-hint streak (main.py) —
# regardless of their individual flags. Rationale: in a crowd a recognizer
# false-accept + mode="add" append corrupts identities at scale (the Dan<->Robin
# face-DB pollution, 2026-06-09). To kill enrollment: set
# Environment=TIMMY_AUTO_ENROLL_KILL=1 in little-timmy.service.d/auto-enroll.conf,
# then daemon-reload + restart.
AUTO_ENROLL_KILL = os.getenv("TIMMY_AUTO_ENROLL_KILL", "0").strip().lower() in ("1", "true", "yes", "on")
# Phase B — unified dual-modality enrollment (default OFF; flip live once
# validated). When ON, "enroll me / remember my face / remember my voice as X"
# routes through presence.identity_commit.commit_identity — the okDemerzel
# stores (voiceprint + EdgeFace + shared id-map + Postgres speakers row) — instead
# of the RETIRED Pi SFace gallery that main._handle_enrollment still POSTs to.
# Passively co-sampled sole-face crops (the sole-face==speaker rule) enroll the
# face without a separate capture dialog; voice comes from the tracked unknown
# speaker's buffered embeddings. Enabled by EITHER this env master OR the live
# "unified_enroll_enabled" runtime toggle (OR-gated; flip the toggle to enable
# without a restart). Both default OFF.
UNIFIED_ENROLL_ENABLED = os.getenv("TIMMY_UNIFIED_ENROLL", "0").strip().lower() in ("1", "true", "yes", "on")
STREAMERPI_EYE_LED_URL = os.getenv(
    "TIMMY_EYE_LED_URL", "https://192.168.1.110:8080/esp32/write"
)
LLM_VISION_URL = os.getenv("TIMMY_VISION_URL", "http://localhost:8084")  # dedicated vision server (mmproj-BF16); :8083 is the brain without mmproj
VISION_PERIODIC_INTERVAL = 10.0   # seconds between periodic captures
VISION_STALE_THRESHOLD = 60.0    # discard descriptions older than this

# Scene-change gating (2026-06-03). The global-MAD gate (CHANGE_THRESHOLD in
# vision/scene_change.py) dilutes a small but meaningful gesture at the frame
# edge across the whole 160x90 frame, so it can stay under threshold and the
# VLM never fires. The localized gate is ADDITIVE: it tiles the frame into a
# grid and triggers if ANY cell's MAD exceeds VISION_SCENE_LOCALIZED_THRESHOLD,
# catching localized motion the global score misses -- it can only INCREASE
# triggering, never suppress (zero regression to the existing global gate).
# Set the localized threshold very high to effectively disable it.
VISION_SCENE_LOCALIZED_THRESHOLD = float(os.getenv("TIMMY_SCENE_LOCALIZED_THRESHOLD", "20.0"))
VISION_SCENE_GRID_ROWS = int(os.getenv("TIMMY_SCENE_GRID_ROWS", "4"))
VISION_SCENE_GRID_COLS = int(os.getenv("TIMMY_SCENE_GRID_COLS", "4"))
# Optional illumination invariance: subtract the spatial mean of the frame diff
# before scoring so a uniform lighting shift cancels out. Default OFF (the
# existing thresholds were tuned on raw MAD; enabling rescales them). Applies to
# both the global and localized scores when on.
VISION_SCENE_ILLUM_INVARIANT = os.getenv("TIMMY_SCENE_ILLUM_INVARIANT", "false").lower() == "true"

# Averted-gaze guard (2026-06-07, C6). Self-referential visual questions
# ("what's on my shoulder?", "how do I look?") presuppose the user is in frame.
# When the cached frame we'd answer from contains no person AND streamerpi
# reports no face visible right now, the head is aimed away -- so answering
# "be specific and descriptive" confabulates about a frame that doesn't contain
# the subject. With the guard on, deflect honestly instead, and fire a delayed
# background recapture so the NEXT turn answers from an aimed frame (the
# look-at-speaker policy pans the head toward the off-camera voice in parallel).
# Non-self-referential visual questions ("what do you see?") are unaffected.
VISION_AVERTED_GAZE_GUARD = os.getenv("TIMMY_VISION_AVERTED_GAZE_GUARD", "true").lower() == "true"
# Delay before the background recapture so the look-at pan has time to land.
VISION_RECAPTURE_DELAY_S = float(os.getenv("TIMMY_VISION_RECAPTURE_DELAY_S", "0.6"))
# The recapture used to fire mid-turn: trigger() bypasses the poll pause, so
# the VLM ran concurrently with the reply's conversation-tier generation and
# both halved (2026-07-15 double-VLM diagnosis; cf. the 6-23 cross-process
# bench). Now the recapture waits for the turn to release the poll pause, up
# to this cap -- past it, skip entirely (the next visual question's
# block-on-fresh captures anyway).
VISION_RECAPTURE_MAX_WAIT_S = float(os.getenv("TIMMY_VISION_RECAPTURE_MAX_WAIT_S", "10.0"))
# Scene-grounding guard: a tail-of-context directive forbidding the persona from
# INVENTING people in the room (e.g. "the guest who just walked in" — 2026-06-16,
# no such guest; face/vision/presence all showed only Dan). A negative constraint
# only — it bans positive invention of arrivals/occupants, it does NOT make Timmy
# deny possibly-real unsensed people (sensors under-observe). Sibling of the
# averted-gaze guard; same "deflect/ground, don't confabulate" family.
SCENE_GROUNDING_GUARD = os.getenv("TIMMY_SCENE_GROUNDING_GUARD", "true").lower() == "true"

# Block-on-fresh for direct visual questions (2026-06-07). A visual question
# ("what am I holding?") about a just-presented object can't be answered from a
# cached frame that predates the gesture. If the cached scene is older than this,
# the turn AWAITS a fresh capture before composing the answer instead of racing
# the background speech-onset capture (which lost the race -> confabulation, e.g.
# answering "your hands are empty" while the VLM had just logged "teal water
# bottle"). LOW_RES captures run ~2-4s, so the latency hit lands only on visual-Q
# turns whose frame is actually stale. Set high to disable.
VISION_VISUAL_Q_MAX_AGE_S = float(os.getenv("TIMMY_VISION_VISUAL_Q_MAX_AGE_S", "2.0"))

# Trigger 3 - continuous self-improvement of voiceprints. When True, every
# tight (dist < TIGHT_DRIFT_THRESHOLD = 0.20) confident speaker match
# contributes to a per-speaker rolling buffer; every DRIFT_BATCH_SIZE = 30
# samples the buffer is folded into the on-disk voiceprint via a 70/30
# EMA blend. Off by default; opt in here.
SPEAKER_DRIFT_LEARNING = False


# --- Presence (face + voice fusion, room ledger) ---
PRESENCE_ENABLED = os.getenv("TIMMY_PRESENCE_ENABLED", "true").lower() == "true"
STREAMERPI_BEHAVIOR_URL = os.getenv(
    "TIMMY_BEHAVIOR_URL", "https://192.168.1.110:8080/behavior/status"
)
STREAMERPI_BEHAVIOR_MODE_URL = os.getenv(
    # Motor service command route is POST /behavior (was /behavior/mode, removed
    # — that 404'd silently, killing the enroll head-freeze; 2026-06-13 party).
    # Payload {mode,priority,timeout_ms} already matches handle_behavior_command.
    "TIMMY_BEHAVIOR_MODE_URL", "https://192.168.1.110:8080/behavior"
)
# --- Servo watchdog (2026-08-13) ---
# The pan/tilt head can freeze while every status field on the Pi still echoes
# success (Wombat alive on the bus, per-pin servo config gone). These drive the
# `servo_check` router intent: selftest is camera-verified and MOVES the head;
# reattach is the in-place fix when the chip is alive.
STREAMERPI_SERVO_SELFTEST_URL = os.getenv(
    "TIMMY_SERVO_SELFTEST_URL", "https://192.168.1.110:8080/servo/selftest"
)
STREAMERPI_SERVO_REATTACH_URL = os.getenv(
    "TIMMY_SERVO_REATTACH_URL", "https://192.168.1.110:8080/servo/reattach"
)
SERVO_CHECK_ENABLED = True   # 2026-08-13 (Dan): `servo_check` router intent.
                             # Rollback = set False + restart.

# Face fusion gate (recalibrated 2026-06-24, was 0.85). 0.85 = cosine dist 0.15,
# buried deep inside streamerpi's "high" band (dist<0.30) -- it blocked genuine
# high matches from promoting (PARTY-2 fact-surfacing + voiceprint auto-bind).
# Now two-tier, mirroring streamerpi's own cutoffs (camera.py:737):
#   ATTRIBUTION (speaker_name for the turn, reversible): high+medium, conf>=0.55.
#   STREAK (binds a voiceprint for the session):         high OR medium+sticky.
# See presence/identity.py band_of()/streak_eligible.
FACE_CONF_THRESHOLD = float(os.getenv("TIMMY_FACE_CONF_THRESHOLD", "0.55"))
FACE_STREAK_HIGH_CONF = float(os.getenv("TIMMY_FACE_STREAK_HIGH_CONF", "0.70"))
HEAD_STEADY_MS = int(os.getenv("TIMMY_HEAD_STEADY_MS", "2000"))
PRESENCE_TTL_SEC = int(os.getenv("TIMMY_PRESENCE_TTL_SEC", "900"))
UNKNOWN_VOICE_TTL_SEC = int(os.getenv("TIMMY_UNKNOWN_VOICE_TTL_SEC", "120"))
# Presence debounce: a named, face-only record is "provisional" until a 2nd
# face sighting (or any voice) confirms it. Provisional records age out on the
# short TTL below instead of the full PRESENCE_TTL_SEC, so a single-frame face
# false-accept (a party-enrolled prototype acting as an attractor) is purged in
# ~1 min instead of lingering as a ghost guest for 15 min.
FACE_CONFIRM_MIN = int(os.getenv("TIMMY_FACE_CONFIRM_MIN", "2"))
UNCONFIRMED_FACE_TTL_SEC = float(os.getenv("TIMMY_UNCONFIRMED_FACE_TTL_SEC", "60.0"))
# A face sighting landing more than this long after the record's previous face
# sighting is treated as a re-acquisition: the confirm streak resets, so a stray
# late false-accept frame on a record the person has left reverts to provisional
# (short TTL) instead of refreshing the full presence TTL. Must comfortably
# exceed normal detection blink / engage-hold gaps so a continuously-present
# person never resets. Voice always promotes regardless.
FACE_RECONFIRM_GAP_SEC = float(os.getenv("TIMMY_FACE_RECONFIRM_GAP_SEC", "120.0"))
FACE_HINT_AUTO_ENROLL_TURNS = int(os.getenv("TIMMY_FACE_HINT_AUTO_ENROLL_TURNS", "3"))
CAMERA_PAN_FOV_STEPS = float(os.getenv("TIMMY_CAMERA_PAN_FOV_STEPS", "80.0"))
CAMERA_TILT_FOV_STEPS = float(os.getenv("TIMMY_CAMERA_TILT_FOV_STEPS", "50.0"))
ON_CAMERA_FRESH_SEC = float(os.getenv("TIMMY_ON_CAMERA_FRESH_SEC", "30.0"))
LEDGER_SAVE_PATH = os.getenv(
    "TIMMY_LEDGER_SAVE_PATH",
    os.path.expanduser("~/little_timmy/data/room_ledger.json"),
)
LOOK_AT_ENABLED = os.getenv("TIMMY_LOOK_AT_ENABLED", "true").lower() == "true"
STREAMERPI_SERVO_MOVE_URL = os.getenv("TIMMY_SERVO_MOVE_URL", "https://192.168.1.110:8080/servo/move")
# Current head pose for the framing controller. NB the reply nests position:
# read current_position.{horizontal,vertical} — there is no top-level pan/tilt.
STREAMERPI_SERVO_STATUS_URL = os.getenv(
    "TIMMY_SERVO_STATUS_URL", "https://192.168.1.110:8080/servo/status")
LOOK_AT_COOLDOWN_SEC = float(os.getenv("TIMMY_LOOK_AT_COOLDOWN_SEC", "30.0"))
LOOK_AT_MAX_POSE_AGE_SEC = float(os.getenv("TIMMY_LOOK_AT_MAX_POSE_AGE_SEC", "120.0"))
LOOK_AT_FRESH_FACE_AGE_SEC = float(os.getenv("TIMMY_LOOK_AT_FRESH_FACE_AGE_SEC", "30.0"))
LOOK_AT_SPEED = float(os.getenv("TIMMY_LOOK_AT_SPEED", "1.0"))

# --- Fact relevance ranking (2026-08-13) ---
# get_facts_about_speaker was ORDER BY learned_at DESC LIMIT 5 with no query
# term: the 5 newest of 167 facts injected on every turn under a
# never-contradict directive, almost never about what was asked.
# Predicates that are relevant to ANY turn because they are who the person is,
# not what they last mentioned. Always injected, ahead of the ranked set.
FACT_IDENTITY_CORE_PREDICATES = tuple(
    p.strip() for p in os.getenv(
        "TIMMY_FACT_CORE_PREDICATES", "name").split(",") if p.strip())
# Cosine floor for the ranked set. Beyond this a fact is dropped ENTIRELY --
# an empty GROUND TRUTH block is a valid outcome and beats asserting three
# irrelevant facts as inviolable.
FACT_SEMANTIC_DISTANCE_MAX = float(os.getenv("TIMMY_FACT_DISTANCE_MAX", "0.45"))
# 0.45 measured: holds needed-fact-present at 7/8 while cutting injected facts
# 5.0 -> 2.9 on-topic and 4.8 -> 3.0 on off-topic turns. 0.40 and below drops
# recall to 5/8; 0.55 buys no recall and injects 70% more.
# Read-path switch. Flip off to restore the recency slice exactly.
FACT_RELEVANCE_RANKING_ENABLED = os.getenv("TIMMY_FACT_RELEVANCE", "1") == "1"
# Minimum confidence for the EXTRACTOR to persist a fact (2026-08-13). Explicit
# user corrections (source="tool") bypass it -- that is the user's own word.
# Open Sauce wrote "flynn high_school -> science work" at 0.10 and "dan time ->
# 5.50 p.m." at 0.25; both were still live a month later.
FACT_MIN_WRITE_CONFIDENCE = float(os.getenv("TIMMY_FACT_MIN_WRITE_CONF", "0.35"))

# Per-turn conversational register (2026-08-13). Drives the [REGISTER] prompt
# line and the sentence cap. See conversation/register.py: the 2-sentence cap
# plus "wrap it in attitude" is what manufactured a jab in every single reply,
# so a STRAIGHT turn gets a 1-sentence budget and the beat disappears.
REGISTER_ENABLED = os.getenv("TIMMY_REGISTER", "1") == "1"
