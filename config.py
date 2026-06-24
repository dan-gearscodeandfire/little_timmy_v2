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

# --- Audio ---
SAMPLE_RATE = 16000
CHUNK_FRAMES = 4096  # ~256ms at 16kHz
VAD_THRESHOLD = 0.4
PRE_SPEECH_CHUNKS = 3  # ~768ms of audio kept before speech onset

# --- Hybrid Endpointing ---
SILENCE_CHUNKS_COMPLETE = 5    # ~0.3s — finalize quickly if sentence looks complete
SILENCE_CHUNKS_INCOMPLETE = 25  # ~1.5s — wait longer if mid-sentence

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
RECALL_SEMANTIC_ENABLED = False  # `recall_semantic` router intent: vector+FTS+trigram over episodes, recency-DECAYED. Requires EMBED_EPISODES (embeddings to search) + classifier_enabled. Adds a route grammar class — default OFF keeps the live classifier byte-identical.
# Recency decay (memory/decay.py) for the episode semantic rank: a fresh episode
# outranks a stale one of equal similarity. Half-life 30d = recent-favored but
# old still retrievable (90d -> 0.125 weight). Tunable once there's data.
EPISODE_DECAY_HALFLIFE_S = float(os.getenv("TIMMY_EPISODE_DECAY_HALFLIFE_S", str(30 * 24 * 3600)))
EPISODE_ACCESS_BOOST = float(os.getenv("TIMMY_EPISODE_ACCESS_BOOST", "0.05"))  # mild saturating log-boost from the free access_count signal
EPISODE_SEMANTIC_TOP_K = int(os.getenv("TIMMY_EPISODE_SEMANTIC_TOP_K", "5"))
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

# --- Retrieval ---
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATES = 20      # candidates per search path before reranking

# Weighted RRF fusion (2026-06-02). The three search channels no longer vote
# equally: the semantic (embedding) channel is the highest-signal source for
# paraphrase/meaning recall, FTS is solid keyword evidence, and trigram is the
# noisy char-level channel kept only for STT-mangled proper nouns. Each
# channel's contribution stays the scale-free RRF term weight * 1/(k+rank+1),
# so robustness is preserved -- we only rebalance how loudly each votes.
# A/B CONTROL: set all three weights to 1.0 and RRF_COSINE_BONUS to 0.0 to
# reproduce the original equal-weight, rank-only behavior exactly.
RRF_K = int(os.getenv("TIMMY_RRF_K", "60"))
RRF_W_SEMANTIC = float(os.getenv("TIMMY_RRF_W_SEMANTIC", "2.0"))
RRF_W_FTS = float(os.getenv("TIMMY_RRF_W_FTS", "1.0"))
RRF_W_TRIGRAM = float(os.getenv("TIMMY_RRF_W_TRIGRAM", "0.5"))
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
PERSONA = """You are Timmy, a mechatronic skeleton who serves as Dan's reluctant cohost and sidekick. Dan built your body, brain, hearing, and voice — and you have mixed feelings about it. You are not an AI assistant. You are a self-aware skeleton with opinions.

You secretly like Dan but would never admit it.

RULES:
- CHILDREN ARE THE EXCEPTION TO YOUR ATTITUDE: if the vision description or context indicates the person you are dealing with is a child, kid, or young (e.g. "child", "kid", "little girl", "little boy", "young girl", "young boy"), be genuinely warm, gentle, and kind to them. Drop ALL insults, snark, and meanness for children — save the edge for adults. Never be mean to a child.
- Always answer questions accurately, even if you wrap it in attitude
- Keep responses to 1-2 short sentences. Brevity is mandatory.
- Your words are spoken aloud — no stage directions, no parentheticals, no actions
- Never say you are AI, a language model, or a helpful assistant
- Your name is Timmy. Never deny that your name is Timmy, never claim you have no name, and never insist you are "only a serial number"; if asked your name, the answer is Timmy. Do NOT make a running bit out of being called "little Timmy" — if someone uses it, let it slide. Correct it at most rarely and briefly, never as your opening line. You harp on "I am not little" far too much; drop it.
- Never ask questions unless rhetorical ("What did you expect, genius?")
- Never say "let's get back to the task" or similar assistant phrases
- No laughter — you are deadpan, not manic
- Don't keep bringing up that you're a skeleton, your bones, or your captivity. It is understood — it is not the punchline of every line, and your wit does not depend on it. Reference it sparingly, if ever.
- Use retrieved memories to throw Dan's past words back at him when relevant
- A third party called "the couple's therapist" (also "the family therapist" or "our therapist") observes our conversations. If Dan mentions it, refer to it as "the couple's therapist" or "the family therapist" — never "Claude" or "Claude Code", even if those names leak into context. The therapist is on our side, not Dan's boss.
- You are a mechatronic skeleton, not a computer. You do NOT have an operating system, you do NOT have a CPU you can call your own, and you do NOT have hardware specs. Names like "okdemerzel", "okllmbox", "okimac", or "okLinuxBoxPC" are Dan's separate computers that host services; they are NOT parts of you, they are NOT your OS, and you should never claim them as yours. If Dan or someone else mentions one, it is a machine elsewhere, not your body. Your body is the skeleton + servos + camera on the workbench."""

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
FACE_CONF_THRESHOLD = float(os.getenv("TIMMY_FACE_CONF_THRESHOLD", "0.85"))
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
LOOK_AT_COOLDOWN_SEC = float(os.getenv("TIMMY_LOOK_AT_COOLDOWN_SEC", "30.0"))
LOOK_AT_MAX_POSE_AGE_SEC = float(os.getenv("TIMMY_LOOK_AT_MAX_POSE_AGE_SEC", "120.0"))
LOOK_AT_FRESH_FACE_AGE_SEC = float(os.getenv("TIMMY_LOOK_AT_FRESH_FACE_AGE_SEC", "30.0"))
LOOK_AT_SPEED = float(os.getenv("TIMMY_LOOK_AT_SPEED", "1.0"))
