# little_timmy — CONTEXT

The domain language and load-bearing design decisions for this codebase.
Code wins when it disagrees with this file; this file wins when it disagrees
with older prose notes. Keep it short — it is a map, not a manual.

---

## Domain language

- **Conversation turn** — one round of interaction: Timmy takes in what was
  said (or notices a situation), and produces and speaks a response. The seven
  internal steps are: hear → identify speaker → retrieve memory → build prompt
  → stream from the LLM → speak (TTS, sentence by sentence) → save what was
  learned.

- **`ConversationTurn`** — the deep module that owns the whole turn behind a
  small interface. Two front doors, one shared private pipeline:
  - `respond(words, who)` — reactive: someone spoke (voice→STT) or typed.
    Voice and text collapse into this one door; they differ only in input
    source, not behaviour.
  - `speak_proactively(situation)` — Timmy initiates with no incoming
    utterance (e.g. a new person arrives). Different *trigger*, same guts.
  TTS lives **inside** the turn (it is the deepest step, not a caller concern),
  so a test can drive a full turn and assert exactly what was spoken.

- **`Introductions`** — the multi-turn "what's your name? … did I get that
  right?" sub-dialog. Owns its own between-turns state (who we're confirming,
  awaiting yes/no). The turn consults it but does **not** carry that state,
  so each `ConversationTurn` call stays self-contained.

- **Presence handoff** — *who is talking* is resolved at the doorway via the
  existing pure `fuse_identity` (see `presence/identity.py`) and handed into
  the turn as a resolved identity + presence snapshot (used to build the
  prompt). The **side-effects** of identification — turning the head toward an
  off-camera speaker, auto-learning a voiceprint on a face-match streak — fire
  from that same doorstep, **outside** the turn. The turn never touches servos
  or the voiceprint store.

- **After-answer chores** — work that runs once the reply is spoken:
  - *Saving what was learned* (memory/fact extraction) is the turn's final
    real step, done through an injected memory seam. It is also the slow call
    that gets pre-empted when a new turn arrives (see Priority gate below).
  - *Cosmetic chores* (mood update, compliment log, turn snapshot for review,
    spoken "re-enroll my voice" detection) fire-and-forget **outside** the
    core, so they cannot slow or clutter the turn.

- **Priority gate** *(future — "Candidate 2", not built yet)* — a single seam
  that owns "a new conversation turn pre-empts in-flight slow work (memory
  save / rollup)." Today this is loose primitives in `llm/client.py`.
  `ConversationTurn` must preserve the pre-empt-and-requeue behaviour; the
  gate's own deepening is tracked separately.

## Injected seams (the test surface)

`ConversationTurn` receives three collaborators at construction; each has a
real adapter in production and a fake in tests:

| Seam      | Production            | Test fake records / returns        |
|-----------|-----------------------|------------------------------------|
| Speaker   | Piper TTS             | the sentences spoken, in order     |
| LLM       | local llama/qwen HTTP | a canned token stream              |
| Memory    | retrieval + extract   | what was saved; canned retrievals  |

A turn test reads: *"Dan says 'how are you?' → assert spoken == ['Doing
well.', 'You?'] and a fact about Dan was saved"* — no mic, GPU, speakers, or DB.

## Why "deep"

The codebase already has the target shape in three places — match them:
- `persona/state.py` — `update(x,y) → state` over a mood machine.
- `vision/analyzer.py::analyze_frame` — `jpeg → SceneRecord`, VLM HTTP hidden.
- `presence/identity.py::fuse_identity` — pure logic, gates exposed for tests.

Before this refactor the turn was smeared across the `main.py` Orchestrator
(~1000 LOC, 16 responsibilities, 37 inline config reads, **zero** offline
tests). The Orchestrator's job shrinks to: listen, identify, wire, speak the
hardware — everything else moves behind `ConversationTurn`.

## Prompt shape is load-bearing — do not break when relocating prompt assembly

- `system[0]` = static persona + PROTOCOL_CLAUSE, KV-cached forever.
- History is stored **wrap-free** (`add_user_turn` raises on `[CONTEXT]` /
  `[UTTERANCE]` markers).
- The current turn is rendered as
  `[CONTEXT]…[/CONTEXT][UTTERANCE]…[/UTTERANCE]` by `wrap_user_message()`.

## Settings

Config is injected once at construction, not re-read per line (was 37 inline
`config.*` reads in `main.py`).
