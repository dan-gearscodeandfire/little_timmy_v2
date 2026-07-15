# Little Timmy — Claude Code Primer

> **This file is the upstream source of truth for the Little Timmy stack.** It is auto-loaded by any Claude Code session running in `~/little_timmy/` (i.e. `localclaude` on okdemerzel). From okLinuxBoxPC, fetch it on session start with `ssh okdemerzel cat ~/little_timmy/CLAUDE.md`. When CLAUDE.md disagrees with code, CODE wins — fix CLAUDE.md in the same session. When CLAUDE.md disagrees with Obsidian / MEMORY.md, CLAUDE.md wins — Obsidian + MEMORY.md propagate FROM here, not into here.

---

## Maintenance protocol (read before editing this file)

**Order of writes when LT behavior changes:**

1. **Edit this file** with the new facts. Bump `Verified against code on:` (footer) and the `Last edited:` date.
2. **Commit** the code change and the CLAUDE.md edit **together** so the doc never drifts from the SHA that produced it.
3. **If you have Obsidian MCP available** (i.e. you are on okLinuxBoxPC, not localclaude): refresh `Zettelkasten/little-timmy-primer.md` to mirror this file. Set its `verified-against-code-on:` to today and `expires:` to today + 14 days. Append a one-line entry to `Areas/session-history.md`.
4. **Update MEMORY.md** if any top-level pointer in the LT Stack section needs to move.
5. **If you do not have Obsidian MCP** (localclaude on okdemerzel): leave a `[CLAUDE.md changed YYYY-MM-DD — Obsidian primer needs refresh]` marker at the top of step 3's note via `ssh okLinuxBoxPC` is not viable; instead, the next session on okLinuxBoxPC will diff the CLAUDE.md `Verified against code on:` date against the Obsidian primer's `verified-against-code-on:` and resync if stale.

**Refresh trigger checklist** — bump this file when any of these change:
- Service is added/removed/renamed, or a port moves.
- Conversation tier model swaps, or the prompt-shape is restructured.
- A discipline-level invariant is added or revoked (e.g. strip-on-store assert, priority gate, KV-cache contract).
- A top-level repo directory is added/removed.
- A new operating gotcha is discovered that future sessions need to know cold.

---

## Quick orientation

Little Timmy is a voice-interactive mechatronic skeleton assistant. This repo is the **brain** on okdemerzel (Strix Halo, 96 GB UMA GPU split, Vulkan). The **body** (servos, camera, WebRTC, face DB, eye LEDs) lives on streamerpi (RPi 4) in the [`little_timmy_motor_raspi`](https://github.com/dan-gearscodeandfire/little_timmy_raspi_motor_v2) repo. All inference is local — no cloud APIs.

`README.md` is the contributor-facing intro; this file is the working set for Claude Code. The README **may be stale**; treat this file as the freshest snapshot.

---

## Services / ports — verified against code on 2026-05-30

### okDemerzel (Strix Halo, headless, 192.168.1.156)

| Port | Unit | Purpose | Default state |
|---|---|---|---|
| 5432 | `postgresql@16-main.service` | Memory DB (pgvector + pg_trgm) | active |
| 8081 | `llama-3b-server.service` | Conversation tier alternative (Llama 3.2 3B Q4) | **inactive** (LT-OS dropdown spins it up on demand) |
| 8083 | `qwen36-server.service` | **Brain + conversation tier** (Qwen3.6-35B-A3B Q4_K_M, thinking via per-request kwargs) — fact extraction, DWU router, **and** conversation since the 2026-05-15 flip | active |
| 8084 | `qwen36-vision-server.service` | Vision tier (Qwen3.6 + mmproj-BF16) — scene captioning, thinking-off | active |
| 8085 | `booth-display.service` | Legacy Open Sauce visitor/operator screens | **inactive** (superseded by booth-mockup) |
| 8090 | `booth-mockup.service` | Concept B visitor overlay — full-bleed WebRTC + face-id annotations, HTTPS self-signed | active |
| 8891 | `whisper-server.service` | STT (whisper.cpp HTTP) | active |
| 8893 | `little-timmy.service` | LT orchestrator (FastAPI main event loop) | active |
| 8894 | `little-timmy-os.service` | LT-OS operator dashboard + service manager | active |
| 11434 | `ollama.service` | Embeddings (`nomic-embed-text`, 768-dim) | active |

### streamerpi (RPi 4, 192.168.1.110, ssh via ProxyJump from okdemerzel)

| Port | Unit | Purpose |
|---|---|---|
| 8080 | `little-timmy-motor.service` | Camera frame buffer, YuNet face detector + SFace face-id, pan/tilt servos via Serial Wombat (NOT ESP32), WebRTC peer, behavior state machine, `/faces` API |

ESP32 on streamerpi is **eye-LED only** (`POST /esp32/write` on streamerpi:8080). Servos = Serial Wombat from the Pi.

---

## Conversation tier — Qwen3.6 + system-first prompt shape (the 2026-05-15 flip)

**Active model:** Qwen3.6-35B-A3B Q4_K_M on `qwen36-server.service` :8083, the same server that handles fact extraction / rollup / DWU routing. The LT-OS dropdown can swap to `llama3.2-3b` (spawns `llama-3b-server.service`) or other ggufs in `~/models/`. Choice persists to `data/lt_runtime_toggles.json` (`conversation_model_id` + `conversation_url_override`). Static `config.LLM_CONVERSATION_URL` defaults to :8081 but the runtime override wins at request time via `llm/client.py:_current_conversation_url()`.

### Prompt shape (NOT "ephemeral system at tail")

Qwen3.6's Jinja template positions `system` at chat-start; the prior Llama-style ephemeral-system-at-tail layout was reordered or broken on Qwen, killing KV cache. The replacement (`llm/prompt_builder.py`):

```
[0] system  = persona + PROTOCOL_CLAUSE                 (truly static, KV-cached forever)
[1..M-1]   = history (synthetic summary pair on rollup; hot turns raw, wrap-free)
[M] user   = [CONTEXT]<ephemeral_block>[/CONTEXT][UTTERANCE]<user_text>[/UTTERANCE]
```

**Critical invariants** — break any of these and you reintroduce the old failure modes:

1. **system[0] is truly static.** No clock, no mood, no per-turn signal. `build_static_persona_system()` returns `config.PERSONA + PROTOCOL_CLAUSE` only. Mutates on persona edit (restart-level event), not per turn.
2. **The `[CONTEXT]`/`[UTTERANCE]` wrap is render-time only.** Applied by `wrap_user_message()` inside `build_messages()` for the **current** user turn alone.
3. **History is stored wrap-free.** `conversation/manager.py:add_user_turn` raises `ValueError` if the input contains `[CONTEXT]` or `[UTTERANCE]` markers. The stored `Turn.content` is the raw utterance; the speaker is a separate `Turn.speaker` field. Render-time, past turns get the speaker prefix re-prepended; current turn gets the context wrap re-applied.
4. **PROTOCOL_CLAUSE teaches the model** that `[CONTEXT]` = its own perception (not user speech, don't quote / acknowledge it as a message) and `[UTTERANCE]` = the human's actual words. MOOD inside CONTEXT is for embodiment, not narration.

If you find yourself rebuilding the system message per turn, or storing wrapped text in history, you are recreating the bugs that the 2026-05-15 flip fixed. See Obsidian `little-timmy-conversation-tier-qwen36-shipped-2026-05-28` for the full rationale + commit trail.

### Conversation-priority gate

Because conversation and brain now share a single `-np 1` server, a user reply would otherwise FIFO-serialize behind a 15–45 s thinking-on extraction. Mitigation in `llm/client.py`:

- `stream_conversation` calls `_cancel_in_flight_slow_calls()` at start → in-flight `extract_and_store` / rollup raise `CancelledError`; their `finally` blocks release locks.
- `_conversation_in_flight` event is set during streaming.
- `generate_memory` (brain-tier slow path) blocks on `_wait_for_conversation_idle()` and registers itself via `_register_slow_call()` so it is cancellable.
- All four guards are **no-ops** when `_conversation_shares_brain()` is false (i.e. Llama 3B selected).

Dan's rule: *"conversational call always takes preference over summarization."*

### Thinking gating

`stream_conversation` injects `chat_template_kwargs: {enable_thinking: false}` when routed to the brain. Llama 3B ignores the kwarg. `generate_memory` passes `thinking=True/False` per-call; the two-pass extractor is thinking-off classifier → thinking-on JSON. Vision tier stays thinking-off.

---

## Pipeline (per user turn, abridged)

```
USB mic 48 kHz → Silero VAD → 16 kHz buffer
  → whisper :8891 STT → user_text
  → voice-print speaker_id (WeSpeaker cosine — `pyannote/wespeaker-voxceleb-resnet34-LM`, migrated off Resemblyzer 2026-06-17; voiceprints `models/speaker/<name>_wespeaker.npy`)
  → broadcast turn event (WS fanout: web /ws + booth_display /ws)
  → eye_led AI_THINKING signal (LT → streamerpi → ESP32)
  → parallel:
      - hybrid retrieval (pgvector + FTS + trigram → WEIGHTED RRF → top-K memories;
        semantic channel query is coreference-augmented with last N turns)
      - get_facts_about_speaker (alias-aware: subject ∈ {canonical, 'user', 'i', 'me'} gated by speaker_id)
      - fetch streamerpi /faces → presence ledger update
  → ephemeral_block assembly (mood + ground-truths + memories + WHO PRESENT)
  → build_messages(history, ephemeral_block, user_text)
  → stream_conversation (Qwen3.6 :8083 SSE, enable_thinking:false)
  → filtered_assistant_stream (narration veto + max-sentences cap)
      → sentence-boundary chunks → Piper TTS → sounddevice
  → eye_led SPEAKING → AI_CONNECTED on TTS-end
  → broadcast metrics (est_prompt_tokens / est_completion_tokens)
  → fire-and-forget:
      - extract_and_store (Qwen3.6 :8083 two-pass; canonical subject)
      - mood update (VADER + nomic-embed → 3×3 axis signals)
      - rollup (idle-windowed; cancellable by next user turn)
      - compliment / 👍👎 detection → flagged.jsonl / feedback_inbox.jsonl
```
(Note: the speech-triggered vision capture now fires at **VAD speech-onset**, earlier than this fire-and-forget block — see below.)

Vision pipeline runs independently at 1 fps with scene-change gating, plus event-driven captures. Behavior state machine runs on streamerpi (`behavior.py`) — IDLE / SCAN / TRACK / ENGAGE / LOOK_AROUND / HOLD / SLEEP with transition-cause attribution.

**Scene-change gating (2026-06-03):** `vision/scene_change.py` keeps the global whole-frame MAD gate (`CHANGE_THRESHOLD`) unchanged, plus an **additive localized gate** — it tiles the 160×90 frame into a `VISION_SCENE_GRID_ROWS`×`COLS` grid (default 4×4) and also triggers if any cell's MAD ≥ `VISION_SCENE_LOCALIZED_THRESHOLD` (default 20), catching small/edge motion the global score dilutes below threshold. Additive = can only *increase* triggering, never suppress (zero regression). Optional `VISION_SCENE_ILLUM_INVARIANT` (default off) subtracts the spatial mean of the frame diff so uniform lighting shifts cancel. **Speech-onset capture:** `audio/capture.py` fires a no-arg `set_speech_onset_callback` the instant VAD detects onset; `main()` wires it to `vision.trigger_capture("speech_onset")`, ~1–2 s earlier than the old STT-end trigger (which was removed), so the cached scene is fresher when a visual question lands. Runs against the :8084 vision server, so no contention with the :8083 brain.

**Averted-gaze guard for visual questions (2026-06-07):** a prompted question *about the user* ("what's on my shoulder?", "what am I wearing?", "how do I look?") is only answerable when the head is actually aimed at them. Two parts: (1) `vision/visual_question.py` adds `is_self_referential_visual_question` — `main()` ORs it into `visual_q`, because plain `is_visual_question` missed self-referential phrasings and they fell through to the *background-awareness* vision branch and confabulated. (2) For a self-referential visual Q, if the cached frame contains no person **and** streamerpi reports no live face (`face_obs.behavior.face_visible`), the head is averted — `build_ephemeral_block(..., vision_subject_absent=True)` suppresses the (wrong) cached scene and injects an honest "I'm not looking your way" deflection instead of "be specific and descriptive", and `main()` fires a delayed `trigger_capture("visual_question_recapture")` (`VISION_RECAPTURE_DELAY_S`, 0.6 s) so the next turn answers from an aimed frame (look-at-speaker pans the head meanwhile). **Scene questions ("what do you see", "describe the room") are deliberately unaffected** — not self-referential, answerable from any frame (narration-veto rule). Env: `TIMMY_VISION_AVERTED_GAZE_GUARD` (default on). **Live-validated 2026-06-07** (deflected honestly with `frame_people=[]`/`face_visible=False`). The live test found a detector gap — natural phrasing "the thermos **I'm holding**" has no "my"/"am i" and a `what..i` span >20 chars, so it slipped `is_self_referential_visual_question`; fixed `4ae4c75` by adding an `i'm <presenting-verb>` pattern (straight + curly apostrophe).

**Visual-question grounding — block-on-fresh + raw injection (2026-06-07, C8):** distinct from the averted guard (which is the *can't-see-you* case). When Timmy *can* see the user but answers a direct visual question ("what am I holding?") wrong. Two stacked bugs, both found in the C6 live test: (1) **stale-frame race** — the turn snapshotted the cached scene record while the background speech-onset capture was still in flight, so it answered from a frame predating the gesture ("empty hands" while the VLM logged "teal water bottle" microseconds later). Fix: `main()` now `await`s its own `trigger_capture("visual_question")` when `VisionContext.scene_age()` exceeds `TIMMY_VISION_VISUAL_Q_MAX_AGE_S` (2.0 s) — HIGH_RES (the ~9 s path that justified never blocking) is retired, LOW_RES runs ~2-4 s. (2) **relevance starvation** — even with a fresh frame, `get_description()` returns None for a low-novelty scene (a bottle in a cluttered workshop), so the prompt carried no `[WHAT YOU SEE]` block and the brain confabulated. Fix: `get_raw_description()` bypasses the relevance gate — that gate is for *unsolicited* observations; an explicit question is the wrong place to apply it. So `main()` routes `visual_q` through `get_raw_description()`, everything else through `get_description()`. Live-validated: "A teal bottle, obviously." (age 2.4 s→0.0 s, e2e 4.1 s). Tests `tests/test_visual_question_grounding.py`.

**LT-OS vision-bar frame freshness (2026-06-07):** `VisionContext.trigger_capture()` previously updated `_current` but **not** `_last_jpeg` (only the 1 fps poll did), so the dashboard thumbnail showed the last *poll* frame while a reply was driven by a *trigger* frame. Both write paths now set `_last_jpeg` + `_last_frame_source` in the same locked block; `/api/vision` exposes `frame_source`, and the LT-OS vision bar reads `Frame: <age> (<source>)`. The thumbnail is now guaranteed to be the exact frame the VLM consumed — the diagnostic instrument for the averted-gaze guard above.

---

## Proactive (unprompted) speech (2026-06-03)

Timmy can react verbally to a high-urgency visual event (someone entering) without being addressed first. `Orchestrator.maybe_speak_proactively()` is called from `vision_people_monitor` (~every 2 s) and is **heavily gated**, in order: `config.PROACTIVE_SPEECH_ENABLED` (static **kill-switch**, default **allow/ON**; set env `TIMMY_PROACTIVE_SPEECH_ENABLED=false` to forbid entirely) → `proactive_speech_enabled` runtime toggle (the **live** control, default **OFF**) → hearing not muted → **not barging in** (`capture.user_speaking` false AND last voiced chunk older than `PROACTIVE_USER_SPEECH_GRACE_SEC`, 2 s) → rising edge (`is_new_arrival`) **or** `record.speak_now` / `urgency_score ≥ PROACTIVE_URGENCY_THRESHOLD` (0.8) → `PROACTIVE_COOLDOWN_SEC` (120 s) → `PROACTIVE_MAX_PER_MIN` (1). **BOTH** the kill-switch and the runtime toggle must be true; effective default is silent (toggle off).

**Operator control:** the runtime toggle is surfaced on the **LT-OS dashboard** (:8894) as the "Proactive Speech (unprompted)" switch, alongside the hearing / vision-auto-poll toggles. Chain: dashboard → LT-OS `POST /api/proactive/toggle` → `services.toggle_proactive_speech` → LT `POST /api/proactive` (:8893) → `runtime_toggles.set`. When the kill-switch is off the dashboard card shows amber + "Disabled by config". Live; no restart needed (read per-decision).

Key invariants (don't regress these):
- **One turn-lock for all spoken turns.** `Orchestrator._turn_lock` is held by reactive turns (the main loop wraps `process_speech`; `process_text_input` wraps its body) AND by the proactive path. Proactive **try-acquires non-blocking** (`if _turn_lock.locked(): return`) and **drops, never queues** — a stale remark must not fire late behind a real conversation.
- **Barge-in guard (2026-06-06).** The turn-lock above only protects a turn that is *already finalized* — the main loop acquires it when a segment lands on `speech_queue`, i.e. **after** STT. It does NOT cover an in-progress utterance, so on its own the proactive path talks right over the user mid-sentence (observed twice in one session, both vision-triggered). Fix: `maybe_speak_proactively` also gates on live VAD state — `capture.user_speaking` (True from VAD onset until finalize/discard) plus a `PROACTIVE_USER_SPEECH_GRACE_SEC` (2 s) window off `capture.last_voice_ts` (last genuine, non-suppressed voiced chunk). The grace covers the finalize→turn-lock handoff gap and natural mid-thought pauses VAD may endpoint. Set the grace to 0 for a pure binary gate (A/B control). **Don't regress to relying on the turn-lock alone for barge-in.**
- **No user UTTERANCE.** A proactive turn reuses the normal `[CONTEXT]/[UTTERANCE]` wrap via `build_proactive_messages()` (in `llm/prompt_builder.py`): the visual trigger sits in `[CONTEXT]`, and the fixed `PROACTIVE_SELF_PROMPT` fills `[UTTERANCE]`. system[0] stays byte-identical (KV cache survives). This follows the existing synthetic-prompt precedent of `_ask_speaker_name` / `_confirm_name`.
- **Only the assistant side is stored** (`add_assistant_turn`); the synthetic self-prompt is NEVER passed to `add_user_turn` (it would trip the strip-on-store assert and pollute history).
- Echo suppression + the conversation-priority gate are **inherited** for free (same `_stream_to_tts` → `tts.speak()` path; it's a conversation-tier call). Vision polling is paused for the duration to free the GPU.

Tunables (env-overridable in `config.py`): `PROACTIVE_SPEECH_ENABLED`, `PROACTIVE_URGENCY_THRESHOLD`, `PROACTIVE_COOLDOWN_SEC`, `PROACTIVE_MAX_PER_MIN`, `PROACTIVE_MAX_SENTENCES` (1), `PROACTIVE_USER_SPEECH_GRACE_SEC` (2.0). **Live in-frame validation still pending** (needs a real walk-into-view test with the feature enabled) — including the 2026-06-06 barge-in guard: talk through a deliberate pause with something interesting in frame and confirm the remark waits for ~2 s of silence instead of stepping on you.

---

## Face & voice identity (Phase B + LED-mic anchor, verified 2026-07-15)

**Authority:** okDemerzel owns face identity since 2026-07-05 (`face_authority` toggle = `"okdemerzel"`, live). Per turn: multi-frame `/capture` grab (`face_authority_frames`=3) → YuNet detect → align → **EdgeFace-S on CPU** → `FaceIdentifier`; accept = cosine distance ≤ `face_threshold` (0.50, live-tunable). The Pi's SFace `/faces` remains detection/behavior-fallback only. `face_shadow_enabled` (OFF) was the Phase A A/B instrument.

**Storage (embeddings only, no crops persisted):** ≤12 L2-normed 512-dim prototypes per person in `models/face/<name>_edgeface.npy` (dedup 0.07, min 3 samples — `presence/face_thresholds.py`; voice twin: ≤12, dedup 0.06, `models/speaker/<name>_wespeaker.npy`). Plus shared id-map `models/speaker/_id_map.json` + Postgres `speakers` row (the `facts.speaker_id` FK). **Face is enroll-time only** — no update-on-sight; voice T2 re-enroll / T3 drift have NO face analog. Face identity does NOT write `facts`/`episodes`/`memories`.

**Sole writer:** `presence.identity_commit.commit_identity` — guards: mismatch (known name must match ≥1 enrolled modality), lookalike (unverified modality too close to an existing identity), retired-name tombstone. Call sites: unified enroll (`main._handle_enrollment`, `unified_enroll_enabled`=true) and the introductions name-tell triple below.

**Per-turn co-sampling (feeds enrollment):** the doorway buffers aligned face crops under the pre-fusion voice key (`CoSampleBuffer`). Rule: **sole-face==speaker** (exactly 1 face in frame) — except when the LED-mic anchor is fresh, whose crops WIN over sole-face (crowd-safe).

**LED-mic anchor (EXPO engagement token, 2026-07-06→13):** the green multi-LED handheld mic marks the engaged speaker. CV detector `presence/led_detect.py` (HSV h 65–85 / s≥60 / v≥160, area 4–1500 px, 60 px cluster-merge; exactly ONE cluster else abstain) finds the lit mic; `presence/anchor.pick_anchored_face` picks the sole face directly above it (x-tol 0.25×frame; ambiguity = abstain). State is in-process (`presence/anchor.py`, TTL 30 s, refreshed by a 2 s `anchor_poll_monitor`) — restart wipes to the dark gate by design. **Binding rule (F1/F7):** anchored crops buffer/commit only when face-ID and voice-ID agree — anchored face recognized AS this speaker, or unrecognized face + `unknown_N` voice (the visitor case). **Mic-in-hand = implied consent** (Dan 7-06): the anchor un-darks the SPEECH identity dialogs under EXPO (`anchor.speech_dialogs_allowed`) but never the face-consent FSM.

**Name-tell triple (the constant face↔voice link for new speakers):** unknown_N speaks → intro name-ask, or volunteered "my name is X" (`passive_self_intro_enabled`) → spoken confirm → yes ⇒ `assign_name` (voice T1 persist) + `_maybe_commit_face` → `commit_identity` binds **name↔voiceprint↔faceprint** (crops = LED-anchored at EXPO, sole-face in Shop). Gate: `intro_face_commit_enabled`. **Both toggles flipped ON 2026-07-15 (Dan: "constantly link faceprints with voiceprints for new speakers").** A refused name (tombstone/reserved/taken) commits nothing and the speaker stays unknown_N (F2).

**Mode effects on face memory:** `guest_mode` withholds `facts.sensitive` at prompt-injection ONLY — face storage/recognition unchanged. The EXPO proximity vision gate changes WHEN the VLM polls, never WHAT is stored. EXPO regime darkens identity-MUTATION dialogs; recognition stays read-only-on; the anchor (or `identity_dialogs_override`) re-opens them.

---

## Repo layout (essentials)

```
~/little_timmy/
  main.py                      # Orchestrator event loop
  config.py                    # All env-overridable config; PERSONA constant lives here
  eye_led.py                   # LT → streamerpi → ESP32 eye-LED state feedback
  data/
    lt_runtime_toggles.json    # vision_auto_poll, hearing, conversation_url_override, conversation_model_id
    mood_state.json            # persisted persona axes
    mood_debug.jsonl           # per-turn instrumentation (Bundle C)

  llm/
    client.py                  # stream_conversation (SSE), generate_memory; conv-priority gate
    prompt_builder.py          # build_static_persona_system, wrap_user_message, build_messages

  memory/
    manager.py                 # memory CRUD + Ollama embeddings
    retrieval.py               # hybrid pgvector + FTS + trigram → weighted RRF (+ cosine fold-in, coreference query)
    facts.py                   # facts table + get_facts_about_speaker (alias-aware, Bundle B)
    extraction.py              # two-pass extractor; canonical subject normalization (Bundle B option b)
    rollup.py                  # sliding-window hot → warm → cold

  conversation/
    manager.py                 # add_user_turn (strip-on-store assert!), add_assistant_turn, idle rollup
    models.py                  # Turn / WarmSummary / ConversationState
    enroll_intent.py           # "remember my face/voice" intent matcher

  audio/                       # PipeWire capture, Silero VAD, hybrid endpointing, sounddevice playback
  stt/client.py                # whisper.cpp async HTTP client
  tts/engine.py                # Piper in-process ONNX, sentence-boundary streaming
  speaker/                     # voice-print ID + voice intents

  vision/
    capture.py                 # FrameCapture (1fps + scene-change)
    analyzer.py                # Qwen3.6 :8084 multimodal client
    context.py                 # VisionContext (boot-race retry, face-id enrichment)
    face_remote.py             # streamerpi /faces client
    relevance.py               # classifier: which VLM outputs inject into prompt
    scene_change.py            # frame-diff gating
    supervisor.py              # behavioral supervisor

  presence/                    # face/voice fusion → RoomLedger; canonical names; auto-enroll; look_at
                               # + Phase B identity: identity_commit.py (sole writer), face_recognize.py /
                               # face_identifier.py (EdgeFace), face_thresholds.py, anchor.py + led_detect.py
                               # (EXPO LED-mic anchor), prototype_base.py (shared .npy store + id-map)

  persona/
    state.py                   # deterministic 3×3 mood axes (X engagement, Y warmth)
    updater.py                 # per-turn signal computation
    render.py                  # ephemeral-prompt mood block

  persistence/runtime_toggles.py   # JSON-backed runtime toggles (vision_auto_poll, hearing, conv override)

  feedback/                    # 👍👎 + compliment capture; flagged.jsonl + feedback_inbox.jsonl

  web/app.py                   # Legacy FastAPI mount; /ws, /api/health, /api/conversation, /api/last_payload, /api/presence, /api/vision/auto_poll, /api/hearing, /api/timmy/toggles

  little_timmy_os/             # Separate FastAPI service on :8894 (the operator dashboard)
    main.py                    # Dashboard HTML+JS+routes; Booth Display panel added 2026-05-30 (5b435d3)
    services.py                # proxies + systemd / runtime-toggle helpers
    config.py                  # SERVICES dict (incl. booth_mockup), CONVERSATION_MODELS dropdown

  booth_display/               # Legacy Open Sauce visitor screen (server.py + static/); service inactive
  booth_mockup/                # Active Concept B visitor overlay on :8090 (HTTPS self-signed); index.html wears the "Combat Cogitator" skin since 2026-06-16 (see Operating → Booth visitor skin)
```

Backup files named `*.bak.<reason>-YYYY-MM-DD` are intentional — Dan keeps a deep history of prior states. Don't delete them.

---

## Operating

### Start everything from cold
```bash
sudo systemctl start postgresql ollama \
                    qwen36-server.service qwen36-vision-server.service \
                    whisper-server.service \
                    little-timmy.service little-timmy-os.service \
                    booth-mockup.service
```
Then open `http://localhost:8894` for the LT-OS dashboard.

### Stop everything
```bash
sudo systemctl stop little-timmy.service little-timmy-os.service \
                   qwen36-server.service qwen36-vision-server.service \
                   whisper-server.service booth-mockup.service
```

### Conversation model swap
LT-OS dashboard dropdown. Choice persists to `data/lt_runtime_toggles.json`. Picking `llama3.2-3b` will start `llama-3b-server.service` and write `conversation_url_override=http://localhost:8081`; picking `qwen36` stops it and writes `http://localhost:8083`.

### Open the booth display
LT-OS dashboard → **Booth Display** panel → **🎪 Open booth display**. Opens `https://<host>:8090/` in a named popup window (`lt_booth_display`). First time per browser: accept the self-signed cert. Requires `booth-mockup.service` active (toggle in services table if not).

### Booth visitor skin ("Combat Cogitator", 2026-06-16)
`booth_mockup/index.html` is reskinned: red T-800 × Warhammer-40K Adeptus-Mechanicus look (Copperplate Gothic headers + scene, brass ✠ litany band `++ AVE OMNISSIAH · THE FLESH IS WEAK · THE MACHINE ENDURES ++`, targeting reticle, scanlines). **Served via FileResponse → HTML edits need NO restart, just reload.**
- **All features preserved** — only the `<style>` block was swapped, 3 decorative divs injected; the entire `<script>` (WebRTC, presence/context/face polling, takeover) is byte-identical. Repaint trick: the page's JS sets inline colors via CSS vars (`--bone*`, `--signal`, `--shadow*`), so the porter keeps those var NAMES and only changes their VALUES → JS-driven content recolors for free.
- **Fonts self-hosted base64-embedded** in the HTML (this server has no `/static` route). Copperplate woff2 sourced from okiMac, converted under `~/claude/demerzel/.staging/booth_display/static/fonts/`.
- **Gotcha:** use ✠ Maltese cross (U+2720), NOT ☠ skull (U+2620) — Windows Chrome renders ☠ as a color emoji (looked like a creature).
- **Build/revert:** porter `~/claude/demerzel/.staging/booth_display/_port_cogitator_to_8090.py` (run it to (re)apply; reads live, writes live). Original backed up at `booth_mockup/index.html.bak.2026-06-16-pre-cogitator` (one `cp` to revert). Six alternate skins (Aliens/Mechanicus/T-800/Dune/Futura/Cogitator) + generator live in `.staging/booth_display/static/`.
- Note: the LT-OS-launched display is **:8090 booth_mockup**, NOT the legacy :8085 booth_display/visitor.html — don't reskin the wrong surface.

### Booth metrics HUD (2026-06-16)
`booth_mockup/index.html` carries a live operational HUD over the Combat Cogitator skin (new panels in commented `added 2026-06-16` blocks; centre `.reticle` removed). **HTML edits need only a reload; `server.py` edits need `sudo systemctl restart booth-mockup.service`.**
- **`/ltos/{path}` GET proxy** added to `booth_mockup/server.py` (`LTOS_URL=http://127.0.0.1:8894`, mirrors the existing `/lt/`→:8893 proxy) so the page can reach host/GPU telemetry on **LT-OS :8894**.
- **Panels & sources:** vitals donuts VRAM/SYS-RAM/CPU/GPU-load ← `/ltos/api/host` (2s, shared sysfs helper `ops/gpu_sysfs.py`); cogitator telemetry payload-tokens/latency-segments(1st|gen|tts)/completion ← `/lt/api/metrics` (4s); 3×3 mood matrix ← `/lt/api/mood`; 4 health pips ← `/lt/api/health` (`llm_3b` down → intentionally not shown); uptime/turns/GEN-t/s meta row. The sample face-ID bracket is a static demo (live faces still draw their own carets).
- **Layout (2026-06-16):** TR "retrieved memories" panel `.context-panel.tr` `bottom` raised 200→300px so it clears the `.telemetry` block below it.
- **Don't re-run the cogitator porter** — it asserts a byte-identical `<script>` and the JS/DOM is now hand-edited. Live `index.html` is source-of-truth. Backups: `index.html.bak.2026-06-16-pre-metrics-hud`, `…-pre-cogitator`. Rationale/history: Obsidian `Zettelkasten/booth-metrics-hud-2026-06-16.md`.

### Health / logs
```bash
curl http://localhost:8894/api/health
curl http://localhost:8893/api/health

journalctl -u little-timmy.service -f
journalctl -u qwen36-server.service -f
tail -F ~/demerzel/logs/little-timmy-os.log
```

### Chat log + service control via API (from anywhere)
```bash
curl http://localhost:8893/api/chatlog
curl http://localhost:8894/api/timmy/conversation | python3 -m json.tool
curl -X POST http://localhost:8894/api/service/little_timmy/restart
```

### Desktop quick-ref
`~/Desktop/little-timmy-startup.txt` mirrors the startup commands above for operator use without a Claude Code session. Keep it in sync when ports/units change.

---

## Disciplines & known gotchas

- **PipeWire env required for systemd** — `little-timmy.service` unit hard-codes `XDG_RUNTIME_DIR=/run/user/1000` + `PULSE_SERVER=...`. Without them, `sounddevice.play()` / Piper raise `PaErrorCode -9997` and TTS is silent.
- **whisper required for STT** — turning whisper off in LT-OS makes LT crash-loop on STT `ConnectError` until it comes back. Known harden-LT-against-missing-STT TODO.
- **Strix Halo Vulkan: do NOT bump `-np 1` → `2`** on llama.cpp. Per-call latency 2.5–3× worse; can't parallelize matmul across slots.
- **Qwen3.6 thinking_budget is silently dropped** by llama.cpp Jinja. Use `max_tokens` to cap thinking instead.
- **streamerpi single-client WebRTC lock** — multiple `/visitor` tabs cause the gray-restart cycle. Guarded by session-token + 409. Check tabs first before suspecting ICE.
- **Face identity authority = okDemerzel EdgeFace since 2026-07-05** (`face_authority` runtime toggle, live `"okdemerzel"`). The 2026-05-07 "face DB lives on streamerpi only" era is OVER: the Pi's SFace `/faces` is now detection + legacy-fallback only (it still supplies the BehaviorSnapshot); identities minted by the unified path NEVER reach the Pi face_db. Enroll via `presence.identity_commit.commit_identity` (see **Face & voice identity** section above), not the Pi `/face_db/enroll`.
- **LT does NOT depend on `demerzel-vision.service` (:8895)** since 2026-05-05. That service is the DeepStack-compatible API for Blue Iris, not LT.
- **GPT-OSS-120B is retired** (2026-05-24). Local frontier = Qwen3.6 only. Don't suggest it as an alternative.
- **Retrieval fusion is weighted, not equal-rank** (2026-06-03). `memory/retrieval.py:_fuse` weights channels (default semantic 2.0 / fts 1.0 / trigram 0.5) and folds the semantic cosine distance back in as a tiebreaker (`RRF_COSINE_BONUS`). The `<SEMANTIC_DISTANCE_MAX` (0.50) floor is unchanged; the bonus normalizes *within* the kept band, so re-tuning the floor and the bonus are coupled — change them together. **A/B control:** set `TIMMY_RRF_W_*=1.0` + `TIMMY_RRF_COSINE_BONUS=0.0` + `TIMMY_COREFERENCE_ENABLED=false` to reproduce the old equal-weight, rank-only, bare-utterance behavior exactly. All knobs are env-overridable in `config.py`.
- **Coreference query affects the semantic channel only** (2026-06-03). `retrieve(query, context_turns=...)` blends the last `CONTEXT_TURNS` (default 2) hot turns into the *embedding* query so elliptical follow-ups ("what about her?") resolve; FTS/trigram still get the bare utterance. Nothing stored changes — `recent_turns_excluding_current()` drops the current utterance (which `add_user_turn` already appended before retrieval runs).
- **Memory extraction is a queue, not single-flight drop** (2026-06-03). `memory/extraction.py` enqueues each exchange into a bounded FIFO (`EXTRACTION_QUEUE_MAX`=32) drained one-at-a-time by `_pump()` (one `_do_extraction` task at a time — the old single-flight guarantee, minus the dropping). A cancelled extraction (priority gate killing it when the user speaks) is **re-enqueued** via `_requeue()` (front of line, bounded by `EXTRACTION_MAX_RETRIES`=5), then parks on `generate_memory`'s existing `_wait_for_conversation_idle` until the conversation lulls — so turns' facts aren't lost during lively chat. Queue overflow / retry exhaustion drop with a WARN (no silent caps). `_do_extraction` must stay its OWN task (not awaited by the pump) so the gate cancels the extraction child, not the pump; its `finally` re-pumps.

---

## Where stuff is when you need rationale / history (Obsidian)

These notes live in the Nexus vault on okLinuxBoxPC. From there: `searchManager.searchContent`. From okdemerzel: use `searxng` MCP or grep the vault via ssh to okLinuxBoxPC.

- **Architecture deep-dive (pre-2026-05-15 flip):** `Zettelkasten/little-timmy-architecture-okdemerzel-2026-05-14.md`. Partially superseded for conversation tier + prompt shape; other sections (services, DB schema, pipeline) still authoritative.
- **Conversation tier + prompt-shape rationale (current):** `Zettelkasten/little-timmy-conversation-tier-qwen36-shipped-2026-05-28.md`.
- **Startup commands (operator reference):** `Zettelkasten/little-timmy-startup-commands-2026-05-14.md`.
- **Open todos / backlog:** `Zettelkasten/little-timmy-stack-open-todos-2026-05-14.md`.
- **Face-id refactor:** session note from 2026-05-07; current state captured in `face-pipeline-streamerpi-only`.
- **Presence v1 closure:** `Zettelkasten/presence-feature-roadmap-2026-05-05-update-2026-05-05-evening.md`.
- **Mood axes design:** `Zettelkasten/lt-deterministic-mood-state-axes-2026-05-06.md`.
- **Hybrid retrieval (cosine floor + RRF):** `Zettelkasten/lt-semantic-retrieval-cosine-floor-nomic-embed-2026-05-06.md`.
- **Visual pipeline baseline:** `Zettelkasten/lt-visual-pipeline-baseline-2026-05-07-update-2026-05-08.md`.
- **Behavior state machine:** see streamerpi repo `behavior.py` + Obsidian `data-age-liveness-pattern-2026-05-14.md`.

On okLinuxBoxPC, `MEMORY.md` carries one-line pointers to each.

---

## Provenance footer

- **Last edited:** 2026-07-15 by Claude (Opus 4.8), with Dan in the loop — face-identity drift correction: new **Face & voice identity** section (Phase B okDemerzel EdgeFace authority, LED-mic anchor, name-tell triple), stale "face DB lives on streamerpi only" gotcha rewritten, pipeline voice-print line ECAPA-TDNN→WeSpeaker, presence/ layout entry expanded. Same session: `intro_face_commit_enabled` + `passive_self_intro_enabled` flipped ON in `data/lt_runtime_toggles.json` (Dan: constant face↔voice linking for new speakers).
- **Verified against code on:** 2026-07-15 (face-identity sections only — `presence/{anchor,identity_commit,face_thresholds,face_shadow}.py`, `conversation/introductions.py`, `main.py` co-sample doorway, `persistence/runtime_toggles.py`, live `data/lt_runtime_toggles.json`). Prior full pass: 2026-06-07 (`main`; vision-freshness Group C — **averted-gaze guard** `f78731b` + self-ref detector gap `4ae4c75` + **visual-question grounding** block-on-fresh & raw injection `7b67a44` + **LT-OS frame-source** `98e8c15`; all deployed and **live-validated in front of Timmy**). Prior: 2026-06-06 (`main`; proactive-speech **barge-in guard** added — `capture.user_speaking`/`last_voice_ts` + `PROACTIVE_USER_SPEECH_GRACE_SEC`, supervisor issue #1, deployed live, live in-frame test pending). Prior: 2026-06-03 (`main`; weighted-RRF + coreference `d2af1e1`, proactive-speech + LT-OS toggle `696a961`, extraction queue/re-enqueue `31ed259`; vision localized scene-gate + speech-onset capture). 2026-05-30 (HEAD `5b435d3`).
- **Spawned this primer:** session 2026-05-29/30 (conv-tier memory refresh + Booth Display button + primer creation).
- **Next refresh expected:** when any item in the "Refresh trigger checklist" above fires. Do **not** wait for a calendar interval — drift in this file directly mis-leads future sessions.
