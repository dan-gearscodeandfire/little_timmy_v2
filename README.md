# Little Timmy v2

Voice-interactive mechatronic skeleton assistant running on local LLMs. Built by Dan (gearscodeandfire).

This repo holds the **brain** — the orchestrator, audio pipeline, memory, vision/perception, persona, and the Open Sauce booth display — running on okdemerzel (Strix Halo, 96 GB unified RAM, Vulkan). The **body** — pan/tilt servos, camera, WebRTC, face DB, eye LEDs, fire effects — lives on a Raspberry Pi 4 ("streamerpi") at the [`little_timmy_motor_raspi`](https://github.com/dan-gearscodeandfire/little_timmy_raspi_motor_v2) repo.

All inference runs locally. No cloud APIs.

---

## Pipeline at a glance

```
Mic ─► PipeWire ─► Silero VAD ─► whisper.cpp ─► Speaker ID
                                                    │
                            Voice-trigger enrollment? ─► streamerpi /face_db/enroll
                                                    │
                                                    ▼
              Memory retrieval (hybrid) ◄─► PostgreSQL + pgvector
                                                    │
                  Ephemeral system prompt ◄── persona/3×3 mood + facts + memories + vision + presence
                                                    │
                                                    ▼
                                Llama 3.2 3B (conversation) ─► Piper TTS ─► PipeWire
                                                    │
                          ╔═══════════════════════╗ │
                          ║  fire-and-forget ─────╫─┤
                          ╚═══════════════════════╝ │
                                                    ▼
                          Qwen3.6-35B-A3B :8083 (thinking-on)
                              memory extraction · summary rollup · DWU router
```

In parallel, two background loops feed context into the next turn:

- **Vision** (`vision/`) — pulls a 320×180 JPEG from streamerpi `/capture`, gates on scene change, calls Qwen3.6 vision (`:8084`, mmproj attached), parses a 7-field structured prompt, and decides whether to inject a `[WHAT YOU SEE]` block on the next user turn.
- **Presence** (`presence/`) — reads streamerpi `/faces` (single-source face DB), translates servo pose to room coordinates, persists a room ledger, drives the look-at-speaker indicator and `on_camera` smoothing.

---

## Local services (okdemerzel)

| Port  | Unit                              | What it serves |
|-------|-----------------------------------|----------------|
| 5432  | PostgreSQL                         | Memory store (pgvector + pg_trgm) |
| 8081  | `llama-3b.service`                | **Llama 3.2 3B Q4** — LT conversation tier |
| 8083  | `qwen36-server.service`           | **Qwen3.6-35B-A3B Q4_K_M** — memory extraction (thinking-on), summary rollup, DWU router, local Claude Code, n8n |
| 8084  | `qwen36-vision-server.service`    | **Qwen3.6 + mmproj-BF16** — vision-only VLM (thinking-off) |
| 8085  | `booth-display.service`           | Open Sauce visitor + operator screens (HTTP fan-out, see [`booth_display/`](booth_display/)) |
| 8891  | whisper.cpp                       | STT (shared with demerzel-stt) |
| 8893  | `little-timmy.service`            | Orchestrator + dashboard (this repo's `main.py`) |
| 8894  | `little-timmy-os.service`         | Service-management dashboard ([`little_timmy_os/`](little_timmy_os/)) |
| 11434 | Ollama                             | `nomic-embed-text` embeddings (768-dim) |

The orchestrator runs as a systemd unit; the canonical launch path is `sudo systemctl restart little-timmy.service`. Don't run `python main.py` directly unless debugging — see [`little-timmy-runtime-gotchas`](https://github.com/dan-gearscodeandfire/little_timmy/blob/main/services/little-timmy.service) (notes in Obsidian).

---

## Repo layout

```
~/little_timmy/
  main.py                  # Orchestrator event loop
  config.py                # All env-overridable config (URLs, thresholds, persona)
  audio/                   # PipeWire capture, Silero VAD, hybrid endpointing, sounddevice playback
  stt/                     # whisper.cpp client
  speaker/                 # Resemblyzer voiceprint matching (cosine, threshold 0.38)
  llm/
    client.py              # llama.cpp clients: SSE streaming (3B) + non-streaming Qwen3.6
    prompt_builder.py      # Ephemeral system prompt assembly
  memory/
    manager.py             # Memory CRUD + Ollama embeddings
    retrieval.py           # Hybrid: pgvector + FTS + trigram → RRF (cosine floor, top 5)
    facts.py               # Subject/predicate/value triples + entity resolution
    rollup.py              # Sliding window: hot (verbatim 5 min) → warm (Qwen3.6 summarized) → cold (DB)
    extraction.py          # Async memory formation via Qwen3.6 :8083 thinking-on
  conversation/
    manager.py             # Turn tracking, history assembly for KV cache
    enroll_intent.py       # Regex matcher for "Little Timmy, learn my face, my name is X"
    models.py              # Turn / WarmSummary / ConversationState dataclasses
  vision/
    capture.py             # 1 fps JPEG poll from streamerpi (320×180), pause-during-speech, cooldown
    scene_change.py        # Pillow+numpy MAD gate (CHANGE_THRESHOLD 12.0, NOISE_FLOOR 2.0)
    analyzer.py            # Qwen3.6 :8084 → 7-field SceneRecord (people/objects/actions/scene_state/change_from_prior/novelty/speak_now)
    relevance.py           # Urgency scoring → inject vs skip
    context.py             # VisionContext, get_vision_debug, pause_polling delegate
    face_remote.py         # RemoteFaceClient — reads streamerpi /faces (replaces in-tree YuNet)
    visual_question.py     # User-utterance matcher for explicit visual questions
  presence/
    face_client_local.py   # Per-turn face observation via streamerpi /faces
    pose.py                # Servo-coord → world-coord translation
    ledger.py              # Persistent room_ledger.json (~/little_timmy/data/)
  persona/
    state.py               # Deterministic 3×3 mood axes (X engagement / Y warmth)
                           # Signals: VADER + nomic-embed similarity
  feedback/
    detector.py            # Meta-feedback "Little Timmy, that was [critique]" → Qwen3.6 confirm
                           # Persists to feedback_inbox.jsonl + /api/feedback endpoint
  tts/
    engine.py              # Piper in-process ONNX, raw PCM, sentence-boundary streaming
  web/
    app.py                 # FastAPI on :8893 — /ws, /api/{mood,vision,presence,feedback}, dashboard
  booth_display/
    server.py              # Fan-out bridge: LT /ws + streamerpi /faces+/behavior + /api/* → browser /ws
    static/visitor.html    # Visitor screen — WebRTC video + face boxes + look-at badge + scene/reply/audit
    static/operator.html   # Operator dashboard
  little_timmy_os/         # Service-management dashboard on :8894
  enroll_face_remote.py    # CLI for streamerpi /face_db/* (stdlib-only)
  enroll_speaker.py        # Voice enrollment (Resemblyzer voiceprint capture)
  models/tts/              # Piper Skeletor ONNX voice model
  services/                # systemd unit files (mirror of /etc/systemd/system/)
  data/                    # Persistent state (room_ledger.json, etc.)
  db/schema.sql            # PostgreSQL schema
  tests/                   # pytest suites (currently 82+ unit tests)
```

---

## Database

| Table | Notes |
|-------|-------|
| `speakers` | id, name, voice_id, created_at |
| `memories` | id, type (episodic/semantic/procedural/conversation_summary), content, speaker_id, embedding `vector(768)`, timestamps, metadata jsonb, generated `content_tsv` |
| `facts` | id, subject, predicate, value, source_memory_id, speaker_id, learned_at, confidence, superseded_by |

Indexes: HNSW on embedding, GIN on tsvector, GIN trigram on `content` and `facts.subject`.

---

## Conversation memory tiers

- **Hot** — last ~5 min of turns, verbatim in context.
- **Warm** — older turns summarized by Qwen3.6 :8083 thinking-on (`memory/rollup.py`). Postprocessing strips reasoning preamble before persistence.
- **Cold** — PostgreSQL with hybrid retrieval (pgvector cosine + FTS + trigram → RRF). Cosine floor enforced to suppress weak hits.

Facts are extracted as subject/predicate/value triples and treated as **ground truth** in the prompt — they precede memories and are not subject to recency decay.

---

## Persona / mood

Persona is **not a static system-prompt block**. It's a deterministic 3×3 axis system in `persona/state.py`:

- **X** axis: engagement (low / medium / high) — driven by VADER sentiment and nomic-embed similarity to engagement exemplars.
- **Y** axis: warmth (cold / neutral / warm) — same signal sources, different exemplars.
- Cell selection picks one of nine `(engagement, warmth)` snippets that are concatenated into the ephemeral system prompt for the current turn.

New tone behavior = a new axis or a new signal source, **not** a new prompt-string trigger.

---

## Vision pipeline

Five-stage frame path:

1. streamerpi camera (`libcamera-vid`, 640×360 @ 15 fps native) → H264 → in-memory frame buffer
2. okdemerzel pulls a JPEG via HTTP `GET /capture?w=320&h=180` (1 fps poll, paused during speech)
3. Scene-change gate (`vision/scene_change.py`): Pillow+numpy mean-abs-diff on grayscale; below threshold → skip VLM
4. VLM call to Qwen3.6 :8084 (mmproj attached, thinking-off, `max_tokens=200`) → 7-field structured JSON
5. Relevance scoring (`vision/relevance.py`) → either inject `[WHAT YOU SEE]` block on next user turn or wait

End-to-end VLM cycle: ~2.0 s on the workshop scene. Capture payload: ~8 KB. See [Obsidian: `lt-visual-pipeline-baseline-2026-05-07-update-2026-05-08`].

---

## Face recognition (single-source on streamerpi)

There is **one** face database, on streamerpi at `~/little_timmy_motor_raspi/face_db/embeddings.json`. okdemerzel's in-tree YuNet+SFace path was retired in 2026-04-27; `vision/face_id.py` + the legacy `enroll_face.py` script were deleted 2026-05-14 (the streamerpi-remote path is the only enrollment route now).

Live consumers on okdemerzel:

- `vision/face_remote.py` (`RemoteFaceClient`) — VLM cycle calls `/faces` to enrich `SceneRecord` with identities.
- `presence/face_client_local.py` — per-turn face observation via `/faces`, replaces local capture+detect.

Enrollment paths:

- **Voice** — say "Little Timmy, learn my face, my name is X". `conversation/enroll_intent.py` matches the regex pre-LLM, `Orchestrator._handle_enrollment` POSTs `/face_db/enroll`, Timmy speaks the result. Speaker-known fallback if name omitted.
- **CLI** — `python enroll_face_remote.py <name> [--count N --interval S --stream]`. `--stream` uses the SSE endpoint for per-sample progress.
- **Manage** — `enroll_face_remote.py --list` / `--delete <name>`.

---

## Booth display (Open Sauce)

`booth_display/` is a thin fan-out layer on `:8085` that bridges Little Timmy (`:8893`), streamerpi (`192.168.1.110:8080`), and a browser kiosk:

- `GET /visitor` — visitor-facing screen: WebRTC video from streamerpi (with face-box overlay + look-at badge driven by behavior-mode), live VLM scene caption, live reply typewriter (Llama 3B), colored audit log.
- `GET /operator` — operator dashboard (heartbeats, latency, overrides — placeholder in current revision).
- `WS /ws` — fan-out. Browser receives `snapshot` on connect, then deltas. `lt_*` events forwarded from LT, `faces`/`behavior` from streamerpi, `mood`/`vision`/`presence` from LT polled HTTP.
- `POST /streamerpi/offer` — proxies WebRTC SDP to streamerpi `/offer` over HTTPS with `verify=False` (avoids self-signed-cert friction in the browser).

---

## Configuration

All config is in `config.py` and overridable via environment. Most-used:

| Env var | Default | Notes |
|---|---|---|
| `TIMMY_LLM_URL` | `http://localhost:8081` | Llama 3B conversation tier |
| `TIMMY_MEMORY_URL` | `http://localhost:8083` | Qwen3.6 for memory + summary |
| `TIMMY_VISION_URL` | `http://localhost:8083` (default in code) | **Overridden by `little-timmy.service` to `http://localhost:8084`** — `qwen36-vision-server.service`. A static grep for `:8083` in `analyzer.py`/`config.py` will find it; runtime resolves to `:8084` via the unit's `Environment=` line. |
| `TIMMY_WHISPER_URL` | `http://localhost:8891` | whisper.cpp |
| `TIMMY_FACES_URL` | `https://192.168.1.110:8080/faces` | streamerpi face DB |
| `TIMMY_CAPTURE_URL` | `https://192.168.1.110:8080/capture` | streamerpi JPEG capture |
| `TIMMY_FACE_ENROLL_URL` | `https://192.168.1.110:8080/face_db/enroll` | Voice/CLI enrollment target |
| `TIMMY_BEHAVIOR_URL` | `https://192.168.1.110:8080/behavior/status` | Behavior-mode poll for booth look-at indicator |
| `TIMMY_DB_DSN` | `postgresql://...localhost:5432/...` | PostgreSQL |
| `TIMMY_VISION_ENABLED` | `true` | Master switch for vision pipeline |

---

## Running

```bash
# Production (systemd-managed)
sudo systemctl restart little-timmy.service
journalctl -u little-timmy.service -f

# Logs (file path, not journal — unit redirects StandardOutput/Error)
tail -f ~/demerzel/logs/voice-server.log    # for voice-server (separate)
tail -f ~/little_timmy/data/lt.log          # if configured
```

Restart paths for downstream LLMs after config changes:

```bash
sudo systemctl restart qwen36-server.service          # :8083 (memory + router + Claude Code + n8n)
sudo systemctl restart qwen36-vision-server.service   # :8084 (vision)
sudo systemctl restart llama-3b.service               # :8081 (conversation)
```

The `little_timmy_os` dashboard on `:8894` provides a UI for service start/stop/restart.

---

## Cross-references

- **Body / streamerpi**: [`little_timmy_motor_raspi`](https://github.com/dan-gearscodeandfire/little_timmy_raspi_motor_v2)
- **Booth display state**: see Obsidian notes `booth-display-server-state-2026-05-10` and `booth-narrator-not-deployed-2026-05-10`
- **LLM stack snapshot**: Obsidian `little-timmy-qwen36-swap-2026-05-04-update-2026-05-05`
- **Vision pipeline**: Obsidian `lt-visual-pipeline-baseline-2026-05-07-update-2026-05-08`
- **Persona / mood axes**: Obsidian `lt-deterministic-mood-state-axes-2026-05-06`
- **Sweep verifiers**: `Areas/sweeps/llm-stack-okdemerzel.md` in the Demerzel memory vault — weekly `cron`-style verification of every claim above
