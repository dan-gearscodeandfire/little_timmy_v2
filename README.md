# Little Timmy v2

Voice-interactive mechatronic skeleton assistant running on local LLMs. Built by Dan (gearscodeandfire).

## Architecture Overview

Little Timmy is a real-time voice assistant with persistent memory, speaker identification, computer vision, and a custom persona. All inference runs locally on a GMKtec Evo 2 (Strix Halo, 96GB unified RAM, Vulkan) — no cloud APIs.

### Core Pipeline

```
Mic → PipeWire → VAD → whisper.cpp (STT) → Speaker ID → Memory Retrieval
  → Prompt Builder → llama.cpp 3B (conversation) → Piper TTS → Speaker
```

### Module Map

| Module | Path | Purpose |
|--------|------|---------|
| **main.py** | `main.py` | Orchestrator — wires all subsystems, runs the main loop |
| **audio/capture** | `audio/capture.py` | PipeWire mic capture, Silero VAD, hybrid endpointing |
| **audio/playback** | `audio/playback.py` | Audio output via sounddevice |
| **stt** | `stt/client.py` | whisper.cpp client (port 8891) |
| **speaker** | `speaker/identifier.py` | Resemblyzer-based speaker ID with unknown tracking |
| **llm** | `llm/client.py` | Streaming llama.cpp client for conversation + extraction |
| **llm/prompt_builder** | `llm/prompt_builder.py` | Ephemeral system prompt assembly (persona + facts + memories + vision) |
| **conversation** | `conversation/manager.py` | Hot/warm/cold conversation tiers, turn management |
| **memory** | `memory/` | PostgreSQL-backed episodic/semantic memory with embedding retrieval |
| **memory/extraction** | `memory/extraction.py` | LLM-based fact and memory extraction from conversations |
| **memory/facts** | `memory/facts.py` | Ground truth fact store (subject-predicate-value triples) |
| **memory/retrieval** | `memory/retrieval.py` | Hybrid retrieval: embedding similarity + keyword + RRF reranking |
| **memory/rollup** | `memory/rollup.py` | Conversation summarization for warm/cold tiers |
| **vision** | `vision/` | 4-layer tiered vision pipeline (see below) |
| **tts** | `tts/engine.py` | Piper TTS with custom Skeletor voice model |
| **web** | `web/app.py` | FastAPI endpoints: metrics, conversation, vision debug, chatlog |
| **config** | `config.py` | All configuration: URLs, thresholds, persona, model params |
| **db** | `db/` | PostgreSQL connection, schema migrations |

### Vision Pipeline (4 layers)

The vision system captures frames from a Raspberry Pi camera (streamerpi) and decides what visual context to inject into conversation.

| Layer | File | Purpose |
|-------|------|---------|
| 1. VLM Analysis | `vision/analyzer.py` | Qwen2.5-VL 7B via llama.cpp → structured JSON SceneRecord |
| 2. Scene-Change Gate | `vision/scene_change.py` | Pillow+numpy grayscale MAD — only triggers VLM on significant change |
| 3. Relevance Classifier | `vision/relevance.py` | Scores novelty/persistence/urgency → decides inject vs skip |
| 4. Context Builder | `vision/context.py` | Manages history, provides filtered summaries to prompt builder |
| — Capture | `vision/capture.py` | 1fps JPEG polling from streamerpi with cooldown logic |

### External Services (all local)

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Memory storage (episodes, facts, embeddings) |
| Ollama | 11434 | Embedding model (nomic-embed-text) |
| llama.cpp (GPT-OSS-120B) | 8080 | Memory extraction LLM |
| llama.cpp (Llama 3.2 3B) | 8081 | Conversation LLM |
| llama.cpp (Qwen2.5-VL 7B) | 8082 | Vision analysis |
| whisper.cpp | 8891 | Speech-to-text |
| Piper TTS | local | Text-to-speech (custom Skeletor voice) |
| streamerpi | 192.168.1.110:8080 | Raspberry Pi camera + servo body |

### Speaker Identification

Uses Resemblyzer embeddings with cosine distance matching. Voiceprints stored as `.npy` files in `models/speaker/`. Threshold: 0.38. Unknown speakers are tracked and Timmy can ask their name after stable utterances.

Enrollment: `python enroll_speaker.py <name> <seconds> --save`

### Conversation Memory

Three-tier system:
- **Hot**: Raw turns from last 5 minutes (verbatim in context)
- **Warm**: LLM-summarized older turns (compressed)
- **Cold**: PostgreSQL with embedding retrieval (long-term)

Facts are extracted as subject-predicate-value triples and used as GROUND TRUTH in prompts.

### Configuration

All config is in `config.py` via environment variables with sensible defaults. Key env vars:
- `TIMMY_WHISPER_URL`, `TIMMY_LLM_URL`, `TIMMY_VISION_URL` — service endpoints
- `TIMMY_VISION_ENABLED` — toggle vision pipeline
- `TIMMY_DB_DSN` — PostgreSQL connection string

### Running

```bash
cd ~/little_timmy
source .venv/bin/activate
python main.py
```

Requires all external services running. See Little Timmy OS (port 8894) for service management dashboard.
