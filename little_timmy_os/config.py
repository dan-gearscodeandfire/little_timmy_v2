"""Service definitions for Little Timmy OS."""

import os

WEB_HOST = "0.0.0.0"
WEB_PORT = 8894
HEALTH_POLL_INTERVAL = 10  # seconds between automatic health checks
LOG_DIR = os.path.expanduser("~/little_timmy/little_timmy_os/logs")

# Little Timmy's dashboard for proxying metrics
TIMMY_WS_URL = "ws://localhost:8893/ws"
TIMMY_METRICS_URL = "http://localhost:8893/api/metrics"
TIMMY_CONVERSATION_URL = "http://localhost:8893/api/conversation"

# --- Conversation LLM Model Selection ---
MODELS_DIR = "/home/gearscodeandfire/models"
LLAMA_SERVER_BIN = "/home/gearscodeandfire/llama.cpp/build-vulkan/bin/llama-server"
CONVERSATION_PORT = 8081

CONVERSATION_MODELS = {
    "qwen2.5-7b": {
        "name": "Qwen 2.5 7B Q4",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "params": "-ngl 99 -c 8192 -np 1",
    },
    "llama3.2-3b": {
        "name": "Llama 3.2 3B Q4",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "params": "-ngl 99 -c 8192 -np 1",
    },
    "llama3.1-8b": {
        "name": "Llama 3.1 8B Q4",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "params": "-ngl 99 -c 8192 -np 1",
    },
    "mistral-7b": {
        "name": "Mistral 7B v0.3 Q4",
        "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "params": "-ngl 99 -c 8192 -np 1",
    },
}

# Track which model is currently loaded
current_conversation_model = "llama3.2-3b"

SERVICES = {
    "postgresql": {
        "name": "PostgreSQL",
        "port": 5432,
        "health_url": None,  # TCP check only
        "systemd": "postgresql",
        "description": "Memory database (pgvector + pg_trgm)",
    },
    "ollama": {
        "name": "Ollama",
        "port": 11434,
        "health_url": "http://localhost:11434/api/tags",
        "systemd": "ollama",
        "description": "Embeddings (nomic-embed-text, 768-dim)",
    },
    "conversation_llm": {
        "name": "Llama 3.2 3B Q4",
        "port": 8081,
        "health_url": "http://localhost:8081/health",
        "systemd": None,
        "start_cmd": None,  # built dynamically from current_conversation_model
        "description": "Conversation LLM",
    },
    "whisper": {
        "name": "whisper.cpp",
        "port": 8891,
        "health_url": "http://localhost:8891/health",
        # Delegate to systemd so LT-OS Start/Stop matches the systemd-managed
        # process and doesn't shell-spawn a duplicate that races for :8891.
        # Same dual-manager fix pattern used for little-timmy.service on
        # 2026-05-05. Note: turning whisper OFF will make LT crash-loop on
        # STT ConnectError until whisper is back — separate bug to harden
        # LT against missing STT.
        "systemd": "whisper-server.service",
        # start_cmd retained as reference for if the unit is ever removed.
        "start_cmd": (
            "/home/gearscodeandfire/whisper-cpp/build/bin/whisper-server "
            "-m /home/gearscodeandfire/whisper-cpp/models/ggml-base.en.bin "
            "--host 0.0.0.0 --port 8891 -t 4 -l en"
        ),
        "description": "Speech-to-text (GPU)",
    },
    "qwen36": {
        "name": "Qwen3.6 brain (:8083)",
        "port": 8083,
        "health_url": "http://localhost:8083/health",
        "systemd": "qwen36-server.service",
        "start_cmd": None,
        "description": (
            "Brain server: fact extraction (thinking=on) + DWU router first "
            "tier + local Claude Code. Shared with non-LT consumers."
        ),
    },
    "qwen36_vision": {
        "name": "Qwen3.6 vision (:8084)",
        "port": 8084,
        "health_url": "http://localhost:8084/health",
        "systemd": "qwen36-vision-server.service",
        "start_cmd": None,
        "description": (
            "Vision-dedicated Qwen3.6 instance with mmproj-BF16 attached. "
            "Used by LT vision/analyzer.py (TIMMY_VISION_URL=:8084). "
            "Stopping this disables Timmy's scene captioning."
        ),
    },
    "little_timmy": {
        "name": "Little Timmy",
        "port": 8893,
        "health_url": "http://localhost:8893/api/health",
        # Delegate launch/kill to systemd so the unit Environment vars
        # (XDG_RUNTIME_DIR, PULSE_SERVER) reach LT and PipeWire-routed
        # TTS actually plays. Subprocess-spawning bypassed those and
        # produced silent audio. Going through the unit also keeps a
        # single source of truth and avoids dual-manager port contention.
        "systemd": "little-timmy.service",
        # start_cmd retained as reference. Not used while systemd is set;
        # if the unit ever goes away, flip systemd back to None to re-enable.
        "start_cmd": (
            "/home/gearscodeandfire/little_timmy/.venv/bin/python "
            "/home/gearscodeandfire/little_timmy/main.py"
        ),
        "start_cwd": "/home/gearscodeandfire/little_timmy",
        "description": "Voice assistant orchestrator",
    },
}


def get_conversation_start_cmd(model_id: str | None = None) -> str:
    """Build the llama-server launch command for the given model."""
    mid = model_id or current_conversation_model
    model = CONVERSATION_MODELS[mid]
    return (
        f"{LLAMA_SERVER_BIN} "
        f"-m {MODELS_DIR}/{model['file']} "
        f"--host 0.0.0.0 --port {CONVERSATION_PORT} {model['params']}"
    )
TIMMY_BASE_URL = "http://localhost:8893"

# streamerpi (Little Timmy's body - Raspberry Pi 4)
STREAMERPI_URL = "https://192.168.1.110:8080"
