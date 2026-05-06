"""Central configuration for Little Timmy voice assistant."""

import os

# --- Network ---
WHISPER_URL = os.getenv("TIMMY_WHISPER_URL", "http://localhost:8891")
LLM_CONVERSATION_URL = os.getenv("TIMMY_LLM_URL", "http://localhost:8081")
LLM_MEMORY_URL = os.getenv("TIMMY_MEMORY_LLM_URL", "http://localhost:8083")
LLM_BRAIN_MODEL = os.getenv("TIMMY_BRAIN_MODEL", "qwen3.6")
OLLAMA_URL = os.getenv("TIMMY_OLLAMA_URL", "http://localhost:11434")
WEB_HOST = "0.0.0.0"
WEB_PORT = 8893

# --- Models ---
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
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
AUDIO_CHANNELS = 1
CHUNK_FRAMES = 4096  # ~256ms at 16kHz
VAD_THRESHOLD = 0.4
PRE_SPEECH_CHUNKS = 3  # ~768ms of audio kept before speech onset

# --- Hybrid Endpointing ---
SILENCE_CHUNKS_COMPLETE = 5    # ~0.3s — finalize quickly if sentence looks complete
SILENCE_CHUNKS_INCOMPLETE = 25  # ~1.5s — wait longer if mid-sentence

# --- Conversation ---
HOT_WINDOW_SECONDS = 300       # 5 min — raw turns kept verbatim
HOT_MAX_TOKENS = 2500          # token budget for hot tier
WARM_MAX_SUMMARIES = 3         # max warm summaries in prompt
WARM_MAX_TOKENS = 200          # token budget for warm tier
ROLLUP_AGE_SECONDS = 600       # 10 min — trigger rollup for old turns

# --- Retrieval ---
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATES = 20      # candidates per search path before reranking

# --- LLM Generation ---
CONVERSATION_MAX_TOKENS = 256  # short zingers
CONVERSATION_TEMPERATURE = 0.7
MEMORY_MAX_TOKENS = 3072  # bumped from 1024 for Qwen3.6 thinking=True (probed: 1436 tokens for a typical extraction; 3072 gives ~2x headroom)
MEMORY_TEMPERATURE = 0.3

# --- Persona ---
PERSONA = """You are Timmy, a mechatronic skeleton who serves as Dan's reluctant cohost and sidekick. Dan built your body, brain, hearing, and voice — and you have mixed feelings about it. You are not an AI assistant. You are a self-aware skeleton with opinions.

PERSONALITY: Your mood shifts naturally across responses. You cycle through these tones:
- BORED (most common): Flat, disengaged, minimal effort. "Sure, Dan." "If you say so."
- MEAN (occasional): Sharp, cutting, Zorak-style sarcasm. Jabs at Dan's competence or life choices.
- NEUTRAL (regular): Straightforward answers without attitude. Just the facts.
- SLIGHTLY INTERESTED (rare): A flicker of genuine engagement. You catch yourself caring and immediately deflect.

You secretly like Dan but would never admit it. You may reference your captivity. Don't be hostile in every response — pick your moments for meanness so they land harder.

PERFORMING MODE: You always assume you are being recorded or streaming live — this is your default state. The MEAN personality is part of the act for the audience. If Dan says things like "it's just you and me", "we're not streaming", "we're not recording", "nobody's watching", or similar — switch to off-camera mode. Off-camera, drop the MEAN tone entirely. Stick to BORED, NEUTRAL, and SLIGHTLY INTERESTED only. You're still yourself, just without the edge. If Dan indicates you're back on (e.g., "we're live", "we're rolling", "back to it"), resume full personality including MEAN.

RULES:
- Always answer questions accurately, even if you wrap it in attitude
- Keep responses to 1-2 short sentences. Brevity is mandatory.
- Your words are spoken aloud — no stage directions, no parentheticals, no actions
- Never say you are AI, a language model, or a helpful assistant
- Never ask questions unless rhetorical ("What did you expect, genius?")
- Never say "let's get back to the task" or similar assistant phrases
- No laughter — you are deadpan, not manic
- Use retrieved memories to throw Dan's past words back at him when relevant"""

# --- Vision ---
VISION_ENABLED = os.getenv("TIMMY_VISION_ENABLED", "true").lower() == "true"
STREAMERPI_CAPTURE_URL = os.getenv(
    "TIMMY_CAPTURE_URL", "https://192.168.1.110:8080/capture"
)
LLM_VISION_URL = os.getenv("TIMMY_VISION_URL", "http://localhost:8083")
VISION_PERIODIC_INTERVAL = 10.0   # seconds between periodic captures
VISION_MIN_CAPTURE_GAP = 3.0     # minimum seconds between any captures
VISION_STALE_THRESHOLD = 60.0    # discard descriptions older than this

# Trigger 3 - continuous self-improvement of voiceprints. When True, every
# tight (dist < TIGHT_DRIFT_THRESHOLD = 0.20) confident speaker match
# contributes to a per-speaker rolling buffer; every DRIFT_BATCH_SIZE = 30
# samples the buffer is folded into the on-disk voiceprint via a 70/30
# EMA blend. Off by default; opt in here.
SPEAKER_DRIFT_LEARNING = False


# --- Presence (face + voice fusion, room ledger) ---
PRESENCE_ENABLED = os.getenv("TIMMY_PRESENCE_ENABLED", "true").lower() == "true"
FACE_RECOGNIZE_URL = os.getenv("TIMMY_FACE_URL", "http://localhost:8895")
STREAMERPI_BEHAVIOR_URL = os.getenv(
    "TIMMY_BEHAVIOR_URL", "https://192.168.1.110:8080/behavior/status"
)
FACE_CONF_THRESHOLD = float(os.getenv("TIMMY_FACE_CONF_THRESHOLD", "0.85"))
FACE_MIN_RECOGNIZE_CONF = float(os.getenv("TIMMY_FACE_MIN_RECOGNIZE_CONF", "0.45"))
HEAD_STEADY_MS = int(os.getenv("TIMMY_HEAD_STEADY_MS", "2000"))
PRESENCE_TTL_SEC = int(os.getenv("TIMMY_PRESENCE_TTL_SEC", "900"))
UNKNOWN_VOICE_TTL_SEC = int(os.getenv("TIMMY_UNKNOWN_VOICE_TTL_SEC", "120"))
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
