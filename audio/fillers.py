"""Filler-phrase selection for the THINKING phase.

Pre-rendered audio clips play between the moment we commit to a brain reply
and the LLM's first sentence, so the silent gap reads as a natural "let me
think" beat rather than dead latency. Picked at a 50% rate; skipped on very
short prompts where a filler would over-pad a curt reply.

2026-06-25 (Dan): phrases reworked to be longer + more natural. The old set
("Eh." / "Heh." / "Huh.") were terse interjections that landed as tics, not
thinking. These are short conversational thinking-beats that genuinely buy
time while the brain spins up. Commas synthesize as natural Piper pauses, so
they read well aloud.

2026-06-27 (Dan): the clips are now frozen, committed .wav assets (see
WAV_DIR / wav_path below), NOT synthesized at startup. After editing this
tuple, re-render the audio so the loader has matching files:

    .venv/bin/python -m audio.render_fillers
"""

import hashlib
import random
from pathlib import Path

# Frozen, pre-rendered filler audio lives here as committed .wav assets (Dan
# 2026-06-27). Rendered ONCE from the FILLERS tuple by audio/render_fillers.py
# with the skeletor_v1 voice, then loaded from disk at startup — so the clips
# are locked/curated and never silently re-synthesize across boots or if the
# Piper voice model changes. Filenames are content-addressed by the exact
# filler text, so the renderer and the loader (TTSEngine.prewarm_fillers) agree
# without a manifest lookup; order/count in the tuple don't matter.
WAV_DIR = Path(__file__).resolve().parent / "fillers_wav"


def wav_path(text: str) -> Path:
    """Deterministic .wav path for a filler string (content-addressed)."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return WAV_DIR / f"filler_{digest}.wav"


# Natural thinking-beats, biased toward the longer end (Dan 2026-06-25). Tone
# stays dry/deadpan to fit the skeleton persona but the job here is purely to
# mask the pre-first-token gap, so they're deliberately low-commitment (they
# precede an answer we don't have yet). Edit freely — order/count don't matter.
FILLERS: tuple[str, ...] = (
    "Hmm, let me think about that for a second.",
    "Okay, give me a moment here.",
    "Right, let me chew on that.",
    "Oh, hang on, let me think.",
    "Well, let me work that one out.",
    "Hmm, good question. Let me think.",
    "Let me dig into that for a sec.",
    "Okay, let me put this together.",
    "Give me just a moment.",
    "Let me see what I've got.",
)

DEFAULT_RATE = 0.5
MIN_USER_WORDS = 4


def should_fire(user_text: str, rate: float = DEFAULT_RATE) -> bool:
    """Coin-flip + curt-prompt guard. False means: skip the filler this turn."""
    if not user_text:
        return False
    if len(user_text.split()) < MIN_USER_WORDS:
        return False
    return random.random() < rate


def pick() -> str:
    """Uniform random pick from FILLERS."""
    return random.choice(FILLERS)
