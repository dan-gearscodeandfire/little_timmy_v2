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


# Two length-tiered pools, selected by how slow we expect the turn to be (Dan
# 2026-06-30, after measuring filler overrun). The asymmetry that drives the
# split: a LONG clip on a turn that turns out fast DELAYS the answer (it sits in
# the serial playback queue ahead of the ready reply — measured ~0.5-2.0s of
# overrun on normal turns); a SHORT clip on a turn that turns out slow merely
# UNDER-fills (the clip ends, a little quiet, then the answer plays the instant
# it's ready — never delayed). So we bias SHORT and only go LONG when we're
# CONFIDENT the turn is slow. The one high-precision "slow" signal available for
# free at fire time is a visual question landing on a stale frame, which forces a
# blocking VLM capture (+2.4-2.85s measured) — see the call site in main.py.

# LONG (~1.75-2.93s): the original 2026-06-25 natural beats. Used ONLY on turns
# we know will block on a fresh VLM capture, where even the longest clip can't
# outrun the answer. Tone stays dry/deadpan. Order/count don't matter.
LONG_FILLERS: tuple[str, ...] = (
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

# SHORT (~0.65-0.8s, busy <=1.2s): the DEFAULT for normal turns. The measured
# filler-start->answer-ready gap on no-VLM turns is min ~1.0s / median ~2.0s, so
# to NEVER delay the answer (Dan 2026-06-30) a clip's busy time (audio + 0.4s
# cooldown) must stay under ~1.0-1.2s — worst-case overrun then ~0.1s on the
# very fastest turn, ~0 on a median turn. These are deliberately curt: on a
# no-VLM turn the real answer is only ~1-2s away, so a brief acknowledgement is
# the right register; the long "let me think about that" beats (LONG_FILLERS)
# would land absurdly before a one-second answer and would also overrun it.
# Curt acknowledgements, NOT the pre-2026-06-25 "Eh./Huh." nonsense tics.
SHORT_FILLERS: tuple[str, ...] = (
    "Alright.",
    "One sec.",
    "Hang on.",
    "Right then.",
    "Okay.",
    "Sure.",
)

# Union — render_fillers freezes ALL of these to .wav and prewarm_fillers loads
# ALL of them; pick() routes to the right pool per turn.
FILLERS: tuple[str, ...] = LONG_FILLERS + SHORT_FILLERS

DEFAULT_RATE = 0.5
MIN_USER_WORDS = 4


def should_fire(user_text: str, rate: float = DEFAULT_RATE) -> bool:
    """Coin-flip + curt-prompt guard. False means: skip the filler this turn."""
    if not user_text:
        return False
    if len(user_text.split()) < MIN_USER_WORDS:
        return False
    return random.random() < rate


def pick(long: bool = False) -> str:
    """Uniform random pick. long=True -> the 2-3s pool (only when the turn is
    known-slow, e.g. a visual question forcing a fresh VLM capture); long=False
    (default, fail-safe) -> the ~1.1s pool for normal turns."""
    return random.choice(LONG_FILLERS if long else SHORT_FILLERS)
