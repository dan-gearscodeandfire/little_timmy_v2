"""Brief filler-word selection for the THINKING phase.

Pre-rendered short audio clips (<=400 ms) play between STT-finalize and
the LLM's first sentence so the silent gap reads as personality rather
than latency. Picked at a 50% rate; skipped on very short prompts where
a filler would over-pad a curt reply ("Hey" -> "Hmm, hi" is too much).
"""

import random

# Core 10. Tone calibrated to the deadpan-skeleton persona (mean+interested
# mood pinning). Dan's call after the 2026-05-14 design discussion -- the
# sardonic-set extras like "Yeah, no" / "Riiight" deferred to a later pass
# so the first version doesn't front-load too much attitude.
FILLERS: tuple[str, ...] = (
    "Eh.",
    "Heh.",
    "Huh.",
    "Right.",
    "Oh.",
    "Well\u2026",
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
