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

# Beyond the length split, the pools are also REGISTER-split (Dan 2026-06-30):
# what fits a declarative statement ("I see.", "Got it.") is wrong before/while
# answering a question ("Let me see."), and vice-versa. register() below is a
# free lexical classifier (question vs statement; default statement). Tool/command
# turns are deliberately OUT of scope here — they route through tool_router with
# their own flow we'll revisit separately, so they fall to the statement default.

# --- SHORT pools (~0.4-0.9s, busy <=~1.2s): the DEFAULT for normal turns. The
# measured filler-start->answer-ready gap on no-VLM turns is min ~1.0s / median
# ~2.0s, so to NEVER delay the answer a clip's busy time (audio + 0.4s cooldown)
# stays ~<=1.2s. Curt acknowledgement/thinking tokens, NOT the pre-2026-06-25
# "Eh./Huh." dismissive tics. ---

# Declarative reply: you TOLD Timmy something; he acknowledges / takes it in.
SHORT_STATEMENT: tuple[str, ...] = (
    "I see.",
    "Got it.",
    "Right.",
    "Ah.",
    "Noted.",
    "Mm, okay.",
    "Hmmmm.",
    "Um.",
)

# Interrogative reply: you ASKED Timmy something; he's about to answer.
SHORT_QUESTION: tuple[str, ...] = (
    "Hmm.",
    "Let me see.",
    "Good question.",
    "Let me think.",
    "Right.",
    "Okay.",
    "Ugh.",
)

# --- LONG pools (~1.5-2.9s): fire ONLY when a turn will block on a fresh VLM
# capture (+2.4-2.85s measured), so even the longest clip can't outrun the
# answer. Because LONG is EXCLUSIVELY a vision-capture beat, the phrasing is
# "let me LOOK", not "let me think". ---

# Declarative + visual: you SHOWED Timmy something ("look at this", "check out
# my X"). Rare today — the detector mostly catches visual *questions*, not
# "look at this" statements — but ready if visual detection widens.
LONG_STATEMENT: tuple[str, ...] = (
    "Hmm, let me take a look at that.",
    "Okay, let me have a look.",
    "Hold on, let me get a proper look.",
    "Let me see what you've got there.",
    "Alright, let me take this in.",
)

# Interrogative + visual: you ASKED a visual question ("what am I holding?",
# "what do you see?"). The live LONG case.
LONG_QUESTION: tuple[str, ...] = (
    "Let me get a good look at that.",
    "Let me take a closer look.",
    "Okay, let me check what I'm seeing.",
    "Hold on, let me focus on that.",
    "Give me a second to look that over.",
)

# Union — render_fillers freezes ALL of these to .wav and prewarm_fillers loads
# ALL of them; pick() routes to the right pool per turn. (Strings shared across
# pools, e.g. "Right.", are content-addressed -> one .wav, rendered/loaded once.)
FILLERS: tuple[str, ...] = (
    SHORT_STATEMENT + SHORT_QUESTION + LONG_STATEMENT + LONG_QUESTION
)

DEFAULT_RATE = 0.5
MIN_USER_WORDS = 4

# Question cues for register(): wh-words + aux-inversion openers. STT rarely
# emits a "?", so we lean on these, not punctuation. Default is "statement".
_QUESTION_OPENERS = (
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "is ", "are ", "am ", "was ", "were ", "do ", "does ", "did ", "can ",
    "could ", "will ", "would ", "should ", "shall ", "may ", "might ",
    "have ", "has ", "had ", "is", "are",
)


def should_fire(user_text: str, rate: float = DEFAULT_RATE) -> bool:
    """Coin-flip + curt-prompt guard. False means: skip the filler this turn."""
    if not user_text:
        return False
    if len(user_text.split()) < MIN_USER_WORDS:
        return False
    return random.random() < rate


def register(user_text: str) -> str:
    """Free lexical register classifier: 'question' vs 'statement' (default).
    Question = ends with '?' OR opens with a wh-word / aux-inversion. STT drops
    punctuation, so openers carry most of the weight. A miss is cheap (you get a
    still-fitting token from the other pool), so we default to the common case."""
    if not user_text:
        return "statement"
    t = user_text.lower().lstrip()
    if user_text.rstrip().endswith("?"):
        return "question"
    if t.startswith(_QUESTION_OPENERS):
        return "question"
    return "statement"


def pick(long: bool = False, reg: str = "statement") -> str:
    """Uniform random pick from the pool for this (length, register). long=True
    is the vision-capture pool (only when a visual question/stale frame forces a
    fresh VLM call); reg comes from register(user_text)."""
    is_q = reg == "question"
    if long:
        pool = LONG_QUESTION if is_q else LONG_STATEMENT
    else:
        pool = SHORT_QUESTION if is_q else SHORT_STATEMENT
    return random.choice(pool)
