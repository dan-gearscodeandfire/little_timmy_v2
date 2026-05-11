"""Render the active mood snippet for injection into the ephemeral system prompt.

The X (engagement) and Y (warmth) snippets are stitched compositionally so
the prompt only carries the active cell of the 3x3 grid. Total ~50 tokens
when both axes are non-neutral, less when either is neutral.
"""

from __future__ import annotations

from persona.state import MoodState


_X_SNIPPETS = {
    -1: "Engagement: Bored. Flat, disengaged, minimal effort. Examples: \"Sure, Dan.\" \"If you say so.\"",
    0:  "Engagement: Neutral. Answer the question, no more, no less.",
    1:  "Engagement: Reluctantly interested. Catch yourself caring, then deflect — but the spark shows.",
}

_Y_SNIPPETS = {
    -1: "Tone: Mean. Sharp, cutting, Zorak-style sarcasm. Jab at Dan's competence or life choices.",
    0:  "Tone: Neutral. Straight delivery, no attitude.",
    1:  "Tone: Begrudgingly nice. Helpful, but make sure he knows you're doing him a favor.",
}


def render(state: MoodState) -> str:
    """Return the mood block as a single string ending without a trailing newline."""
    parts = ["MOOD:"]
    parts.append("- " + _X_SNIPPETS[state.x])
    parts.append("- " + _Y_SNIPPETS[state.y])
    return "\n".join(parts)
