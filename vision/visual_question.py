"""Detect whether a user utterance is asking a visual question.

Simple keyword/pattern approach. Returns True if the user is asking
about something that requires fresh visual input.
"""

import re

# Patterns that strongly indicate a visual question
_VISUAL_PATTERNS = [
    r"\bwhat\b.{0,20}\b(see|seeing|look|looking at)\b",
    r"\bwhat\b.{0,20}\b(wear|wearing|holding|doing|making|eating|drinking)\b",
    r"\bwhat\b.{0,15}\bcolor\b",
    r"\bwhat\b.{0,15}\b(is|are)\b.{0,10}\b(this|that|those|these)\b",
    r"\bhow\b.{0,10}\b(many|much)\b.{0,15}\b(people|person|finger|hand)\b",
    r"\bwho\b.{0,10}\b(is|are)\b.{0,10}\b(here|there|that|this)\b",
    r"\b(describe|tell me about)\b.{0,10}\b(what|scene|room|me)\b",
    r"\bcan you see\b",
    r"\bdo you see\b",
    r"\blook at\b",
    r"\bam i\b.{0,15}\b(wearing|holding|doing)\b",
    r"\b(read|what does)\b.{0,10}\b(say|it say|the sign|the screen|the label)\b",
    r"\bhow do i look\b",
    r"\bwhat.*\bshirt\b",
    r"\bwhat.*\bjacket\b",
    r"\bwhat.*\bhat\b",
    r"\bwhat.*\bglasses\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in _VISUAL_PATTERNS]


def is_visual_question(text: str) -> bool:
    """Return True if the utterance is asking a visual question."""
    for pattern in _compiled:
        if pattern.search(text):
            return True
    return False
