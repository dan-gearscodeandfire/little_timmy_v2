"""Detect whether a user utterance is asking a visual question.

Simple keyword/pattern approach. Returns True if the user is asking
about something that requires fresh visual input.

2026-05-08: OCR-specific patterns ("read this", "what does it say",
"what's on the screen/sign/label") were removed. Owner does not need
OCR; the camera resolution stays at 320x180 across all paths and we
no longer escalate to HIGH_RES on visual questions. The matcher now
covers scene/people/clothing/object questions only.
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
    r"\bhow do i look\b",
    r"\bwhat.*\bshirt\b",
    r"\bwhat.*\bjacket\b",
    r"\bwhat.*\bhat\b",
    r"\bwhat.*\bglasses\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in _VISUAL_PATTERNS]

# Patterns where the question presupposes the USER (or something they're
# presenting) is in the camera frame -- "what am I wearing", "what's on my
# shoulder", "how do I look", "can you see me". These are the only visual
# questions the averted-gaze guard applies to: if the head is looked away,
# answering them confabulates. Scene questions ("what do you see", "describe
# the room") are deliberately excluded -- they're answerable from any frame.
_SELF_REF_PATTERNS = [
    r"\b(am i|do i|how do i)\b",
    r"\bwhat\b.{0,20}\bi\b.{0,15}\b(wearing|holding|doing|making|eating|drinking)\b",
    r"\b(my|me)\b.{0,20}\b(wearing|holding|shoulder|hand|shirt|jacket|hat|glasses|face|hair|look|looking|holding)\b",
    r"\b(on|in)\b.{0,10}\bmy\b",          # "on my shoulder", "in my hand"
    r"\bwhat('?s| is| am i)?\b.{0,15}\bon me\b",
    r"\bcan you see me\b",
    r"\bdo you see me\b",
    r"\bhow do i look\b",
    r"\bwhat.*\b(my)\b.*\b(shirt|jacket|hat|glasses)\b",
]
_self_ref_compiled = [re.compile(p, re.IGNORECASE) for p in _SELF_REF_PATTERNS]


def is_visual_question(text: str) -> bool:
    """Return True if the utterance is asking a visual question."""
    for pattern in _compiled:
        if pattern.search(text):
            return True
    return False


def is_self_referential_visual_question(text: str) -> bool:
    """Return True if the question presupposes the user is in the camera frame.

    Used by the averted-gaze guard: these questions ("what am I wearing",
    "what's on my shoulder", "how do I look") can only be answered honestly
    when the head is actually aimed at the user. Scene questions ("what do you
    see") return False here so they stay answerable from any frame.
    """
    for pattern in _self_ref_compiled:
        if pattern.search(text):
            return True
    return False
