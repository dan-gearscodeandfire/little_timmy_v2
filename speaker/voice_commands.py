"""Voice-command intent detection for live speaker enrollment.

Detects when the user asks Little Timmy to re-enroll a voiceprint mid-
conversation (Trigger 2 in speaker.identifier). Cheap regex pre-filter
modeled on feedback.detector. The orchestrator calls
``detect_reenroll_intent(user_text, default_speaker)`` in the response-
finalize hook and, on a hit, fires ``SpeakerIdentifier.start_reenrollment``.
"""

import re
from typing import Optional

# Phrases that indicate "do something to a voiceprint".
_INTENT_PATTERNS = [
    r"\b(?:re-?enroll|relearn|retrain|update|refresh|tune|tighten|recalibrate)\b"
    r"[^.?!]{0,40}\b(?:voice ?print|voice|ear|voice ?id)\b",
    r"\b(?:learn|hear) (?:my|his|her|their)? ?voice (?:again|better|fresh)\b",
    r"\b(?:redo|re-?do) (?:my|his|her|their)? ?voice ?(?:print)?\b",
    # Permissive: "re-enroll" alone is rare enough in normal speech that it's
    # itself a strong signal -- catches "re-enroll thea" without a voice keyword.
    r"\bre-?enroll\b",
]

# Name extraction patterns: "re-enroll <name>", "<name>'s voice".
_NAME_PATTERNS = [
    r"\b(?:re-?enroll|relearn|retrain|update|refresh)\s+([a-z][a-z0-9_-]{1,31})\b",
    r"\b([a-z][a-z0-9_-]{1,31})'?s voice ?(?:print)?\b",
]

# Pronouns that mean "the current speaker" rather than a literal name.
_PRONOUNS = {"my", "your", "the", "his", "her", "their", "our", "this", "that"}

_INTENT_RE = re.compile("|".join(_INTENT_PATTERNS), re.IGNORECASE)


def detect_reenroll_intent(user_text: str,
                           default_speaker: Optional[str]) -> Optional[str]:
    """If the text is a re-enrollment request, return the target speaker name.

    Returns:
      None if the text is not a re-enrollment request.
      A lowercase name string otherwise. Falls back to ``default_speaker``
      when the request uses a pronoun ("my voice") rather than a literal name.

    The caller is responsible for verifying the returned name is actually a
    known speaker before opening the collection window.
    """
    if not user_text:
        return None
    if not _INTENT_RE.search(user_text):
        return None

    for pat in _NAME_PATTERNS:
        m = re.search(pat, user_text, re.IGNORECASE)
        if not m:
            continue
        candidate = m.group(1).lower()
        if candidate in _PRONOUNS:
            continue
        return candidate

    if default_speaker and default_speaker.lower() not in _PRONOUNS:
        return default_speaker.lower()
    return None
