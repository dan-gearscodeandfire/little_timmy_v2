"""Subscriber-hype: regex-triggered celebration lines (2026-08-18).

Dan announces a new channel subscriber out loud ("we have a new subscriber!",
"somebody just subscribed") and Timmy fires back a randomized victory line.
Pure-lexical detection — no classifier call — so it runs even when the Qwen3-4B
tool classifier is toggled off, and costs nothing on the miss path.

Detection is deliberately ANNOUNCEMENT-SHAPED, not keyword-shaped. The bare
phrase "new subscriber" must NOT trigger (Dan: "we don't have new subscribers"
is a real utterance). Three gates, applied per sentence:

  1. an announcement pattern must match (positive phrasings only);
  2. a negation/hypothetical word anywhere earlier in the same sentence kills
     it ("we don't have...", "if we got...", "I wish we had...");
  3. a leading question auxiliary kills it ("do we have a new subscriber?").

False negatives are acceptable; false positives are the failure mode Dan
called out. When in doubt, stay silent — Dan can re-announce with a canonical
phrasing.

Lines live in prompts/subscriber_hype_lines.txt, re-read on every hit so they
can be edited live (same pattern as store_fact_acks.txt). The picker avoids
repeating the previous line.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

log = logging.getLogger(__name__)

_LINES_PATH = (Path(__file__).resolve().parent.parent
               / "prompts" / "subscriber_hype_lines.txt")

_FALLBACK_LINE = "Yeah, dude! We rock!"

# Announcement phrasings. Matched against a lowercased, apostrophe-normalized
# single sentence. Optional words widen each stem to natural variants of the
# five canonical triggers without opening up bare "new subscriber".
_TRIGGERS = [
    # "we have a new subscriber" / "we've got another new subscriber" /
    # "we just got a new subscriber" / "we got ourselves a new subscriber"
    re.compile(r"\bwe(?:'ve)?\s+(?:just\s+)?(?:have|got|gained|picked\s+up)\s+"
               r"(?:ourselves\s+)?(?:another\s+|a\s+|one\s+more\s+)?"
               r"(?:brand[\s-]*new\s+|new\s+)subscribers?\b"),
    # "look, new subscriber" / "look at that, a new subscriber" / "looky here..."
    re.compile(r"\blook\w*\b[^.!?]{0,25}?\bnew\s+subscribers?\b"),
    # "another new subscriber"
    re.compile(r"\banother\s+(?:brand[\s-]*new\s+)?new\s+subscribers?\b"),
    # "somebody just subscribed" / "someone subscribed"
    re.compile(r"\bsome(?:body|one)\s+(?:just\s+)?subscribed\b"),
    # "hey hey new subscriber" / "hey, hey, a new subscriber"
    re.compile(r"\bhey[\s,!]+hey\b[^.!?]{0,15}?\bnew\s+subscribers?\b"),
    # "new subscriber alert"
    re.compile(r"\bnew\s+subscriber\s+alert\b"),
]

# Negation / hypothetical anywhere in the sentence -> reject. Over-broad on
# purpose (e.g. "no way, a new subscriber!" is a lost trigger): false
# negatives are cheap, false positives are the bug.
_NEGATION = re.compile(
    r"\b(?:don't|do\s+not|not|no|never|haven't|hasn't|hadn't|didn't|doesn't|"
    r"ain't|zero|none|nobody|without|if|unless|until|before|wish|wished|"
    r"hope|hoping|want|wanted|wanna|need|needed)\b")

# Leading question auxiliary ("do we have a new subscriber") -> reject.
_QUESTION_LEAD = re.compile(
    r"^\s*(?:do|does|did|have|has|had|are|is|was|were|will|would|can|"
    r"could|should|any\s+chance)\b")


def detect(user_text: str) -> bool:
    """True iff `user_text` contains a positive new-subscriber announcement."""
    if not user_text:
        return False
    text = user_text.lower().replace("’", "'")
    if "subscrib" not in text:  # free early-out for ~every turn
        return False
    for sentence in re.split(r"[.!?]+", text):
        if "subscrib" not in sentence:
            continue
        if _QUESTION_LEAD.match(sentence):
            continue
        if not any(pat.search(sentence) for pat in _TRIGGERS):
            continue
        # Whole-sentence negation scan: stricter than scoping to words before
        # the match, and that strictness is the point.
        if _NEGATION.search(sentence):
            continue
        return True
    return False


def _load_lines() -> list[str]:
    """Re-read the celebration lines each call so they can be edited live."""
    try:
        raw = _LINES_PATH.read_text(encoding="utf-8")
    except Exception:
        log.exception("[subscriber_hype] lines file unreadable; using fallback")
        return [_FALLBACK_LINE]
    lines = [ln.strip() for ln in raw.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    return lines or [_FALLBACK_LINE]


_last_line: str | None = None


def pick_line() -> str:
    """Random line, never the same one twice in a row (when >1 exists)."""
    global _last_line
    lines = _load_lines()
    candidates = [ln for ln in lines if ln != _last_line] or lines
    _last_line = random.choice(candidates)
    return _last_line
