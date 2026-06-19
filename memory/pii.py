"""Deterministic sensitivity classifier for facts (PII gating).

Runs at fact-creation time inside memory.facts.store_fact (the single chokepoint
both writers pass through: the extraction pipeline AND the :8092 tool-call
classifier). Sets facts.sensitive / facts.pii_category. The consumer
(conversation.turn.RetrievalGatherer) drops sensitive facts from prompt
injection while the guest/privacy gate is active, so the brain never receives
them and cannot speak them via TTS in front of guests.

Heuristic v1 (2026-06-18, Dan). Categories are keyed off the normalized
predicate + a value regex; tune the token sets below as needed. Names,
relationships (other than minor children), pets, and projects are intentionally
NOT sensitive -- Timmy addresses guests by name freely.
"""
import re
import config

# Categories (stored verbatim in facts.pii_category):
CONTACT = "contact"
LOCATION = "location"
FINANCIAL = "financial"
HEALTH_CRED = "health_credentials"
FAMILY_MINOR = "family_minor"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,}\d(?!\d)")
_MONEY_RE = re.compile(r"\$\s?\d|\b\d+\s?(?:dollars|usd|k\b|grand)\b")
# Minor children: gate any fact that reveals a child/daughter, regardless of
# subject. \bson\b excluded (too many false matches: "person", "reason", names).
_CHILD_RE = re.compile(r"\b(daughter|daughters|child|children|kid|kids)\b")

# Per-category word-boundary patterns (matched against the predicate, or the
# predicate+value blob where noted). Word boundaries avoid short-token false
# positives like 'pin' matching 'pinocchio' or 'cell' matching 'excellent'.
# Underscores are word chars, so the alternatives also cover snake_case
# predicates (e.g. \bmobile\b matches 'mobile' in 'mobile_number').
# 'number' alone is too broad ("youtube video number") -- rely on phone/cell/
# mobile + the value phone regex for actual numbers.
_CONTACT_PRED_RE = re.compile(
    r"\b(e-?mail|phone|phone number|cell|mobile|mailing|handle|instagram|"
    r"twitter|snapchat|telegram|discord|contact)\b")
# Require 'lives' or 'live in/at/with' -- bare 'live' matched 'live streaming'.
_LOCATION_PRED_RE = re.compile(
    r"\b(lives|live (?:in|at|with)|home address|home|resid\w*|address|located|"
    r"location|neighborhood|zip|works? at|workplace|office)\b")
_FINANCIAL_RE = re.compile(
    r"\b(salary|income|wage|bank|account|routing|credit card|debit|"
    r"net[_ ]?worth|mortgage|debt|owes?|loan|venmo|paypal)\b")
# NOTE: 'therapy'/'therapist' deliberately excluded -- "couples therapist" is the
# supervisor-mode alias, not medical PII.
_HEALTH_CRED_RE = re.compile(
    r"\b(medication|medicine|medical|diagnos\w*|illness|disease|prescription|"
    r"health condition|allerg\w*|password|passcode|passphrase|pin|ssn|"
    r"social security|api[_ ]?key|secret|credential|token|login)\b")


def classify_sensitivity(subject: str, predicate: str, value: str):
    """Return (sensitive: bool, category: str | None) for a fact.

    Order matters: most-specific / highest-risk categories first so the stored
    pii_category is the most meaningful single label.
    """
    s = (subject or "").strip().lower()
    # Normalize snake_case to spaces so \b word boundaries fire on predicates
    # like 'lives_in' / 'takes_medication' / 'mobile_number'.
    p = (predicate or "").strip().lower().replace("_", " ")
    v = (value or "").strip().lower().replace("_", " ")
    blob = f"{p} {v}"

    # 1. Minor children (Dan's daughters): subject is a known child, or the fact
    #    text reveals a child/daughter relationship.
    daughters = {n.strip().lower() for n in getattr(config, "DAUGHTER_NAMES", ())}
    if s in daughters or _CHILD_RE.search(blob):
        return True, FAMILY_MINOR

    # 2. Contact info (email / phone / handle).
    if _EMAIL_RE.search(value) or _PHONE_RE.search(value) or _CONTACT_PRED_RE.search(p):
        return True, CONTACT

    # 3. Financial.
    if _FINANCIAL_RE.search(blob) or _MONEY_RE.search(v):
        return True, FINANCIAL

    # 4. Health / credentials.
    if _HEALTH_CRED_RE.search(blob):
        return True, HEALTH_CRED

    # 5. Precise location (city-level included per Dan).
    if _LOCATION_PRED_RE.search(p):
        return True, LOCATION

    return False, None
