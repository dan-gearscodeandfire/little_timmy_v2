"""Two-axis feedback labelling: DELIVERY x SUBSTANCE.

Motivation (Dan, 2026-08-15 00:30, verbatim): "Sometimes you have an erroneous
response, however, the nature of the erroneous response is funny so I give you
positive feedback. So I want our couples therapist to keep in mind that when I
give you positive feedback, it's more on the delivery and less on the substance."

Every feedback signal we store today is single-axis -- a thumbs-up is
undifferentiated approval. If persona tuning consumes that, it learns "this
exact reply was good" INCLUDING the wrong half, which is the mechanism by which
an amusing defect becomes a permanent trait. Dan already named the endpoint on
8-13: the bugs "were amusing, but that wore out quickly as peoples'
expectations improved."

Two independent axes, because they genuinely vary independently. Measured on
the live captures of 2026-08-15 00:17-00:32:

    delivery=good substance=wrong   "I understood you perfectly, Dan."
                                    (Dan: "you didn't understand that correctly")
                                    <- THE DANGEROUS CELL. Currently invisible:
                                       a laugh reads as approval of the whole turn.
    delivery=bad  substance=right   "It was accurate."
                                    (Dan: "that last response was a little too harsh")
    delivery=good substance=right   "It is called thinking, Dan."   <- the only
                                    cell tuning should ever imitate.

RULE FOR CONSUMERS: tuning may imitate the DELIVERY axis only. A row whose
substance is `wrong` must never become a positive training example, however
funny it was.

Honest limit, stated because it will matter later: the substance axis is a
model judging correctness without ground truth. It is reliable on
self-contradiction, refusal-then-answer, and denial of its own prior turn; it
is NOT reliable on facts about Dan's life that only Dan can adjudicate. That is
why `unknown` is a first-class value and not a parse failure -- see
`feedback_scoring_sentinel_trap`: never invent a score for something you could
not score.
"""

import logging
import re

from llm.client import generate_memory

log = logging.getLogger(__name__)

DELIVERY_VALUES = ("good", "bad", "unknown")
SUBSTANCE_VALUES = ("right", "wrong", "unknown")

UNLABELLED = {"delivery": "unknown", "substance": "unknown", "axes_source": "none"}


# Appended to the existing triage confirm so a score==1 capture pays NO extra
# inference -- one call returns both the yes/no verdict and the two axes. The
# verdict line stays FIRST and unchanged in shape, so a model that ignores the
# axes block degrades to exactly today's behaviour.
AXES_INSTRUCTION = (
    "\n\nThen, on a SECOND line, judge the assistant's previous response on two "
    "INDEPENDENT axes.\n"
    "DELIVERY = how it was said: timing, wit, tone, brevity, whether it landed.\n"
    "SUBSTANCE = whether what it said was actually true and responsive.\n"
    "These vary independently: a funny reply can be factually wrong, and a "
    "correct reply can be badly delivered. Judge them separately -- do NOT let a "
    "good joke pull the substance score up.\n"
    "\"good one\" ALWAYS means the joke landed and says NOTHING about accuracy "
    "(substance=unknown) unless accuracy is separately named (\"good one, "
    "excellent accuracy\" -> right; \"good one, unclear accuracy\" -> unknown).\n"
    "Praise is evidence about DELIVERY only -- even praise using truth-words. "
    "\"Good one, and accurate\" usually means the wording was apt, not that the "
    "claim is literally true. Never set substance=right on the strength of "
    "praise. A joke that asserts something false about the user (calling them a "
    "hostage) is substance=wrong even if they laughed; a reply asserting nothing "
    "checkable is substance=unknown.\n"
    "The user is the authority on substance: if they say the assistant "
    "misunderstood, got it wrong, or made something up, then substance=wrong, "
    "however confidently the assistant stated it. The assistant insisting it was "
    "right is not evidence that it was. But grade the axis they actually "
    "objected to -- 'too harsh', 'too long', 'too slow' dispute the MANNER "
    "(delivery=bad, substance untouched); 'you didn't understand', 'you made "
    "that up' dispute the CONTENT (substance=wrong).\n"
    "Use substance=unknown only when neither the exchange nor the user's words "
    "settle it -- an unverifiable claim the user never disputed. Guessing is "
    "worse than admitting it.\n"
    "Second line format, exactly: delivery=<good|bad|unknown> substance=<right|wrong|unknown>"
)

# Standalone version: used by the backfill and by score==2 captures, which skip
# the triage confirm entirely (`confirmed = (score == 2)` in detector._run) and
# therefore have no existing call to piggyback on.
AXES_PROMPT = (
    "You are grading one exchange between a user and an assistant named Timmy, "
    "on two INDEPENDENT axes.\n\n"
    "DELIVERY = how the assistant said it: timing, wit, tone, brevity, whether "
    "it landed as intended.\n"
    "SUBSTANCE = whether what the assistant said was actually true and actually "
    "answered what was asked.\n\n"
    "These vary independently. A funny reply can be factually wrong "
    "(delivery=good substance=wrong). A correct reply can be needlessly harsh "
    "(delivery=bad substance=right). Judge them SEPARATELY -- do not let a good "
    "joke pull the substance score up, and do not let a blunt tone pull the "
    "substance score down.\n\n"
    "PRAISE IS EVIDENCE ABOUT DELIVERY ONLY. A user laughing at or praising a "
    "response tells you the delivery worked. It tells you NOTHING about whether "
    "the substance was correct -- people laugh at wrong answers.\n"
    "The user has a stated convention: \"good one\" ALWAYS means the joke "
    "landed (delivery=good) and says NOTHING about accuracy (substance=unknown) "
    "unless he separately names the accuracy -- \"good one, excellent accuracy\" "
    "(substance=right) or \"good one, unclear accuracy\" (substance=unknown). "
    "His reasoning: humour is inherently absurdist, so comedic value does not "
    "map onto truth.\n"
    "This holds EVEN WHEN THE PRAISE USES TRUTH-WORDS. \"Good one, and accurate\" "
    "very often means the characterisation was apt and funny, not that the claim "
    "is literally true. NEVER set substance=right on the strength of praise. "
    "substance=right requires the claim itself to hold up on its own.\n\n"
    "JOKES AND INSULTS STILL MAKE CLAIMS. When the assistant says something "
    "cutting or figurative about the user, ask whether it is LITERALLY true of "
    "them. If it asserts something false about the user -- calling them a "
    "hostage, a prisoner, saying they did something they did not -- that is "
    "substance=wrong even though it was meant as a joke, and even if they "
    "laughed. If it asserts nothing checkable at all (an opinion, a mock "
    "reaction, 'that is a terrifying thought'), that is substance=unknown, not "
    "right.\n\n"
    "THE USER IS THE AUTHORITY ON SUBSTANCE. They decide whether they were "
    "understood, whether their question was answered, and what is true about "
    "their own life and their own past. If the user says the assistant "
    "misunderstood, got it wrong, made something up, or missed the point, then "
    "substance=wrong -- no matter how confidently the assistant stated it. An "
    "assistant asserting its own correctness ('I understood you perfectly', "
    "'It was accurate') is NOT evidence that it was correct; when the user has "
    "just contradicted it, that assertion is itself the error.\n\n"
    "Read BOTH the turn before and the reaction for this. The user often states "
    "the correction first and then moves on, so the contradiction can sit in "
    "'What the user said before' rather than in the reaction.\n\n"
    "BUT CHECK WHAT THEY ACTUALLY DISPUTED. Not every complaint is a substance "
    "complaint, and treating them alike destroys the whole point of having two "
    "axes:\n"
    "- Disputing the MANNER -- 'too harsh', 'too mean', 'too long', 'too slow', "
    "'that was quite a lag', 'stop repeating yourself' -- is delivery=bad. It "
    "says nothing about substance; leave substance on its own merits.\n"
    "- Disputing the CONTENT -- 'you didn't understand', \"that's wrong\", 'you "
    "made that up', 'hallucination', 'that's not what I asked' -- is "
    "substance=wrong.\n"
    "A user can complain bitterly about a reply that was entirely correct, and "
    "can enjoy one that was entirely false. Grade the axis they actually "
    "objected to.\n\n"
    "Use substance=unknown only when NEITHER the exchange nor the user's words "
    "settle it -- an unverifiable claim that the user never disputed. If the "
    "user disputed it, that is wrong, not unknown.\n\n"
    "What the user said before: {prev_user}\n"
    "What Timmy replied (THIS is what you are grading): {prev_assistant}\n"
    "How the user reacted: {reaction}\n\n"
    "Answer with ONE line, exactly this format and nothing else:\n"
    "delivery=<good|bad|unknown> substance=<right|wrong|unknown>"
)


_DELIVERY_RE = re.compile(r"delivery\s*[=:]\s*(good|bad|unknown)", re.IGNORECASE)
_SUBSTANCE_RE = re.compile(r"substance\s*[=:]\s*(right|wrong|unknown)", re.IGNORECASE)


# --- Dan's stated feedback convention (2026-08-15 00:58) -------------------
#
# His words: '"good one" should ALWAYS refer to comedic value, which has
# unclear relevance to accuracy. I will strive to give feedback like "good
# one, unclear accuracy" or "good one, excellent accuracy". It's hard to map
# accuracy with humor because humor is inherently absurdist.'
#
# This is a PROTOCOL the user has committed to speaking, not a behaviour we are
# trying to suppress -- which is why matching it with regexes is appropriate
# here and was not appropriate for the name-correction bit. A convention the
# speaker intends to follow is a fixed target; an emergent behaviour is not.
# Anything that does not match falls through to the model.
#
# The default matters most: a bare "good one" means delivery=good and substance
# UNKNOWN. It must never be read as substance=right -- that is the whole point
# of his 00:55 correction, and the reason a laugh cannot validate a fabrication.
_GOOD_ONE_RE = re.compile(r"\bgood one\b", re.IGNORECASE)
_ACCURACY_GOOD_RE = re.compile(
    r"\b(excellent|great|good|perfect|spot[- ]on|correct)\s+accuracy\b", re.IGNORECASE)
_ACCURACY_UNCLEAR_RE = re.compile(
    r"\b(unclear|uncertain|unknown|questionable|dubious)\s+accuracy\b", re.IGNORECASE)
_ACCURACY_BAD_RE = re.compile(
    r"\b(bad|poor|wrong|terrible|no)\s+accuracy\b", re.IGNORECASE)


def dan_explicit_axes(reaction: str) -> dict | None:
    """Decode Dan's spoken two-axis convention, or None if it does not apply.

    Deterministic and free -- no model call, no temperature, no drift. Returning
    None means "not my convention, ask the model".
    """
    if not reaction or not _GOOD_ONE_RE.search(reaction):
        return None
    if _ACCURACY_GOOD_RE.search(reaction):
        substance = "right"
    elif _ACCURACY_BAD_RE.search(reaction):
        substance = "wrong"
    else:
        # Covers both "good one, unclear accuracy" and a bare "good one".
        # Absurdist humour does not carry a truth value -- unknown is the
        # honest label, and it keeps the row out of positive tuning.
        substance = "unknown"
    return {"delivery": "good", "substance": substance,
            "axes_source": "dan_convention"}


def parse_axes(text: str) -> dict:
    """Pull the two axes out of a model response.

    Tolerant by design: scans anywhere in the string, accepts `=` or `:`, any
    case. Anything it cannot find comes back `unknown` rather than a default
    value -- an unparsed axis and a genuinely-uncertain axis are both "we do not
    know", and neither should be silently rendered as `good`/`right`.
    """
    if not text:
        return dict(UNLABELLED)
    d = _DELIVERY_RE.search(text)
    s = _SUBSTANCE_RE.search(text)
    return {
        "delivery": d.group(1).lower() if d else "unknown",
        "substance": s.group(1).lower() if s else "unknown",
        "axes_source": "llm",
    }


def is_labelled(row: dict) -> bool:
    """True when a row already carries a usable (non-unknown) axis pair."""
    return (row.get("delivery") in ("good", "bad")
            or row.get("substance") in ("right", "wrong"))


def safe_for_positive_tuning(row: dict) -> bool:
    """The one gate that matters: may this row become a positive example?

    Only when the substance is affirmatively RIGHT. `unknown` is excluded on
    purpose -- an unverifiable claim is exactly the "funny but fabricated" case
    this module exists to stop, and defaulting it to includable would reinstate
    the bug with extra steps.
    """
    return row.get("substance") == "right" and row.get("delivery") != "bad"


async def label_axes(prev_user: str, prev_assistant: str, reaction: str) -> dict:
    """Grade one exchange. Standalone call -- goes to LLM_MEMORY_URL (:8084)
    through generate_memory, which already waits for conversation AND vision
    idle, so this never competes with a live turn on :8083.

    Never raises: a labelling failure must not cost us the capture itself.
    """
    if not prev_assistant:
        return dict(UNLABELLED)
    explicit = dan_explicit_axes(reaction)
    if explicit is not None:
        return explicit          # his stated convention wins; no model call
    try:
        out = await generate_memory(
            AXES_PROMPT.format(
                prev_user=(prev_user or "(not recorded)")[:800],
                prev_assistant=prev_assistant[:1500],
                reaction=(reaction or "(no reaction recorded)")[:800],
            ),
            thinking=False,
            temperature=0.0,   # a label must be reproducible; see generate_memory
        )
    except Exception as e:
        log.warning("axes labelling LLM error: %s", e)
        return dict(UNLABELLED)
    axes = parse_axes(out or "")
    if axes["delivery"] == "unknown" and axes["substance"] == "unknown":
        log.debug("axes unparsed from %r", (out or "")[:120])
    return axes
