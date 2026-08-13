"""Per-turn conversational register (2026-08-13).

WHY THIS EXISTS. Two measured facts from the Open Sauce audit:

  1. `_REPLY_MAX_SENTENCES = 2` combined with the persona rule "Always answer
     questions accurately, even if you wrap it in attitude" leaves EXACTLY one
     free sentence per turn and instructs the model to fill it with attitude.
     Every reply becomes [answer][jab]. That is a shape imposed by the cap, not
     a personality trait -- which is why "a jab in every response" survived
     direct instruction to stop.

  2. The only per-turn tone control that existed (the 3x3 mood grid) was PINNED
     by manual override for all 1,273 Open Sauce turns, so the updater computed
     a signal every turn and discarded it. Dan reached for the dial by hand 14
     times over two days and reverted to (1,1) within 5-35 minutes every single
     time -- a global mode cannot express "be straight for THIS one answer."

So: derive a register per turn from signals that are already free, and let it
drive BOTH the prompt line and the sentence cap. A STRAIGHT turn gets one
sentence, which removes the beat the jab was living in; a BANTER turn keeps two.

Deliberately a pure function over text + a couple of booleans: no LLM call, no
added latency, and trivially testable. Everything it reads is already computed
by the doorway before the turn runs.
"""

from __future__ import annotations

import re

STRAIGHT = "STRAIGHT"
BANTER = "BANTER"
WARM = "WARM"

# Asks for a FACT -- a thing with a right answer that memory or knowledge
# supplies. These are the turns where a closing jab reads as evasion, and the
# ones Dan's floor feedback kept objecting to ("just answer the question").
_FACTUAL_RE = re.compile(
    r"^\s*(?:hey\s+|ok(?:ay)?\s+|so\s+)?(?:little\s+)?(?:timmy[,\s]+){0,2}"
    r"(?:what(?:'s| is| are| was| were)?|who(?:'s| is| are)?|when(?:'s| is)?|"
    r"where(?:'s| is| are)?|which|how (?:many|much|old|tall|long|far)|"
    r"do you (?:know|remember|recall)|can you (?:tell|remember)|"
    r"tell me (?:what|who|when|where|how many))\b",
    re.IGNORECASE,
)

# A question asked as a STATEMENT + TAG -- "..., right?", "..., didn't you?".
# _FACTUAL_RE is ^-anchored on fronted wh-/aux- forms, so none of these matched
# and every one fell through to BANTER, which grants the 2-sentence jab budget.
# Measured 2026-08-13: a question battery is all fronted wh-questions, so this
# was invisible until the test was styled as ordinary conversation. The live
# cost was a fabrication in the bought second sentence -- retrieval had already
# missed on "somebody walked off with one of your microphones at that party,
# right?" and the spare sentence became "Dan probably lost them again."
_TAG_QUESTION_RE = re.compile(
    # The main clause must NOT itself be a fronted question -- without this
    # guard "how are you?" parses as <clause>+<are you?> and gets the
    # one-sentence budget, which is the opposite of the intent. Caught by
    # tests/test_conversation_turn.py, which drives exactly that utterance.
    r"^(?!\s*(?:hey\s+|ok(?:ay)?\s+|so\s+)?(?:little\s+)?(?:timmy[,\s]+){0,2}"
    r"(?:what|who|whom|whose|when|where|which|why|how|is|are|was|were|do|does|"
    r"did|can|could|will|would|should|shall|have|has|had|am|may|might)\b)"
    r"\S.*?,?\s*(?:"
    r"right|correct|yeah|no|eh|"
    r"(?:is|isn'?t|are|aren'?t|was|wasn'?t|were|weren'?t|do|don'?t|does|doesn'?t|"
    r"did|didn'?t|has|hasn'?t|have|haven'?t|had|hadn'?t|can|can'?t|could|couldn'?t|"
    r"will|won'?t|would|wouldn'?t|should|shouldn'?t|am|ain'?t)\s+"
    r"(?:it|he|she|they|you|we|i|there|that|this)"
    r")\s*\?\s*$",
    re.IGNORECASE,
)

# Imperative recall -- asks memory for a fact without ever forming a question.
# "Tell me about the party." / "Remind me what Sierra ordered." Deliberately
# NOT folded into _FACTUAL_RE: the opinion guard below must still be able to
# veto "tell me a joke", which is the same frame asking for the opposite thing.
_RECALL_ASK_RE = re.compile(
    r"^\s*(?:hey\s+|ok(?:ay)?\s+|so\s+)?(?:little\s+)?(?:timmy[,\s]+){0,2}"
    r"(?:tell me about|remind me|refresh my memory|"
    r"walk me through|fill me in on|catch me up on)\b",
    re.IGNORECASE,
)

# Explicitly invites opinion / performance / play. A right answer does not
# exist, so wit is the point rather than an evasion of the point.
_OPINION_RE = re.compile(
    r"\b(?:what do you think|how do you feel|do you like|do you enjoy|"
    r"your opinion|favorite|favourite|would you rather|imagine|pretend|"
    r"tell me a (?:joke|story|poem)|sing|rate|guess|make fun|roast)\b",
    re.IGNORECASE,
)

# The user is correcting Timmy. A jab on top of a correction is what produced
# "I have your name exactly right, Dan" (7-18) and "I am Timmy. Fine, I will
# stop." (7-19) -- the two worst-received replies in the whole audit.
_CORRECTION_RE = re.compile(
    r"\b(?:no,? (?:that's|thats|it's|its|you)|that's (?:not|wrong|incorrect)|"
    r"thats (?:not|wrong)|you(?:'re| are) wrong|you got (?:it|that) wrong|"
    r"wrong|incorrect|not (?:my|his|her|their) name|stop (?:doing|saying|being)|"
    r"quit it|knock it off|don't (?:do|say) that|"
    # Repetition complaints (added after live acoustic test 2026-08-13):
    # "You said that already." classified BANTER, so Timmy got a second
    # sentence and spent it on "I have said nothing. You are misinterpreting
    # my silence." -- wrong AND a jab, on a turn where the user was correcting
    # him. Being told you are repeating yourself is a correction.
    r"you (?:said|already said)|said that already|already told|"
    r"repeating yourself|same thing again|you keep saying)\b",
    re.IGNORECASE,
)

# Vision-side child cue. The persona already has a children-are-the-exception
# rule, but it fires only when the VLM caption happens to contain one of these
# words -- and at Open Sauce Dan had to intervene VERBALLY three separate times
# ("there's a kid here", "you're talking to a kid right now, you have to be
# nice"), which means the caption route was not reaching it. Reading the same
# cue here at least makes the trigger explicit and loggable.
_CHILD_RE = re.compile(
    r"\b(?:child|children|kid|kids|little (?:girl|boy)|young (?:girl|boy)|"
    r"toddler|boy|girl)\b",
    re.IGNORECASE,
)


def classify(user_text: str,
             *,
             vision_description: str | None = None,
             speaker_is_unknown: bool = False,
             turns_with_speaker: int = 0) -> str:
    """Return STRAIGHT, WARM, or BANTER for this turn.

    Precedence is deliberate, most-protective first:
      WARM     a child is in frame, or a stranger's first exchange -- the two
               cases where a put-down lands worst and Dan intervened by hand.
      STRAIGHT a turn that asks for a real answer and is not an opinion prompt
               -- a fronted question ("who directed X"), a statement+tag
               ("..., right?"), or an imperative recall ("remind me ...") --
               or the user correcting Timmy.
      BANTER   everything else, which is where the wit belongs.
    """
    text = user_text or ""

    if vision_description and _CHILD_RE.search(vision_description):
        return WARM
    # A stranger's opening turn: they have not earned the edge yet and it reads
    # as hostility rather than a bit. After a couple of exchanges, banter is fine.
    if speaker_is_unknown and turns_with_speaker <= 1:
        return WARM

    if _CORRECTION_RE.search(text):
        return STRAIGHT
    # All three "asks for a real answer" shapes share ONE opinion veto, so
    # "tell me a joke" and "what's your favorite album?" still land in BANTER.
    asks_for_an_answer = (_FACTUAL_RE.search(text)
                          or _TAG_QUESTION_RE.search(text)
                          or _RECALL_ASK_RE.search(text))
    if asks_for_an_answer and not _OPINION_RE.search(text):
        return STRAIGHT

    return BANTER


# Sentence budget per register. STRAIGHT gets ONE -- that is the whole point:
# with a single sentence there is no second beat for the reflexive jab to
# occupy, so brevity does the work that instruction could not.
SENTENCE_CAP = {STRAIGHT: 1, WARM: 2, BANTER: 2}

REGISTER_TEXT = {
    STRAIGHT: (
        "[REGISTER] This is a direct question with a real answer. Answer it "
        "plainly in ONE sentence and stop. No closing remark, no jab, no "
        "commentary on the asker. Dry delivery is fine; a put-down here reads "
        "as dodging the question."
    ),
    WARM: (
        "[REGISTER] You are talking to a child or to someone you have just "
        "met. Be genuinely warm and welcoming. No insults, no sarcasm at their "
        "expense, no put-downs — save the edge for Dan and for people you know."
    ),
    BANTER: (
        "[REGISTER] Ordinary conversation — your wit belongs here. Land it in "
        "the first sentence if you land it at all, and only when it actually "
        "earns its place; a remark every single turn stops being funny."
    ),
}


def register_line(register: str | None) -> str | None:
    """The [REGISTER] prompt line, or None for an unknown/empty register."""
    if not register:
        return None
    return REGISTER_TEXT.get(register)
