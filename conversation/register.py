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
    r"where(?:'s| is| are)?|which|why|"
    # Bare "how" -- how-TO questions ("how are you supposed to knock these
    # out?", "how do I wire this outlet?") are the practical ones Dan most
    # needs a real answer to, and until 2026-08-14 only the measurement forms
    # (how many/much/old/...) were listed, so every one of them graded BANTER
    # and bought a jab. Live: "You hit them with a screwdriver and a hammer,
    # Dan. It's not advanced physics." -- right answer, taxed.
    #
    # "how" was excluded here to protect the phatic "how are you?", which
    # tests/test_register.py pins as BANTER. _PHATIC_RE now owns that case
    # explicitly, so the exclusion is no longer load-bearing and the veto below
    # keeps the greetings out.
    r"how|"
    r"do you (?:know|remember|recall)|can you (?:tell|remember)|"
    r"tell me (?:what|who|when|where|how many))\b",
    re.IGNORECASE,
)

# Polar (yes/no) questions -- "Is Alien Earth in your training data?", "Did
# you already move the servos?". _FACTUAL_RE is a wh-word list, so every
# aux-fronted question fell through to BANTER and bought the jab sentence.
# Measured 8-13: "Is Aliens Earth in your training data?" only reached STRAIGHT
# by accident, because the same utterance opened with "No, no it's not" and
# tripped _CORRECTION_RE. Kept separate from _FACTUAL_RE because this branch
# terminates on "?" rather than a word boundary -- folding it into that group's
# trailing \b makes it match nothing at all.
#
# The 4-token floor after the auxiliary is load-bearing, not a heuristic hedge:
# tests/test_register.py pins short fronted polars to BANTER as social phatics
# ("Can you hear me?", "Is that so?", "Was it you?", "Are you serious?"), and
# without the floor this branch collapses all of them to a one-sentence answer.
# Substantive polars clear it comfortably -- "Is Alien Earth in your training
# data?" (6), "Did you already move the servos?" (5), "Have you ever met X?" (4)
# -- while every phatic in that test is 2-3.
_POLAR_Q_RE = re.compile(
    r"^\s*(?:hey\s+|ok(?:ay)?\s+|so\s+)?(?:little\s+)?(?:timmy[,\s]+){0,2}"
    r"(?:is|are|was|were|do|does|did|can|could|will|would|should|have|has|am)\s+"
    r"(?:\S+\s+){3,}[^?]*\?",
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
    r"(?:tell me (?:something|more|anything|what you know) about|tell me about|"
    r"what do you know about|do you know anything about|"
    r"remind me|refresh my memory|"
    r"walk me through|fill me in on|catch me up on)\b",
    re.IGNORECASE,
)

# Explicitly invites opinion / performance / play. A right answer does not
# exist, so wit is the point rather than an evasion of the point.
_OPINION_RE = re.compile(
    r"\b(?:what do you think|how do you feel|do you like|do you enjoy|do you want|"
    r"your opinion|favorite|favourite|would you rather|imagine|pretend|"
    r"tell me a (?:joke|story|poem)|sing|rate|guess|make fun|roast)\b",
    re.IGNORECASE,
)

# Polar questions that are not questions. _POLAR_Q_RE has to admit proper nouns
# ("Is Alien Earth in your training data?"), which also admits set-phrase
# questions that want no answer. Vetoed here rather than by narrowing that
# branch, because narrowing is what made it miss the real question. Exasperation
# ("are you kidding me") is deliberately NOT in this list -- see the next block.
_RHETORICAL_RE = re.compile(
    r"\b(?:do you mind|can you believe|what did you expect|am i right)\b",
    re.IGNORECASE,
)

# Exasperation. Reads as a rhetorical question and is anything but: from Dan to
# Timmy, "Are you kidding me right now?" is a complaint about the CODEBASE --
# Timmy has just done something wrong and Dan is naming it. Dan's correction,
# 2026-08-13, after this classifier's first draft filed it under the rhetorical
# veto: "it's a strong 'what's wrong with you' indicator, in re: LT's codebase."
#
# It therefore belongs with _CORRECTION_RE, not with "tell me a joke": the reply
# that must NOT follow it is a jab, which is exactly what the 2-sentence BANTER
# budget buys. Note this deliberately MOVES a boundary that tests/test_register.py
# pinned ("Are you serious?" -> BANTER); that assertion was written from the
# tag-question angle and is updated with this rationale.
_EXASPERATION_RE = re.compile(
    r"\b(?:are you (?:kidding|serious|for real)|you kidding me|"
    r"what(?:'s| is) wrong with you|do you even (?:hear|listen)|"
    r"are you (?:happy|proud) now|what are you (?:doing|talking about)|"
    r"seriously\?|you'?ve got to be kidding)",
    re.IGNORECASE,
)

# Phatic greetings. These are questions in shape only -- "what's shaking?" wants
# hello, not a status report. _FACTUAL_RE matches on the bare "what's", so every
# one of them graded STRAIGHT, which caps the reply at one sentence and forbids
# the closing remark. Live cost 2026-08-14 19:49, the FIRST turn of the evening:
#
#   Dan: "Hey Timmy, what's shaking?"  ->  STRAIGHT  ->  "Nothing."
#
# Dan had asked, one message earlier, for exactly the opposite: "it's supposed
# to sound like talking to a person with awesome delivery." A greeting is where
# delivery matters most and where a right answer does not exist -- so it belongs
# in BANTER with the rest of the wit. Note "how are you?" / "how's it going?"
# escaped only by accident: _FACTUAL_RE's wh-list has no bare "how", so they
# were never at risk. This closes the same hole for the "what's ..." family.
_PHATIC_RE = re.compile(
    r"^\s*(?:hey\s+|hi\s+|yo\s+|ok(?:ay)?\s+|so\s+)?(?:little\s+)?"
    r"(?:timmy[,\s]+){0,2}"
    r"(?:what(?:'s| is)\s+(?:shaking|up|new|good|happening|going on|cooking|"
    r"the word|crackin['g]?|poppin['g]?)|"
    r"how(?:'s| is| goes| are)\s+(?:it going|it|things|life|tricks|"
    r"you(?:\s+doing)?(?!\s+\w))|"
    r"you (?:doing )?(?:ok|okay|alright|good)|how you doing)\b",
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
    # Being told you FABRICATED something. Live 2026-08-15 00:07: "the couples
    # therapist told me that you invented the Frank thing" matched nothing here
    # -- no "no", no "wrong", no "that's not" -- so it graded BANTER, bought a
    # second sentence, and spent it on "I never invented it." Fifth register gap
    # of the session with the same shape: the formal phrasing was covered and
    # the conversational one was not.
    r"you (?:invented|fabricated|made (?:that|it|this) up|"
    r"(?:just )?made (?:that|it|this) up)|"
    r"(?:that|it|this) (?:was|is) (?:made up|invented|a fabrication)|"
    r"you'?re making (?:that|it|this) up|didn'?t happen|never happened|"
    r"repeating yourself|same thing again|you keep saying|"
    # Decorum / content-safety corrections (added after 8-13 live audit).
    # "It sounded slightly sexualized and this needs to be an all ages thing."
    # classified BANTER, so the reply spent its second sentence on "maybe you
    # should be the one screening the chat" -- arguing with a stated content
    # constraint. On a livestream that is the clip that gets screenshotted.
    r"that was (?:a little|a bit|kind of|too|way too|sort of)|"
    # Bare judgment forms. The patterns above are anchored on "that was ...",
    # so the shapes people actually use when something lands badly slipped
    # through. Live 2026-08-14 23:54: Dan said "Too mean, too mean there." ->
    # BANTER -> "I am not being mean. I am being accurate." -- arguing with a
    # correction, which is the exact reply this register exists to prevent.
    r"too (?:mean|harsh|much|far|rude|cruel|nasty|sharp|personal|aggressive)|"
    r"that was (?:mean|harsh|rude|cruel|nasty|cold|brutal)|"
    # Judgments of the REPLY itself. "That's a bizarre response." was the
    # phrasing Dan used on 2026-08-14 when a phantom speaker made Timmy greet
    # him as a stranger; it matched nothing and scored 0 for feedback capture.
    r"(?:that'?s|that was|this is) (?:a )?(?:bizarre|weird|strange|nonsense|"
    r"gibberish|word salad|non sequitur|incoherent|confusing)|"
    r"you(?:'re| are| were)?\s*(?:being\s+)?(?:too\s+)?"
    r"(?:mean|harsh|rude|cruel|nasty) (?:to|about)|"
    r"too far|went there|uncalled for|inappropriate|not appropriate|"
    r"sexualiz|racy|keep it (?:clean|pg|family)|all[- ]ages|"
    r"watch (?:your|the) (?:language|mouth)|tone it down|"
    # Clarification requests. Being asked to restate is not an invitation to
    # refuse: "I don't completely follow" drew "I don't have the patience to
    # repeat myself for the half-brained."
    r"i don'?t (?:completely |quite |really )?follow|"
    r"say (?:that|it) again|repeat that|come again|restate|"
    r"i didn'?t (?:catch|get) that|what do you mean)\b",
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


# The user is challenging something Timmy JUST said. Structurally the
# highest-stakes turn type there is: the answer is verifiable from his own hot
# history one turn back, so a jab here is not edge, it is a denial of the
# record. Live cost 8-13 22:33: "Why the hell did you just call me Nathan?"
# classified BANTER -> 2-sentence budget -> "I didn't call you Nathan, Dan.
# You're just projecting your own confusion onto me." Both sentences false.
_OWN_TURN_CHALLENGE_RE = re.compile(
    r"\b(?:"
    r"why did you (?:just )?(?:call|say|do|tell|ask)|"
    r"what did you (?:just )?(?:call|say)|"
    r"did you just (?:call|say)|"
    r"you just (?:called|said)|"
    r"(?:why|what) the (?:hell|heck|fuck) did you|"
    r"you (?:did|do) so|you did too|"
    r"that'?s not what (?:i|you) said"
    r")\b",
    re.IGNORECASE,
)


# Real speech does not arrive one sentence at a time. STT hands over the whole
# turn, and the question is usually LAST: "Timmy, you legitimately just saw me
# trying to drill a hole in a pottery pot. Does that remind you of any stories
# with my wife, Erin?" _FACTUAL_RE / _POLAR_Q_RE / _RECALL_ASK_RE are all
# ^-anchored, so all three looked at "Timmy, you legitimately just saw me..."
# and saw no question -> BANTER -> the 2-sentence budget -> and the bought
# second sentence was "And yes, it reminds me of the time she tried to fix the
# sink and flooded the kitchen", a story that exists nowhere in any store
# (prop_search top score 0.057). Dan, live 2026-08-14: "Boo! Hallucination."
#
# The 79-turn replay that validated those patterns could not surface this: it
# measured which TURNS were reclassified, never where in a turn the question
# sat. So: run the anchored patterns against the final sentence as well.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _last_sentence(text: str) -> str:
    """The final sentence of a multi-sentence turn, or the text itself."""
    parts = [p for p in _SENTENCE_SPLIT_RE.split((text or "").strip()) if p.strip()]
    return parts[-1] if parts else (text or "")


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

    # Challenging what Timmy JUST said outranks everything, including the
    # child/stranger WARM guards -- there is no register in which denying the
    # record is the right move, and STRAIGHT is not unkind (one plain sentence,
    # dry delivery is fine). A child who says "why did you call me that?"
    # deserves the true answer as much as Dan does.
    if _OWN_TURN_CHALLENGE_RE.search(text):
        return STRAIGHT

    if vision_description and _CHILD_RE.search(vision_description):
        return WARM
    # A stranger's opening turn: they have not earned the edge yet and it reads
    # as hostility rather than a bit. After a couple of exchanges, banter is fine.
    if speaker_is_unknown and turns_with_speaker <= 1:
        return WARM

    if _CORRECTION_RE.search(text) or _EXASPERATION_RE.search(text):
        return STRAIGHT
    # All three "asks for a real answer" shapes share ONE opinion veto, so
    # "tell me a joke" and "what's your favorite album?" still land in BANTER.
    # Anchored patterns get two shots: the turn as spoken, and its last
    # sentence (see _last_sentence above -- the question is usually there).
    _tail = _last_sentence(text)
    asks_for_an_answer = any(
        rx.search(candidate)
        for rx in (_FACTUAL_RE, _POLAR_Q_RE, _TAG_QUESTION_RE, _RECALL_ASK_RE)
        for candidate in (text, _tail)
    )
    if (asks_for_an_answer
            and not _OPINION_RE.search(text)
            and not _RHETORICAL_RE.search(text)
            and not _PHATIC_RE.search(text)
            and not _PHATIC_RE.search(_tail)):
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
