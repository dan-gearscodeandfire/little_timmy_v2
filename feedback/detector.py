"""Detect meta-feedback turns ("Little Timmy, that was too long...") and
persist (prev_user, prev_assistant, feedback_text, current_assistant) so
Claude Code can review and tune the persona/system prompt later.

Two-pass design (mirrors memory.extraction):
  1. Cheap keyword pre-filter -- rejects ~99% of normal turns instantly.
  2. Thinking-OFF Qwen3.6 confirm -- rules out false positives like
     "that was too long ago" or "your guardrails on the truck".

Single-flight: drops new detections while one is running.
"""

import asyncio
import logging
import re
import time

from llm.client import generate_memory
from feedback.storage import append_event

log = logging.getLogger(__name__)

_detector_running = False


_ADDRESS_OR_REFER_PATTERNS = [
    r"\blittle timmy\b",
    r"\btimmy\b",
    r"\bthat (was|response|reply|answer)\b",
    r"\byour (last|previous|response|reply|answer|tone|persona|prompt|system prompt|guardrails|guard rails|meta|instructions|setup)\b",
    r"\bthe (last|previous) (response|reply|answer|thing you said)\b",
    r"\bnext time\b",
    r"\bin the future\b",
    r"\bdon'?t (do|say|be|respond|answer) (that|this|it) again\b",
    r"\bfrom now on\b",
]

_CRITIQUE_PATTERNS = [
    r"\b(too|overly|way too) (long|short|verbose|terse|wordy|brief|slow|fast|formal|casual|loud|quiet|dramatic|robotic)\b",
    r"\b(more|less) (concise|brief|terse|verbose|wordy|formal|casual|natural|chatty|direct|polite|friendly)\b",
    r"\b(stop|quit|cut it out|knock it off) (saying|doing|being|using)\b",
    r"\b(don'?t|do not) (say|do|be|use|mention|repeat|start with|end with)\b",
    r"\b(meta|guard ?rails?|system prompt|persona|character|tone|style)\b",
    r"\b(wrong|incorrect|mistaken|off-base|missed the point|missed the mark|hallucinat\w+|confabulat\w+|made (that|it) up)\b",
    r"\b(could have|should have|shoulda|coulda) (been|said|told|answered|responded)\b",
    r"\b(annoying|patronizing|condescending|preachy|repetitive|long-winded|verbose|boring)\b",
    r"\b(be|sound|act) (more|less|like) [a-z]+\b",
    r"\b(simpler|shorter|longer|warmer|colder|friendlier|snappier|funnier)\b",
    r"\bthat'?s (not|wrong|too|the wrong)\b",
    r"\b(quit|stop) (introducing|repeating|doing|saying)\b",
    r"\b(just|only) (say|answer|tell|give)\b",
    r"\bbe more [a-z]+\b",
    r"\bbe less [a-z]+\b",
]

_ADDRESS_RE = re.compile("|".join(_ADDRESS_OR_REFER_PATTERNS), re.IGNORECASE)
_CRITIQUE_RE = re.compile("|".join(_CRITIQUE_PATTERNS), re.IGNORECASE)


def _keyword_score(user_text: str) -> int:
    if not user_text:
        return 0
    addr = bool(_ADDRESS_RE.search(user_text))
    crit = bool(_CRITIQUE_RE.search(user_text))
    if addr and crit:
        return 2
    if addr or crit:
        return 1
    return 0


_CONFIRM_PROMPT = (
    "You are a triage step. Read the user's latest message and the assistant "
    "response that came BEFORE it. Decide if the user's latest message is "
    "META-FEEDBACK -- a critique, correction, or instruction about the "
    "assistant's persona, response style, length, tone, accuracy, or "
    "behavior -- as opposed to a normal request for information or action.\n\n"
    "Answer YES if the user is telling the assistant how to behave differently, "
    "complaining about the previous response, or correcting it. Answer NO for "
    "normal questions, commands, chitchat, or factual corrections about the "
    "world (not about the assistant).\n\n"
    "Examples that should be YES:\n"
    "- \"Little Timmy, that was an overly long response\"\n"
    "- \"You always start with 'I think' -- stop doing that\"\n"
    "- \"Be more concise next time\"\n"
    "- \"Stop using that phrase\"\n"
    "- \"That's not what I asked\"\n\n"
    "Examples that should be NO:\n"
    "- \"What's the weather?\"\n"
    "- \"Turn off the lights\"\n"
    "- \"That was a long time ago\" (about the world, not the assistant)\n"
    "- \"You're right\" / \"thanks\"\n"
    "- \"Tell me more\"\n\n"
    "Previous assistant response: {prev_assistant}\n"
    "Latest user message: {user_text}\n\n"
    "Answer (one word, yes or no):"
)


async def maybe_capture_feedback(
    user_text: str,
    current_assistant: str,
    messages: list[dict],
    speaker: str | None = None,
):
    """Fire-and-forget entry point called from main.py response-finalize.

    `messages` is the prompt history sent to the LLM, ordered oldest-first.
    The current `user_text` is messages[-1]; the previous assistant turn
    is messages[-2]; the previous user prompt that elicited it is
    messages[-3].
    """
    global _detector_running
    if _detector_running:
        log.debug("feedback detector busy; skipping")
        return
    score = _keyword_score(user_text)
    if score == 0:
        return
    _detector_running = True
    asyncio.create_task(
        _run(user_text, current_assistant, messages, speaker, score)
    )


async def _run(
    user_text: str,
    current_assistant: str,
    messages: list[dict],
    speaker: str | None,
    score: int,
):
    global _detector_running
    try:
        prev_user, prev_assistant = _extract_prev_pair(messages)
        if not prev_assistant:
            log.debug("feedback: no prior assistant turn -- skipping")
            return

        confirmed = (score == 2)
        if not confirmed:
            try:
                verdict = await generate_memory(
                    _CONFIRM_PROMPT.format(
                        prev_assistant=(prev_assistant[:1500] or ""),
                        user_text=user_text,
                    ),
                    thinking=False,
                )
            except Exception as e:
                log.warning("feedback confirm LLM error: %s", e)
                return
            verdict_clean = (verdict or "").strip().lower()
            first = verdict_clean.split()[0].rstrip(".,!?:;") if verdict_clean else ""
            confirmed = first == "yes"
            if not confirmed:
                log.debug("feedback verdict: %r (skipping)",
                          first or verdict_clean[:30])
                return

        entry = {
            "ts": time.time(),
            "speaker": speaker,
            "feedback_text": user_text,
            "current_assistant": current_assistant,
            "prev_user": prev_user,
            "prev_assistant": prev_assistant,
            "keyword_score": score,
            "llm_confirmed": (score < 2),
        }
        try:
            event_id = await asyncio.to_thread(append_event, entry)
            log.info("[FEEDBACK] captured event id=%s score=%d feedback=%r",
                     event_id, score, user_text[:80])
        except Exception as e:
            log.warning("feedback persist error: %s", e)
    finally:
        _detector_running = False


def _extract_prev_pair(messages: list[dict]) -> tuple[str, str]:
    """Find the user-prompt / assistant-response pair that came BEFORE
    the current user turn (which is messages[-1]). Skips system messages.
    """
    prev_user = ""
    prev_assistant = ""
    for m in reversed(messages[:-1]):
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant" and not prev_assistant:
            prev_assistant = content
        elif role == "user" and prev_assistant and not prev_user:
            prev_user = content
            break
    return prev_user, prev_assistant
