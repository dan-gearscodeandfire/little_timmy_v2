"""Relevance classifier for vision pipeline (Step 3).

Scores a SceneRecord against recent history and conversation state
to decide what's worth injecting into the prompt. Filters out noise
and repetition so the 3B model only gets meaningful visual context.

Scoring dimensions:
  - novelty:     How different is this from recent records?
  - persistence: Has this been true across multiple frames? (filters transient noise)
  - urgency:     VLM flagged speak_now or high novelty

Output: RelevanceResult with an overall score and per-dimension breakdown,
plus a filtered summary that omits stale/redundant information.
"""

import logging
from dataclasses import dataclass, field

from vision.analyzer import SceneRecord

log = logging.getLogger(__name__)

# Thresholds
INJECT_THRESHOLD = 0.3       # below this, don't inject vision into prompt at all
DETAIL_THRESHOLD = 0.6       # above this, include full detail (objects, scene_state)
SPEAK_THRESHOLD = 0.8        # above this, vision record suggests proactive comment


@dataclass
class RelevanceResult:
    """Scoring result from the relevance classifier."""
    overall: float = 0.0          # 0-1, weighted composite
    novelty_score: float = 0.0    # 0-1, how new is this vs recent history
    persistence_score: float = 0.0  # 0-1, how stable across frames
    urgency_score: float = 0.0    # 0-1, VLM-flagged importance
    should_inject: bool = False   # overall >= INJECT_THRESHOLD
    detail_level: str = "none"    # "none", "minimal", "full"
    filtered_summary: str = ""    # what actually goes into the prompt
    new_people: list[str] = field(default_factory=list)
    new_actions: list[str] = field(default_factory=list)
    changed: str = ""


def _set_overlap(a: list[str], b: list[str]) -> float:
    """Fraction of items in `a` that also appear in `b` (case-insensitive)."""
    if not a:
        return 1.0  # empty set is "fully covered" by anything
    a_lower = {x.lower() for x in a}
    b_lower = {x.lower() for x in b}
    if not a_lower:
        return 1.0
    return len(a_lower & b_lower) / len(a_lower)


def _new_items(current: list[str], prior: list[str]) -> list[str]:
    """Items in current that weren't in prior (case-insensitive)."""
    prior_lower = {x.lower() for x in prior}
    return [x for x in current if x.lower() not in prior_lower]


def score_novelty(record: SceneRecord, history: list[SceneRecord]) -> float:
    """How different is this record from recent history?

    Compares people, actions, objects, and scene_state against the last
    few records. High overlap = low novelty.
    """
    if not history:
        return 1.0  # first record is always novel

    # Compare against most recent record
    prev = history[-1]

    people_overlap = _set_overlap(record.people, prev.people)
    action_overlap = _set_overlap(record.actions, prev.actions)
    object_overlap = _set_overlap(record.objects, prev.objects)
    scene_same = (record.scene_state.lower().strip()
                  == prev.scene_state.lower().strip())

    # Weighted: people and actions matter more than objects
    overlap = (
        0.35 * people_overlap
        + 0.35 * action_overlap
        + 0.15 * object_overlap
        + 0.15 * (1.0 if scene_same else 0.0)
    )

    # Invert: high overlap = low novelty
    novelty = 1.0 - overlap

    # Boost if VLM itself flagged high novelty
    vlm_novelty = record.novelty  # 0-1 from VLM
    novelty = max(novelty, vlm_novelty)

    return min(novelty, 1.0)


def score_persistence(record: SceneRecord, history: list[SceneRecord]) -> float:
    """How stable is the current observation across recent frames?

    High persistence = this has been true for several frames = more reliable.
    Low persistence = just appeared = might be transient noise.

    We WANT persistent observations (reliable) but also want NEW observations
    (novel). This score is used to filter out single-frame glitches, not to
    suppress new information.
    """
    if len(history) < 2:
        return 0.5  # not enough history to judge

    # Check how many of the last N records share current people/actions
    lookback = min(len(history), 5)
    recent = history[-lookback:]

    people_count = 0
    action_count = 0
    for prev in recent:
        if _set_overlap(record.people, prev.people) > 0.5:
            people_count += 1
        if _set_overlap(record.actions, prev.actions) > 0.5:
            action_count += 1

    persistence = max(people_count, action_count) / lookback
    return persistence


def score_urgency(record: SceneRecord) -> float:
    """How urgent/important did the VLM flag this?

    `humor_potential` and `store_as_memory` were removed from the VLM
    schema 2026-05-07 (limited utility, +600 ms per cycle); urgency now
    falls back to speak_now + high novelty alone.
    """
    score = 0.0
    if record.speak_now:
        score = max(score, 0.9)
    if record.novelty > 0.7:
        score = max(score, 0.5)
    return min(score, 1.0)


def classify(record: SceneRecord, history: list[SceneRecord]) -> RelevanceResult:
    """Score a SceneRecord for prompt injection relevance.

    Args:
        record: The current SceneRecord from VLM.
        history: Recent SceneRecords (oldest first).

    Returns:
        RelevanceResult with scores and filtered summary.
    """
    novelty = score_novelty(record, history)
    persistence = score_persistence(record, history)
    urgency = score_urgency(record)

    # Weighted composite — novelty is most important, urgency is a strong boost
    overall = (
        0.45 * novelty
        + 0.15 * persistence
        + 0.40 * urgency
    )

    # Clamp
    overall = max(0.0, min(1.0, overall))

    should_inject = overall >= INJECT_THRESHOLD

    # Determine detail level
    if overall >= DETAIL_THRESHOLD:
        detail_level = "full"
    elif should_inject:
        detail_level = "minimal"
    else:
        detail_level = "none"

    # Find what's actually new
    prev = history[-1] if history else None
    new_people = _new_items(record.people, prev.people if prev else [])
    new_actions = _new_items(record.actions, prev.actions if prev else [])
    changed = record.change_from_prior if record.change_from_prior.lower() != "none" else ""

    # Build filtered summary based on detail level
    filtered_summary = _build_filtered_summary(
        record, detail_level, new_people, new_actions, changed
    )

    result = RelevanceResult(
        overall=round(overall, 2),
        novelty_score=round(novelty, 2),
        persistence_score=round(persistence, 2),
        urgency_score=round(urgency, 2),
        should_inject=should_inject,
        detail_level=detail_level,
        filtered_summary=filtered_summary,
        new_people=new_people,
        new_actions=new_actions,
        changed=changed,
    )

    log.info(
        "[RELEVANCE] overall=%.2f (nov=%.2f pers=%.2f urg=%.2f) -> %s | %s",
        result.overall, novelty, persistence, urgency,
        detail_level, filtered_summary[:80] if filtered_summary else "(skip)"
    )

    return result


def _build_filtered_summary(
    record: SceneRecord,
    detail_level: str,
    new_people: list[str],
    new_actions: list[str],
    changed: str,
) -> str:
    """Build a prompt-ready summary based on detail level.

    - "none": empty string (don't inject)
    - "minimal": only new/changed info (people, actions, change)
    - "full": complete scene summary (people, actions, objects, scene_state)
    """
    if detail_level == "none":
        return ""

    parts = []

    if detail_level == "minimal":
        # Only what's new or changed
        if new_people:
            parts.append("New: " + ", ".join(new_people))
        if new_actions:
            parts.append("Doing: " + ", ".join(new_actions))
        elif record.actions:
            # If no new actions but we have actions, include them briefly
            parts.append("Doing: " + ", ".join(record.actions[:2]))
        if changed:
            parts.append("Changed: " + changed)
        if record.speak_now:
            parts.append("[ATTENTION: something notable happening]")

    elif detail_level == "full":
        # Full scene record
        if record.people:
            parts.append("People: " + ", ".join(record.people))
        if record.actions:
            parts.append("Activity: " + ", ".join(record.actions))
        if record.objects:
            parts.append("Objects: " + ", ".join(record.objects[:5]))
        if record.scene_state:
            parts.append("Scene: " + record.scene_state)
        if changed:
            parts.append("Changed: " + changed)
        if record.speak_now:
            parts.append("[ATTENTION: something notable happening]")

    return "; ".join(parts) if parts else ""
