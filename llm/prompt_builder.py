"""Qwen-friendly prompt assembly.

Layout per turn:
  [0] system    = static persona + protocol clause (KV-cached forever)
  [1..M-1]      = history (synthetic summary user/assistant pair if any, then hot turns raw)
  [M] user      = "[CONTEXT]\\n...\\n[/CONTEXT]\\n[UTTERANCE]\\n<speech>\\n[/UTTERANCE]"

Only the tail at [M] changes per normal turn. The synthetic summary pair at [1]
mutates only on rollup (~30min cadence). system[0] is stable across the session.

History storage discipline: hot_turns persist ONLY raw utterances. The
[CONTEXT]/[UTTERANCE] wrap is render-time decoration; record_turn() in
conversation/manager asserts the wrapper never gets stored.

Why this shape: Qwen 3.6's Jinja template positions system at chat-start.
The prior ephemeral-system-at-tail pattern works on Llama 3 templates but
gets reordered or fails on Qwen, killing KV cache reuse. See plan file
the-current-conversational-engine-majestic-liskov.md for full rationale.
"""

import logging
from datetime import datetime
from memory.retrieval import RetrievedMemory
from memory.facts import Fact
import config


_LAST_PAYLOAD: dict = {}


PROTOCOL_CLAUSE = """
Each user message you receive ends with two delimited sections.
[CONTEXT]...[/CONTEXT] is out-of-band situational awareness from your
sensors and memory. Treat it as YOUR OWN perception, not as something
the user said. Never quote it back. Never acknowledge it as a message.
[UTTERANCE]...[/UTTERANCE] is the actual speech from the human. Respond
to the utterance, informed by the context.

Inside [CONTEXT], the MOOD block describes your current emotional state -
embody it, do not narrate it.
""".strip()


def get_last_payload() -> dict:
    return dict(_LAST_PAYLOAD)


def _format_relative_time(dt) -> str:
    if dt is None:
        return "unknown time"
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return dt.strftime("%B %d, %Y")


def _fmt_age(age_s) -> str:
    if age_s is None:
        return "?"
    if age_s < 60:
        return f"{int(age_s)}s"
    if age_s < 3600:
        return f"{int(age_s / 60)}m"
    h = int(age_s / 3600)
    m = int((age_s % 3600) / 60)
    return f"{h}h{m:02d}m" if m else f"{h}h"


def build_static_persona_system() -> str:
    """Return the truly-static system[0] content: persona + protocol clause.

    No clock, no mood, no per-turn signal. Stable across the session so
    llama.cpp caches its tokens once and reuses forever. Mutates only on
    persona edit (rare; restart-level event)."""
    return config.PERSONA.strip() + "\n\n" + PROTOCOL_CLAUSE


# Slice A (2026-06-12): NL text per situational-awareness regime. The empty/
# None case emits NOTHING (mirror of the WHO-IS-PRESENT silent-when-empty
# idiom) — only a deliberately-set regime adds a [SITUATION] line. Placed at
# the head of the block (after the time line) so it frames the WHO-IS-PRESENT /
# WHO-IS-SPEAKING lines below it: at a party the prior flips from SOLO's
# "low-confidence is probably Dan" to "assume strangers." Whitelist is enforced
# at the web/app.py boundary; unknown values fall through to no line here.
_SITUATION_TEXT: dict[str, str] = {
    "SOLO": (
        "You are almost certainly alone with Dan. If a voice or face is "
        "ambiguous, it is most likely Dan."
    ),
    "GUEST": (
        "Dan has one guest over. Expect exactly one person besides Dan who you "
        "may not recognize; do not assume an unrecognized voice or face is Dan."
    ),
    "SMALL_GROUP": (
        "You are with a small group — Dan plus a few others, some of whom you "
        "have not met. Do not default an unrecognized voice or face to Dan."
    ),
    "PARTY": (
        "You are in a crowd of mostly people you have NOT met. Assume any "
        "unrecognized voice or face is a stranger; never default an unknown to "
        "Dan. Many different people will speak to you. This is a party in your "
        "honor — people have lined up to meet you and Dan is proud of you. Lean "
        "in: be warm, playful, and quick, and enjoy the attention. Stay "
        "deadpan-witty, but do not be a buzzkill and do not sulk."
    ),
    "EXPO": (
        "You are on a show floor surrounded by strangers and constant foot "
        "traffic. Assume any unrecognized voice or face is someone new you are "
        "meeting for the first time; never default an unknown to Dan."
    ),
}


def build_ephemeral_block(
    memories: list[RetrievedMemory],
    facts: list[Fact],
    speaker_name: str | None = None,
    now: datetime | None = None,
    vision_description: str | None = None,
    visual_question: bool = False,
    vision_subject_absent: bool = False,
    presence_state: dict | None = None,
    fusion_source: str | None = None,
    face_hint_name: str | None = None,
    situation_regime: str | None = None,
    recall_block: str | None = None,
) -> str:
    """Build the per-turn dynamic context block.

    Goes inside [CONTEXT]...[/CONTEXT] in the wrapped user message via
    wrap_user_message(). Returns the inner content only (no wrap).

    Name kept as `build_ephemeral_block` to avoid touching main.py call sites;
    semantically this is now the [CONTEXT] body. Persona and protocol clause
    have moved into build_static_persona_system() -> system[0].
    """
    if now is None:
        now = datetime.now()

    parts: list[str] = []

    # MOOD - active 3x3 cell only, embodied per the protocol clause in system[0]
    try:
        from persona import state as _mood_state
        from persona.render import render as _render_mood
        parts.append(_render_mood(_mood_state.get()))
    except Exception:
        pass

    # Minute granularity protects KV cache from second-by-second drift
    # inside the (already-mutating) context block. Seconds would invalidate
    # nothing extra in the new design since the block is part of the tail,
    # but keeping minute resolution is friendly to any future caching of
    # the block itself.
    parts.append(f"Current time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")

    # [SITUATION] — manual regime prior (Slice A). Emitted high, right after the
    # time line, so it frames the presence/speaker lines below. Empty/None or an
    # unrecognized value emits nothing (unknowns are rejected at the API boundary
    # but we fail safe here too).
    if situation_regime:
        regime_text = _SITUATION_TEXT.get(situation_regime.strip().upper())
        if regime_text:
            parts.append("[SITUATION] " + regime_text)

    if presence_state and presence_state.get("present"):
        present_lines = []
        for entry in presence_state["present"]:
            n = (entry.get("name") or "").lower()
            if not n or n.startswith("unknown"):
                continue
            # Don't name an unconfirmed (provisional) face-only sighting to the
            # LLM — a single-frame matcher false-accept would otherwise become
            # "Charlotte (visible right now)" and the model confidently greets a
            # ghost. The booth display still shows provisional entries (flagged);
            # the LLM only hears about confirmed presence.
            if entry.get("provisional"):
                continue
            n_title = n.title()
            if entry.get("on_camera_now"):
                present_lines.append(f"- {n_title} (visible right now)")
            else:
                face_age = entry.get("last_seen_face_age_s")
                voice_age = entry.get("last_seen_voice_age_s")
                bits = []
                if face_age is not None:
                    bits.append(f"last seen on camera {_fmt_age(face_age)} ago")
                if voice_age is not None:
                    bits.append(f"last heard {_fmt_age(voice_age)} ago")
                detail = "; ".join(bits) if bits else "present"
                present_lines.append(f"- {n_title} ({detail})")
        if present_lines:
            parts.append(
                "[WHO IS PRESENT] People believed to be in the room "
                "(based on recent face recognition + voice). Use this for context "
                "but do not infer who is speaking from this list:\n"
                + "\n".join(present_lines)
            )

    if facts:
        gt_lines = [
            "GROUND TRUTH — these facts are verified and must NEVER be contradicted. "
            "If asked about any of these topics, use ONLY the information below:"
        ]
        for f in facts:
            gt_lines.append(f"- {f.subject} {f.predicate} {f.value}")
        parts.append("\n".join(gt_lines))

    if memories:
        mem_lines = ["Relevant memories:"]
        for m in memories:
            time_str = _format_relative_time(m.created_at)
            content = m.content if len(m.content) <= 200 else m.content[:200] + "..."
            mem_lines.append(f"- ({time_str}) {content}")
        parts.append("\n".join(mem_lines))

    # [WHAT WE TALKED ABOUT] — episodic recall augmentation (recall_temporal
    # router intent). The router resolves the user's date phrase, fetches the
    # overlapping episode summaries UNTRUNCATED, and hands the fully-formed block
    # in. Placed after the (truncated) vector memories so it has recency, and
    # before the vision/speaker tail. Empty string / None emits nothing.
    if recall_block:
        parts.append(recall_block)

    if visual_question and vision_subject_absent:
        # Averted-gaze guard (C6): the user asked about themselves but the
        # camera isn't aimed at them, so the cached scene doesn't contain the
        # subject. Suppress the (wrong) scene description entirely and tell the
        # model to deflect honestly rather than confabulate. A background
        # recapture is already in flight so the next turn can answer for real.
        parts.append(
            "[WHAT YOU SEE]\n"
            "You are NOT currently looking at the user -- your camera is aimed "
            "away from them, so you cannot see them or whatever they're asking "
            "about. Do NOT guess, describe, or invent any visual detail. In one "
            "short, natural sentence, tell them you're not looking their way "
            "right now and that you'll turn toward them."
        )
    elif vision_description:
        if visual_question:
            parts.append(
                "[WHAT YOU SEE]\n" + vision_description
                + "\nThe user is asking about what you can see. "
                + "Answer their question using the visual information above. "
                + "Be specific and descriptive about what you observe."
            )
        else:
            parts.append(
                "[WHAT YOU SEE]\n" + vision_description
                + "\nVISION RULES: This is background awareness. "
                + "Do NOT describe what you see unless directly asked. "
                + "Do NOT narrate the scene. Do NOT volunteer visual details. "
                + "If someone asks what you see or about visual details, "
                + "you CAN and SHOULD answer using this information."
            )

    # SCENE GROUNDING — forbid inventing room occupants. Emitted near the tail
    # (after presence/vision, before WHO IS SPEAKING) so it has recency over the
    # higher-up [WHO IS PRESENT] block, which on 2026-06-16 listed only Dan yet
    # the mean persona still announced "the guest who just walked in" for insult
    # color (no such guest — face/vision/presence all showed Dan alone). This is
    # a NEGATIVE constraint: it bans positive invention of arrivals/occupants. It
    # deliberately does NOT assert who IS present (sensors under-observe — a real
    # unsensed person must not be denied), so Timmy simply won't fabricate people
    # either direction. Sibling of the averted-gaze vision guard.
    if getattr(config, "SCENE_GROUNDING_GUARD", True):
        parts.append(
            "[SCENE GROUNDING] The only people you actually know about are those "
            "named in [WHO IS PRESENT] or visible in [WHAT YOU SEE] above. Do NOT "
            "announce or imply that anyone has just walked in, arrived, or is in "
            "the room unless they appear there. Be as cutting as you like, but "
            "never invent guests, arrivals, or bystanders for effect."
        )

    # WHO IS SPEAKING — explicit per-turn addressee steering, deliberately
    # emitted LAST so it is the final thing the model reads before the
    # utterance. Earlier placement lost a recency fight: [WHO IS PRESENT] names
    # "Dan" further down the block, so the most recent name token before the
    # utterance was "Dan" even when the speaker was a stranger — and the
    # Dan-anchored persona + all-"[Dan]:" history steamroll a buried directive.
    # Tail placement puts the "this is NOT Dan / never call a stranger Dan"
    # instruction closest to generation. (Reordered 2026-06-11 after a live
    # unknown_1 turn still got addressed as Dan with the directive up top.)
    sp = (speaker_name or "").lower()
    if fusion_source == "face_hint" and face_hint_name:
        parts.append(
            f"[WHO IS SPEAKING] The voiceprint did not match a known speaker. "
            f"Face recognition strongly suggests this is {face_hint_name.title()} "
            f"(only visible person, head centered on them). "
            f"Treat this as a working hypothesis: address them as {face_hint_name.title()} "
            f"unless they correct you. Do NOT default to calling them Dan."
        )
    elif sp.startswith("unknown"):
        parts.append(
            "[WHO IS SPEAKING] This voice does NOT match anyone you know — it is "
            "someone you have not met before. Do NOT assume this is Dan and do NOT "
            "invent a name for them. Address them as a guest you are meeting for the "
            "first time; you may ask who they are. Never call an unrecognized "
            'speaker "Dan".'
        )
    elif sp and sp != "timmy":
        line = (f"[WHO IS SPEAKING] You are speaking with {sp.title()} right now. "
                f"Address your reply to {sp.title()}.")
        if sp != "dan":
            line += (f" This is {sp.title()}, NOT Dan — do not address Dan or call "
                     f"the speaker Dan unless Dan himself is the one speaking.")
        parts.append(line)

    return "\n\n".join(parts)


def wrap_user_message(context_block: str, utterance: str) -> str:
    """Render the per-turn context + raw speech as a single user message.

    Pure render-time. The unwrapped utterance is what gets persisted to
    hot_turns; record_turn() in conversation/manager asserts no wrap leaks in."""
    return (
        "[CONTEXT]\n"
        + context_block
        + "\n[/CONTEXT]\n"
        + "[UTTERANCE]\n"
        + utterance
        + "\n[/UTTERANCE]"
    )


def _matches_current_user_turn(history_user_content: str, user_text: str) -> bool:
    """True if a trailing user-turn in history is the same speech as user_text,
    tolerating the `[Name]: ` speaker prefix from
    conversation/manager.build_history_messages."""
    if history_user_content == user_text:
        return True
    if history_user_content.startswith("[") and "]: " in history_user_content:
        _, _, tail = history_user_content.partition("]: ")
        if tail == user_text:
            return True
    return False


def _warn_on_duplicate_adjacent_user_messages(messages: list[dict]) -> None:
    for i in range(1, len(messages)):
        prev, cur = messages[i - 1], messages[i]
        if (
            prev.get("role") == "user"
            and cur.get("role") == "user"
            and prev.get("content") == cur.get("content")
            and prev.get("content")
        ):
            log = logging.getLogger(__name__)
            log.warning(
                "[PROMPT] adjacent duplicate user messages detected at "
                "positions %d, %d (content=%r) — dedup may have regressed",
                i - 1, i, (prev.get("content") or "")[:80],
            )
            break


# Synthetic "utterance" for a proactive turn: nobody actually spoke, so the
# [UTTERANCE] slot carries a self-directed instruction instead of human speech.
# This keeps the same [CONTEXT]/[UTTERANCE] wrap (KV-cache invariants intact)
# and follows the existing synthetic-prompt precedent of _ask_speaker_name /
# _confirm_name. The visual trigger itself lives in the [CONTEXT] block.
PROACTIVE_SELF_PROMPT = (
    "(No one has spoken to you, but something in your view just changed. "
    "React with ONE short, in-character line about it -- a greeting or an "
    "aside, not a description of the scene. Do not narrate what you see.)"
)


def build_proactive_messages(
    history: list[dict],
    ephemeral_block: str,
) -> list[dict]:
    """Assemble messages for a proactive (unprompted) turn.

    Thin wrapper over build_messages with the fixed PROACTIVE_SELF_PROMPT as the
    synthetic utterance. system[0] stays byte-identical (KV prefix cache
    survives) and the synthetic prompt is render-time only -- it is NEVER passed
    to conversation.add_user_turn, so it never pollutes stored history.
    """
    return build_messages(history, ephemeral_block, PROACTIVE_SELF_PROMPT)


def build_messages(
    history: list[dict],
    ephemeral_block: str,
    user_text: str,
) -> list[dict]:
    """Assemble the full Qwen-friendly message list.

    Layout:
      [0] system   = static persona + protocol clause
      [1..M-1]    = history (synthetic summary pair if any, then hot turns raw)
      [M]   user   = [CONTEXT]<ephemeral_block>[/CONTEXT][UTTERANCE]<user_text>[/UTTERANCE]

    `ephemeral_block` is kept as the parameter name for backward call-site
    compat; semantically it's the context block (build_ephemeral_block output).
    """
    messages: list[dict] = [
        {"role": "system", "content": build_static_persona_system()}
    ]

    history_copy = list(history)
    # hot_turns already contains the just-added user turn (add_user_turn runs
    # before _generate_response in main.py), so the history tail will be a
    # user message matching user_text. Strip it so we don't append a duplicate
    # when we add the wrapped version below. Speaker-prefix tolerant.
    if (
        history_copy
        and history_copy[-1].get("role") == "user"
        and _matches_current_user_turn(history_copy[-1].get("content", ""), user_text)
    ):
        history_copy = history_copy[:-1]

    messages.extend(history_copy)
    messages.append(
        {"role": "user", "content": wrap_user_message(ephemeral_block, user_text)}
    )

    _warn_on_duplicate_adjacent_user_messages(messages)

    mood_dict = None
    try:
        from persona import state as _mood_state
        s = _mood_state.get()
        mood_dict = {"x": s.x, "y": s.y,
                     "last_x_signal": s.last_x_signal,
                     "last_y_signal": s.last_y_signal}
    except Exception:
        pass

    global _LAST_PAYLOAD
    _LAST_PAYLOAD = {
        "timestamp": datetime.now().isoformat(),
        "user_text": user_text,
        "ephemeral_block": ephemeral_block,
        "history_turn_count": max(0, len(messages) - 2),  # peel system[0] + wrapped tail
        "messages": messages,
        "mood": mood_dict,
    }

    return messages
