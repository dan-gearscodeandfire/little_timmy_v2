"""Ephemeral system prompt assembly.

Prompt structure (for KV cache efficiency):
  [conversation history]  <-- stable prefix, KV cached
  [ephemeral system block] <-- rebuilt each turn (~300 tokens)
  [user message]
  [assistant]              <-- generation starts here

The ephemeral block is injected as a system message right before the user's
current turn. This exploits LLM recency bias (instructions closer to
generation = stronger adherence) while keeping history as a cacheable prefix.
"""

from datetime import datetime
from memory.retrieval import RetrievedMemory
from memory.facts import Fact
import config


def _format_relative_time(dt) -> str:
    """Format a datetime as a human-readable relative time."""
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
    """Compact age formatter for presence annotations."""
    if age_s is None:
        return "?"
    if age_s < 60:
        return f"{int(age_s)}s"
    if age_s < 3600:
        return f"{int(age_s / 60)}m"
    h = int(age_s / 3600)
    m = int((age_s % 3600) / 60)
    return f"{h}h{m:02d}m" if m else f"{h}h"


def build_ephemeral_block(
    memories: list[RetrievedMemory],
    facts: list[Fact],
    speaker_name: str | None = None,
    now: datetime | None = None,
    vision_description: str | None = None,
    visual_question: bool = False,
    presence_state: dict | None = None,
    fusion_source: str | None = None,
    face_hint_name: str | None = None,
) -> str:
    """Build the ephemeral system context block."""
    if now is None:
        now = datetime.now()

    parts = [config.PERSONA.strip()]
    parts.append(f"\nCurrent time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")

    if speaker_name:
        if fusion_source == "face_hint" and face_hint_name:
            parts.append(
                f"\n[WHO IS SPEAKING] The voiceprint did not match a known speaker. "
                f"Face recognition strongly suggests this is {face_hint_name.title()} "
                f"(only visible person, head centered on them). "
                f"Treat this as a working hypothesis: address them as {face_hint_name.title()} "
                f"unless they correct you."
            )
        else:
            parts.append(
                f"\nIMPORTANT: The person speaking to you right now is {speaker_name.title()}. "
                f"Address them as {speaker_name.title()}. Do NOT confuse them with anyone else "
                f"mentioned in the conversation. Even if other names come up in discussion, "
                f"you are talking to {speaker_name.title()}."
            )

    if presence_state and presence_state.get("present"):
        present_lines = []
        for entry in presence_state["present"]:
            n = (entry.get("name") or "").lower()
            if not n or n.startswith("unknown"):
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
                "\n[WHO IS PRESENT] People believed to be in the room "
                "(based on recent face recognition + voice). Use this for context "
                "but do not infer who is speaking from this list:"
            )
            parts.extend(present_lines)

    if facts:
        parts.append(
            "\nGROUND TRUTH \u2014 these facts are verified and must NEVER be contradicted. "
            "If asked about any of these topics, use ONLY the information below:"
        )
        for f in facts:
            parts.append(f"- {f.subject} {f.predicate} {f.value}")

    if memories:
        parts.append("\nRelevant memories:")
        for m in memories:
            time_str = _format_relative_time(m.created_at)
            # Truncate long memories
            content = m.content if len(m.content) <= 200 else m.content[:200] + "..."
            parts.append(f"- ({time_str}) {content}")

    if vision_description:
        if visual_question:
            # User asked a visual question — encourage answering with visual details
            parts.append(
                "\n[WHAT YOU SEE]\n" + vision_description
                + "\nThe user is asking about what you can see. "
                + "Answer their question using the visual information above. "
                + "Be specific and descriptive about what you observe."
            )
        else:
            # Background awareness — don't volunteer visual details
            parts.append(
                "\n[WHAT YOU SEE]\n" + vision_description
                + "\nVISION RULES: This is background awareness. "
                + "Do NOT describe what you see unless directly asked. "
                + "Do NOT narrate the scene. Do NOT volunteer visual details. "
                + "If someone asks what you see or about visual details, "
                + "you CAN and SHOULD answer using this information."
            )

    return "\n".join(parts)


def build_messages(
    history: list[dict],
    ephemeral_block: str,
    user_text: str,
) -> list[dict]:
    """Assemble the full message list for the LLM.

    Order: [history prefix] [ephemeral system] [current user turn]
    """
    messages = list(history)  # copy -- this is the KV-cached prefix
    messages.append({"role": "system", "content": ephemeral_block})
    messages.append({"role": "user", "content": user_text})
    return messages
