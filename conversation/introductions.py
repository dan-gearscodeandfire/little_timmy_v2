"""Introductions — the multi-turn "what's your name?" sub-dialog.

Split out of the Orchestrator (CONTEXT.md, decision 2). It owns the only
cross-turn state in the conversation path: whether we're waiting for an unknown
speaker to state their name (`_pending_capture`) or to confirm a guessed one
(`_pending_confirm`). The doorway consults `handle()` at the top of each turn;
`ConversationTurn` itself stays stateless per call.

Speaking is delegated to ConversationTurn.say() (a minimal prompted utterance),
so this module never touches TTS/LLM directly. Identity assignment goes through
the injected speaker-id module.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from persistence import runtime_toggles

log = logging.getLogger("timmy")

_AFFIRMATIVE = ("yes", "yeah", "yep", "correct", "that's right",
                "right", "sure", "yup", "exactly", "mhm")
_NEGATIVE = ("no", "nope", "nah", "wrong")


@dataclass(frozen=True)
class IntroOutcome:
    """Result of consulting Introductions at the top of a turn.

    handled=True  -> Introductions spoke a follow-up; the doorway should return
                     without running a normal turn.
    handled=False -> fall through to a normal turn, using `speaker_name`
                     (which may have been promoted to a just-confirmed name)."""
    handled: bool
    speaker_name: str


class Introductions:
    def __init__(self, *, speaker_id_module, turn, cosample=None, committer=None):
        self._spk = speaker_id_module
        self._turn = turn               # ConversationTurn, for .say()
        # Three-way link on a name-tell (2026-07-06): the passively co-sampled
        # face crops (CoSampleBuffer) + commit_identity turn a confirmed name
        # into name<->voiceprint<->faceprint instead of voice-only. `committer`
        # is injectable for tests; None -> lazy presence.identity_commit import
        # at the confirm (keeps this module import-light).
        self._cosample = cosample
        self._committer = committer
        self._pending_capture: str | None = None          # temp_id awaiting a name
        self._pending_confirm: dict | None = None         # {"temp_id", "name"}

    @property
    def awaiting(self) -> bool:
        return self._pending_capture is not None or self._pending_confirm is not None

    def drop_pending(self) -> None:
        """Silently drop an in-flight name exchange when the identity-dialog
        gate closes mid-dialog (EXPO regime, Dan 2026-07-06). SILENT on
        purpose — the visitor's next utterance falls through to the LLM as
        ordinary speech with no sign a name-ask was ever pending."""
        if self.awaiting:
            log.info("[INTRO] identity-dialog gate closed with name exchange "
                     "pending -> silent drop")
            self._pending_capture = None
            self._pending_confirm = None

    async def ask_name(self, unknown_info) -> None:
        """Ask an unknown speaker for their name, then await it next utterance."""
        known_names = [ks.name for ks in self._spk._known_speakers]
        known_str = ", ".join(n.title() for n in known_names if n != "timmy")
        last_quote = unknown_info.last_text[:80] if unknown_info.last_text else "something"
        prompt_text = (
            f"A new person has joined the conversation. I know {known_str} is here, "
            f"but someone new just said: \"{last_quote}\". "
            f"Ask them for their name in a friendly, in-character way."
        )
        result = await self._turn.say(prompt_text)
        log.info("[TIMMY] %s (name solicitation)", result.text)
        # Next utterance from this unknown triggers name capture.
        self._pending_capture = unknown_info.temp_id

    async def offer_confirm(self, temp_id: str, name: str) -> None:
        """Arm the confirm flow from a PASSIVE self-intro (2026-07-06): an
        unknown speaker volunteered "my name is X" without being asked, so
        skip the ask and go straight to the spoken confirm. The visitor's
        "yes" then flows through the existing confirm branch in handle() —
        assign_name, and (toggle-gated) the co-sampled face commit."""
        self._pending_capture = None
        self._pending_confirm = {"temp_id": temp_id, "name": name}
        await self._say_confirm(name)

    async def handle(self, user_text: str, speaker_name: str) -> IntroOutcome:
        """Process a heard utterance against any in-progress name exchange."""
        # --- waiting for yes/no on a guessed name ---
        if self._pending_confirm and speaker_name.startswith("unknown_"):
            lower = user_text.lower().strip().rstrip(".!?,")
            if any(w in lower for w in _AFFIRMATIVE):
                name = self._pending_confirm["name"]
                temp_id = self._pending_confirm["temp_id"]
                # assign_name's verdict is load-bearing (2026-07-06): a
                # tombstoned / reserved / already-taken name returns False,
                # and a refused name must NOT get a face committed under it.
                ok = self._spk.assign_name(temp_id, name)
                log.info("Confirmed name: %s for %s (assign ok=%s)",
                         name, temp_id, ok)
                self._pending_confirm = None
                if ok:
                    # Name-tell -> full triple: also bind the co-sampled face
                    # crops (LED-anchored at EXPO — implied consent; sole-face
                    # in Shop) so the name links voice AND face, not voice
                    # only. Never blocks or breaks the promotion.
                    await self._maybe_commit_face(temp_id, name)
                # Promote the speaker and continue into a normal turn.
                return IntroOutcome(handled=False, speaker_name=name)
            elif any(w in lower for w in _NEGATIVE):
                log.info("Name rejected by user, will re-ask next stable utterance")
                temp_id = self._pending_confirm["temp_id"]
                self._pending_confirm = None
                # Allow re-asking by resetting name_asked.
                for us in self._spk._unknown_speakers:
                    if us.temp_id == temp_id:
                        us.name_asked = False
                        break
                return IntroOutcome(handled=False, speaker_name=speaker_name)
            else:
                # They said something else — maybe the actual name this time.
                name = _extract_name_from_response(user_text)
                if name:
                    self._pending_confirm["name"] = name
                    await self._say_confirm(name)
                    return IntroOutcome(handled=True, speaker_name=speaker_name)
                else:
                    self._pending_confirm = None

        # --- waiting for the speaker to state their name ---
        if self._pending_capture and speaker_name.startswith("unknown_"):
            name = _extract_name_from_response(user_text)
            if name:
                # Confirm before committing.
                self._pending_confirm = {
                    "temp_id": self._pending_capture,
                    "name": name,
                }
                self._pending_capture = None
                await self._say_confirm(name)
                return IntroOutcome(handled=True, speaker_name=speaker_name)
            else:
                log.info("Could not extract name from: %r", user_text)
            self._pending_capture = None

        return IntroOutcome(handled=False, speaker_name=speaker_name)

    async def _maybe_commit_face(self, temp_id: str, name: str) -> None:
        """Bind co-sampled face crops to a just-confirmed name (the triple).

        Fires only when the intro_face_commit_enabled toggle is on and the
        cosample buffer holds crops for this speaker — buffered under the
        temp_id before the promotion, or under the name after it. Errors and
        refusals (commit_identity's mismatch/lookalike/tombstone guards) are
        logged and swallowed: a failed face bind must never break the name
        promotion that already happened."""
        if not runtime_toggles.get("intro_face_commit_enabled"):
            return
        if self._cosample is None:
            return
        crops = (self._cosample.crops_for(temp_id)
                 or self._cosample.crops_for(name))
        if not crops:
            log.info("[INTRO] no co-sampled crops for %s/%s — voice-only "
                     "promotion", temp_id, name)
            return
        try:
            committer = self._committer
            if committer is None:
                from presence.identity_commit import commit_identity
                committer = commit_identity
            res = await committer(name, face_crops=crops,
                                  speaker_identifier=self._spk)
            if getattr(res, "face_committed", False):
                self._cosample.clear_speaker(temp_id)
                self._cosample.clear_speaker(name)
                # id split-brain sync (2026-07-06): assign_name minted a
                # session-local _next_known_id(), but commit_identity just
                # allocated the AUTHORITATIVE shared id (id-map + Postgres
                # speakers row — the facts.speaker_id FK). Face-only commits
                # skip identity_commit's _refresh_voice, so refresh the
                # in-memory KnownSpeaker here or facts stored this session
                # key off the wrong id.
                sid = getattr(res, "speaker_id", None)
                if sid is not None:
                    for ks in self._spk._known_speakers:
                        if ks.name == name and getattr(ks, "speaker_id",
                                                       sid) != sid:
                            log.info("[INTRO] speaker_id sync %s: %s -> %s",
                                     name, ks.speaker_id, sid)
                            ks.speaker_id = sid
                            break
                log.info("[INTRO] name-tell triple: %s face committed "
                         "(id=%s, %d crops)", name,
                         getattr(res, "speaker_id", None), len(crops))
            else:
                log.info("[INTRO] face commit declined for %s (status=%s) — "
                         "voice-only promotion stands", name,
                         getattr(res, "status", "?"))
        except Exception:
            log.exception("[INTRO] face commit failed for %s — voice-only "
                          "promotion stands", name)

    async def _say_confirm(self, name: str) -> None:
        confirm_prompt = (
            f'You just heard someone say their name is "{name.title()}". '
            f'Repeat the name back to confirm, like "Did you say {name.title()}?" '
            f'Keep it brief and in-character.'
        )
        result = await self._turn.say(confirm_prompt)
        log.info("[TIMMY] %s (name confirmation)", result.text)


def _extract_name_from_response(text: str) -> str | None:
    """Try to extract a name from a short response like 'I'm Erin' or 'My name
    is Erin'. Conservative: rejects evasive, playful, or non-name responses.
    Moved verbatim from main.Orchestrator (2026-06-06 refactor)."""
    text = text.strip().rstrip(".!?,")
    lower = text.lower()

    # Reject obviously evasive/playful responses early
    _EVASIVE_PHRASES = [
        "not sure", "don't know", "i'm not", "none of your",
        "wouldn't you", "guess", "figure it out", "not telling",
        "not allowed", "can't tell", "secret", "classified",
        "why do you", "does it matter", "who cares", "i don't",
        "i can't", "i won't", "not going to", "rather not",
    ]
    if any(phrase in lower for phrase in _EVASIVE_PHRASES):
        return None

    patterns = [
        r"(?:my name is|i'm|i am|it's|call me|they call me|name's|i go by)\s+(\w+)",
        r"^(\w+)$",  # just a single word
    ]

    # Expanded rejection set
    _NOT_NAMES = {
        # Fillers & affirmations
        "yes", "no", "yeah", "yep", "nope", "nah", "sure", "ok", "okay",
        "hi", "hey", "hello", "bye", "thanks",
        # Articles & pronouns
        "the", "a", "an", "this", "that", "it", "i",
        # Question words
        "what", "who", "why", "how", "when", "where", "which",
        # Common verbs/adverbs
        "well", "just", "um", "uh", "like", "really", "actually",
        "here", "there", "going", "doing", "trying", "thinking",
        # Adjectives that match "I'm X" but aren't names
        "not", "fine", "good", "great", "tired", "busy", "sorry",
        "happy", "sad", "bored", "confused", "lost", "done",
        "sure", "ready", "allowed", "able", "afraid", "certain",
        "kidding", "joking", "serious", "curious", "interested",
        # Negative constructs
        "nobody", "nothing", "none", "never",
    }

    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            name = m.group(1)
            if name not in _NOT_NAMES and len(name) >= 2:
                return name
    return None
