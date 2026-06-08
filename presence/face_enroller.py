"""Interactive auto-enrollment FSM — LT notices a genuinely new person who is
talking to him, asks consent + name, guides a head-pose capture, and verifies
the face now recognises before announcing.

This is the dialog/orchestration layer that sits on top of the read-only
discriminator in `new_face_trigger.py`. The trigger answers the hard question
("is this a new person or a known one at a bad angle?"); this module runs the
human interaction once a CANDIDATE fires.

Locked design decisions (Dan, 2026-06-07)
-----------------------------------------
* Autonomy:    FULLY AUTONOMOUS — offer to anyone who triggers, Dan needn't be present.
* Consent:     ALWAYS ASK first-time. A "no" persists a short-lived decline cooldown.
* Engagement:  MUST BE TALKING TO LT — only offer when a recent unknown-voice utterance
               coincides with the new-face track (gated by the `engaged` flag the
               poll loop passes in). Background passers-by never get an offer.

Two drivers, one state machine
------------------------------
The FSM is advanced from two places, mirroring how `Introductions` is consulted
at the top of each turn:

  * `observe_faces(faces, image_size, engaged)` — called from the face-poll loop
    (~2 Hz). Runs the trigger; on a fresh CANDIDATE while engaged + IDLE it makes
    the spoken consent offer. Also enforces response timeouts and kicks the
    capture/verify tail once a name is confirmed.
  * `handle(user_text, speaker_name)` — called from the conversation doorway when
    a heard utterance lands. Advances OFFERING / ASK_NAME / CONFIRM_NAME. Returns
    `EnrollOutcome(handled=True)` to suppress the normal LLM turn while a consent
    dialog is in flight.

All speaking goes through injected seams (`say` for in-character LLM+TTS lines,
`speak` for tightly-timed fixed lines during capture) so the FSM unit-tests with
fakes and never imports TTS/LLM directly. Run `python -m presence.face_enroller
--selftest` for the synthetic flow tests.

Provenance: auto-enrolled identities are recorded via the `on_enrolled` callback
(name, source="auto", timestamp, samples) so they can be audited / pruned and a
future "forget me" command can target them.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from .new_face_trigger import NewFaceTrigger, TriggerConfig

log = logging.getLogger("timmy")


# Reuse the conservative name extractor the voice-introduction dialog already uses.
try:  # pragma: no cover - import shape differs only under direct module run
    from conversation.introductions import _extract_name_from_response
except Exception:  # pragma: no cover
    _extract_name_from_response = None  # selftest injects a stub


import re as _re

# Consent parsing, negation-aware ordering (never substring). STT audits out a
# bare "yes", so real consent replies are LONG and keyword-laden ("yes, you can
# remember my face", "I give you affirmative consent, Timmy") — they must parse
# yes, while "I do not consent" / "no thanks" must parse no. Order matters:
#   1. affirmative-despite-negative idioms ("don't mind", "why not") -> yes
#   2. explicit negation (idiom / word-boundary / "do not") -> no
#   3. affirmative idioms + words (incl. consent vocabulary) -> yes
# See live false-declines 2026-06-07 ("I don't mind…", "That is not how…").
_YES_OVERRIDE = (
    "don't mind", "dont mind", "do not mind", "wouldn't mind", "wouldnt mind",
    "don't care", "dont care", "why not", "no problem", "no objection",
)
_NO_IDIOM = (
    "rather not", "no thanks", "no thank", "not now", "maybe later", "leave it",
    "skip it", "forget it", "please don't", "please dont", "i'd prefer not",
    "id prefer not", "absolutely not", "no way", "i'd rather", "id rather not",
    "do not", "did not", "didn't",
)
_NO_WORDS = ("no", "nope", "nah", "don't", "dont", "never", "stop")
_YES_IDIOM = (
    "affirmative", "consent", "permission", "you have my", "i agree", "agreed",
    "i'm okay with", "im okay with", "fine by me", "feel free", "be my guest",
    "go ahead", "go right ahead", "go for it", "knock yourself", "of course",
    "please do", "sounds good", "that's fine", "thats fine", "i'm fine with",
    "im fine with", "i'd love", "id love", "that's right", "thats right",
    "remember me", "remember my", "memoriz", "do it", "you can",
)
_YES_WORDS = ("yes", "yeah", "yep", "yup", "ok", "okay", "alright", "sure",
              "absolutely", "fine", "correct", "right", "exactly", "agree")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


class State(str, Enum):
    IDLE = "IDLE"
    OFFERING = "OFFERING"          # consent question asked, awaiting yes/no
    ASK_NAME = "ASK_NAME"          # consent given, awaiting a name
    CONFIRM_NAME = "CONFIRM_NAME"  # name heard, awaiting yes/no
    ENROLLING = "ENROLLING"        # capture + verify task running
    COOLDOWN = "COOLDOWN"          # recently declined/failed, suppress re-offer


@dataclass(frozen=True)
class EnrollerConfig:
    enabled: bool = _b("TIMMY_AUTO_ENROLL_ENABLED", False)
    # How long after the most recent unknown-voice utterance a CANDIDATE still
    # counts as "talking to me" (engagement gate, decision 3).
    engagement_window_s: float = _f("TIMMY_AE_ENGAGEMENT_WINDOW_S", 12.0)
    # How long to wait for a verbal reply at each dialog step before giving up.
    response_timeout_s: float = _f("TIMMY_AE_RESPONSE_TIMEOUT_S", 45.0)
    # After a decline / failure / completion, suppress any new offer this long.
    cooldown_s: float = _f("TIMMY_AE_OFFER_COOLDOWN_S", 90.0)
    # Capture parameters handed to /face_db/enroll/stream.
    capture_count: int = _i("TIMMY_AE_CAPTURE_COUNT", 24)  # ~5 frames/zone over frontal+L/R/U/D
    capture_interval_s: float = _f("TIMMY_AE_CAPTURE_INTERVAL_S", 0.7)
    # Verify step: poll /faces this many times looking for the new name.
    verify_polls: int = _i("TIMMY_AE_VERIFY_POLLS", 8)
    verify_interval_s: float = _f("TIMMY_AE_VERIFY_INTERVAL_S", 0.6)
    # A verify face counts as "recognises now" if its label matches OR its
    # distance dips below this (mirrors streamerpi match threshold).
    verify_match_dist: float = _f("TIMMY_AE_VERIFY_MATCH_DIST", 0.45)


@dataclass(frozen=True)
class EnrollOutcome:
    """Result of consulting the enroller at the top of a turn.

    handled=True  -> a consent-dialog follow-up was spoken (or the turn consumed);
                     the doorway should return without running a normal turn.
    handled=False -> not our turn; fall through to the normal pipeline."""
    handled: bool


# Type aliases for the injected seams.
SayFn = Callable[[str], Awaitable]          # in-character LLM+TTS line; returns TurnResult-ish
SpeakFn = Callable[[str], Awaitable]        # fixed TTS line
EnrollStreamFn = Callable[..., "AsyncIterator"]  # (name, count, interval_s, mode) -> async iter of (evt, payload)
VerifyFn = Callable[[], Awaitable]          # () -> list[face dict]
OnEnrolledFn = Callable[[str, dict], None]


class FaceEnroller:
    def __init__(
        self,
        *,
        say: SayFn,
        speak: SpeakFn,
        enroll_stream: EnrollStreamFn,
        verify_faces: VerifyFn,
        turn_lock: asyncio.Lock,
        on_enrolled: Optional[OnEnrolledFn] = None,
        now: Callable[[], float] = None,
        cfg: Optional[EnrollerConfig] = None,
        trigger: Optional[NewFaceTrigger] = None,
        name_extractor: Optional[Callable[[str], Optional[str]]] = None,
    ):
        import time as _time
        self._say = say
        self._speak = speak
        self._enroll_stream = enroll_stream
        self._verify = verify_faces
        self._turn_lock = turn_lock
        self._on_enrolled = on_enrolled
        self._now = now or _time.time
        self.cfg = cfg or EnrollerConfig()
        self._trigger = trigger or NewFaceTrigger(TriggerConfig())
        self._extract_name = name_extractor or _extract_name_from_response

        self.state = State.IDLE
        self._candidate_track: Optional[int] = None
        self._pending_name: Optional[str] = None
        self._deadline: float = 0.0      # response timeout for the active dialog step
        self._cooldown_until: float = 0.0
        self._capture_task: Optional[asyncio.Task] = None
        self._consent_nudged: bool = False  # one clarify nudge per offer

    # ------------------------------------------------------------------ #
    @property
    def awaiting(self) -> bool:
        """True while a consent dialog owns the conversation turns."""
        return self.state in (State.OFFERING, State.ASK_NAME, State.CONFIRM_NAME)

    @property
    def busy(self) -> bool:
        return self.state != State.IDLE

    def _reset(self, cooldown: bool = True) -> None:
        if cooldown:
            self._cooldown_until = self._now() + self.cfg.cooldown_s
        self.state = State.IDLE
        self._candidate_track = None
        self._pending_name = None
        self._deadline = 0.0
        self._consent_nudged = False

    # ------------------------------------------------------------------ #
    # Driver 1: the face-poll loop
    # ------------------------------------------------------------------ #
    async def observe_faces(self, faces: list, image_size, engaged: bool) -> None:
        """Ingest one /faces tick. Runs the trigger and, on a fresh CANDIDATE
        while engaged + IDLE, makes the spoken consent offer. Also enforces the
        per-step response timeout so a person who walks away doesn't wedge us."""
        if not self.cfg.enabled:
            return
        now = self._now()
        decisions = self._trigger.update(faces, image_size, now)

        # Timeout a stalled dialog (person stopped responding / left).
        if self.awaiting and now >= self._deadline:
            log.info("[AUTOENROLL] no reply at %s within %.0fs -> abort",
                     self.state.value, self.cfg.response_timeout_s)
            await self._safe_speak("It's alright, maybe another time.")
            self._reset(cooldown=True)
            return

        # Only originate a new offer from IDLE, when engaged, off cooldown, and
        # while no spoken turn is in flight (mirrors the proactive-speech guard).
        if self.state != State.IDLE:
            return
        if not engaged:
            return
        if now < self._cooldown_until:
            return
        if self._turn_lock.locked():
            return  # a real turn is happening; try again next tick

        fresh = next((d for d in decisions if d.is_candidate), None)
        if fresh is None:
            return

        self._candidate_track = fresh.track_id
        log.info("[AUTOENROLL] CANDIDATE trk%d (min_dist=%.2f, h=%.0fpx) -> offering consent",
                 fresh.track_id, fresh.min_dist, fresh.median_h)
        await self._make_offer()

    async def _make_offer(self) -> None:
        """Acquire the turn lock and speak the consent question."""
        if self._turn_lock.locked():
            self._candidate_track = None
            return
        await self._turn_lock.acquire()
        try:
            # Plain, fixed wording (NOT persona-reworded say()) — a consent ask
            # must be unmistakable. Live-proven 2026-06-07 that LLM-reworded
            # offers read as banter and the user misses the question entirely.
            await self._safe_speak(
                "Hey, quick question — I don't recognise your face yet. Can I "
                "save it so I know you next time? Say something like 'yes, you "
                "have my consent', or 'no thanks'."
            )
            self.state = State.OFFERING
            self._deadline = self._now() + self.cfg.response_timeout_s
            log.info("[AUTOENROLL] offer spoken (trk%s), awaiting consent",
                     self._candidate_track)
        finally:
            self._turn_lock.release()

    # ------------------------------------------------------------------ #
    # Driver 2: the conversation doorway (a heard utterance landed)
    # ------------------------------------------------------------------ #
    async def handle(self, user_text: str, speaker_name: str) -> EnrollOutcome:
        """Advance an in-flight consent dialog with a heard utterance. Runs
        inside the turn lock (process_speech holds it), so it may speak directly."""
        if not self.cfg.enabled or not self.awaiting:
            return EnrollOutcome(handled=False)

        # Refresh the deadline — they're still here and talking.
        self._deadline = self._now() + self.cfg.response_timeout_s

        if self.state == State.OFFERING:
            verdict = self._consent_verdict(user_text)
            if verdict == "no":
                log.info("[AUTOENROLL] consent declined: %r", user_text)
                await self._safe_speak("No problem at all.")
                self._reset(cooldown=True)
                return EnrollOutcome(handled=True)
            if verdict == "yes":
                # A name volunteered alongside consent ("sure, I'm Sarah") skips ASK_NAME.
                name = self._extract(user_text)
                if name:
                    self._pending_name = name
                    self.state = State.CONFIRM_NAME
                    await self._safe_speak(
                        f"Great. Did you say your name is {name.title()}? "
                        f"Tell me 'yes that's right' or 'no'."
                    )
                else:
                    self.state = State.ASK_NAME
                    await self._safe_speak(
                        "Great. What's your name? Tell me, like 'my name is Dan'."
                    )
                return EnrollOutcome(handled=True)
            # Unclear: nudge once for a clear yes/no rather than abandoning.
            if not self._consent_nudged:
                self._consent_nudged = True
                log.info("[AUTOENROLL] unclear consent %r -> nudge", user_text)
                await self._safe_speak("Sorry, I need a clear yes or no — can I save your face?")
            else:
                log.info("[AUTOENROLL] still unclear %r -> abort", user_text)
                await self._safe_speak("No worries, maybe another time.")
                self._reset(cooldown=True)
            return EnrollOutcome(handled=True)

        if self.state == State.ASK_NAME:
            name = self._extract(user_text)
            if name:
                self._pending_name = name
                self.state = State.CONFIRM_NAME
                await self._safe_speak(
                    f"Did you say {name.title()}? Tell me 'yes that's right' or 'no'."
                )
            else:
                log.info("[AUTOENROLL] no name in %r -> abort", user_text)
                await self._safe_speak("No worries, we can sort that out later.")
                self._reset(cooldown=True)
            return EnrollOutcome(handled=True)

        if self.state == State.CONFIRM_NAME:
            verdict = self._consent_verdict(user_text)
            if verdict == "yes":
                name = self._pending_name
                log.info("[AUTOENROLL] name confirmed: %s -> capture", name)
                self.state = State.ENROLLING
                # Capture is long (~15s) and paces poses; run it off the turn so
                # this turn can release the lock, which the capture task re-takes.
                self._capture_task = asyncio.create_task(self._run_capture(name))
                return EnrollOutcome(handled=True)
            if verdict == "no":
                # Wrong guess — re-ask the name.
                self._pending_name = None
                self.state = State.ASK_NAME
                await self._safe_speak("Sorry about that. What's your name?")
                return EnrollOutcome(handled=True)
            # They said something else — maybe the actual name this time.
            name = self._extract(user_text)
            if name:
                self._pending_name = name
                await self._safe_speak(f'Ah — {name.title()}, is that right? Yes or no.')
            else:
                await self._safe_speak("Sorry, I didn't catch that. What's your name?")
                self.state = State.ASK_NAME
            return EnrollOutcome(handled=True)

        return EnrollOutcome(handled=False)

    # ------------------------------------------------------------------ #
    # Capture + verify tail (runs as its own task, owns the turn lock)
    # ------------------------------------------------------------------ #
    async def _run_capture(self, name: str) -> None:
        await self._turn_lock.acquire()
        try:
            await self._safe_speak(
                f"Perfect. Hold still and look right at me, {name.title()}. "
                "I'm going to take a few looks."
            )
            saved = await self._drive_enroll(name)
            if not saved:
                await self._safe_speak(
                    "Hmm, I couldn't get a clear look. We can try again later."
                )
                self._reset(cooldown=True)
                return

            ok = await self._verify_recognition(name)
            if ok:
                await self._safe_say(f"Got it — nice to meet you, {name.title()}.")
                if self._on_enrolled:
                    try:
                        self._on_enrolled(name, {
                            "source": "auto",
                            "enrolled_at": self._now(),
                            "samples": self.cfg.capture_count,
                        })
                    except Exception:
                        log.exception("[AUTOENROLL] on_enrolled callback failed")
                log.info("[AUTOENROLL] enrolled + verified: %s", name)
            else:
                await self._safe_speak(
                    "I saved that, though I'm not certain it took — I'll let you "
                    "know if I don't recognise you."
                )
                log.warning("[AUTOENROLL] %s enrolled but verify did not confirm", name)
        except Exception:
            log.exception("[AUTOENROLL] capture failed")
        finally:
            self._reset(cooldown=True)
            self._turn_lock.release()

    # Pose cues spread across the capture so head movement yields distinct
    # prototypes (the multi-prototype DB clusters by angle -> min-distance match
    # kills the recognition flap). Slow movement only — fast motion blurs frames
    # the Pi skips. (fraction-of-count, line).
    _POSE_CUES = (
        (0.20, "Now turn your head slowly to the left."),
        (0.40, "And slowly to the right."),
        (0.60, "Now tip your chin up a little."),
        (0.80, "And down, like you're looking just below me."),
    )

    async def _drive_enroll(self, name: str) -> bool:
        """Run the SSE enroll stream, pacing left/right/up/down pose cues off the
        progress events. Returns True if the Pi reported saved."""
        saved = False
        count = max(1, self.cfg.capture_count)
        idx = 0
        try:
            async for evt, payload in self._enroll_stream(
                name, self.cfg.capture_count, self.cfg.capture_interval_s, "add"
            ):
                if evt == "progress":
                    captured = int(payload.get("captured", 0))
                    while idx < len(self._POSE_CUES) and captured >= self._POSE_CUES[idx][0] * count:
                        await self._safe_speak(self._POSE_CUES[idx][1])
                        idx += 1
                elif evt == "complete":
                    saved = bool(payload.get("saved"))
                elif evt == "error":
                    log.warning("[AUTOENROLL] enroll error: %s", payload.get("error"))
                    return False
        except Exception:
            log.exception("[AUTOENROLL] enroll stream failed")
            return False
        return saved

    async def _verify_recognition(self, name: str) -> bool:
        """Poll /faces a few times; pass if the new identity now reads back."""
        target = name.strip().lower()
        for _ in range(self.cfg.verify_polls):
            faces = await self._verify() or []
            for fdict in faces:
                fname = str(fdict.get("name", "")).strip().lower()
                dist = float(fdict.get("distance", 1.0))
                if fname == target:
                    return True
                if dist < self.cfg.verify_match_dist:
                    return True
            await asyncio.sleep(self.cfg.verify_interval_s)
        return False

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _consent_verdict(self, text: str) -> str:
        """Classify a yes/no reply: 'yes' | 'no' | 'unclear'. Idiom phrases win
        first (so "I don't mind" = yes), then word-boundary tokens (so "no" does
        not match inside "not"/"know")."""
        lower = text.lower().strip().rstrip(".!?,")
        # 1. affirmative-despite-negative idioms win first
        if any(p in lower for p in _YES_OVERRIDE):
            return "yes"
        # 2. explicit negation
        if any(p in lower for p in _NO_IDIOM):
            return "no"
        if any(_re.search(rf"\b{_re.escape(w)}\b", lower) for w in _NO_WORDS):
            return "no"
        # 3. affirmatives (idioms incl. consent vocab, then plain words)
        if any(p in lower for p in _YES_IDIOM):
            return "yes"
        if any(_re.search(rf"\b{_re.escape(w)}\b", lower) for w in _YES_WORDS):
            return "yes"
        return "unclear"

    def _extract(self, text: str) -> Optional[str]:
        if not self._extract_name:
            return None
        try:
            return self._extract_name(text)
        except Exception:
            return None

    async def _safe_say(self, prompt: str):
        try:
            return await self._say(prompt)
        except Exception:
            log.exception("[AUTOENROLL] say failed")
            return None

    async def _safe_speak(self, text: str):
        try:
            return await self._speak(text)
        except Exception:
            log.exception("[AUTOENROLL] speak failed")
            return None


# --------------------------------------------------------------------------
# Self-test — synthetic end-to-end flow with fakes (no network, no asyncio loop
# blocking). Run: python -m presence.face_enroller --selftest
# --------------------------------------------------------------------------

def _run_selftest() -> int:
    import asyncio as aio

    W, H = 640, 360

    def box(cx, h=90):
        w = int(h * 0.8)
        return [int(cx - w / 2), int(H / 2 - h / 2), w, h]

    class Clock:
        def __init__(self):
            self.t = 1000.0
        def __call__(self):
            return self.t
        def tick(self, dt):
            self.t += dt

    class Harness:
        def __init__(self, clock, enroll_saved=True, verify_name=True):
            self.said = []
            self.spoke = []
            self.enrolled = []
            self._enroll_saved = enroll_saved
            self._verify_name = verify_name
            self.lock = aio.Lock()
            self.clock = clock

        async def say(self, prompt):
            self.said.append(prompt)
            return type("R", (), {"text": prompt})()

        async def speak(self, text):
            self.spoke.append(text)

        async def enroll_stream(self, name, count, interval_s, mode):
            yield ("started", {"name": name})
            for i in range(count):
                yield ("progress", {"captured": i + 1, "ok": True})
            yield ("complete", {"saved": self._enroll_saved, "name": name})

        async def verify_faces(self):
            if self._verify_name:
                return [{"name": "Sarah", "distance": 0.2, "confidence": "high",
                         "bbox": box(W / 2)}]
            return [{"name": "unknown", "distance": 0.8, "confidence": "low",
                     "bbox": box(W / 2)}]

        def on_enrolled(self, name, meta):
            self.enrolled.append((name, meta))

    def build(clock, **kw):
        h = Harness(clock, **kw)
        cfg = EnrollerConfig(enabled=True, verify_polls=2, verify_interval_s=0.0,
                             capture_count=6, capture_interval_s=0.0,
                             cooldown_s=90.0, response_timeout_s=25.0,
                             engagement_window_s=12.0)
        # Name extractor stub mirroring the real conservative one for tests.
        import re

        def extract(text):
            m = re.search(r"(?:i'm|i am|my name is|it's|call me)\s+(\w+)", text.lower())
            if m:
                return m.group(1)
            m = re.search(r"^(\w+)$", text.strip().lower())
            if m and m.group(1) not in ("yes", "no", "yeah", "sure", "nope"):
                return m.group(1)
            return None

        e = FaceEnroller(
            say=h.say, speak=h.speak, enroll_stream=h.enroll_stream,
            verify_faces=h.verify_faces, turn_lock=h.lock,
            on_enrolled=h.on_enrolled, now=clock, cfg=cfg,
            name_extractor=extract,
        )
        return h, e

    async def feed_stranger(e, clock, ticks=40, engaged=True):
        """Drive a true-stranger /faces stream until a CANDIDATE offer fires."""
        for i in range(ticks):
            faces = [{"name": "unknown", "distance": 0.75, "confidence": "low",
                      "bbox": box(W / 2)}]
            await e.observe_faces(faces, (W, H), engaged=engaged)
            clock.tick(0.25)
            if e.state == State.OFFERING:
                return True
        return False

    passed = failed = 0

    def check(label, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    async def scenario_happy():
        print("Scenario A: stranger talks -> consent -> name -> capture -> verify -> announce")
        clock = Clock()
        h, e = build(clock)
        fired = await feed_stranger(e, clock)
        check("offer made (state OFFERING)", fired and e.state == State.OFFERING)
        check("consent question spoken",
              any("save it" in s.lower() and "consent" in s.lower() for s in h.spoke))

        out = await e.handle("yeah sure", "unknown_3")
        check("consent yes handled", out.handled and e.state == State.ASK_NAME)

        out = await e.handle("I'm Sarah", "unknown_3")
        check("name captured -> CONFIRM_NAME", out.handled and e.state == State.CONFIRM_NAME)
        check("pending name is Sarah", e._pending_name == "sarah")

        out = await e.handle("yes", "unknown_3")
        check("confirm yes handled", out.handled)
        # capture task scheduled; await it
        if e._capture_task:
            await e._capture_task
        check("enrolled callback fired for sarah",
              any(n == "sarah" and m["source"] == "auto" for n, m in h.enrolled))
        check("announce spoken", any("nice to meet you" in s.lower() for s in h.said))
        check("returned to IDLE", e.state == State.IDLE)
        check("pose cues spoken", any("left" in s.lower() for s in h.spoke))

    async def scenario_decline():
        print("Scenario B: stranger declines -> cooldown, no enroll")
        clock = Clock()
        h, e = build(clock)
        await feed_stranger(e, clock)
        out = await e.handle("no thanks", "unknown_4")
        check("decline handled", out.handled)
        check("no enrollment", not h.enrolled)
        check("IDLE + cooldown set", e.state == State.IDLE and e._cooldown_until > clock())
        # An immediate re-trigger must be suppressed by cooldown.
        fired = await feed_stranger(e, clock, ticks=30)
        check("re-offer suppressed during cooldown", not fired)

    async def scenario_consent_with_name():
        print("Scenario C: name volunteered with consent -> skips ASK_NAME")
        clock = Clock()
        h, e = build(clock)
        await feed_stranger(e, clock)
        out = await e.handle("sure, I'm Marcus", "unknown_5")
        check("jumped to CONFIRM_NAME", e.state == State.CONFIRM_NAME and e._pending_name == "marcus")
        check("handled", out.handled)

    async def scenario_not_engaged():
        print("Scenario D: stranger present but NOT talking to LT -> no offer")
        clock = Clock()
        h, e = build(clock)
        fired = await feed_stranger(e, clock, engaged=False)
        check("no offer when not engaged", not fired and e.state == State.IDLE)
        check("nothing spoken", not h.said and not h.spoke)

    async def scenario_timeout():
        print("Scenario E: offer made, no reply -> timeout abort")
        clock = Clock()
        h, e = build(clock)
        await feed_stranger(e, clock)
        check("offering", e.state == State.OFFERING)
        # Advance past the response deadline and tick the poller.
        clock.tick(30.0)
        await e.observe_faces(
            [{"name": "unknown", "distance": 0.75, "confidence": "low", "bbox": box(W / 2)}],
            (W, H), engaged=True,
        )
        check("timed out to IDLE", e.state == State.IDLE)
        check("graceful timeout line spoken", any("another time" in s.lower() for s in h.spoke))

    async def scenario_verify_fail():
        print("Scenario F: capture saves but verify fails -> hedged announce, no callback")
        clock = Clock()
        h, e = build(clock, verify_name=False)
        await feed_stranger(e, clock)
        await e.handle("yes", "unknown_6")
        await e.handle("Dana", "unknown_6")
        await e.handle("yes", "unknown_6")
        if e._capture_task:
            await e._capture_task
        check("no verified-enroll callback", not h.enrolled)
        check("hedged line spoken", any("not certain it took" in s.lower() for s in h.spoke))
        check("back to IDLE", e.state == State.IDLE)

    async def run_all():
        await scenario_happy()
        await scenario_decline()
        await scenario_consent_with_name()
        await scenario_not_engaged()
        await scenario_timeout()
        await scenario_verify_fail()

    aio.run(run_all())
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Interactive auto-enrollment FSM")
    p.add_argument("--selftest", action="store_true", help="run synthetic flow tests")
    args = p.parse_args()
    if args.selftest:
        return _run_selftest()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
