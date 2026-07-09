"""LED-mic anchor — the booth engagement token (Dan's design, 2026-07-06).

At EXPO a green-LED handheld mic marks the person Timmy is engaged with:
vision detects the LIT mic in frame and the face directly above it is the
speaker's. This module holds the anchor STATE (in-process, deliberately not
in runtime_toggles: a restart wipes it, which degrades to the shipped
global-dark identity gate — the safe default) plus the pure geometry that
picks the anchored face.

Two writers share the state: the ``POST /api/anchor`` stub (pre-hardware
bench driving) and the CV LED detector (per-turn, on the /capture frames the
face recognizer already grabs). Freshness is TTL-based — the anchor is
"active" only while recently refreshed, so a racked mic / occluded LED decays
back to dark with no further code.

Knobs (enable flag, TTL, geometry tolerance) live in runtime_toggles so
they're live-tunable at the booth; only the volatile state lives here.

Consent note (Dan ruling 2026-07-06): mic-in-hand is IMPLIED consent to store
the anchored face. The anchor un-darks the speech identity dialogs
(introductions, misID correction, enroll intent) but NOT the face-consent
FSM — asking permission the mic already granted is redundant, so FaceEnroller
stays on the pure regime+override predicate.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from persistence import runtime_toggles

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorState:
    captured_at: float                       # unix ts of the last refresh
    led_xy: Optional[tuple] = None           # (x, y) px of the lit LED, if known
    anchored_bbox: Optional[tuple] = None    # (x_min, y_min, x_max, y_max) of the paired face
    source: str = "stub"                     # "stub" (API-declared) | "cv" (LED detector)
    ttl_s: Optional[float] = None            # per-set override; None -> anchor_ttl_s toggle
    # Recognized (enrolled) name of the anchored face, or None when the face
    # under the LED is unrecognized (the ordinary visitor case). Written by the
    # CV detector path only — a stub anchor carries None. This is what lets the
    # gate/crop consumers BIND the anchor to a turn's voice attribution
    # (binding_ok below) instead of trusting any speaker inside the TTL window.
    anchored_name: Optional[str] = None


_state: Optional[AnchorState] = None
_lock = threading.Lock()


def set_anchor(led_xy=None, anchored_bbox=None, *, source: str, ttl_s: Optional[float] = None,
               anchored_name: Optional[str] = None) -> None:
    """Declare/refresh the anchor. Each call restamps captured_at (TTL restart)."""
    global _state
    with _lock:
        was = _state
        _state = AnchorState(
            captured_at=time.time(),
            led_xy=tuple(led_xy) if led_xy is not None else None,
            anchored_bbox=tuple(anchored_bbox) if anchored_bbox is not None else None,
            source=source,
            ttl_s=ttl_s,
            anchored_name=anchored_name,
        )
        st = _state
    # INFO only on a state TRANSITION (dark->active, source or identity flip);
    # steady refreshes log at debug — the CV path republishes every poll tick /
    # per-turn grab and was spamming INFO per frame (code review 7-07).
    was_fresh = was is not None and (time.time() - was.captured_at) <= (
        was.ttl_s if was.ttl_s is not None else float(runtime_toggles.get("anchor_ttl_s")))
    if (not was_fresh or was.source != st.source
            or was.anchored_name != st.anchored_name):
        log.info("[ANCHOR] set source=%s led_xy=%s ttl=%s name=%s", source, led_xy,
                 ttl_s or "toggle", anchored_name or "-")
    else:
        log.debug("[ANCHOR] refresh source=%s led_xy=%s", source, led_xy)


def clear_anchor() -> None:
    """Drop the anchor immediately (mic racked / operator clear)."""
    global _state
    with _lock:
        was = _state
        _state = None
    if was is not None:
        log.info("[ANCHOR] cleared (was source=%s, age=%.1fs)", was.source, time.time() - was.captured_at)


def get_anchor() -> Optional[AnchorState]:
    """Current state regardless of freshness (None when never set/cleared)."""
    with _lock:
        return _state


def anchor_active() -> bool:
    """True while the anchor was refreshed within its TTL."""
    with _lock:
        st = _state
    if st is None:
        return False
    ttl = st.ttl_s if st.ttl_s is not None else float(runtime_toggles.get("anchor_ttl_s"))
    return (time.time() - st.captured_at) <= ttl


def gate_disjunct(speaker_name: Optional[str] = None) -> bool:
    """The identity-dialog gate's third input (regime + override + THIS).

    True only when the anchor feature is enabled AND the anchor is fresh —
    ORed into main's ``_dialogs_ok`` so the verified mic-holder gets identity
    dialogs back under EXPO. Deliberately NOT folded into
    runtime_toggles.identity_dialogs_allowed(): that predicate stays pure
    regime+override (it also guards the face-consent FSM, which the anchor
    must NOT un-dark), and persistence must not import presence.

    ``speaker_name`` (F7 binding, code review 7-07): when given, the fresh
    anchor must also BIND to this turn's voice attribution (binding_ok) —
    without it, a fresh anchor un-darked the identity-MUTATION surfaces for
    EVERY speaker in the TTL window (an off-mic bystander could open the
    misID confirm FSM on the mic-holder's anchor). None preserves the plain
    freshness verdict (status surfaces, non-speaker consumers).
    """
    if not (bool(runtime_toggles.get("anchor_enabled")) and anchor_active()):
        return False
    return speaker_name is None or binding_ok(speaker_name)


def binding_ok(speaker_name: str) -> bool:
    """Is this turn's VOICE attribution consistent with the ANCHORED face?

    The anchor marks a face (position above the LED); the turn's speaker comes
    from voice. They are different sensors and can disagree — an off-mic
    bystander's voice with the mic-holder's face was the wrong-face-commit
    vector (F1) and the gate-for-everyone hole (F7). Consistency contract:

    - stub anchor: True (operator-declared bench state — the operator owns
      attribution; binding would break every stub-driven test flow).
    - anchored face RECOGNIZED as enrolled X: speaker must be X.
    - anchored face UNRECOGNIZED (None): speaker must be an unknown_N — the
      ordinary visitor case. An ENROLLED voice with an unrecognized anchored
      face is exactly the off-mic-collapse signature, so it does NOT bind.
      (Cost: an on-mic visitor whose voice misIDs as enrolled loses the misID
      protest under EXPO — acceptable at the booth; Shop is ungated and the
      supervised override restores everything.)
    """
    with _lock:
        st = _state
    if st is None:
        return False
    if st.source == "stub":
        return True
    if st.anchored_name is None:
        return speaker_name.startswith("unknown_")
    return speaker_name == st.anchored_name


def speech_dialogs_allowed(speaker_name: Optional[str] = None) -> bool:
    """THE gate for the SPEECH identity dialogs (introductions, misID
    correction, enroll intent): pure regime+override OR a fresh, bound anchor.
    Named helper for the disjunction that was previously inlined at each
    doorway site (code review 7-07). The face-consent FSM must keep using
    consent_allowed() — the anchor never un-darks it (implied consent)."""
    return runtime_toggles.identity_dialogs_allowed() or gate_disjunct(speaker_name)


def consent_allowed() -> bool:
    """Gate for the face-consent FSM: pure regime+override, NO anchor disjunct
    (mic-in-hand is implied consent — the offer would be noise). Alias of
    runtime_toggles.identity_dialogs_allowed() so doorway code reads as the
    consent/speech split it implements."""
    return runtime_toggles.identity_dialogs_allowed()


def pick_anchored_face(face_bboxes, led_xy, image_size, *, x_tol_frac: Optional[float] = None):
    """Pick the face directly above the lit LED — or abstain.

    A candidate face's bbox horizontal center must be within
    ``x_tol_frac * frame_width`` of the LED x AND its bbox vertical CENTER
    must sit above (smaller y than) the LED y. Center, not bottom (F8 fix,
    live 7-08): a mic held at the MOUTH puts the LED at chin height — right
    at bbox bottom — so the old ``y_max < led_y`` flickered by a few px on
    the natural speaking hold; the LED-in-lower-half rule passes every live
    mouth-hold frame while still rejecting a face BELOW the LED. EXACTLY one
    candidate -> its index in ``face_bboxes``; zero or two-plus -> None. No
    nearest-wins among two: ambiguity is an abstain, the same contract as
    the sole-face rule (never guess which face is talking).

    Pure function (hermetic-testable). bboxes are (x_min, y_min, x_max, y_max)
    px; led_xy is (x, y) px; image_size is (width, height).
    """
    if not face_bboxes or led_xy is None or not image_size:
        return None
    if x_tol_frac is None:
        x_tol_frac = float(runtime_toggles.get("anchor_x_tol_frac"))
    led_x, led_y = led_xy
    x_tol = x_tol_frac * float(image_size[0])
    candidates = []
    for i, (x_min, y_min, x_max, y_max) in enumerate(face_bboxes):
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        if abs(center_x - led_x) <= x_tol and center_y < led_y:
            candidates.append(i)
    if len(candidates) == 1:
        return candidates[0]
    return None
