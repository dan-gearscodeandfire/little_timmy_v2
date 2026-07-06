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


_state: Optional[AnchorState] = None
_lock = threading.Lock()


def set_anchor(led_xy=None, anchored_bbox=None, *, source: str, ttl_s: Optional[float] = None) -> None:
    """Declare/refresh the anchor. Each call restamps captured_at (TTL restart)."""
    global _state
    with _lock:
        _state = AnchorState(
            captured_at=time.time(),
            led_xy=tuple(led_xy) if led_xy is not None else None,
            anchored_bbox=tuple(anchored_bbox) if anchored_bbox is not None else None,
            source=source,
            ttl_s=ttl_s,
        )
    log.info("[ANCHOR] set source=%s led_xy=%s ttl=%s", source, led_xy, ttl_s or "toggle")


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


def gate_disjunct() -> bool:
    """The identity-dialog gate's third input (regime + override + THIS).

    True only when the anchor feature is enabled AND the anchor is fresh —
    ORed into main's ``_dialogs_ok`` so the verified mic-holder gets identity
    dialogs back under EXPO. Deliberately NOT folded into
    runtime_toggles.identity_dialogs_allowed(): that predicate stays pure
    regime+override (it also guards the face-consent FSM, which the anchor
    must NOT un-dark), and persistence must not import presence.
    """
    return bool(runtime_toggles.get("anchor_enabled")) and anchor_active()


def pick_anchored_face(face_bboxes, led_xy, image_size, *, x_tol_frac: Optional[float] = None):
    """Pick the face directly above the lit LED — or abstain.

    A candidate face's bbox horizontal center must be within
    ``x_tol_frac * frame_width`` of the LED x AND its bbox bottom must sit
    above (smaller y than) the LED y. EXACTLY one candidate -> its index in
    ``face_bboxes``; zero or two-plus -> None. No nearest-wins among two:
    ambiguity is an abstain, the same contract as the sole-face rule
    (never guess which face is talking).

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
        if abs(center_x - led_x) <= x_tol and y_max < led_y:
            candidates.append(i)
    if len(candidates) == 1:
        return candidates[0]
    return None
