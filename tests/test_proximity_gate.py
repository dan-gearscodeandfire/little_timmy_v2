"""Hermetic test for the EXPO face-proximity vision-poll gate.

Drives FrameCapture._poll_proximity() directly with a scripted fake /faces
client and asserts the debounced rising-edge firing behavior. No live servers,
no frame transfer, no restart — importing config/capture is enough.

Run standalone:  .venv/bin/python tests/test_proximity_gate.py
"""

import asyncio
import sys

from vision import capture as capture_mod
from vision.capture import FrameCapture


class _FakeFaceClient:
    """Returns scripted /faces states in fetch_full() shape.

    behavior_mode is what fetch_behavior_mode() returns for the trust-the-Pi
    disengage gate; default "idle" so the pure-window disengage path is
    exercised unless a test opts into "track"/"scan".
    """

    def __init__(self, states, behavior_mode="idle"):
        self._states = list(states)
        self.i = 0
        self.behavior_mode = behavior_mode

    async def fetch_full(self):
        s = self._states[min(self.i, len(self._states) - 1)]
        self.i += 1
        return s

    async def fetch_behavior_mode(self):
        return self.behavior_mode

    async def close(self):
        pass


def _state(*heights, frame_h=360):
    """Build a /faces state with faces of the given pixel heights."""
    return {
        "faces": [{"name": "x", "distance": 0.3, "confidence": "high",
                   "bbox": [0, 0, 100, h]} for h in heights],
        "image_size": (640, frame_h),
        "age_s": 0.1,
    }


def _patch_toggles(monkey_values):
    real_get = capture_mod.runtime_toggles.get
    capture_mod.runtime_toggles.get = (  # type: ignore
        lambda k: monkey_values[k] if k in monkey_values else real_get(k))


async def _run_seq(cap, states, behavior_mode="idle"):
    """Feed each state through one proximity poll; return list of fire reasons."""
    fires = []
    async def _rec(jpeg):
        fires.append(cap._last_face_height_frac)
    cap._on_frame = _rec  # type: ignore
    # _fire_proximity fetches a frame; stub it to a constant jpeg.
    async def _fake_fetch(reason, res=None):
        return b"jpeg"
    cap._fetch_frame = _fake_fetch  # type: ignore
    # Preserve an existing fake client (and its behavior mode) across calls on
    # the same cap so a test can flip the Pi mode mid-sequence.
    if not isinstance(getattr(cap, "_face_client", None), _FakeFaceClient):
        cap._face_client = _FakeFaceClient(states, behavior_mode)
    else:
        cap._face_client._states = list(states)
        cap._face_client.i = 0
        cap._face_client.behavior_mode = behavior_mode
    for _ in states:
        await cap._poll_proximity()
    return fires


def main():
    # 20% of 360px = 72px. debounce 2-of-6. refresh off.
    _patch_toggles({
        "vision_proximity_gate_enabled": True,
        "vision_proximity_height_frac": 0.20,
        "vision_proximity_debounce_m": 6,
        "vision_proximity_debounce_n": 2,
        "vision_proximity_refresh_s": 0.0,
        "vision_proximity_pi_track_hold": True,
    })
    THR_PX = 72  # >=72px => qualifies at 360px frame

    ok = True

    # --- Case 1: far standing (all below threshold) never fires ---------------
    cap = FrameCapture()
    fires = asyncio.run(_run_seq(cap, [_state(50)] * 8))       # ~14% height
    assert len(fires) == 0, f"far-standing should not fire, got {len(fires)}"
    assert not cap._prox_engaged
    print("case1 far-standing: no fire OK")

    # --- Case 2: arrival fires ONCE on the rising edge (2nd qualifying poll) ---
    cap = FrameCapture()
    # 3 far polls, then close faces (200px ~= 56%): should fire on the 2nd close.
    seq = [_state(50), _state(50), _state(50),
           _state(200), _state(200), _state(200), _state(200)]
    fires = asyncio.run(_run_seq(cap, seq))
    assert len(fires) == 1, f"arrival should fire exactly once, got {len(fires)}"
    assert cap._prox_engaged, "should be latched engaged after arrival"
    print("case2 arrival: single rising-edge fire OK")

    # --- Case 3: intermittent detection (dropouts) still counts N-of-M --------
    cap = FrameCapture()
    # close, MISS, close within a 6-window => 2-of-6 => engage on 3rd poll.
    seq = [_state(200), _state(), _state(200), _state(200)]
    fires = asyncio.run(_run_seq(cap, seq))
    assert len(fires) == 1, f"intermittent should still fire once, got {len(fires)}"
    print("case3 intermittent (dropout tolerated): fire OK")

    # --- Case 4: disengage after window clears, then re-arm + re-fire ---------
    cap = FrameCapture()
    fires = asyncio.run(_run_seq(cap, [_state(200), _state(200)]))   # engage + fire
    assert len(fires) == 1 and cap._prox_engaged
    # 6 empty polls clear the window -> disengage (re-arm the rising edge).
    asyncio.run(_run_seq(cap, [_state()] * 6))
    assert not cap._prox_engaged, "should disengage after window clears"
    # Reset cooldown, then a fresh arrival must fire AGAIN.
    cap._last_vlm_time = 0.0
    fires2 = asyncio.run(_run_seq(cap, [_state(200), _state(200)]))
    assert len(fires2) == 1, f"re-arrival should fire again, got {len(fires2)}"
    print("case4 disengage -> re-arm -> re-fire: OK")

    # --- Case 5: faces-state callback feeds ledger without VLM fires ----------
    # Far faces (below threshold, never fires VLM) must STILL reach the
    # callback — presence freshness is decoupled from gate engagement.
    cap = FrameCapture()
    fed = []
    cap.set_faces_state_callback(lambda results, img_size: fed.append((results, img_size)))
    fires = asyncio.run(_run_seq(cap, [_state(50)] * 4))        # ~14%, no VLM
    assert len(fires) == 0, "far faces must not fire VLM"
    assert len(fed) == 1, f"throttle: expected 1 callback in a fast burst, got {len(fed)}"
    assert fed[0][0][0]["name"] == "x" and fed[0][1] == (640, 360)
    # With throttle off, every poll with faces feeds; empty polls never do.
    cap2 = FrameCapture()
    fed2 = []
    cap2.set_faces_state_callback(lambda results, img_size: fed2.append(results))
    cap2._faces_state_min_interval = 0.0
    asyncio.run(_run_seq(cap2, [_state(50), _state(), _state(50)]))
    assert len(fed2) == 2, f"expected 2 callbacks (empty poll skipped), got {len(fed2)}"
    print("case5 faces-state callback (no VLM, throttled, skips empty): OK")

    # --- Case 6: trust-the-Pi hold — empty window does NOT disengage while the
    # Pi tracker is still holding the person (mode track/scan); disengages once
    # the Pi reaches idle. This is the working-in-profile / head-turn fix. ------
    cap = FrameCapture()
    fires = asyncio.run(_run_seq(cap, [_state(200), _state(200)]))  # engage + fire
    assert len(fires) == 1 and cap._prox_engaged
    # 6 empty polls, but the Pi is still tracking -> must STAY engaged.
    asyncio.run(_run_seq(cap, [_state()] * 6, behavior_mode="track"))
    assert cap._prox_engaged, "should hold engaged while Pi mode=track"
    # Pi still searching (scan) -> still held.
    asyncio.run(_run_seq(cap, [_state()] * 6, behavior_mode="scan"))
    assert cap._prox_engaged, "should hold engaged while Pi mode=scan"
    # Pi gives up (idle) with the window empty -> now disengage.
    asyncio.run(_run_seq(cap, [_state()] * 6, behavior_mode="idle"))
    assert not cap._prox_engaged, "should disengage once Pi reaches idle"
    print("case6 trust-the-Pi hold (track/scan hold, idle releases): OK")

    # --- Case 7: kill switch OFF -> pure-window disengage regardless of Pi -----
    _patch_toggles({
        "vision_proximity_gate_enabled": True,
        "vision_proximity_height_frac": 0.20,
        "vision_proximity_debounce_m": 6,
        "vision_proximity_debounce_n": 2,
        "vision_proximity_refresh_s": 0.0,
        "vision_proximity_pi_track_hold": False,
    })
    cap = FrameCapture()
    fires = asyncio.run(_run_seq(cap, [_state(200), _state(200)]))
    assert len(fires) == 1 and cap._prox_engaged
    # Pi says track, but the hold is disabled -> window clear must disengage.
    asyncio.run(_run_seq(cap, [_state()] * 6, behavior_mode="track"))
    assert not cap._prox_engaged, "kill switch off -> disengage on empty window"
    print("case7 kill-switch off (pure-window hysteresis restored): OK")

    print("\nALL PROXIMITY-GATE CASES PASSED" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
