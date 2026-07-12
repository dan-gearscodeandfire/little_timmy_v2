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
    """Returns scripted /faces states in fetch_full() shape."""

    def __init__(self, states):
        self._states = list(states)
        self.i = 0

    async def fetch_full(self):
        s = self._states[min(self.i, len(self._states) - 1)]
        self.i += 1
        return s

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


async def _run_seq(cap, states):
    """Feed each state through one proximity poll; return list of fire reasons."""
    fires = []
    async def _rec(jpeg):
        fires.append(cap._last_face_height_frac)
    cap._on_frame = _rec  # type: ignore
    # _fire_proximity fetches a frame; stub it to a constant jpeg.
    async def _fake_fetch(reason, res=None):
        return b"jpeg"
    cap._fetch_frame = _fake_fetch  # type: ignore
    cap._face_client = _FakeFaceClient(states)
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

    print("\nALL PROXIMITY-GATE CASES PASSED" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
