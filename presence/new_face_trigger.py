"""New-face trigger spike (MVP) — distinguish a genuinely new person from a
known person seen at a bad angle, using ONLY the existing streamerpi /faces
stream. No Pi changes, no enrollment, read-only.

Why this is the hard part
-------------------------
SFace's per-frame identity label flaps `Dan <-> unknown` constantly at booth
distance (good frames ~0.40 cosine, bad frames 0.77-0.94). So a single
`unknown` frame means nothing. Naive "I see an unknown face" triggering would
keep trying to enroll Dan as a stranger.

The discriminator
-----------------
The /faces `distance` field is the cosine distance to the NEAREST enrolled
identity, and the Pi reports it even when the label is `unknown`. So:

  * A KNOWN person at bad angles still throws the occasional good frame -> over
    a few seconds their MINIMUM distance dips under the match threshold (~0.45).
  * A TRUE stranger never does -> their min-over-window to every enrolled
    identity stays above a margin (NEW_FACE_MIN_DIST, ~0.55) the whole time.

So the trigger is NOT "an unknown frame appeared", it is:
  "this spatially-stable, close-enough face's BEST distance to any enrolled
   identity has stayed above NEW_FACE_MIN_DIST for WINDOW_S seconds."

That single rule kills the false-positive on Dan. The band between the match
threshold and NEW_FACE_MIN_DIST is treated as HOLD (ambiguous) — neither fire
nor declare known.

This module is pure logic (no I/O, no numpy) so it unit-tests deterministically.
Run `python -m presence.new_face_trigger --selftest` for the synthetic
discrimination tests, or `--live` to watch real /faces at the booth.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------
# Config (all env-overridable, per LT convention)
# --------------------------------------------------------------------------

def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class TriggerConfig:
    # A track whose best distance to any enrolled identity stays >= this for the
    # whole window is "not anyone we know" -> a new-face candidate.
    new_face_min_dist: float = _f("TIMMY_AE_NEW_FACE_MIN_DIST", 0.55)
    # Mirrors streamerpi FACE_ID_MATCH_THRESHOLD. If the window's best distance
    # ever dips below this, the face matched an enrolled identity -> suppress.
    known_suppress_dist: float = _f("TIMMY_AE_KNOWN_DIST", 0.45)
    # How long a track must persist (and how far back the window looks).
    window_s: float = _f("TIMMY_AE_WINDOW_S", 5.0)
    # Minimum observations within the window (guards against a blip).
    min_samples: int = _i("TIMMY_AE_MIN_SAMPLES", 6)
    # Minimum face box height in pixels — only offer to remember someone close
    # enough to be in a conversation, not a face across the room.
    min_face_px: int = _i("TIMMY_AE_MIN_FACE_PX", 50)
    # Center-association gating radius as a fraction of frame width.
    assoc_radius_frac: float = _f("TIMMY_AE_ASSOC_RADIUS_FRAC", 0.18)
    # Drop a track unseen for this long.
    track_ttl_s: float = _f("TIMMY_AE_TRACK_TTL_S", 2.0)
    # After a track fires (or is explicitly handled), don't re-fire it for this
    # long. In the spike this just stops log spam; v1 uses it for decline cooldown.
    refire_cooldown_s: float = _f("TIMMY_AE_REFIRE_COOLDOWN_S", 120.0)


# --------------------------------------------------------------------------
# Observation + track state
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceObs:
    """One face from one /faces tick. bbox is [x, y, w, h] in pixels."""
    ts: float
    name: str
    distance: float
    confidence: str
    bbox: tuple  # (x, y, w, h)

    @property
    def center(self) -> tuple:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def height(self) -> float:
        return self.bbox[3]

    @property
    def is_known_label(self) -> bool:
        return bool(self.name) and self.name.lower() != "unknown"


@dataclass
class Track:
    track_id: int
    created_ts: float
    last_ts: float
    last_center: tuple
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    fired_ts: Optional[float] = None

    def add(self, obs: FaceObs) -> None:
        self.history.append(obs)
        self.last_ts = obs.ts
        self.last_center = obs.center

    def _window(self, now: float, window_s: float) -> list:
        cutoff = now - window_s
        return [o for o in self.history if o.ts >= cutoff]

    def age(self, now: float) -> float:
        return now - self.created_ts


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TriggerDecision:
    track_id: int
    is_candidate: bool
    verdict: str          # CANDIDATE | KNOWN | HOLD | WARMING | SMALL | SPARSE | COOLDOWN
    reason: str
    samples: int
    min_dist: float       # best (lowest) distance to any enrolled identity in window
    median_h: float       # median face-box height in px (proximity proxy)
    age_s: float


def _median(vals: list) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


class NewFaceTrigger:
    """Spatially tracks faces across /faces ticks and decides, per track,
    whether it's a genuinely new person worth offering to enroll.

    Stateful: call update() once per /faces tick. Returns the list of decisions
    for all currently-live tracks (so a live monitor can show why a track is or
    isn't firing); newly-fired CANDIDATE tracks also carry is_candidate=True
    exactly once (then enter refire cooldown).
    """

    def __init__(self, cfg: Optional[TriggerConfig] = None):
        self.cfg = cfg or TriggerConfig()
        self._tracks: dict = {}
        self._next_id = 1

    @property
    def tracks(self) -> dict:
        return self._tracks

    def _associate(self, obs_list: list, frame_w: int, now: float) -> list:
        """Greedy nearest-center association of this tick's faces to live tracks.
        Returns the list of (Track, FaceObs) pairs touched this tick."""
        radius = max(1.0, self.cfg.assoc_radius_frac * max(frame_w, 1))
        used = set()
        touched = []
        # Match existing tracks first, nearest pair greedily.
        candidates = []
        for obs in obs_list:
            for tid, tr in self._tracks.items():
                dx = obs.center[0] - tr.last_center[0]
                dy = obs.center[1] - tr.last_center[1]
                d = (dx * dx + dy * dy) ** 0.5
                if d <= radius:
                    candidates.append((d, id(obs), tid, obs))
        candidates.sort(key=lambda c: c[0])
        matched_obs = set()
        for _d, oid, tid, obs in candidates:
            if oid in matched_obs or tid in used:
                continue
            matched_obs.add(oid)
            used.add(tid)
            self._tracks[tid].add(obs)
            touched.append((self._tracks[tid], obs))
        # Unmatched observations -> new tracks.
        for obs in obs_list:
            if id(obs) in matched_obs:
                continue
            tr = Track(track_id=self._next_id, created_ts=now,
                       last_ts=now, last_center=obs.center)
            tr.add(obs)
            self._tracks[self._next_id] = tr
            touched.append((tr, obs))
            self._next_id += 1
        return touched

    def _prune(self, now: float) -> None:
        dead = [tid for tid, tr in self._tracks.items()
                if now - tr.last_ts > self.cfg.track_ttl_s]
        for tid in dead:
            del self._tracks[tid]

    def _decide(self, tr: Track, now: float) -> TriggerDecision:
        cfg = self.cfg
        win = tr._window(now, cfg.window_s)
        samples = len(win)
        dists = [o.distance for o in win]
        heights = [o.height for o in win]
        min_dist = min(dists) if dists else 1.0
        median_h = _median(heights)
        age = tr.age(now)

        def dec(is_cand, verdict, reason):
            return TriggerDecision(tr.track_id, is_cand, verdict, reason,
                                   samples, round(min_dist, 3), round(median_h, 1),
                                   round(age, 2))

        if tr.fired_ts is not None and (now - tr.fired_ts) < cfg.refire_cooldown_s:
            return dec(False, "COOLDOWN", f"fired {now - tr.fired_ts:.0f}s ago")
        if age < cfg.window_s:
            return dec(False, "WARMING", f"age {age:.1f}s < window {cfg.window_s:.0f}s")
        if samples < cfg.min_samples:
            return dec(False, "SPARSE", f"{samples} samples < {cfg.min_samples}")
        if median_h < cfg.min_face_px:
            return dec(False, "SMALL", f"median h {median_h:.0f}px < {cfg.min_face_px}px")
        # Belt-and-suspenders: a confident known label in-window => enrolled.
        if any(o.is_known_label and o.confidence in ("high", "medium") for o in win):
            return dec(False, "KNOWN", "confident known label in window")
        if min_dist < cfg.known_suppress_dist:
            return dec(False, "KNOWN", f"best dist {min_dist:.2f} < {cfg.known_suppress_dist:.2f}")
        if min_dist < cfg.new_face_min_dist:
            return dec(False, "HOLD", f"best dist {min_dist:.2f} in ambiguous band")
        return dec(True, "CANDIDATE", f"best dist {min_dist:.2f} >= {cfg.new_face_min_dist:.2f} for {age:.1f}s")

    def update(self, faces: list, image_size, now: float) -> list:
        """Ingest one /faces tick.

        faces: list of {name, distance, confidence, bbox:[x,y,w,h]} dicts.
        image_size: (w, h). now: unix timestamp of this tick.
        Returns: list[TriggerDecision] for all live tracks.
        """
        frame_w = int(image_size[0]) if image_size and image_size[0] else 640
        obs_list = [
            FaceObs(
                ts=now,
                name=f.get("name", "unknown"),
                distance=float(f.get("distance", 1.0)),
                confidence=f.get("confidence", "none"),
                bbox=tuple(f.get("bbox", (0, 0, 0, 0))),
            )
            for f in faces
        ]
        self._associate(obs_list, frame_w, now)
        self._prune(now)

        decisions = []
        for tr in self._tracks.values():
            d = self._decide(tr, now)
            if d.is_candidate:
                tr.fired_ts = now  # enter cooldown so we report it once
            decisions.append(d)
        return decisions


# --------------------------------------------------------------------------
# Self-test (synthetic /faces streams) — no network
# --------------------------------------------------------------------------

def _run_selftest() -> int:
    cfg = TriggerConfig()
    W, H = 640, 360
    DT = 0.25  # 4 Hz, like the booth poller

    def box(cx, h=90):
        w = int(h * 0.8)
        return [int(cx - w / 2), int(H / 2 - h / 2), w, h]

    def stream(trig, frames, t0=1000.0):
        """frames: list of (distance, name, conf) for a single centered face."""
        last = []
        for i, (dist, name, conf) in enumerate(frames):
            now = t0 + i * DT
            last = trig.update(
                [{"name": name, "distance": dist, "confidence": conf, "bbox": box(W / 2)}],
                (W, H), now,
            )
        return last

    passed, failed = 0, 0

    def check(label, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    # --- Scenario 1: Dan at a bad angle (the flap). Mostly unknown+high, but a
    # few good frames dip under 0.45. Must NEVER fire. ---
    print("Scenario 1: known person flapping (Dan at booth) -> must NOT fire")
    t = NewFaceTrigger(cfg)
    flap = []
    pattern = [(0.85, "unknown", "low"), (0.88, "unknown", "low"),
               (0.41, "Dan", "medium"), (0.79, "unknown", "low"),
               (0.90, "unknown", "low"), (0.43, "Dan", "medium")]
    for r in range(8):  # ~12s of flapping
        flap += pattern
    decs = stream(t, flap)
    d = decs[0]
    check(f"verdict KNOWN (got {d.verdict}, min_dist={d.min_dist})", d.verdict == "KNOWN")
    check("no candidate ever fired", all(not x.is_candidate for x in decs))

    # --- Scenario 2: true stranger. Always unknown, distance always high. Must fire. ---
    print("Scenario 2: true stranger -> must fire CANDIDATE once")
    t = NewFaceTrigger(cfg)
    fired = 0
    stranger = [(0.72 + (i % 5) * 0.03, "unknown", "low") for i in range(40)]
    for i, (dist, name, conf) in enumerate(stranger):
        now = 2000.0 + i * DT
        decs = t.update([{"name": name, "distance": dist, "confidence": conf,
                          "bbox": box(W / 2)}], (W, H), now)
        fired += sum(1 for x in decs if x.is_candidate)
    check(f"fired exactly once (got {fired})", fired == 1)

    # --- Scenario 3: stranger too far away (small box) -> must NOT fire. ---
    print("Scenario 3: stranger but face too small/far -> must NOT fire")
    t = NewFaceTrigger(cfg)
    fired = 0
    for i in range(40):
        now = 3000.0 + i * DT
        small = [int(W / 2 - 15), int(H / 2 - 18), 30, 36]  # 36px tall < 50
        decs = t.update([{"name": "unknown", "distance": 0.8, "confidence": "low",
                          "bbox": small}], (W, H), now)
        fired += sum(1 for x in decs if x.is_candidate)
    last = decs[0]
    check(f"verdict SMALL (got {last.verdict})", last.verdict == "SMALL")
    check("never fired", fired == 0)

    # --- Scenario 4: ambiguous band (distance ~0.50) -> HOLD, must NOT fire. ---
    print("Scenario 4: ambiguous distance band -> HOLD, must NOT fire")
    t = NewFaceTrigger(cfg)
    decs = stream(t, [(0.50, "unknown", "low")] * 40, t0=4000.0)
    check(f"verdict HOLD (got {decs[0].verdict})", decs[0].verdict == "HOLD")
    check("never fired", all(not x.is_candidate for x in decs))

    # --- Scenario 5: stranger then Dan walks in beside them — two tracks,
    # only the stranger fires. ---
    print("Scenario 5: stranger + Dan side by side -> only stranger fires")
    t = NewFaceTrigger(cfg)
    fired_names = []
    for i in range(40):
        now = 5000.0 + i * DT
        stranger_face = {"name": "unknown", "distance": 0.78, "confidence": "low",
                         "bbox": box(W * 0.30)}
        dan_dist, dan_name = (0.41, "Dan") if i % 3 == 0 else (0.83, "unknown")
        dan_face = {"name": dan_name, "distance": dan_dist, "confidence": "medium",
                    "bbox": box(W * 0.70)}
        decs = t.update([stranger_face, dan_face], (W, H), now)
        for x in decs:
            if x.is_candidate:
                # which track is near x=0.30*W (stranger)?
                tr = t.tracks[x.track_id]
                fired_names.append("stranger" if tr.last_center[0] < W / 2 else "dan")
    check(f"only stranger fired (fired={fired_names})", fired_names == ["stranger"])

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


# --------------------------------------------------------------------------
# Live monitor (read-only; polls real /faces)
# --------------------------------------------------------------------------

async def _run_live(interval: float, duration: float) -> int:
    import time
    import config  # noqa: WPS433 (in-tree only when running live)
    from vision.face_remote import RemoteFaceClient

    cfg = TriggerConfig()
    trig = NewFaceTrigger(cfg)
    client = RemoteFaceClient(max_age_s=cfg.track_ttl_s)
    print(f"[new-face-trigger] live, polling {client.url} every {interval}s "
          f"(new_face>={cfg.new_face_min_dist}, known<{cfg.known_suppress_dist}, "
          f"window={cfg.window_s}s, min_face={cfg.min_face_px}px). Ctrl-C to stop.\n")
    t_end = time.time() + duration if duration > 0 else float("inf")
    try:
        while time.time() < t_end:
            now = time.time()
            full = await client.fetch_full()
            if full is None:
                print(f"{now:.1f}  streamerpi unreachable")
            else:
                decs = trig.update(full["faces"], full["image_size"], now)
                if not decs:
                    print(f"{now:.1f}  (no faces)")
                for d in decs:
                    flag = "  <<< NEW-FACE CANDIDATE" if d.is_candidate else ""
                    print(f"{now:.1f}  trk{d.track_id} {d.verdict:9s} "
                          f"min_d={d.min_dist:<5} h={d.median_h:<5} n={d.samples:<2} "
                          f"age={d.age_s:<5} | {d.reason}{flag}")
            import asyncio
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        await client.close()
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="New-face trigger spike (MVP)")
    p.add_argument("--selftest", action="store_true", help="run synthetic discrimination tests")
    p.add_argument("--live", action="store_true", help="poll real /faces read-only")
    p.add_argument("--interval", type=float, default=0.3, help="live poll interval (s)")
    p.add_argument("--duration", type=float, default=0.0, help="live run seconds (0 = forever)")
    args = p.parse_args()
    if args.selftest:
        return _run_selftest()
    if args.live:
        import asyncio
        return asyncio.run(_run_live(args.interval, args.duration))
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
