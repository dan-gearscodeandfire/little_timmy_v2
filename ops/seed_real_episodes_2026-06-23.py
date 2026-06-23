"""One-off: seed three REAL episodic memories Dan requested (2026-06-23).

These are durable, intentional seeds (NOT synth-speech test data), written
through the production `store_episode()` path so hashing / redaction / source
shape match real warm-eviction rollups. EMBED_EPISODES is off, so embeddings
stay NULL — consistent with current prod and irrelevant to date-range recall.

source.trigger = "seed_dan_2026-06-23" so they can be located/removed later.
Run: cd ~/little_timmy && .venv/bin/python -m ops.seed_real_episodes_2026-06-23
"""
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from memory.manager import store_episode

TZ = ZoneInfo("America/New_York")


def ts(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp()


# (span_start, span_end, text) — phrased as conversation/event summaries, the
# voice the rollup writer produces and the recall block renders.
EPISODES = [
    # June 10 — Knicks Game 4 (evening game)
    (ts(2026, 6, 10, 21, 0), ts(2026, 6, 10, 22, 45),
     "Talked about the Knicks winning Game 4 of their NBA playoff series. Dan "
     "was worried the deciding final game might land on the day of his upcoming "
     "birthday party."),
    # June 13 — birthday party + Little Timmy demo (Open Sauce warm-up)
    (ts(2026, 6, 13, 17, 0), ts(2026, 6, 13, 20, 0),
     "Dan threw his birthday party and showed off Little Timmy to a bunch of "
     "guests, demoing the assistant as a warm-up for presenting it at Open Sauce."),
    # June 13 — a microphone went missing during the party
    (ts(2026, 6, 13, 18, 30), ts(2026, 6, 13, 19, 15),
     "During the party one of Little Timmy's microphones went missing. We used "
     "Supervisor Mode to try to track it down, but didn't manage to find it."),
]

SOURCE = {"trigger": "seed_dan_2026-06-23", "kind": "real_event_seed"}


async def main():
    ids = []
    for span_start, span_end, text in EPISODES:
        eid = await store_episode(
            span_start, span_end, text,
            token_count=max(1, len(text) // 4),
            source=dict(SOURCE),
        )
        ids.append(eid)
        print(f"stored episode id={eid}: {text[:60]}...")
    print(f"\nseeded ids: {ids}")


if __name__ == "__main__":
    asyncio.run(main())
