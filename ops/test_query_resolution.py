"""End-to-end test of the query-resolution change through the REAL retrieve()
path against the live DB. No LT restart; toggle is monkeypatched in-process so
no shared on-disk state is touched (the running LT is unaffected)."""

import asyncio
import sys
import time
import types

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")
from memory import retrieval  # noqa: E402
from persistence import runtime_toggles  # noqa: E402


def turn(role, content):
    return types.SimpleNamespace(role=role, content=content)


CASES = [
    ([turn("user", "I was thinking about my iguana.")], "what's its name again?"),
    ([turn("user", "I have a friend named Thomas who's a YouTuber.")], "what did he give me?"),
    ([turn("user", "Remember that big party I had?")], "what did I lose there?"),
    ([turn("user", "My partner and I have been working on our relationship.")],
     "who are we seeing for that?"),
    ([turn("user", "I once had a beloved companion.")], "what happened to him?"),
    # control: self-contained query with NO deixis -> gate should skip resolution
    ([turn("user", "random prior chatter")], "what are my cats named?"),
]

_orig = runtime_toggles.get


async def run(query, ctx):
    t0 = time.perf_counter()
    res = await retrieval.retrieve(query, top_k=3, context_turns=ctx)
    ms = (time.perf_counter() - t0) * 1000
    return ms, [(r.id, r.content[:58]) for r in res]


async def main():
    for ctx, q in CASES:
        runtime_toggles.get = _orig  # OFF = current blend behavior
        off_ms, off = await run(q, ctx)
        # ON: resolution enabled (monkeypatch, no disk write)
        runtime_toggles.get = lambda k: True if k == "query_resolution_enabled" else _orig(k)
        on_ms, on = await run(q, ctx)
        runtime_toggles.get = _orig

        deictic = retrieval._needs_resolution(q)
        print(f"\nQ: {q!r}   (deixis-gate fires: {deictic})")
        print(f"   ctx: {ctx[-1].content!r}")
        print(f"   OFF (blend)     {off_ms:6.0f}ms  top: {off[0] if off else None}")
        print(f"   ON  (resolved)  {on_ms:6.0f}ms  top: {on[0] if on else None}")
        if [i for i, _ in off] != [i for i, _ in on]:
            print(f"   ↳ ranking changed: OFF={[i for i,_ in off]}  ON={[i for i,_ in on]}")


if __name__ == "__main__":
    asyncio.run(main())
