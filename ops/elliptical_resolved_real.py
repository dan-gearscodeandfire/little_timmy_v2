"""Replace the hand-authored 'resolved' oracle with a REAL model resolver (:8092).

Same elliptical cases / corpus as ops/elliptical_experiment.py. Adds resolved_4b:
feed the prior turn + follow-up to :8092 and have it REWRITE the follow-up as a
standalone query (extractive coref substitution), then embed that. Compares against
bare, the live blend, the hand-authored oracle, and hyde -- so we see how close a
real small-model resolver gets to the ceiling.
"""

import asyncio
import sys
import types
import httpx
import numpy as np

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")
sys.path.insert(0, "/home/gearscodeandfire/little_timmy/ops")
import asyncpg  # noqa: E402
from memory.manager import embed  # noqa: E402
from memory.retrieval import _build_semantic_query  # noqa: E402
import config  # noqa: E402
from prefix_scale_experiment import hard_negatives, filler  # noqa: E402
from elliptical_experiment import CASES  # noqa: E402

GEN_URL = "http://localhost:8092/v1/chat/completions"
RESOLVE_SYS = (
    "Rewrite the user's LAST message as a single standalone search query. Replace "
    "every pronoun or vague reference (it, its, them, that, there, him, her) with "
    "the specific thing it refers to, taken from the conversation. Keep it a short "
    "question. Output ONLY the rewritten query, nothing else."
)
HYDE_SYS = ('You write a single short declarative fact sentence that would be a stored '
            'memory answering the user. Match this style: "The user\'s iguana is named '
            'Nacho." Output ONLY the sentence.')


async def call(client, sys_prompt, user):
    r = await client.post(GEN_URL, json={
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        "temperature": 0.1, "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    t = r.json()["choices"][0]["message"]["content"]
    return (t.split("</think>")[-1] if "</think>" in t else t).strip().strip('"')


def cos(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def turn(role, content):
    return types.SimpleNamespace(role=role, content=content)


async def main():
    conn = await asyncpg.connect(config.DB_DSN)
    rows = await conn.fetch("SELECT id, content FROM memories ORDER BY id")
    await conn.close()
    real = {r["id"]: r["content"] for r in rows}
    hard = hard_negatives()
    distractors = [(-(i + 1), t) for i, t in enumerate(hard + filler(700 - len(hard)))]
    SIZE = 650

    cache = {}
    async def emb(text):
        if text not in cache:
            cache[text] = await embed(text)
        return cache[text]

    # build conversation transcript the resolver sees (prior turn + current)
    res4b = {}
    hyp = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=== :8092 model resolution (no oracle) vs my hand-authored oracle ===")
        for ctx, fu, oracle, _ in CASES:
            convo = f"User: {ctx}\nUser: {fu}"
            res4b[fu] = await call(client, RESOLVE_SYS, convo)
            hyp[fu] = await call(client, HYDE_SYS, f"{ctx} {fu}")
            print(f"  follow-up   : {fu!r}  (context: {ctx!r})")
            print(f"    oracle    : {oracle!r}")
            print(f"    resolved_4b: {res4b[fu]!r}")
        print()

    strat = {
        "bare": lambda ctx, fu, orc: fu,
        "lt_blend (LIVE)": lambda ctx, fu, orc: _build_semantic_query(fu, [turn("user", ctx)]),
        "resolved_oracle": lambda ctx, fu, orc: orc,
        "resolved_4b (REAL)": lambda ctx, fu, orc: res4b[fu],
        "hyde_ctx": lambda ctx, fu, orc: hyp[fu],
    }

    corpus = list(real.items()) + distractors[:SIZE - len(real)]
    ids = [d for d, _ in corpus]
    txt_of = dict(corpus)
    for _, t in corpus:
        await emb(t)

    for name, fn in strat.items():
        rel_d, irr_d, ranks, margins = [], [], [], []
        r1 = r5 = 0
        for ctx, fu, orc, rel in CASES:
            qe = await emb(fn(ctx, fu, orc))
            scored = sorted(((cos(qe, cache[txt_of[d]]), d) for d in ids), key=lambda x: x[0])
            order = [d for _, d in scored]
            dist_of = {d: dd for dd, d in scored}
            for dd, d in scored:
                (rel_d if d in rel else irr_d).append(dd)
            br = min(order.index(d) + 1 for d in rel)
            ranks.append(br); r1 += br == 1; r5 += br <= 5
            margins.append(min(dist_of[d] for d in ids if d not in rel) - min(dist_of[d] for d in rel))
        disp = sum(1 for m in margins if m < 0)
        print(f"  {name:<20} margin={np.mean(margins):+.3f}  displaced={disp}/10  "
              f"R@1={r1}/10  R@5={r5}/10  MRR={np.mean([1/r for r in ranks]):.3f}  meanrank={np.mean(ranks):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
