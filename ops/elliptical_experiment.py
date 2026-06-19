"""Elliptical-query test: where query-side intelligence SHOULD pay.

Self-contained queries ("what's my iguana's name?") already embed well -> prefixes
and HyDE failed to beat bare. This tests the opposite regime: degenerate follow-ups
("what's its name again?") that carry little standalone signal but have a prior turn
supplying the antecedent. Four strategies for embedding the semantic query:

  bare            : embed only the elliptical follow-up (expected: bad)
  lt_blend (LIVE) : memory.retrieval._build_semantic_query -- LT's actual coref path
                    (role-tagged prior turn prepended, follow-up last)
  resolved        : embed the coref-RESOLVED clean query ("what's my iguana's name?")
  hyde_ctx        : LLM hypothetical (:8092) generated from context + follow-up

Scored against the scale corpus (real targets + hard negs + filler) at n=650.
Metrics: sep / margin / displaced / R@1 / R@5 / MRR / mean-rank.
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
from memory.retrieval import _build_semantic_query  # LT's live coref builder  # noqa: E402
import config  # noqa: E402
from prefix_scale_experiment import hard_negatives, filler  # noqa: E402

GEN_URL = "http://localhost:8092/v1/chat/completions"
SYS = ('You write a single short declarative fact sentence that would be a stored '
       'memory answering the user. Match this style: "The user\'s iguana is named '
       'Nacho." Output ONLY the sentence, no preamble.')

# context (prior user turn) + elliptical follow-up + resolved form + relevant real ids
CASES = [
    ("I was thinking about my iguana.", "what's its name again?",
     "what is my iguana's name?", {85}),
    ("Let's talk about my pets.", "what are they called?",
     "what are my cats named?", {40}),
    ("I have a friend named Thomas who's a YouTuber.", "what did he give me?",
     "what did Thomas give me?", {79, 88}),
    ("Let's talk about the conference this summer.", "when is it?",
     "when is the OpenSauce conference?", {80, 49}),
    ("My partner and I have been working on our relationship.", "who are we seeing for that?",
     "are we seeing a couples therapist?", {69, 71}),
    ("I built a heater for my shop.", "what did I make it out of?",
     "what is the shop heater made of?", {74}),
    ("Remember that big party I had?", "what did I lose there?",
     "what did I lose at the party?", {158, 152, 136}),
    ("I had a milestone birthday recently.", "when was it?",
     "when was my 50th birthday?", {110, 140}),
    ("I once had a beloved companion.", "what happened to him?",
     "what happened to Winston?", {58}),
    ("That fire-shooting maker project of mine.", "what was it called?",
     "what is my fire-shooting project called?", {76, 87}),
]


async def hyde(client, text):
    r = await client.post(GEN_URL, json={
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": text}],
        "temperature": 0.2, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    t = r.json()["choices"][0]["message"]["content"]
    return (t.split("</think>")[-1] if "</think>" in t else t).strip()


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

    assert config.COREFERENCE_ENABLED, "enable TIMMY_COREFERENCE_ENABLED for lt_blend"

    # build the query string per strategy
    async with httpx.AsyncClient(timeout=60.0) as client:
        hyp = {}
        print("=== strategy inputs per case ===")
        for ctx, fu, res, _ in CASES:
            hyp[fu] = await hyde(client, f"{ctx} {fu}")
            print(f"  follow-up : {fu!r}")
            print(f"    bare    -> {fu!r}")
            print(f"    lt_blend-> {_build_semantic_query(fu, [turn('user', ctx)])!r}")
            print(f"    resolved-> {res!r}")
            print(f"    hyde_ctx-> {hyp[fu]!r}")
        print()

    strat = {
        "bare": lambda ctx, fu, res: fu,
        "lt_blend (LIVE)": lambda ctx, fu, res: _build_semantic_query(fu, [turn("user", ctx)]),
        "resolved": lambda ctx, fu, res: res,
        "hyde_ctx": lambda ctx, fu, res: hyp[fu],
    }

    # corpus (docs always embedded bare -- today's store path)
    corpus = list(real.items()) + distractors[:SIZE - len(real)]
    ids = [d for d, _ in corpus]
    txt_of = dict(corpus)
    for _, t in corpus:
        await emb(t)

    for name, fn in strat.items():
        rel_d, irr_d, ranks, margins = [], [], [], []
        r1 = r5 = 0
        for ctx, fu, res, rel in CASES:
            qe = await emb(fn(ctx, fu, res))
            scored = sorted(((cos(qe, cache[txt_of[d]]), d) for d in ids), key=lambda x: x[0])
            order = [d for _, d in scored]
            dist_of = {d: dd for dd, d in scored}
            for dd, d in scored:
                (rel_d if d in rel else irr_d).append(dd)
            br = min(order.index(d) + 1 for d in rel)
            ranks.append(br); r1 += br == 1; r5 += br <= 5
            margins.append(min(dist_of[d] for d in ids if d not in rel) - min(dist_of[d] for d in rel))
        disp = sum(1 for m in margins if m < 0)
        print(f"  {name:<16} sep={np.mean(irr_d)-np.mean(rel_d):.3f}  margin={np.mean(margins):+.3f}  "
              f"displaced={disp}/10  R@1={r1}/10  R@5={r5}/10  MRR={np.mean([1/r for r in ranks]):.3f}  "
              f"meanrank={np.mean(ranks):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
