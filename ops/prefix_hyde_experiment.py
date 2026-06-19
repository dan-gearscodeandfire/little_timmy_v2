"""HyDE arm: does generating a hypothetical declarative fact (via :8092 Qwen3-4B)
and embedding THAT beat embedding the bare query?

Reuses the scale corpus (real targets + hard negatives + filler) from
ops/prefix_scale_experiment.py. Compares, at the decisive sizes 47 and 650:

  bare/bare (TODAY)         : embed the bare query
  doc=D / query=Q (PROPOSED): nomic prefixes (reference; already rejected)
  HyDE-bare                 : embed an LLM-generated hypothetical fact (bare), docs bare
  HyDE+query (blend)        : mean(embed(query), embed(hypothetical)), docs bare

Hypotheticals generated once via :8092 (the production-candidate generator),
thinking suppressed, and printed so the artifact records them. Docs are the real
DB rows (transient embed). Distractors synthetic. Embeddings memoized.
"""

import asyncio
import sys
import httpx
import numpy as np

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")
sys.path.insert(0, "/home/gearscodeandfire/little_timmy/ops")
import asyncpg  # noqa: E402
from memory.manager import embed  # noqa: E402
import config  # noqa: E402
from prefix_scale_experiment import PAIRS, hard_negatives, filler  # noqa: E402

DOC_PREFIX = "search_document"
QUERY_PREFIX = "search_query"
GEN_URL = "http://localhost:8092/v1/chat/completions"
SYS = ('You write a single short declarative fact sentence that would be a stored '
       'memory answering the user question. Match this style: "The user\'s iguana '
       'is named Nacho." Output ONLY the sentence, no preamble.')


async def hyde(client, q: str) -> str:
    r = await client.post(GEN_URL, json={
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        "temperature": 0.2, "max_tokens": 80,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    if "</think>" in txt:  # strip any stray thinking
        txt = txt.split("</think>")[-1]
    return txt.strip()


def cos(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def main():
    conn = await asyncpg.connect(config.DB_DSN)
    rows = await conn.fetch("SELECT id, content FROM memories ORDER BY id")
    await conn.close()
    real = {r["id"]: r["content"] for r in rows}
    real_ids = list(real.keys())
    hard, fill = hard_negatives(), filler(700 - len(hard_negatives()))
    distractors = [(-(i + 1), t) for i, t in enumerate(hard + fill)]

    cache = {}
    async def emb(text, prefix=None):
        key = (text, prefix)
        if key not in cache:
            s = f"{prefix}: {text}" if prefix else text
            cache[key] = await embed(s)
        return cache[key]

    # generate hypotheticals once
    hyp = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=== HyDE hypotheticals (generated via :8092) ===")
        for q, _ in PAIRS:
            hyp[q] = await hyde(client, q)
            print(f"  Q: {q}\n   -> {hyp[q]}")
        print()

    SIZES = [47, 650]
    # query-embedding strategies: name -> async fn(q) -> vector ; plus doc prefix
    async def vec_bare(q):       # TODAY
        return await emb(q, None)
    async def vec_prefix(q):     # PROPOSED
        return await emb(q, QUERY_PREFIX)
    async def vec_hyde(q):       # pure HyDE
        return await emb(hyp[q], None)
    async def vec_hyde_blend(q): # HyDE + query, mean-pooled
        a = await emb(q, None); b = await emb(hyp[q], None)
        return (a + b) / 2.0

    ARMS = [
        ("bare/bare (TODAY)", vec_bare, None),
        ("doc=D / query=Q (PROPOSED)", vec_prefix, DOC_PREFIX),
        ("HyDE-bare", vec_hyde, None),
        ("HyDE+query (blend)", vec_hyde_blend, None),
    ]

    for label, qvec, doc_pref in ARMS:
        print(f"================ {label} ================")
        # pre-embed all docs for this arm's doc prefix
        for did, txt in list(real.items()) + distractors:
            await emb(txt, doc_pref)
        for size in SIZES:
            corpus = list(real.items())
            nd = size - len(real_ids)
            if nd > 0:
                corpus += distractors[:nd]
            ids = [d for d, _ in corpus]
            txt_of = dict(corpus)
            rel_d, irr_d, ranks, margins = [], [], [], []
            r1 = r5 = 0
            for q, rel in PAIRS:
                qe = await qvec(q)
                scored = sorted(((cos(qe, cache[(txt_of[did], doc_pref)]), did)
                                 for did in ids), key=lambda x: x[0])
                order = [d for _, d in scored]
                dist_of = {d: dd for dd, d in scored}
                for dd, d in scored:
                    (rel_d if d in rel else irr_d).append(dd)
                br = min(order.index(d) + 1 for d in rel)
                ranks.append(br); r1 += br == 1; r5 += br <= 5
                mr = min(dist_of[d] for d in rel)
                mi = min(dist_of[d] for d in ids if d not in rel)
                margins.append(mi - mr)
            disp = sum(1 for m in margins if m < 0)
            print(f"  n={size:<4} sep={np.mean(irr_d)-np.mean(rel_d):.3f}  "
                  f"margin={np.mean(margins):+.3f}  displaced={disp}/10  "
                  f"R@1={r1}/10  R@5={r5}/10  MRR={np.mean([1/r for r in ranks]):.3f}  "
                  f"meanrank={np.mean(ranks):.1f}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
