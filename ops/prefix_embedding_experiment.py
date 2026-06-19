"""Step-1 experiment (handoff-lt-memory-embedding-retrieval-prompts-2026-06-18).

Quantify whether nomic-embed-text task prefixes (search_document: / search_query:)
improve retrieval separation on the live LT `memories` corpus BEFORE committing to
the embed(task=...) + re-embed change.

Measures, per prefix scheme, against the REAL corpus (transient in-memory embedding,
nothing persisted; artifact logs only doc IDs + distances, not raw doc content):
  - relevant-pair mean cosine distance (lower better)
  - irrelevant-pair mean cosine distance
  - separation = mean_irrelevant - mean_relevant (higher better)
  - recall@5 and MRR (does the relevant doc rank higher among all 47 docs?)

Schemes:
  bare/bare            : today's behavior (no prefix either side)
  doc=D / query=Q      : proposed (search_document on docs, search_query on queries)
  doc=D / query=bare   : doc-only prefix (ablation)
  doc=bare / query=Q   : query-only prefix (ablation)
  crossed (D<->Q)      : sanity control (wrong prefixes swapped)
"""

import asyncio
import sys
import numpy as np

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")
import asyncpg  # noqa: E402
from pgvector.asyncpg import register_vector  # noqa: E402
from memory.manager import embed  # bare embed() — the live chokepoint  # noqa: E402
import config  # noqa: E402

DOC_PREFIX = "search_document"
QUERY_PREFIX = "search_query"


async def embed_p(text: str, prefix: str | None) -> np.ndarray:
    s = f"{prefix}: {text}" if prefix else text
    return await embed(s)


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    # Matches pgvector <=>: cosine distance = 1 - cosine similarity.
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# query (short utterance, mirrors real query shape) -> set of relevant doc ids
PAIRS = [
    ("did I lose my microphone at the party?", {158, 152, 136}),
    ("how big is the AI's context window?", {154}),
    ("who won the basketball finals?", {144}),
    ("what are my cats named?", {40}),
    ("what's my iguana's name?", {85}),
    ("when is the OpenSauce conference?", {80, 49}),
    ("tell me about the shop heater project", {74, 75}),
    ("what fire-shooting projects have I built?", {76, 87, 79}),
    ("are we seeing a couples therapist?", {69, 71}),
    ("when was my 50th birthday?", {110, 140}),
]

# (doc_prefix, query_prefix, label)
SCHEMES = [
    (None, None, "bare/bare (TODAY)"),
    (DOC_PREFIX, QUERY_PREFIX, "doc=D / query=Q (PROPOSED)"),
    (DOC_PREFIX, None, "doc=D / query=bare"),
    (None, QUERY_PREFIX, "doc=bare / query=Q"),
    (QUERY_PREFIX, DOC_PREFIX, "crossed D<->Q (control)"),
]


async def main():
    conn = await asyncpg.connect(config.DB_DSN)
    await register_vector(conn)
    rows = await conn.fetch("SELECT id, type, content FROM memories ORDER BY id")
    await conn.close()
    doc_ids = [r["id"] for r in rows]
    doc_text = {r["id"]: r["content"] for r in rows}
    print(f"corpus: {len(doc_ids)} docs; {len(PAIRS)} query/relevant pairs\n")

    # Sanity: confirm the prefix actually changes the embedding at all.
    a = await embed_p("the iguana is named Nacho", None)
    b = await embed_p("the iguana is named Nacho", DOC_PREFIX)
    print(f"[sanity] same text bare-vs-prefixed cosine distance = {cos_dist(a, b):.4f} "
          f"(0 => prefix ignored by model)\n")

    for doc_pref, q_pref, label in SCHEMES:
        # Embed all docs under this scheme's doc prefix.
        doc_emb = {}
        for did in doc_ids:
            doc_emb[did] = await embed_p(doc_text[did], doc_pref)

        rel_dists, irr_dists = [], []
        ranks = []  # rank (1-based) of best relevant doc per query
        recall5 = 0
        per_query = []

        for q, rel in PAIRS:
            qe = await embed_p(q, q_pref)
            dists = sorted(((cos_dist(qe, doc_emb[did]), did) for did in doc_ids),
                           key=lambda x: x[0])
            order = [did for _, did in dists]
            for d, did in dists:
                (rel_dists if did in rel else irr_dists).append(d)
            best_rank = min(order.index(did) + 1 for did in rel)
            ranks.append(best_rank)
            if best_rank <= 5:
                recall5 += 1
            best_rel_dist = min(cos_dist(qe, doc_emb[did]) for did in rel)
            per_query.append((q[:38], best_rank, best_rel_dist))

        mean_rel = float(np.mean(rel_dists))
        mean_irr = float(np.mean(irr_dists))
        sep = mean_irr - mean_rel
        mrr = float(np.mean([1.0 / r for r in ranks]))
        print(f"=== {label} ===")
        print(f"  relevant mean dist : {mean_rel:.4f}")
        print(f"  irrelevant mean dist: {mean_irr:.4f}")
        print(f"  separation (irr-rel): {sep:.4f}")
        print(f"  recall@5: {recall5}/{len(PAIRS)}   MRR: {mrr:.4f}   "
              f"mean best-rank: {np.mean(ranks):.2f}")
        for qtxt, r, d in per_query:
            flag = "" if r <= 5 else "  <-- MISS"
            print(f"     r@{r:<2} d={d:.3f}  {qtxt}{flag}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
