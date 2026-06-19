"""Scale-stress experiment: do nomic task prefixes help retrieval as the corpus
grows and the real hit must beat near-neighbor distractors?

Extends ops/prefix_embedding_experiment.py. The 47-doc test was "easy" (recall@5
saturated). Here we keep the 10 real query->target pairs but flood the corpus with:
  - HARD NEGATIVES: near-paraphrases of each target topic with the key fact flipped
    (wrong owner / wrong animal / wrong model / wrong event) -> genuine competition
  - TOPICAL ADJACENTS + FILLER: synthetic personal facts to pad corpus size

We grow the corpus across sizes and, per prefix scheme, measure:
  - separation  = mean(irrelevant dist) - mean(relevant dist)        [coarse]
  - margin      = mean over queries of (nearest-distractor dist - nearest-relevant dist)
                  NEGATIVE margin => a distractor out-ranked the real hit  [the sharp one]
  - displaced   = # queries where margin < 0
  - recall@1 / recall@5 / MRR

All distractors are synthetic (no PII). Real targets come from the live DB
(transient embed; artifact logs only metrics + query text, no doc content).
Embeddings are memoized per (text, prefix) and reused across corpus sizes.
"""

import asyncio
import sys
import random
import numpy as np

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")
import asyncpg  # noqa: E402
from memory.manager import embed  # noqa: E402
import config  # noqa: E402

random.seed(42)
DOC_PREFIX = "search_document"
QUERY_PREFIX = "search_query"

# query -> set of relevant REAL doc ids (same as the 47-doc test)
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

SCHEMES = [
    (None, None, "bare/bare (TODAY)"),
    (DOC_PREFIX, QUERY_PREFIX, "doc=D / query=Q (PROPOSED)"),
    (None, QUERY_PREFIX, "doc=bare / query=Q"),
    (QUERY_PREFIX, DOC_PREFIX, "crossed (control)"),
]

# ---- synthetic distractor generation (deterministic) ----------------------
NAMES = ["Marcus", "Lena", "Priya", "Omar", "Sofia", "Hiro", "Greta", "Tobias",
         "Nadia", "Ivan", "Yuki", "Rosa", "Felix", "Wendy", "Dario", "Aisha",
         "Bjorn", "Clara", "Dara", "Esma"]

# HARD NEGATIVES per query index: same shape/topic, key fact flipped.
def hard_negatives():
    H = []
    # 0: lost item at an event (not Dan's lavalier at the birthday)
    items = ["wireless headset", "camera tripod", "name badge", "laser pointer",
             "bluetooth speaker", "presentation clicker", "lapel pin"]
    events = ["wedding reception", "retirement dinner", "conference mixer",
              "graduation party", "company offsite"]
    for it in items:
        for ev in events[:3]:
            H.append(f"A {it} went missing during the {ev} and was later returned.")
    # 1: context window of a different model / size
    for m, n in [("Llama 4", 32), ("GPT-5", 256), ("Gemini 3", 1000),
                 ("Mistral Large", 64), ("Claude Sonnet", 200), ("Phi-4", 16)]:
        H.append(f"The {m} model runs with roughly a {n}k-token context window.")
    # 2: a different sports result
    for t, r in [("the Lakers won", "playoffs"), ("the Celtics lost", "semifinals"),
                 ("the Yankees won", "world series"), ("the Rangers lost", "stanley cup"),
                 ("the Heat won", "conference finals")]:
        H.append(f"In a recent game {t} the {r}.")
    # 3: pet ownership, wrong owner/animal/name
    for nm in NAMES[:8]:
        an = random.choice(["ferrets", "rabbits", "parakeets", "turtles"])
        p1, p2 = random.sample(["Biscuit", "Pepper", "Mochi", "Tango", "Olive", "Cosmo"], 2)
        H.append(f"{nm} has two {an} named {p1} and {p2}.")
    # 4: reptile/exotic pet, wrong name
    for nm, pn in [("Lena", "Spike"), ("Omar", "Rex"), ("Priya", "Noodle"),
                   ("Felix", "Gizmo"), ("Rosa", "Pancake")]:
        H.append(f"{nm}'s pet gecko is named {pn}.")
    # 5: a different conference / dates
    for c, d in [("the Maker Faire", "in May"), ("DEF CON", "in August"),
                 ("CES", "in January"), ("PyCon", "in April"),
                 ("a robotics expo", "next spring")]:
        H.append(f"{random.choice(NAMES)} plans to attend {c} {d}.")
    # 6: a different DIY heating/cooling build
    for x in ["a solar water heater from black hose", "a wood-stove backup heater",
              "a geothermal loop for the garage", "a swamp cooler from a fan and ice",
              "a pellet stove install in the basement"]:
        H.append(f"{random.choice(NAMES)} built {x} as a winter project.")
    # 7: a different maker/electronics project (no fire)
    for x in ["a LED matrix that displays the weather", "a CNC router from kit parts",
              "a 3D-printed prosthetic hand", "an automatic plant-watering rig",
              "a Raspberry-Pi retro arcade cabinet", "a Tesla coil music synth"]:
        H.append(f"{random.choice(NAMES)} made {x}.")
    # 8: a different appointment / relationship service
    for x in ["a financial advisor", "a personal trainer", "a marriage counselor for friends",
              "a real-estate agent", "a tax accountant", "a physical therapist"]:
        H.append(f"{random.choice(NAMES)} has been seeing {x} this year.")
    # 9: a different milestone birthday/anniversary
    for nm, age in [("Lena", "40th"), ("Omar", "60th"), ("Priya", "30th"),
                    ("Felix", "70th"), ("Rosa", "25th wedding anniversary")]:
        H.append(f"{nm} celebrated their {age} last month.")
    return H

def filler(n):
    hobbies = ["pottery", "rock climbing", "beekeeping", "watercolor painting",
               "sourdough baking", "trail running", "birdwatching", "woodturning",
               "kite surfing", "astrophotography", "fly fishing", "calligraphy"]
    places = ["Lisbon", "Kyoto", "Reykjavik", "Oaxaca", "Tallinn", "Hanoi",
              "Marrakesh", "Ljubljana", "Bergen", "Cape Town"]
    months = ["January", "March", "June", "September", "November"]
    cats = ["coffee", "music", "footwear", "houseplants", "board games",
            "cheese", "podcasts", "tea", "novels", "bicycles"]
    things = ["espresso", "vinyl jazz", "leather boots", "ferns", "catan",
              "gouda", "true crime", "oolong", "mystery", "a steel road bike"]
    out = []
    templates = [
        lambda nm: f"{nm} recently took up {random.choice(hobbies)}.",
        lambda nm: f"{nm} traveled to {random.choice(places)} last {random.choice(months)}.",
        lambda nm: f"{nm}'s favorite {random.choice(cats)} is {random.choice(things)}.",
        lambda nm: f"{nm} switched jobs and now commutes by train.",
        lambda nm: f"{nm} is learning to play the {random.choice(['cello','banjo','drums','piano'])}.",
        lambda nm: f"{nm} adopted a rescue dog from the shelter in {random.choice(places)}.",
    ]
    while len(out) < n:
        out.append(random.choice(templates)(random.choice(NAMES)))
    return out


async def main():
    conn = await asyncpg.connect(config.DB_DSN)
    rows = await conn.fetch("SELECT id, content FROM memories ORDER BY id")
    await conn.close()
    real = {r["id"]: r["content"] for r in rows}
    real_ids = list(real.keys())

    hard = hard_negatives()
    fill = filler(700 - len(hard))
    # distractor docs get synthetic negative ids so they never collide with real
    distractors = [(-(i + 1), t) for i, t in enumerate(hard + fill)]
    print(f"real docs: {len(real_ids)}  hard-negs: {len(hard)}  filler: {len(fill)}  "
          f"total pool: {len(real_ids) + len(distractors)}\n")

    all_docs = [(did, txt) for did, txt in real.items()] + distractors

    # memoize embeddings per (text, prefix)
    cache: dict[tuple[str, str | None], np.ndarray] = {}

    async def emb(text, prefix):
        key = (text, prefix)
        if key not in cache:
            s = f"{prefix}: {text}" if prefix else text
            cache[key] = await embed(s)
        return cache[key]

    def cos(a, b):
        return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    SIZES = [47, 150, 350, 650]  # corpus sizes to evaluate (subsample distractors)

    for doc_pref, q_pref, label in SCHEMES:
        print(f"================ {label} ================")
        # pre-embed every doc once for this scheme's doc prefix
        for did, txt in all_docs:
            await emb(txt, doc_pref)
        for size in SIZES:
            n_distract = size - len(real_ids)
            corpus = [(did, txt) for did, txt in real.items()]
            if n_distract > 0:
                corpus += distractors[:n_distract]
            corpus_ids = [d for d, _ in corpus]
            txt_of = dict(corpus)

            rel_d, irr_d, ranks, margins = [], [], [], []
            r1 = r5 = 0
            for q, rel in PAIRS:
                qe = await emb(q, q_pref)
                scored = sorted(((cos(qe, cache[(txt_of[did], doc_pref)]), did)
                                 for did in corpus_ids), key=lambda x: x[0])
                order = [did for _, did in scored]
                dist_of = {did: d for d, did in scored}
                for d, did in scored:
                    (rel_d if did in rel else irr_d).append(d)
                best_rank = min(order.index(did) + 1 for did in rel)
                ranks.append(best_rank)
                r1 += best_rank == 1
                r5 += best_rank <= 5
                min_rel = min(dist_of[did] for did in rel)
                min_irr = min(dist_of[did] for did in corpus_ids if did not in rel)
                margins.append(min_irr - min_rel)
            displaced = sum(1 for m in margins if m < 0)
            print(f"  n={size:<4} sep={np.mean(irr_d)-np.mean(rel_d):.3f}  "
                  f"margin={np.mean(margins):+.3f}  displaced={displaced}/10  "
                  f"R@1={r1}/10  R@5={r5}/10  MRR={np.mean([1/r for r in ranks]):.3f}  "
                  f"meanrank={np.mean(ranks):.1f}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
