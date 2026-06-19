"""Latency delta of folding a `resolved_query` field into the :8092 classifier.

Compares, against the LIVE classifier server, three call shapes on identical inputs:

  A  route-only (CURRENT)      : real classify_route prompt, fixed-output grammar,
                                 max_tokens=16 -- the per-turn cost LT pays today.
  B  route + resolved_query    : same call, grammar extended to also emit a
                                 free-text resolved_query; context turns added to
                                 the prompt so resolution is possible. ONE call.
  C  separate resolution call  : the route call (A) PLUS a second unconstrained
                                 rewrite call (what we'd pay if NOT folded in).

Reports p50/p90 over N warm iterations. B-A = the delta Dan asked for; C shows
the two-call alternative folding-in avoids.

Mirrors llm/client.classify_constrained exactly (grammar field, temp 0.0,
enable_thinking False, /v1/chat/completions). Live-server caveat: keep N modest.
"""

import asyncio
import sys
import time
import statistics
import httpx
from pathlib import Path

sys.path.insert(0, "/home/gearscodeandfire/little_timmy")
import config  # noqa: E402

URL = f"{config.LLM_CLASSIFIER_URL}/v1/chat/completions"
ROUTE_PROMPT = (Path("/home/gearscodeandfire/little_timmy/prompts/classify_route.txt")
                .read_text(encoding="utf-8"))

ROUTE_GRAMMAR = r'''
root        ::= store-call | none-call
store-call  ::= "{\"tool\":\"store_fact\"}"
none-call   ::= "{\"tool\":\"none\"}"
'''
# route + free-text resolved_query in one constrained object
COMBINED_GRAMMAR = r'''
root  ::= "{\"tool\":\"" tool "\",\"resolved_query\":\"" str "\"}"
tool  ::= "store_fact" | "none"
str   ::= char+
char  ::= [^"\\]
'''
COMBINED_PROMPT = ROUTE_PROMPT + (
    "\n\nALSO output `resolved_query`: rewrite the user's LAST message as a "
    "standalone search query, replacing pronouns/vague references (it, that, "
    "there, him, her, them) with what they refer to from the conversation. If "
    "nothing needs resolving, repeat the message as-is.")
RESOLVE_PROMPT = (
    "Rewrite the user's LAST message as a single standalone search query, "
    "replacing pronouns/vague references with what they refer to from the "
    "conversation. Output ONLY the rewritten query.")

# realistic inputs: prior turn (context) + elliptical follow-up
INPUTS = [
    ("I was thinking about my iguana.", "what's its name again?"),
    ("I have a friend named Thomas who's a YouTuber.", "what did he give me?"),
    ("Let's talk about the conference this summer.", "when is it?"),
    ("Remember that big party I had?", "what did I lose there?"),
    ("My partner and I have been working on our relationship.", "who are we seeing for that?"),
]
N = 12  # iterations per input (keep modest -- live server, Dan testing in shop)


async def one(client, messages, grammar, max_tokens):
    payload = {"messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
               "stream": False}
    if grammar:
        payload["grammar"] = grammar
    t0 = time.perf_counter()
    r = await client.post(URL, json=payload, timeout=10.0)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    out_toks = r.json().get("usage", {}).get("completion_tokens")
    content = (r.json()["choices"][0]["message"].get("content") or "").strip()
    return ms, out_toks, content


def convo(ctx, fu):
    return f"User: {ctx}\nUser: {fu}"


async def main():
    async with httpx.AsyncClient() as client:
        # warmup
        for _ in range(3):
            await one(client, [{"role": "system", "content": ROUTE_PROMPT},
                               {"role": "user", "content": "hello there"}], ROUTE_GRAMMAR, 16)

        results = {"A route-only (CURRENT)": [], "B route+resolved_query (1 call)": [],
                   "C route + separate resolve (2 calls)": []}
        toks = {k: [] for k in results}
        sample_out = {}

        for ctx, fu in INPUTS:
            for _ in range(N):
                # A: current -- route only sees the bare utterance (as today)
                a_ms, a_tok, a_out = await one(
                    client, [{"role": "system", "content": ROUTE_PROMPT},
                             {"role": "user", "content": fu}], ROUTE_GRAMMAR, 16)
                results["A route-only (CURRENT)"].append(a_ms)
                toks["A route-only (CURRENT)"].append(a_tok or 0)

                # B: one call, route + resolved_query, with context
                b_ms, b_tok, b_out = await one(
                    client, [{"role": "system", "content": COMBINED_PROMPT},
                             {"role": "user", "content": convo(ctx, fu)}], COMBINED_GRAMMAR, 96)
                results["B route+resolved_query (1 call)"].append(b_ms)
                toks["B route+resolved_query (1 call)"].append(b_tok or 0)
                sample_out[fu] = b_out

                # C: route (bare) + separate unconstrained resolve (with context)
                c1_ms, _, _ = await one(
                    client, [{"role": "system", "content": ROUTE_PROMPT},
                             {"role": "user", "content": fu}], ROUTE_GRAMMAR, 16)
                c2_ms, c2_tok, _ = await one(
                    client, [{"role": "system", "content": RESOLVE_PROMPT},
                             {"role": "user", "content": convo(ctx, fu)}], None, 60)
                results["C route + separate resolve (2 calls)"].append(c1_ms + c2_ms)
                toks["C route + separate resolve (2 calls)"].append(c2_tok or 0)

        def pct(xs, p):
            return statistics.quantiles(xs, n=100)[p - 1]
        print(f"server {URL}  |  {len(INPUTS)} inputs x {N} iters\n")
        base = None
        for k, xs in results.items():
            p50, p90 = pct(xs, 50), pct(xs, 90)
            mt = statistics.median(toks[k])
            if base is None:
                base = p50
            delta = p50 - base
            print(f"  {k:<38} p50={p50:6.1f}ms  p90={p90:6.1f}ms  "
                  f"out_tok(med)={mt:4.0f}  Δp50={delta:+6.1f}ms")
        print("\n  sample B resolved_query outputs:")
        for fu, out in sample_out.items():
            print(f"    {fu!r} -> {out[:90]!r}")


if __name__ == "__main__":
    asyncio.run(main())
