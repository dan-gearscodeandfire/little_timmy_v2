#!/usr/bin/env python3
"""Extraction-path value-confidence gate (2026-06-21).

The synchronous store_fact TOOL path value-confidence-gates the values it stores
(ops/test_stt_confidence_gate.py). But a value that MISSES the tool route and is
mined by the BACKGROUND extractor was stored with no confidence -> default 1.0 =
verified, a silent FALSE for a misheard value (found live 2026-06-21: "My dog is
named Max" router-missed -> extracted at conf 1.0). This gates that path too:
the extractor scores each mined value against the exchange's whisper word-probs
and stores the result as confidence, so recall hedges instead of asserting. No
read-back (extraction is background -- nothing to confirm into); the tag is the
whole fix.

Hermetic: the LLM (generate_memory) and store_fact are monkeypatched, so this
runs with no model/DB. Run: .venv/bin/python ops/test_extraction_value_confidence.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory.extraction as ex

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)

# whisper words: "Remember my horse is named Thorn." with a LOW 'Thorn' (mishear
# of Thorne) and a separate high-confidence run for 'Rex'.
WORDS_LOW = [(" Remember", 0.9), (" my", 0.5), (" horse", 0.9), (" is", 0.99),
             (" named", 0.99), (" Thorn", 0.30), (".", 0.6)]
WORDS_HIGH = [(" my", 0.5), (" dog", 0.93), (" is", 0.99), (" named", 0.99),
              (" Rex", 0.95), (".", 0.6)]


def _patch_llm(fact_value):
    """generate_memory: 1st call (classifier) -> 'yes'; 2nd (extraction) -> JSON
    with one fact carrying fact_value."""
    calls = {"n": 0}
    async def fake_generate_memory(prompt, thinking=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return "yes"
        return ('{"facts": [{"subject": "user\'s pet", "predicate": "name", '
                f'"value": "{fact_value}"}}], "memories": []}}')
    return fake_generate_memory


async def _run_extraction(value, words):
    """Drive _do_extraction once with patched LLM + store_fact; return the
    confidence store_fact was called with (or None if never called)."""
    captured = {}
    async def fake_store_fact(subject, predicate, val, **kw):
        captured["confidence"] = kw.get("confidence")
        captured["value"] = val
    orig_gen, orig_store = ex.generate_memory, ex.store_fact
    ex.generate_memory = _patch_llm(value)
    ex.store_fact = fake_store_fact
    ex._extraction_running = True  # _do_extraction resets it + pumps in finally
    try:
        await ex._do_extraction({
            "user_text": f"Remember my pet is named {value}.",
            "assistant_text": "ok",
            "speaker_id": None, "speaker_name": None,
            "stt_words": words, "ts": 1000.0, "retries": 0,
        })
    finally:
        ex.generate_memory, ex.store_fact = orig_gen, orig_store
    return captured


async def main():
    print("Extraction-path value-confidence gate:")

    # 1. low-confidence mishear -> stored LOW (not the 1.0 default that made it
    #    a silent verified FALSE).
    c = await _run_extraction("Thorn", WORDS_LOW)
    check("misheard extracted value scored low (not default 1.0)",
          c.get("confidence") is not None and abs(c["confidence"] - 0.30) < 1e-6,
          f"got {c.get('confidence')}")

    # 2. confident correct value -> stored high.
    c = await _run_extraction("Rex", WORDS_HIGH)
    check("confident extracted value scored high",
          c.get("confidence") is not None and abs(c["confidence"] - 0.95) < 1e-6,
          f"got {c.get('confidence')}")

    # 3. text path / no audio -> None -> 1.0 default (prior behavior preserved).
    c = await _run_extraction("Rex", None)
    check("no words -> confidence defaults to 1.0",
          c.get("confidence") == 1.0, f"got {c.get('confidence')}")

    # 4. value not locatable in the words (paraphrased) -> None -> 1.0.
    c = await _run_extraction("Bramble", WORDS_LOW)  # 'Bramble' absent from WORDS_LOW
    check("unlocatable value -> 1.0 (fallback, not a spurious low)",
          c.get("confidence") == 1.0, f"got {c.get('confidence')}")

    # 5. coalesce concatenates per-turn word lists in order.
    coalesced = ex._coalesce_by_speaker([
        {"user_text": "a", "assistant_text": "x", "speaker_id": 1,
         "speaker_name": "dan", "stt_words": WORDS_HIGH, "ts": 1.0},
        {"user_text": "b", "assistant_text": "y", "speaker_id": 1,
         "speaker_name": "dan", "stt_words": WORDS_LOW, "ts": 2.0},
    ])
    check("one coalesced pass for same speaker", len(coalesced) == 1)
    check("coalesced stt_words = concatenation of both turns",
          coalesced[0]["stt_words"] == WORDS_HIGH + WORDS_LOW,
          f"len {len(coalesced[0]['stt_words'] or [])}")
    # a value from EITHER turn is still locatable in the concatenated list
    from stt.client import value_confidence
    check("value from 2nd turn locatable in concatenated words",
          value_confidence(coalesced[0]["stt_words"], "Thorn") == 0.30)

    # 6. all-text coalesce -> stt_words None (no spurious empty list)
    coalesced2 = ex._coalesce_by_speaker([
        {"user_text": "a", "assistant_text": "x", "speaker_id": 1,
         "speaker_name": "dan", "stt_words": None, "ts": 1.0},
    ])
    check("coalesced stt_words is None when no turn had audio",
          coalesced2[0]["stt_words"] is None)

    print()
    print("ALL PASS" if not fails else f"FAILED ({len(fails)}): {', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    asyncio.run(main())
