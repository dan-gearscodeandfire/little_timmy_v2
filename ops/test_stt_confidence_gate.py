#!/usr/bin/env python3
"""STT value-confidence framework (2026-06-21).

Addresses the dominant FALSE source found in acoustic testing: a misheard fact
VALUE (Bolt->Volt, Blaze->Blazed) silently committing as a confident "verified
fact". The framework scores the acoustic confidence of the value word(s) against
whisper's per-word probabilities and, below threshold, (a) tags the stored fact
low-confidence, (b) reads the value back in the ACK, and (c) surfaces it to the
brain as HEARD-BUT-UNCONFIRMED so recall hedges instead of asserting. The goal:
turn a silent FALSE into a surfaced AMBIGUITY (TRUE > AMBIGUITY > FALSE).

Covers all three layers hermetically:
  1. stt.client.value_confidence  (pure)
  2. memory.facts.store_fact persists `confidence`  (DB, isolated subject)
  3. llm.prompt_builder splits verified vs unconfirmed facts  (pure render)

Run: .venv/bin/python ops/test_stt_confidence_gate.py
"""
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from stt.client import value_confidence, is_name_like_value
from memory.facts import store_fact, Fact
from db.connection import get_pool
from llm.prompt_builder import build_ephemeral_block

SUBJ = "__test_conf__"
THR = config.STT_VALUE_CONFIDENCE_THRESHOLD

# whisper word list from a real probe ("Remember my beacon is named glow.")
WORDS = [(" Remember", 0.94), (" my", 0.49), (" beacon", 0.93), (" is", 0.99),
         (" named", 0.99), (" glow", 0.645), (".", 0.63)]

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def layer1_value_confidence():
    print("Layer 1 — value_confidence (pure):")
    check("single value word returns its prob", value_confidence(WORDS, "glow") == 0.645)
    check("case-insensitive", value_confidence(WORDS, "Glow") == 0.645)
    check("higher-prob value", value_confidence(WORDS, "beacon") == 0.93)
    check("value word absent -> None (fallback to utterance)",
          value_confidence(WORDS, "Voyager") is None)
    w = WORDS + [(" Pioneer", 0.7), (" Now", 0.4)]
    check("multiword value -> weakest word", value_confidence(w, "Pioneer Now") == 0.4)
    check("no words -> None", value_confidence([], "glow") is None)
    # whisper splits uncommon proper nouns into sub-word pieces; the value must
    # still be scored by reassembling them (regression: live 2026-06-21 these
    # fell through to 1.0 and bypassed the gate).
    subword = [("named", 0.99), ("On", 0.65), ("yx", 0.84), (".", 0.64)]
    check("sub-word split value (On+yx=Onyx) -> min piece",
          value_confidence(subword, "Onyx") == 0.65)
    zsplit = [("Z", 0.9), ("olt", 0.7), ("an", 0.98)]
    check("3-piece split (Z+olt+an=Zoltan) -> min piece",
          value_confidence(zsplit, "Zoltan") == 0.7)


def layer1b_name_like():
    # read-back-always-on-novel-noun (v2): a *confident* homophone (Thorne->Thorn
    # 0.721) clears value_confidence but is still a misheard name. Names/proper
    # nouns must be read back regardless of confidence. COMMON is injected so the
    # proper-noun branch is hermetic (independent of /usr/share/dict/words).
    print("Layer 1b — is_name_like_value (proper-noun read-back trigger):")
    COMMON = frozenset({"blue", "teal", "pizza", "red", "thorn", "wren"})
    # (a) name slot fires regardless of the value or whether it's a real word --
    #     this is the whole point: Thorne->Thorn lands a real word ("thorn") but
    #     the predicate marks it a name, so we still read it back.
    check("name predicate -> True", is_name_like_value("name", "Thorn", COMMON))
    check("dotted name predicate -> True", is_name_like_value("robot.name", "Prax", COMMON))
    check("'first name' predicate -> True", is_name_like_value("first name", "Max", COMMON))
    # (b) proper-noun value under a NON-name predicate
    check("coined value under non-name pred -> True",
          is_name_like_value("company", "Praxton", COMMON))
    check("capitalized COMMON word under non-name pred -> False",
          not is_name_like_value("color", "Blue", COMMON))
    check("lowercase common value -> False",
          not is_name_like_value("color", "teal", COMMON))
    check("multi-word non-name value -> False (v1 single-token only)",
          not is_name_like_value("favorite food", "deep dish", COMMON))
    check("empty value -> False", not is_name_like_value("name", "", COMMON))
    # no dictionary available -> proper-noun branch disabled, name slot still works
    check("name slot works with empty dict", is_name_like_value("name", "Zephyr", frozenset()))
    check("proper-noun branch off when dict empty",
          not is_name_like_value("city", "Zephyr", frozenset()))


async def layer2_persist():
    print("Layer 2 — store_fact persists confidence (DB):")
    pool = await get_pool()
    await pool.execute("DELETE FROM facts WHERE subject=$1", SUBJ)
    try:
        await store_fact(SUBJ, "robot", "Sparky", source="tool", confidence=0.42)
        row = await pool.fetchrow(
            "SELECT value, confidence FROM facts WHERE subject=$1 AND predicate='robot' "
            "AND superseded_by IS NULL", SUBJ)
        check("low-confidence value stored (not lost)", row and row["value"] == "Sparky")
        check("confidence persisted ~0.42",
              row and abs(row["confidence"] - 0.42) < 1e-3,
              f"got {row['confidence'] if row else None}")
        # a confident correction lifts it back
        await store_fact(SUBJ, "robot", "Rusty", source="tool", confidence=1.0)
        row = await pool.fetchrow(
            "SELECT value, confidence FROM facts WHERE subject=$1 AND predicate='robot' "
            "AND superseded_by IS NULL", SUBJ)
        check("confident correction overwrites + lifts confidence",
              row and row["value"] == "Rusty" and row["confidence"] >= 0.99)
    finally:
        await pool.execute("DELETE FROM facts WHERE subject=$1", SUBJ)


def layer3_prompt_split():
    print("Layer 3 — prompt_builder splits verified vs unconfirmed:")
    now = datetime(2026, 6, 21, 2, 0)
    hi = Fact(1, "dan", "robot", "Rusty", now, 1.0)
    lo = Fact(2, "dan", "drone", "Volt", now, 0.40)
    block = build_ephemeral_block([], [hi, lo], speaker_name="dan", now=now)
    check("verified block present for high-confidence fact",
          "GROUND TRUTH" in block and "robot Rusty" in block)
    check("unconfirmed block present for low-confidence fact",
          "HEARD BUT UNCONFIRMED" in block and "drone Volt" in block)
    # the low one must NOT appear under verified ground truth
    gt = block.split("HEARD BUT UNCONFIRMED")[0]
    check("low-confidence value NOT in the verified section", "Volt" not in gt)


async def main():
    layer1_value_confidence()
    layer1b_name_like()
    await layer2_persist()
    layer3_prompt_split()
    print()
    print("ALL PASS" if not fails else f"FAILED ({len(fails)}): {', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    asyncio.run(main())
