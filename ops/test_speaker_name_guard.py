#!/usr/bin/env python3
"""Hermetic unit test for the speaker-name-overwrite collapse guard
(_speaker_name_overwrite_collapse) added 2026-06-20.

Validates the CATASTROPHIC path that a guest voice cannot reproduce acoustically:
speaker=dan + "remember my <thing> is named X" must NOT overwrite dan.name.
Pure function; no DB, no live LLM. Run: .venv/bin/python ops/test_speaker_name_guard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conversation.tool_router import _speaker_name_overwrite_collapse as G

# (utterance, subject_after_normalize, predicate, speaker_name, expect_block)
CASES = [
    # --- CATASTROPHIC: collapse onto speaker's own name -> MUST block ---
    ("remember my robot is named Sparky", "dan", "name", "dan", True),
    ("my dog is named Rex",               "dan", "name", "dan", True),
    ("my friend is named Sparky",         "dan", "name", "dan", True),
    ("remember my spaceship is called Voyager", "dan", "name", "dan", True),
    ("my horse is named Apollo",          "dan", "is",   "dan", True),
    # --- LEGITIMATE self-naming -> MUST allow ---
    ("remember my name is Dan",           "dan", "name", "dan", False),
    ("call me Danny",                     "dan", "name", "dan", False),
    ("I'm Dan",                           "dan", "name", "dan", False),
    ("I am Daniel",                       "dan", "name", "dan", False),
    # --- LEGITIMATE entity name (subject != speaker) -> MUST allow ---
    ("my dog's name is Rex",              "dan's dog", "name", "dan", False),
    ("my robot is named Sparky",          "dan's robot", "name", "dan", False),
    # --- guest speaker: nothing to protect -> MUST allow ---
    ("remember my robot is named Sparky", "user", "name", "unknown_1", False),
    # --- non-name predicate: out of scope -> MUST allow ---
    ("my robot is named Sparky",          "dan", "has_robot", "dan", False),
    ("my car's make is Toyota",           "dan", "make", "dan", False),
    # --- no speaker name -> allow ---
    ("my robot is named Sparky",          "dan", "name", None, False),
]

fails = 0
for utt, subj, pred, spk, expect in CASES:
    got = G(utt, subj, pred, spk)
    ok = (got == expect)
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] block={got!s:5} want={expect!s:5} "
          f"| spk={spk!r} {subj}.{pred}  <- {utt!r}")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'} ({len(CASES)} cases)")
sys.exit(1 if fails else 0)
