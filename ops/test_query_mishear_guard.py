#!/usr/bin/env python3
"""Query-side mishear guard (2026-06-22).

The store-side gate protects WRITES; this protects the READ path. A misheard
CONTENT word in a question ("what's my mail" -> "my male") silently keys
retrieval on the wrong term, so Timmy answers confidently wrong or falsely
denies knowledge. low_confidence_query_term() flags a sub-threshold content word
(skipping function words, whose low prob is acoustic noise, not a misheard query
term), and build_ephemeral_block injects an [UNCERTAIN INPUT] hint telling the
brain to confirm rather than guess/deny. Goal: turn a query-side FALSE into a
surfaced AMBIGUITY (TRUE > AMBIGUITY > FALSE).

Hermetic (pure). Run: .venv/bin/python ops/test_query_mishear_guard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt.client import low_confidence_query_term
from llm.prompt_builder import build_ephemeral_block

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)

THR = 0.55


def layer1_detect():
    print("Layer 1 — low_confidence_query_term (pure):")
    # "what's my mail" with a LOW content word 'mail' (misheard) -> flagged.
    w_low = [(" what's", 0.95), (" my", 0.49), (" mail", 0.33)]
    check("low content word flagged", low_confidence_query_term(w_low, THR) == "mail")
    # function word 'my' is low but must be IGNORED (not a query term).
    w_fn = [(" what's", 0.95), (" my", 0.30), (" robot", 0.92)]
    check("low function word ignored (only content words count)",
          low_confidence_query_term(w_fn, THR) is None)
    # all content words clear threshold -> None (no needless hint).
    w_hi = [(" what's", 0.95), (" my", 0.49), (" dog's", 0.88), (" name", 0.91)]
    check("clean content words -> None", low_confidence_query_term(w_hi, THR) is None)
    # picks the WORST content word when several dip.
    w_two = [(" my", 0.4), (" otter", 0.50), (" Quill", 0.42)]
    check("returns the worst content word", low_confidence_query_term(w_two, THR) == "Quill")
    # 'name'/'named'/'remember' are treated as function/stop words here.
    w_name = [(" what's", 0.95), (" my", 0.49), (" name", 0.30)]
    check("'name' itself is a stopword (not the query subject)",
          low_confidence_query_term(w_name, THR) is None)
    check("empty words -> None", low_confidence_query_term([], THR) is None)
    # threshold is exclusive-ish: a word AT threshold does not fire (< only).
    w_at = [(" robot", 0.55)]
    check("content word exactly at threshold does NOT fire",
          low_confidence_query_term(w_at, THR) is None)
    # sub-word pieces: whisper splits a rare noun and a fragment is low. Must
    # return the REASSEMBLED word, never the fragment (live bug 2026-06-22: it
    # surfaced 's' from 'micro-santhemums', 'ab' from 'boulabes').
    subword = [(" micro", 0.9), ("-", 0.8), ("s", 0.33), ("anth", 0.7), ("emums", 0.6)]
    check("reassembles sub-word pieces, returns whole word not fragment",
          low_confidence_query_term(subword, THR) == "micro-santhemums")
    boul = [(" Did", 0.9), (" I", 0.8), (" hear", 0.9), (" bou", 0.7), ("lab", 0.42), ("es", 0.6)]
    check("multi-piece rare noun -> whole word",
          low_confidence_query_term(boul, THR) == "boulabes")
    # a lone short fragment is never surfaced (len < 3 after reassembly).
    frag = [(" hmm", 0.9), (" s", 0.2)]
    check("lone sub-3-char fragment not flagged", low_confidence_query_term(frag, THR) is None)


def layer2_inject():
    print("Layer 2 — build_ephemeral_block injects the guard:")
    blk = build_ephemeral_block([], [], speaker_name="dan", uncertain_query_term="male")
    check("[UNCERTAIN INPUT] present when term set", "[UNCERTAIN INPUT]" in blk)
    check("heard word echoed for read-back", '"male"' in blk and "did you say male" in blk)
    blk2 = build_ephemeral_block([], [], speaker_name="dan")
    check("no guard when term is None", "[UNCERTAIN INPUT]" not in blk2)


if __name__ == "__main__":
    layer1_detect()
    layer2_inject()
    print()
    print("ALL PASS" if not fails else f"FAILED ({len(fails)}): {', '.join(fails)}")
    sys.exit(1 if fails else 0)
