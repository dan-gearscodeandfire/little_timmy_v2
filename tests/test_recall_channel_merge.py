"""The two-retrieval-channel merge (2026-08-18, 'option C').

When a recall tool fires, the turn used to carry the always-on RECALLED block
AND the tool's [WHAT WE TALKED ABOUT] block, largely word-for-word duplicates
under contradictory preambles. build_ephemeral_block now folds the always-on
items into the tool block: dedupe by content, keep unique items, one preamble.

Run: .venv/bin/pytest tests/test_recall_channel_merge.py -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.prompt_builder import build_ephemeral_block
from memory.retrieval import RetrievedMemory


def _mem(i, text, days=2):
    return RetrievedMemory(id=i, type="proposition", content=text, score=0.05,
                           created_at=datetime.now().astimezone() - timedelta(days=days))


TOOL_BLOCK = (
    "[WHAT WE TALKED ABOUT] Specific things you recorded from past "
    "conversations, most relevant first. Answer using ONLY these; if they "
    "don't actually cover it, say so rather than guessing:\n"
    "- (Aug 13) Timmy jokes that he would order a bucket of screws at Taco Bell.\n"
    "- (Aug 10) Dan identified the cats as Dexter and Preston."
)


def test_duplicates_are_absorbed_not_repeated():
    mems = [_mem(1, "Timmy jokes that he would order a bucket of screws at Taco Bell."),
            _mem(2, "Dan identified the cats as Dexter and Preston.")]
    blk = build_ephemeral_block(memories=mems, facts=[], speaker_name="Dan",
                                recall_block=TOOL_BLOCK)
    assert "RECALLED FROM PAST CONVERSATIONS" not in blk
    assert blk.count("bucket of screws") == 1
    assert blk.count("Dexter and Preston") == 1


def test_unique_items_are_kept():
    mems = [_mem(1, "Timmy jokes that he would order a bucket of screws at Taco Bell."),
            _mem(3, "Dan re-enrolled every voiceprint after swapping microphones.")]
    blk = build_ephemeral_block(memories=mems, facts=[], speaker_name="Dan",
                                recall_block=TOOL_BLOCK)
    # unique item survives, inside the single tool block
    assert "re-enrolled every voiceprint" in blk
    assert "RECALLED FROM PAST CONVERSATIONS" not in blk
    # exactly one preamble
    assert blk.count("[WHAT WE TALKED ABOUT]") == 1


def test_no_tool_block_means_no_change():
    mems = [_mem(1, "Dan identified the cats as Dexter and Preston.")]
    blk = build_ephemeral_block(memories=mems, facts=[], speaker_name="Dan",
                                recall_block=None)
    assert "RECALLED FROM PAST CONVERSATIONS" in blk
    assert "Dexter and Preston" in blk


def test_tool_block_without_memories_unchanged():
    blk = build_ephemeral_block(memories=[], facts=[], speaker_name="Dan",
                                recall_block=TOOL_BLOCK)
    assert blk.count("[WHAT WE TALKED ABOUT]") == 1
    assert "RECALLED FROM PAST CONVERSATIONS" not in blk


def test_contradictory_preamble_gone_on_merged_turns():
    """The RECALLED preamble ('say nothing about them') must not coexist with
    the tool preamble ('answer using ONLY these') on the same turn."""
    mems = [_mem(9, "Something entirely unrelated happened in the shop.")]
    blk = build_ephemeral_block(memories=mems, facts=[], speaker_name="Dan",
                                recall_block=TOOL_BLOCK)
    assert "say nothing about them" not in blk
    assert blk.count("Answer using ONLY these") == 1
