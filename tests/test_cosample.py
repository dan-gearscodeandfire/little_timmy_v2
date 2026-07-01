"""Tests for the passive co-sample crop buffer."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presence.cosample import CoSampleBuffer  # noqa: E402


def _crop(v):
    return np.full((112, 112, 3), v, dtype=np.uint8)


def test_add_and_retrieve_per_speaker():
    b = CoSampleBuffer()
    b.add("unknown_1", [_crop(1), _crop(2)])
    b.add("unknown_2", [_crop(3)])
    assert len(b.crops_for("unknown_1")) == 2
    assert len(b.crops_for("unknown_2")) == 1
    assert b.crops_for("nobody") == []


def test_noop_on_empty():
    b = CoSampleBuffer()
    b.add("", [_crop(1)])
    b.add("unknown_1", [])
    b.add("unknown_1", [None])
    assert len(b) == 0


def test_per_speaker_cap():
    b = CoSampleBuffer(max_crops=100, max_per_speaker=3)
    b.add("unknown_1", [_crop(i) for i in range(10)])
    assert len(b.crops_for("unknown_1")) == 3


def test_global_ring_cap():
    b = CoSampleBuffer(max_crops=4, max_per_speaker=100)
    for i in range(6):
        b.add(f"s{i}", [_crop(i)])
    assert len(b) == 4                    # oldest evicted
    assert b.crops_for("s0") == []        # first speaker dropped


def test_clear_speaker():
    b = CoSampleBuffer()
    b.add("unknown_1", [_crop(1)])
    b.add("unknown_2", [_crop(2)])
    b.clear_speaker("unknown_1")
    assert b.crops_for("unknown_1") == []
    assert len(b.crops_for("unknown_2")) == 1
