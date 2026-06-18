"""Deterministic tests for the open-set anti-model / s-norm guard.

No audio, no encoder, no I/O — synthetic embeddings constructed so the geometry
reproduces the production failure mode:

  - a GENUINE utterance sits near the enrolled prototype and FAR from the
    impostor cohort -> high s-norm, positive anti-model margin -> ACCEPT.
  - a NOISE-COLLAPSE utterance sits in the cohort blob (close to everything,
    including, by bad luck, just inside 0.30 of one enrolled print) -> low
    s-norm, non-positive margin -> REJECT. This is the Erin false-accept.

Also pins the default-OFF invariant: with the open-set flag unset, identify()
behaves byte-for-byte as before (the guard is never constructed or consulted).

Run:
    .venv/bin/pytest tests/test_open_set.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speaker.open_set import OpenSetScorer, OpenSetScore  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _make_cohort(dim=64, n=16, seed=0):
    """A tight 'noise/impostor' blob: vectors clustered around one direction,
    mimicking how degraded captures collapse together (the A/B finding)."""
    rng = np.random.default_rng(seed)
    center = _unit(rng.standard_normal(dim))
    rows = []
    for _ in range(n):
        rows.append(_unit(center + 0.15 * rng.standard_normal(dim)))
    return np.vstack(rows), center


def test_genuine_accepts_noise_rejects():
    dim = 64
    cohort, noise_center = _make_cohort(dim=dim)
    scorer = OpenSetScorer(cohort)

    # Enrolled identity: a direction well away from the cohort blob.
    rng = np.random.default_rng(99)
    proto = _unit(rng.standard_normal(dim))
    # ensure the prototype isn't accidentally near the noise center
    while abs(float(proto @ noise_center)) > 0.2:
        proto = _unit(rng.standard_normal(dim))
    prototypes = proto[None, :]

    # Genuine: very close to the prototype, away from cohort.
    genuine = _unit(proto + 0.05 * rng.standard_normal(dim))
    g = scorer.score(genuine, prototypes)

    # Noise-collapse: sits in the cohort blob but nudged toward the prototype
    # just enough to be "close" in raw terms (the sub-0.30 false accept).
    collapse = _unit(noise_center + 0.30 * proto)
    c = scorer.score(collapse, prototypes)

    # raw similarity can be comparable / the collapse can even look "close",
    # but s-norm and the anti-model margin must separate them.
    assert g.snorm > c.snorm, (g.snorm, c.snorm)
    assert g.am_margin > 0 > c.am_margin or g.am_margin > c.am_margin
    # a threshold exists between them
    t = 0.5 * (g.snorm + c.snorm)
    assert scorer.accept(genuine, prototypes, t_snorm=t)
    assert not scorer.accept(collapse, prototypes, t_snorm=t)


def test_anti_model_margin_rejects_cohort_member():
    cohort, _ = _make_cohort()
    scorer = OpenSetScorer(cohort)
    proto = _unit(np.r_[1.0, np.zeros(cohort.shape[1] - 1)])
    # A test that IS essentially a cohort member -> margin must be <= ~0.
    member = cohort[3]
    sc = scorer.score(member, proto[None, :])
    assert sc.max_cohort_sim > 0.95          # it is in the cohort
    assert sc.am_margin <= 0.05              # not closer to proto than to cohort
    assert not scorer.accept(member, proto[None, :], t_snorm=0.0, min_am_margin=0.1)


def test_kprototype_uses_best_matching():
    cohort, _ = _make_cohort()
    dim = cohort.shape[1]
    rng = np.random.default_rng(7)
    p1 = _unit(rng.standard_normal(dim))
    p2 = _unit(rng.standard_normal(dim))
    protos = np.vstack([p1, p2])
    # test near p2 only -> s_raw should reflect the p2 match, not the mean
    test = _unit(p2 + 0.02 * rng.standard_normal(dim))
    sc = scorer = OpenSetScorer(cohort).score(test, protos)
    assert sc.s_raw > 0.9


def test_from_dir_missing_returns_none(tmp_path):
    assert OpenSetScorer.from_dir(tmp_path / "nope") is None
    # empty dir -> None
    (tmp_path / "empty").mkdir()
    assert OpenSetScorer.from_dir(tmp_path / "empty") is None


def test_from_dir_loads_and_stacks(tmp_path):
    d = tmp_path / "cohort"
    d.mkdir()
    np.save(d / "a.npy", _unit(np.r_[1.0, 0, 0, 0]))            # (D,)
    np.save(d / "b.npy", np.vstack([_unit([0, 1.0, 0, 0]),
                                    _unit([0, 0, 1.0, 0])]))     # (K, D)
    s = OpenSetScorer.from_dir(d)
    assert s is not None and s.size == 3 and s.dim == 4


def test_empty_cohort_raises():
    with pytest.raises(ValueError):
        OpenSetScorer(np.zeros((0, 8)))


def test_identifier_default_off_is_unchanged(monkeypatch):
    """With the flag OFF (default), a sub-threshold known match is accepted
    exactly as before — the guard must not run or alter the result."""
    from speaker import identifier as idmod

    # default flag must be False
    assert idmod.OPEN_SET_REJECT_ENABLED is False

    si = idmod.SpeakerIdentifier()
    # Hand-build one known speaker with a single prototype; bypass disk + encoder.
    proto = _unit(np.r_[1.0, np.zeros(255)])[None, :]
    si._known_speakers = [idmod.KnownSpeaker(speaker_id=1, name="dan", prototypes=proto)]
    # Make extract_embedding return a vector ~0.1 from the prototype (a match).
    near = _unit(np.r_[1.0, 0.1, np.zeros(254)])
    monkeypatch.setattr(si, "extract_embedding", lambda audio: near)

    res = si.identify(np.zeros(16000, dtype=np.float32))
    assert res.name == "dan"
    assert res.confidence > 0.7
