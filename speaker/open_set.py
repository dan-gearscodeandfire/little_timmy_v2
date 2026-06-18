"""Open-set speaker rejection: anti-model + s-norm scoring.

Production speaker-ID (``speaker/identifier.py``) accepts the nearest enrolled
identity whenever its min-cosine distance falls below ``KNOWN_SPEAKER_THRESHOLD``
(0.30). That single raw threshold is brittle in the OPEN set: a degraded / far /
quiet / noise capture rots toward a noise-shaped embedding centroid, and
whichever enrolled print sits nearest that centroid (measured 2026-06-17: Erin)
wins a sub-threshold FALSE ACCEPT. See
``memory/project_lt_erin_voiceprint_noise_centroid_bug``.

This module adds an open-set GUARD that an otherwise-accepted match must also
pass. We score the test utterance against an IMPOSTOR COHORT (an "anti-model")
of non-enrolled speakers + room noise, and normalize the claimed-speaker
similarity by that cohort's score distribution (s-norm, the standard
speaker-verification normalization). A noise-collapsed utterance is generically
close to the WHOLE cohort, so its normalized score stays low even when its raw
distance to one enrolled print dips below 0.30 -> it is rejected, not
mis-stamped onto the nearest real person.

Two complementary signals, both returned so the caller (and the calibration
harness) can pick the operating point:

  s-norm   = 0.5 * ( (s_raw - mu_e)/sd_e  +  (s_raw - mu_t)/sd_t )
      s_raw = cosine SIMILARITY of the test to its best-matching prototype
              ( = 1 - best_known_dist, so it tracks the production matcher )
      mu_e,sd_e : distribution of the ENROLLED MODEL's similarity to the cohort
                  ( z-norm term — "how cohort-like is this identity in general" )
      mu_t,sd_t : distribution of the TEST utterance's similarity to the cohort
                  ( t-norm term — "how cohort-like is THIS utterance" — the term
                    that catches a noise utterance that is close to everything )
      Higher s-norm = more confidently a genuine, non-cohort match.

  anti-model margin = s_raw - max_cohort_sim
      How much closer the test sits to the claimed speaker than to its NEAREST
      cohort member. <= 0 means the utterance is at least as close to a known
      impostor / noise sample as to the enrolled speaker -> reject. A simple,
      un-normalized companion to s-norm.

Pure + deterministic (vectors in, decision out), mirroring
``continuity_allowed()``: no audio and no I/O in the hot path, so it is
unit-testable without the encoder. Cohort loading is the only filesystem touch
and is done once at construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_EPS = 1e-8


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + _EPS)


def _l2_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + _EPS)


@dataclass
class OpenSetScore:
    """One scoring of a test utterance against a claimed identity + cohort."""
    s_raw: float          # max cosine similarity to any claimed prototype (=1-dist)
    snorm: float          # s-normalized score (higher = more genuine)
    z_e: float            # z-norm term (enrolled-model normalization)
    z_t: float            # t-norm term (test-utterance normalization)
    max_cohort_sim: float # similarity to the NEAREST cohort member
    am_margin: float      # s_raw - max_cohort_sim (anti-model margin)


class OpenSetScorer:
    """Scores a test embedding against a claimed identity using an impostor
    cohort. Cohort rows are L2-normalized at construction; all scoring is
    cosine similarity on unit vectors."""

    def __init__(self, cohort: np.ndarray):
        C = np.asarray(cohort, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] == 0:
            raise ValueError("cohort must be a non-empty (N, D) array")
        self.cohort = _l2_rows(C)
        self.dim = self.cohort.shape[1]

    @property
    def size(self) -> int:
        return self.cohort.shape[0]

    @classmethod
    def from_dir(cls, cohort_dir) -> "OpenSetScorer | None":
        """Load every ``*.npy`` under ``cohort_dir`` (each a (D,) vector or
        (K, D) set) and stack into one cohort. Returns None if the directory is
        missing or holds no usable embeddings (caller treats None = guard
        unavailable -> fall back to raw threshold)."""
        d = Path(cohort_dir)
        if not d.is_dir():
            return None
        rows = []
        for p in sorted(d.glob("*.npy")):
            try:
                a = np.load(p)
            except Exception as e:  # pragma: no cover - corrupt file
                log.warning("open-set cohort: failed to load %s: %s", p.name, e)
                continue
            if a.ndim == 1:
                a = a[None, :]
            if a.ndim == 2 and a.shape[0] > 0:
                rows.append(a.astype(np.float64))
        if not rows:
            return None
        dims = {r.shape[1] for r in rows}
        if len(dims) != 1:
            log.warning("open-set cohort: mixed embedding dims %s; skipping guard", dims)
            return None
        return cls(np.vstack(rows))

    def score(self, test_emb: np.ndarray, prototypes: np.ndarray) -> OpenSetScore:
        """Score a test embedding against a claimed identity's prototype set
        (shape (K, D); a (D,) vector is accepted and treated as K=1)."""
        t = _l2(test_emb)
        P = np.asarray(prototypes, dtype=np.float64)
        if P.ndim == 1:
            P = P[None, :]
        P = _l2_rows(P)

        proto_sims = P @ t                       # (K,)
        s_raw = float(proto_sims.max())

        tc = self.cohort @ t                     # (N,) test-vs-cohort sims
        mu_t = float(tc.mean()); sd_t = float(tc.std()) + _EPS
        z_t = (s_raw - mu_t) / sd_t

        ec = (self.cohort @ P.T).max(axis=1)     # (N,) model-vs-cohort sims (max over protos)
        mu_e = float(ec.mean()); sd_e = float(ec.std()) + _EPS
        z_e = (s_raw - mu_e) / sd_e

        snorm = 0.5 * (z_e + z_t)
        max_cohort_sim = float(tc.max())
        return OpenSetScore(
            s_raw=s_raw, snorm=snorm, z_e=z_e, z_t=z_t,
            max_cohort_sim=max_cohort_sim, am_margin=s_raw - max_cohort_sim,
        )

    def accept(self, test_emb: np.ndarray, prototypes: np.ndarray,
               t_snorm: float, min_am_margin: float | None = None) -> bool:
        """True iff the test passes the open-set guard for this identity:
        s-norm at/above ``t_snorm`` and (optionally) anti-model margin at/above
        ``min_am_margin``."""
        sc = self.score(test_emb, prototypes)
        if sc.snorm < t_snorm:
            return False
        if min_am_margin is not None and sc.am_margin < min_am_margin:
            return False
        return True
