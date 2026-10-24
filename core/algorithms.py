"""
algorithms.py
=============

UCB-LCB-GS and the five baselines from the experimental section.

All algorithms share:

  * the same ``mu``, ``T``, ``UCB``, ``LCB`` bookkeeping,
  * the same partition optimizer (``ashg_lib.compute_partition``),
  * the same deviation checker  (``ashg_lib.is_stable``),
  * the same numpy ``Generator`` for tie-breaking / random fallbacks.

Every algorithm exposes a uniform interface::

    alg = Algorithm(n, sigma, beta, k, eps, rng, ...)
    pi_t, declared_existence = alg.act(t)        # returns Partition + d_t
    alg.update(observations)                     # noisy pair feedback

so the simulation runner is algorithm-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ashg_lib import (
    ASHGEnvironment,
    Partition,
    compute_partition,
    is_stable,
    random_partition,
)


# ============================================================
# Shared estimator state
# ============================================================


class PairwiseEstimator:
    """Sample mean / count and UCB-LCB confidence intervals per pair."""

    def __init__(self, n: int, sigma: float, beta: float, init_bonus: float = 1e3,
                 directed: bool = False):
        self.n = int(n)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.init_bonus = float(init_bonus)
        # ``directed=False`` (default) reproduces the paper's symmetric
        # estimator exactly: one observation per unordered pair, mirrored into
        # both entries.  ``directed=True`` is used only by the asymmetric
        # robustness check, where each ordered pair (i, j) has its own utility
        # and its own feedback stream, so mirroring would average the two
        # directions and silently collapse back to a symmetric estimate.
        self.directed = bool(directed)
        self.mu = np.zeros((n, n), dtype=np.float64)
        self.T = np.zeros((n, n), dtype=np.int64)

    def update(self, observations: dict) -> None:
        for (i, j), x in observations.items():
            self.T[i, j] += 1
            if not self.directed:
                self.T[j, i] = self.T[i, j]
            # incremental sample mean
            t = self.T[i, j]
            self.mu[i, j] += (x - self.mu[i, j]) / t
            if not self.directed:
                self.mu[j, i] = self.mu[i, j]

    def bounds(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (UCB, LCB) matrices at round ``t`` (1-indexed)."""
        log_t = np.log(max(t, 2))
        denom = np.maximum(1, self.T)
        bonus = np.sqrt(2.0 * self.sigma ** 2 * self.beta * log_t / denom)
        # Force exploration of unseen pairs.
        bonus[self.T == 0] = self.init_bonus
        np.fill_diagonal(bonus, 0.0)  # diagonal never used
        ucb = self.mu + bonus
        lcb = self.mu - bonus
        # diagonal must be exactly 0 (no self-utility)
        np.fill_diagonal(ucb, 0.0)
        np.fill_diagonal(lcb, 0.0)
        return ucb, lcb


# ============================================================
# Base class
# ============================================================


@dataclass
class _AlgConfig:
    n: int
    sigma: float
    beta: float
    k: int
    eps: float
    rng: np.random.Generator
    n_restarts: int = 3
    max_iter: int = 60
    deviation_cap: int = 64
    max_random_coalitions: int = 4
    directed: bool = False          # asymmetric-utility robustness check only


class BaseAlgorithm:
    name: str = "base"

    def __init__(self, **kw):
        self.cfg = _AlgConfig(**kw)
        self.estimator = PairwiseEstimator(
            self.cfg.n, self.cfg.sigma, self.cfg.beta, directed=self.cfg.directed
        )

    # --- helpers -----------------------------------------------------------
    def _bounds(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.estimator.bounds(t)

    def _solve(self, etype: str, ucb: np.ndarray, lcb: np.ndarray) -> Partition:
        return compute_partition(
            etype,
            ucb,
            lcb,
            k=self.cfg.k,
            eps=self.cfg.eps,
            rng=self.cfg.rng,
            n_restarts=self.cfg.n_restarts,
            max_iter=self.cfg.max_iter,
            deviation_cap=self.cfg.deviation_cap,
        )

    def _stable(self, pi: Partition, etype: str, ucb: np.ndarray, lcb: np.ndarray) -> bool:
        if etype == "pessimistic":
            return is_stable(pi, lcb, ucb, self.cfg.k, self.cfg.eps, deviation_cap=self.cfg.deviation_cap)
        if etype == "optimistic":
            return is_stable(pi, ucb, lcb, self.cfg.k, self.cfg.eps, deviation_cap=self.cfg.deviation_cap)
        if etype == "ucb_only":
            return is_stable(pi, ucb, ucb, self.cfg.k, self.cfg.eps, deviation_cap=self.cfg.deviation_cap)
        if etype == "lcb_only":
            return is_stable(pi, lcb, lcb, self.cfg.k, self.cfg.eps, deviation_cap=self.cfg.deviation_cap)
        raise ValueError(etype)

    def _random_partition(self) -> Partition:
        return random_partition(self.cfg.n, self.cfg.max_random_coalitions, self.cfg.rng)

    # --- public ------------------------------------------------------------
    def act(self, t: int) -> Tuple[Partition, int]:
        raise NotImplementedError

    def update(self, observations: dict) -> None:
        self.estimator.update(observations)


# ============================================================
# UCB-LCB-GS  (the paper's algorithm)
# ============================================================


class UCBLCBGS(BaseAlgorithm):
    """
    UCB-LCB-GS.

    Round t:
      1. compute pessimistic partition pi_pess  (Eq. (7))
      2. if pi_pess is *pessimistically* stable -> declare existence,
         play pi_pess  (no need to explore further on this round).
      3. otherwise compute optimistic partition pi_opt  (Eq. (9))
         and play it -- this is the exploration step.
      4. update estimates with the observations of the played partition.
    """

    name = "UCB-LCB-GS"

    def act(self, t: int) -> Tuple[Partition, int]:
        ucb, lcb = self._bounds(t)
        pi_pess = self._solve("pessimistic", ucb, lcb)
        if self._stable(pi_pess, "pessimistic", ucb, lcb):
            return pi_pess, 1
        pi_opt = self._solve("optimistic", ucb, lcb)
        return pi_opt, 0


# ============================================================
# Baseline 1: pessimistic only
# ============================================================


class PessimisticOnly(BaseAlgorithm):
    name = "Pessimistic-only"

    def act(self, t: int) -> Tuple[Partition, int]:
        ucb, lcb = self._bounds(t)
        pi = self._solve("pessimistic", ucb, lcb)
        if self._stable(pi, "pessimistic", ucb, lcb):
            return pi, 1
        return self._random_partition(), 0


# ============================================================
# Baseline 2: optimistic only
# ============================================================


class OptimisticOnly(BaseAlgorithm):
    name = "Optimistic-only"

    def act(self, t: int) -> Tuple[Partition, int]:
        ucb, lcb = self._bounds(t)
        pi = self._solve("optimistic", ucb, lcb)
        if self._stable(pi, "optimistic", ucb, lcb):
            return pi, 1
        return self._random_partition(), 0


# ============================================================
# Baseline 3: UCB only
# ============================================================


class UCBOnly(BaseAlgorithm):
    name = "UCB-only"

    def act(self, t: int) -> Tuple[Partition, int]:
        ucb, lcb = self._bounds(t)
        pi = self._solve("ucb_only", ucb, lcb)
        if self._stable(pi, "ucb_only", ucb, lcb):
            return pi, 1
        return self._random_partition(), 0


# ============================================================
# Baseline 4: LCB only
# ============================================================


class LCBOnly(BaseAlgorithm):
    name = "LCB-only"

    def act(self, t: int) -> Tuple[Partition, int]:
        ucb, lcb = self._bounds(t)
        pi = self._solve("lcb_only", ucb, lcb)
        if self._stable(pi, "lcb_only", ucb, lcb):
            return pi, 1
        return self._random_partition(), 0


# ============================================================
# Baseline 5: mixed (LCB-pessimistic + UCB-optimistic fallback)
# ============================================================


class Mixed(BaseAlgorithm):
    """First try pessimistic; if that's not certified stable, try optimistic;
    if neither is certified stable, declare non-existence and play random."""

    name = "Mixed"

    def act(self, t: int) -> Tuple[Partition, int]:
        ucb, lcb = self._bounds(t)
        pi_pess = self._solve("pessimistic", ucb, lcb)
        if self._stable(pi_pess, "pessimistic", ucb, lcb):
            return pi_pess, 1
        pi_opt = self._solve("optimistic", ucb, lcb)
        if self._stable(pi_opt, "optimistic", ucb, lcb):
            return pi_opt, 1
        return self._random_partition(), 0


# ============================================================
# Baseline 0: random
# ============================================================


class RandomBaseline(BaseAlgorithm):
    name = "Random"

    def act(self, t: int) -> Tuple[Partition, int]:
        return self._random_partition(), 0


# ============================================================
# Registry
# ============================================================


ALGORITHMS = {
    "ucb_lcb_gs": UCBLCBGS,
    "pessimistic": PessimisticOnly,
    "optimistic": OptimisticOnly,
    "ucb_only": UCBOnly,
    "lcb_only": LCBOnly,
    "mixed": Mixed,
    "random": RandomBaseline,
}


__all__ = [
    "PairwiseEstimator",
    "BaseAlgorithm",
    "UCBLCBGS",
    "PessimisticOnly",
    "OptimisticOnly",
    "UCBOnly",
    "LCBOnly",
    "Mixed",
    "RandomBaseline",
    "ALGORITHMS",
]
