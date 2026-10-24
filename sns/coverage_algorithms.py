"""
sns.coverage_algorithms
=======================

UCB-LCB-GS with coverage probes. These variants are not reported in the
paper; the module is included because ``sns.candidate_algorithms`` builds on
its classes and registry.

Identifiers
-----------
    ucb_lcb_gs_original       the original UCB-LCB-GS implementation
                              (core/algorithms.UCBLCBGS)
    ucb_lcb_gs_coverage       coverage probes, probe period L = 2
    ucb_lcb_gs_coverage_only  probe period L = 1: every uncertified round is
                              a coverage probe

Round t:

    if some pair is unobserved:      uncertified (the pessimistic gap is -inf)
    else:
        pi_pess <- SOLVE(LCB, UCB)                     [heuristic oracle]
        if Gap^pess_k(pi_pess) >= eps:  d_t = 1, play pi_pess
                                        [exact or restricted checker, as installed]
    uncertified:
        d_t = 0;  j <- j + 1
        if some pair is unobserved or j = 1 (mod L):  play the grand coalition
                                                      (coverage probe)
        else:                                         play SOLVE(UCB, LCB)
                                                      (optimistic round)

Under pairwise semi-bandit feedback the coverage cycle is the grand
coalition alone (r = 1): one probe observes every pair, so before round t
every count is at least the number of probes played before t.

The exact maximisation over all partitions is replaced by the shared
heuristic SOLVE (ashg_lib.compute_partition), as for every other algorithm.
The pessimistic test is evaluated by whichever checker is bound to
``ashg_lib.gap_k``. The experiment scripts install the exact checker
(sns.full_deviation_checker) for every algorithm.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .paths import ensure_core_on_path

ensure_core_on_path()

from algorithms import ALGORITHMS, BaseAlgorithm, UCBLCBGS  # noqa: E402
from ashg_lib import Partition  # noqa: E402


class UCBLCBGSCoverage(BaseAlgorithm):
    """UCB-LCB-GS with coverage probes, probe period L (default 2)."""

    name = "UCB-LCB-GS (coverage, L=2)"
    probe_period = 2

    def __init__(self, **kw):
        super().__init__(**kw)
        self.L = int(self.probe_period)
        if self.L < 1:
            raise ValueError("probe period L must be >= 1")
        self._grand = Partition(np.zeros(self.cfg.n, dtype=np.int64))
        # Counters, recorded for the experiments.
        self.uncertified_rounds = 0     # j in the pseudocode
        self.n_probes = 0               # p in the pseudocode
        self.n_optimistic = 0
        self.n_certified = 0
        self.first_certified_round = -1

    # A pair (or ordered pair, for the directed estimator) is observed iff
    # its count is positive. The diagonal is ignored.
    def _coverage_complete(self) -> bool:
        T = self.estimator.T
        off = ~np.eye(self.cfg.n, dtype=bool)
        return bool(np.all(T[off] > 0))

    def act(self, t: int) -> Tuple[Partition, int]:
        covered = self._coverage_complete()
        if covered:
            ucb, lcb = self._bounds(t)
            pi_pess = self._solve("pessimistic", ucb, lcb)
            if self._stable(pi_pess, "pessimistic", ucb, lcb):
                self.n_certified += 1
                if self.first_certified_round < 0:
                    self.first_certified_round = t
                return pi_pess, 1
        # uncertified round
        self.uncertified_rounds += 1
        j = self.uncertified_rounds
        if (not covered) or ((j - 1) % self.L == 0):
            self.n_probes += 1
            return self._grand, 0
        self.n_optimistic += 1
        ucb, lcb = self._bounds(t)
        return self._solve("optimistic", ucb, lcb), 0

    def counters(self) -> dict:
        return {
            "n_probes": self.n_probes,
            "n_optimistic": self.n_optimistic,
            "n_certified": self.n_certified,
            "first_certified_round": self.first_certified_round,
            "uncertified_rounds": self.uncertified_rounds,
        }


class UCBLCBGSCoverageOnly(UCBLCBGSCoverage):
    """Coverage-only ablation: probe period L = 1."""

    name = "UCB-LCB-GS (coverage only, L=1)"
    probe_period = 1


class UCBLCBGSOriginal(UCBLCBGS):
    """The paper's UCB-LCB-GS, unchanged, with counters recorded externally."""

    name = "UCB-LCB-GS (without coverage)"


EXTENDED_ALGORITHMS = dict(ALGORITHMS)
EXTENDED_ALGORITHMS.update({
    "ucb_lcb_gs_original": UCBLCBGSOriginal,
    "ucb_lcb_gs_coverage": UCBLCBGSCoverage,
    "ucb_lcb_gs_coverage_only": UCBLCBGSCoverageOnly,
})

__all__ = [
    "UCBLCBGSCoverage",
    "UCBLCBGSCoverageOnly",
    "UCBLCBGSOriginal",
    "EXTENDED_ALGORITHMS",
]
