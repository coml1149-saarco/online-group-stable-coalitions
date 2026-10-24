"""
sns.candidate_algorithms
========================

UCB-LCB-GS with witness probing (Algorithm 1 of the paper) and two
exploration variants. The algorithms in ``core/`` are not modified.

    ucb_lcb_gs_witness_dev    witness probing; a probe plays the binding deviation
                              (the UCB-LCB-GS evaluated in the paper)
    ucb_lcb_gs_witness_grand  witness probing; a probe plays the grand coalition
                              (not reported in the paper)
    ucb_lcb_gs_coverage_sqrt  coverage probes at uncertified rounds j = 1, 4, 9, ...
                              (not reported in the paper)

Witness probing (one round, after full coverage):

    pi_pess <- SOLVE(LCB, UCB); if Gap^pess_k(pi_pess) >= eps: declare, play it
    pi_opt  <- SOLVE(UCB, LCB)
    (C°, pi°) <- exact pessimistic minimiser of the deviations of pi_opt
    if Gap^pess_k(pi_opt) >= eps: declare, play pi_opt
        [implementation safeguard: possible only when the heuristic missed
         the pessimistic optimum]
    S <- {(i, j): i in C°, j in (pi_opt(i) U pi°(i)) \\ {i}}   (the witness pairs)
    e* = (i, j) <- the witness pair with the largest confidence radius
    if j in pi_opt(i):  play pi_opt                  (informative optimistic round)
    else:               play pi°  [dev]  or  N  [grand]   (probe)

For the exact-oracle version, every uncertified round on the confidence event
observes a pair whose radius exceeds Delta*/kappa_n, so the number of
uncertified rounds is logarithmic (Theorems 3.2 and 3.3 of the paper).
"""

from __future__ import annotations

from math import isqrt
from typing import Tuple

import numpy as np

from .paths import ensure_core_on_path

ensure_core_on_path()

from algorithms import BaseAlgorithm  # noqa: E402
from ashg_lib import Partition  # noqa: E402

from .coverage_algorithms import EXTENDED_ALGORITHMS  # noqa: E402
from .full_deviation_checker import full_gap_k_argmin  # noqa: E402


class _CounterMixin:
    def _init_counters(self):
        self._grand = Partition(np.zeros(self.cfg.n, dtype=np.int64))
        self.uncertified_rounds = 0
        self.n_probes = 0
        self.n_optimistic = 0
        self.n_certified = 0
        self.first_certified_round = -1

    def _coverage_complete(self) -> bool:
        T = self.estimator.T
        off = ~np.eye(self.cfg.n, dtype=bool)
        return bool(np.all(T[off] > 0))

    def _certified(self, t, pi):
        self.n_certified += 1
        if self.first_certified_round < 0:
            self.first_certified_round = t
        return pi, 1

    def counters(self) -> dict:
        return {"n_probes": self.n_probes, "n_optimistic": self.n_optimistic,
                "n_certified": self.n_certified, "first_certified_round": self.first_certified_round,
                "uncertified_rounds": self.uncertified_rounds}


class UCBLCBGSWitness(_CounterMixin, BaseAlgorithm):
    """Witness probing; ``probe_kind`` is 'dev' (binding deviation) or 'grand'."""

    name = "UCB-LCB-GS (witness, deviation probe)"
    probe_kind = "dev"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._init_counters()

    def act(self, t: int) -> Tuple[Partition, int]:
        if not self._coverage_complete():
            self.uncertified_rounds += 1
            self.n_probes += 1
            return self._grand, 0
        ucb, lcb = self._bounds(t)
        pi_pess = self._solve("pessimistic", ucb, lcb)
        if self._stable(pi_pess, "pessimistic", ucb, lcb):
            return self._certified(t, pi_pess)
        pi_opt = self._solve("optimistic", ucb, lcb)
        g, C, cid_dev = full_gap_k_argmin(pi_opt, lcb, ucb, self.cfg.k)
        if g >= self.cfg.eps:                      # safeguard (heuristic missed the pessimistic optimum)
            return self._certified(t, pi_opt)
        self.uncertified_rounds += 1
        rad = (ucb - lcb) / 2.0
        cid_opt = pi_opt.cid
        best, e_star = -np.inf, None
        for i in C:                                # witness pairs, deterministic order
            for j in range(self.cfg.n):
                if j == i:
                    continue
                if cid_opt[j] == cid_opt[i] or cid_dev[j] == cid_dev[i]:
                    if rad[i, j] > best:
                        best, e_star = rad[i, j], (i, j)
        i, j = e_star
        if cid_opt[j] == cid_opt[i]:
            self.n_optimistic += 1
            return pi_opt, 0
        self.n_probes += 1
        if self.probe_kind == "dev":
            return Partition(cid_dev), 0
        return self._grand, 0


class UCBLCBGSWitnessGrand(UCBLCBGSWitness):
    name = "UCB-LCB-GS (witness, grand-coalition probe)"
    probe_kind = "grand"


class UCBLCBGSCoverageSqrt(_CounterMixin, BaseAlgorithm):
    """Coverage probes at uncertified indices j that are perfect squares."""

    name = "UCB-LCB-GS (coverage at j = 1, 4, 9, ...)"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._init_counters()

    def act(self, t: int) -> Tuple[Partition, int]:
        covered = self._coverage_complete()
        if covered:
            ucb, lcb = self._bounds(t)
            pi_pess = self._solve("pessimistic", ucb, lcb)
            if self._stable(pi_pess, "pessimistic", ucb, lcb):
                return self._certified(t, pi_pess)
        self.uncertified_rounds += 1
        j = self.uncertified_rounds
        if (not covered) or isqrt(j) ** 2 == j:
            self.n_probes += 1
            return self._grand, 0
        self.n_optimistic += 1
        ucb, lcb = self._bounds(t)
        return self._solve("optimistic", ucb, lcb), 0


CANDIDATE_ALGORITHMS = dict(EXTENDED_ALGORITHMS)
CANDIDATE_ALGORITHMS.update({
    "ucb_lcb_gs_witness_dev": UCBLCBGSWitness,
    "ucb_lcb_gs_witness_grand": UCBLCBGSWitnessGrand,
    "ucb_lcb_gs_coverage_sqrt": UCBLCBGSCoverageSqrt,
})

__all__ = ["UCBLCBGSWitness", "UCBLCBGSWitnessGrand", "UCBLCBGSCoverageSqrt", "CANDIDATE_ALGORITHMS"]
