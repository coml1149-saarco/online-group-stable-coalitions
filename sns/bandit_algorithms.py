"""
sns.bandit_algorithms
=====================

Per-agent coalition-level bandit feedback (Section 4 of the paper): in every
round each agent i reports

    R_i = sum_{j in pi(i) \\ {i}} v_ij + eta_i ,

one noisy aggregate per agent. Nothing else is observed.

Common estimator. Every algorithm here updates the shared PairwiseEstimator
only from agents in two-agent coalitions: if pi(i) = {i, j}, R_i is an
observation of v_ij. Reports of singletons (pure noise) and of agents in
coalitions of three or more are discarded. ``BanditFeedback.observe`` returns
exactly these observations, so no algorithm can receive a pairwise
observation that this feedback model does not provide. With symmetric
utilities (mirrored estimator) a two-agent coalition yields two observations
of the pair (q = 2); with asymmetric utilities (ordered estimator) one
observation of each ordered pair (q = 1).

Algorithms (identifiers used in the experiments):

  bandit_witness        the bandit variant of UCB-LCB-GS evaluated in the
                        paper: witness probing with pair probes
  bandit_rr_L1          round-robin matching coverage, probe period L = 1
                        (not reported in the paper)
  bandit_rr_L2          round-robin matching coverage, probe period L = 2
                        (not reported in the paper)
  pessimistic, optimistic, ucb_only, lcb_only, mixed
                        the semi-bandit baselines' decision rules, unchanged,
                        preceded by one round-robin matching cycle (the minimal
                        initialisation without which no pairwise estimate
                        exists under this feedback model; d_t = 0 during it)
  random                the semi-bandit Random baseline, unchanged: uniformly
                        random partitions, never declares (its decisions do not
                        depend on feedback)
  shared_<baseline>     a baseline that additionally plays the next matching of
                        the cycle on every second round on which it does not
                        declare (shared-exploration control experiment)

The modules in ``core/`` and the semi-bandit modules are not modified.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .paths import ensure_core_on_path

ensure_core_on_path()

from algorithms import (  # noqa: E402
    BaseAlgorithm, LCBOnly, Mixed, OptimisticOnly, PessimisticOnly, RandomBaseline, UCBOnly,
)
from ashg_lib import Partition  # noqa: E402

from .full_deviation_checker import full_gap_k_argmin  # noqa: E402


# ---------------------------------------------------------------- feedback


class BanditFeedback:
    """Wraps an environment (true utilities ``v`` and noise stream ``rng``) and
    replaces its feedback by per-agent aggregate reports. One noise draw per
    agent per round, in agent order, so the noise stream does not depend on
    the partition played."""

    def __init__(self, env, noise_model: str = "gaussian"):
        self.env = env
        self.v = env.v
        self.n = env.n
        self.noise_model = noise_model

    def _noise(self) -> float:
        if hasattr(self.env, "_noise"):
            return self.env._noise()
        return float(self.env.rng.normal(0.0, self.env.noise_std))

    def reports(self, partition: Partition) -> np.ndarray:
        cid = partition.cid
        R = np.empty(self.n)
        for i in range(self.n):
            R[i] = sum(self.v[i, j] for j in range(self.n) if j != i and cid[j] == cid[i]) + self._noise()
        return R

    def observe(self, partition: Partition) -> Dict[Tuple[int, int], float]:
        R = self.reports(partition)
        cid = partition.cid
        out: Dict[Tuple[int, int], float] = {}
        for i in range(self.n):
            mates = [j for j in range(self.n) if j != i and cid[j] == cid[i]]
            if len(mates) == 1:
                out[(i, mates[0])] = float(R[i])
        return out


def round_robin(n: int) -> List[Partition]:
    """Round-robin (circle-method) near-one-factorisation of K_n: r = n - 1
    matchings for even n, r = n for odd n; every unordered pair lies in
    exactly one matching; unmatched agents are singletons."""
    m = n if n % 2 == 0 else n + 1                     # a dummy agent for odd n
    others = list(range(1, m))
    out = []
    for _ in range(m - 1):
        ring = [0] + others
        cid = np.arange(n, dtype=np.int64) + n          # singletons by default
        for a in range(m // 2):
            i, j = ring[a], ring[m - 1 - a]
            if i < n and j < n:
                cid[i] = cid[j] = a
        out.append(Partition(cid))
        others = [others[-1]] + others[:-1]
    return out


# ---------------------------------------------------------------- mixins


class _BanditCore:
    """Matching cycle, coverage test and counters shared by all bandit arms."""

    def _bandit_init(self):
        self._cycle = round_robin(self.cfg.n)
        self._cpos = 0                                  # position in the cycle (never reset)
        self.n_init = 0
        self.n_probes = 0
        self.n_optimistic = 0
        self.n_certified = 0
        self.first_certified_round = -1
        self.uncertified_rounds = 0

    def _covered(self) -> bool:
        T = self.estimator.T
        return bool(np.all(T[~np.eye(self.cfg.n, dtype=bool)] > 0))

    def _next_matching(self) -> Partition:
        p = self._cycle[self._cpos % len(self._cycle)]
        self._cpos += 1
        return p

    def counters(self) -> dict:
        return {"n_init": self.n_init, "n_probes": self.n_probes, "n_optimistic": self.n_optimistic,
                "n_certified": self.n_certified, "first_certified_round": self.first_certified_round,
                "uncertified_rounds": self.uncertified_rounds, "matching_rounds": self._cpos,
                "completed_cycles": self._cpos // len(self._cycle)}


def _with_init(base):
    """A semi-bandit baseline preceded by one matching cycle (own-policy comparison)."""

    class B(_BanditCore, base):
        name = base.name + " (bandit)"

        def __init__(self, **kw):
            super().__init__(**kw)
            self._bandit_init()

        def act(self, t):
            if not self._covered():
                self.n_init += 1
                return self._next_matching(), 0
            pi, d = base.act(self, t)
            if d == 1:
                self.n_certified += 1
                if self.first_certified_round < 0:
                    self.first_certified_round = t
            return pi, d

    B.__name__ = "Bandit" + base.__name__
    return B


def _with_shared(base):
    """Baseline with the shared exploration schedule: on every second round on
    which it does not declare, the next matching of the cycle replaces its play."""

    class S(_BanditCore, base):
        name = base.name + " (bandit, shared exploration)"

        def __init__(self, **kw):
            super().__init__(**kw)
            self._bandit_init()

        def act(self, t):
            if not self._covered():
                self.n_init += 1
                return self._next_matching(), 0
            pi, d = base.act(self, t)
            if d == 1:
                self.n_certified += 1
                if self.first_certified_round < 0:
                    self.first_certified_round = t
                return pi, 1
            self.uncertified_rounds += 1
            if (self.uncertified_rounds - 1) % 2 == 0:
                self.n_probes += 1
                return self._next_matching(), 0
            return pi, 0

    S.__name__ = "BanditShared" + base.__name__
    return S


# ---------------------------------------------------------------- proposed algorithms


class BanditRoundRobin(_BanditCore, BaseAlgorithm):
    """Coverage probes = the next matching of the round-robin cycle, on every
    L-th uncertified round ([C] Algorithm 2 with the cycle of Section 10)."""

    name = "UCB-LCB-GS (round-robin matching coverage)"
    probe_period = 1

    def __init__(self, **kw):
        super().__init__(**kw)
        self._bandit_init()

    def act(self, t):
        covered = self._covered()
        if covered:
            ucb, lcb = self._bounds(t)
            pi_pess = self._solve("pessimistic", ucb, lcb)
            if self._stable(pi_pess, "pessimistic", ucb, lcb):
                self.n_certified += 1
                if self.first_certified_round < 0:
                    self.first_certified_round = t
                return pi_pess, 1
        self.uncertified_rounds += 1
        j = self.uncertified_rounds
        if not covered:
            self.n_init += 1
            return self._next_matching(), 0
        if (j - 1) % self.probe_period == 0:
            self.n_probes += 1
            return self._next_matching(), 0
        self.n_optimistic += 1
        ucb, lcb = self._bounds(t)
        return self._solve("optimistic", ucb, lcb), 0


class BanditRoundRobinL2(BanditRoundRobin):
    name = "UCB-LCB-GS (round-robin matching coverage, L=2)"
    probe_period = 2


class BanditWitness(_BanditCore, BaseAlgorithm):
    """Witness probing under coalition-level feedback: the probe isolates the
    witness pair as a two-agent coalition."""

    name = "UCB-LCB-GS with witness probing (bandit feedback)"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._bandit_init()

    def act(self, t):
        if not self._covered():
            self.n_init += 1
            self.uncertified_rounds += 1
            return self._next_matching(), 0
        ucb, lcb = self._bounds(t)
        pi_pess = self._solve("pessimistic", ucb, lcb)
        if self._stable(pi_pess, "pessimistic", ucb, lcb):
            return self._cert(t, pi_pess)
        pi_opt = self._solve("optimistic", ucb, lcb)
        g, C, cid_dev = full_gap_k_argmin(pi_opt, lcb, ucb, self.cfg.k)
        if g >= self.cfg.eps:                              # safeguard: exact pessimistic test of pi_opt
            return self._cert(t, pi_opt)
        self.uncertified_rounds += 1
        rad = (ucb - lcb) / 2.0
        cid_opt = pi_opt.cid
        best, e_star = -np.inf, None
        for i in C:                                        # witness pairs, deterministic order
            for j in range(self.cfg.n):
                if j != i and (cid_opt[j] == cid_opt[i] or cid_dev[j] == cid_dev[i]) and rad[i, j] > best:
                    best, e_star = rad[i, j], (i, j)
        i, j = e_star
        if cid_opt[i] == cid_opt[j] and int(np.sum(cid_opt == cid_opt[i])) == 2:
            self.n_optimistic += 1                         # pi_opt(i) = {i, j}: informative
            return pi_opt, 0
        self.n_probes += 1
        cid = cid_opt.copy()
        cid[i] = cid[j] = int(cid_opt.max()) + 1           # extract {i, j} as a new coalition
        return Partition(cid), 0

    def _cert(self, t, pi):
        self.n_certified += 1
        if self.first_certified_round < 0:
            self.first_certified_round = t
        return pi, 1


BASELINES = {"pessimistic": PessimisticOnly, "optimistic": OptimisticOnly, "ucb_only": UCBOnly,
             "lcb_only": LCBOnly, "mixed": Mixed}

BANDIT_ALGORITHMS = {
    "bandit_witness": BanditWitness,
    "bandit_rr_L1": BanditRoundRobin,
    "bandit_rr_L2": BanditRoundRobinL2,
    "random": RandomBaseline,
}
BANDIT_ALGORITHMS.update({k: _with_init(v) for k, v in BASELINES.items()})
BANDIT_ALGORITHMS.update({"shared_" + k: _with_shared(v) for k, v in BASELINES.items()})

__all__ = ["BanditFeedback", "round_robin", "BanditWitness", "BanditRoundRobin", "BanditRoundRobinL2",
           "BANDIT_ALGORITHMS"]
