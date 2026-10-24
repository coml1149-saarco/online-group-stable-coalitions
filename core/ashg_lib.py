"""
ashg_lib.py
===========

Core library for additively separable hedonic games (ASHGs) with
symmetric utilities.  Provides:

  * ``ASHGEnvironment``     - random ASHG with noisy feedback.
  * ``Partition``           - lightweight partition representation.
  * ``compute_partition``   - heuristic optimizer for the max-min
                              stability problem (Eqs. (7) / (9) of the paper).
  * ``is_stable`` / ``gap_k`` - shared (epsilon, k)-strong-Nash deviation
                                checker.

Conventions
-----------
A coalition ``C`` deviates from partition ``pi`` to partition ``pi'``.
We use the gap convention

    gap_C(pi, pi', v_curr, v_dev)
        = max_{i in C} [ u_i(pi, v_curr) - u_i(pi', v_dev) ]

i.e. the largest *loss* incurred by any member of ``C`` from the
deviation.  A partition is (epsilon, k)-strong-Nash stable iff

    gap_k(pi, v_curr, v_dev) := min_{|C|<=k, pi'} gap_C(...)   >= epsilon

so the per-round (epsilon, k)-regret is  max(0, epsilon - gap_k).
The four "estimate types" used by the algorithms are mapped to
(v_curr, v_dev) as follows:

    pessimistic : v_curr = LCB,  v_dev = UCB     # safe certificate
    optimistic  : v_curr = UCB,  v_dev = LCB     # exploration
    ucb_only    : v_curr = UCB,  v_dev = UCB
    lcb_only    : v_curr = LCB,  v_dev = LCB
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from typing import List, Sequence, Tuple

import numpy as np


# ============================================================
# Partition representation
# ============================================================


class Partition:
    """A partition of {0, ..., n-1} stored as a vector of coalition ids."""

    __slots__ = ("cid",)

    def __init__(self, cid: Sequence[int]):
        self.cid = np.asarray(cid, dtype=np.int64).copy()
        self._normalize()

    # ---- internal helpers -------------------------------------------------
    def _normalize(self) -> None:
        """Relabel ids so the partition has ids 0..k-1 in order of appearance."""
        seen: dict = {}
        out = np.empty_like(self.cid)
        nxt = 0
        for idx, c in enumerate(self.cid):
            c = int(c)
            if c not in seen:
                seen[c] = nxt
                nxt += 1
            out[idx] = seen[c]
        self.cid = out

    # ---- public API -------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.cid.shape[0])

    @property
    def num_coalitions(self) -> int:
        return int(self.cid.max()) + 1 if self.n else 0

    def coalitions(self) -> List[List[int]]:
        groups: dict = defaultdict(list)
        for agent, c in enumerate(self.cid):
            groups[int(c)].append(agent)
        return [groups[k] for k in sorted(groups)]

    def coalition_of(self, i: int) -> List[int]:
        c = self.cid[i]
        return [j for j in range(self.n) if self.cid[j] == c and j != i]

    def copy(self) -> "Partition":
        return Partition(self.cid)

    def with_moved(self, agent: int, new_cid: int) -> "Partition":
        new = self.cid.copy()
        new[agent] = new_cid
        return Partition(new)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        return isinstance(other, Partition) and np.array_equal(self.cid, other.cid)

    def __repr__(self) -> str:
        return f"Partition({self.coalitions()})"


def random_partition(n: int, max_coalitions: int, rng: np.random.Generator) -> Partition:
    """Assign each agent to one of ``max_coalitions`` coalitions uniformly."""
    max_coalitions = max(1, int(max_coalitions))
    cid = rng.integers(0, max_coalitions, size=n)
    return Partition(cid)


# ============================================================
# Utility / welfare helpers
# ============================================================


def agent_utility(i: int, partition: Partition, v: np.ndarray) -> float:
    return float(sum(v[i, j] for j in partition.coalition_of(i)))


def total_welfare(partition: Partition, v: np.ndarray) -> float:
    """
    Sum_i u_i(pi, v), i.e. the sum over *ordered* same-coalition pairs.

    Written as sum_{i<j} (v[i,j] + v[j,i]) so that it is correct for
    asymmetric ``v`` as well.  For symmetric ``v`` this equals the previous
    ``2 * sum_{i<j} v[i,j]`` exactly, so behaviour on the paper's symmetric
    experiments is unchanged.
    """
    cid = partition.cid
    n = partition.n
    total = 0.0
    for i in range(n):
        ci = cid[i]
        for j in range(i + 1, n):
            if cid[j] == ci:
                total += v[i, j] + v[j, i]
    return total


# ============================================================
# Environment
# ============================================================


class ASHGEnvironment:
    """ASHG with symmetric utilities and i.i.d. Gaussian noise per round."""

    def __init__(
        self,
        n: int,
        model: str = "uniform",
        noise_std: float = 1.0,
        seed: int | None = None,
    ):
        if model not in ("uniform", "gaussian"):
            raise ValueError(f"unknown model {model!r}")
        self.n = int(n)
        self.model = model
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.v = self._draw_utilities()

    def _draw_utilities(self) -> np.ndarray:
        n = self.n
        v = np.zeros((n, n), dtype=np.float64)
        if self.model == "uniform":
            for i in range(n):
                for j in range(i + 1, n):
                    val = float(self.rng.integers(-100, 101))
                    v[i, j] = val
                    v[j, i] = val
        else:  # gaussian
            for i in range(n):
                for j in range(i + 1, n):
                    mu = float(self.rng.integers(-100, 101))
                    val = float(round(self.rng.normal(mu, 10.0)))
                    v[i, j] = val
                    v[j, i] = val
        return v

    def observe(self, partition: Partition) -> dict:
        """Return ``{(i, j): noisy v_{i,j}}`` for every same-coalition pair."""
        cid = partition.cid
        out: dict = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if cid[i] == cid[j]:
                    eps = self.rng.normal(0.0, self.noise_std)
                    out[(i, j)] = self.v[i, j] + eps
        return out


# ============================================================
# Deviation enumeration & gap
# ============================================================


def _enumerate_deviations(
    partition: Partition,
    C: Tuple[int, ...],
    deviation_cap: int = 64,
) -> List[Partition]:
    """
    Enumerate a finite, well-defined set of deviations of coalition ``C``.

    Three families are considered:
      (a) the whole coalition C joins each existing coalition or forms a new
          one (a "block move"),
      (b) the whole coalition C breaks off into its own new coalition,
      (c) each member of C independently chooses an existing/new coalition,
          subject to a combinatorial cap (``deviation_cap``).
    """
    cid = partition.cid
    existing = list(range(int(cid.max()) + 1))
    new_id = (existing[-1] + 1) if existing else 0
    targets = existing + [new_id]
    out: List[Partition] = []

    # (a) block moves
    for t in targets:
        new_cid = cid.copy()
        for i in C:
            new_cid[i] = t
        cand = Partition(new_cid)
        if not np.array_equal(cand.cid, partition.cid):
            out.append(cand)

    # (b) C alone in a brand-new coalition (subsumed by (a) but harmless)

    # (c) per-agent independent re-assignment, capped
    if len(C) >= 2:
        n_combos = len(targets) ** len(C)
        if n_combos <= deviation_cap:
            for assign in product(targets, repeat=len(C)):
                new_cid = cid.copy()
                for idx, i in enumerate(C):
                    new_cid[i] = assign[idx]
                cand = Partition(new_cid)
                if not np.array_equal(cand.cid, partition.cid):
                    out.append(cand)

    # de-duplicate
    seen = set()
    uniq: List[Partition] = []
    for p in out:
        key = p.cid.tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def gap_C(
    C: Sequence[int],
    pi: Partition,
    pi_prime: Partition,
    v_curr: np.ndarray,
    v_dev: np.ndarray,
) -> float:
    """Maximum loss across members of C from deviating pi -> pi'."""
    worst_loss = -np.inf
    cid_old = pi.cid
    cid_new = pi_prime.cid
    n = pi.n
    for i in C:
        co_old = cid_old[i]
        co_new = cid_new[i]
        u_old = 0.0
        u_new = 0.0
        for j in range(n):
            if j == i:
                continue
            if cid_old[j] == co_old:
                u_old += v_curr[i, j]
            if cid_new[j] == co_new:
                u_new += v_dev[i, j]
        loss = u_old - u_new
        if loss > worst_loss:
            worst_loss = loss
    return float(worst_loss)


def gap_k(
    pi: Partition,
    v_curr: np.ndarray,
    v_dev: np.ndarray,
    k: int,
    deviation_cap: int = 64,
) -> float:
    """Worst-case (smallest) gap over all coalitions of size <= k."""
    n = pi.n
    best_min = np.inf
    for sz in range(1, k + 1):
        for C in combinations(range(n), sz):
            for pp in _enumerate_deviations(pi, C, deviation_cap=deviation_cap):
                g = gap_C(C, pi, pp, v_curr, v_dev)
                if g < best_min:
                    best_min = g
    return float(best_min)


def is_stable(
    pi: Partition,
    v_curr: np.ndarray,
    v_dev: np.ndarray,
    k: int,
    eps: float,
    deviation_cap: int = 64,
) -> bool:
    """Return ``True`` iff ``gap_k(pi, v_curr, v_dev) >= eps``."""
    return gap_k(pi, v_curr, v_dev, k, deviation_cap=deviation_cap) >= eps


# ============================================================
# Heuristic partition optimizer (shared by every algorithm)
# ============================================================


def _greedy_local_search(
    n: int,
    v_curr: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 60,
    init: np.ndarray | None = None,
) -> Partition:
    """Greedy hill-climb on welfare under ``v_curr``."""
    if init is None:
        max_c = int(rng.integers(1, n + 1))
        cid = rng.integers(0, max_c, size=n).astype(np.int64)
    else:
        cid = init.astype(np.int64).copy()

    partition = Partition(cid)
    cid = partition.cid  # normalised

    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        order = rng.permutation(n)
        for i in order:
            existing = list(range(int(cid.max()) + 1))
            new_id = existing[-1] + 1
            best_target = int(cid[i])
            cur_w = total_welfare(Partition(cid), v_curr)
            best_w = cur_w
            for t in existing + [new_id]:
                if t == cid[i]:
                    continue
                old = int(cid[i])
                cid[i] = t
                w = total_welfare(Partition(cid), v_curr)
                if w > best_w + 1e-9:
                    best_w = w
                    best_target = t
                cid[i] = old
            if best_target != int(cid[i]):
                cid[i] = best_target
                improved = True
    return Partition(cid)


_ESTIMATE_TYPES = ("pessimistic", "optimistic", "ucb_only", "lcb_only")


def _select_v(
    estimate_type: str, ucb: np.ndarray, lcb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Map an estimate type to (v_curr, v_dev)."""
    if estimate_type == "pessimistic":
        return lcb, ucb
    if estimate_type == "optimistic":
        return ucb, lcb
    if estimate_type == "ucb_only":
        return ucb, ucb
    if estimate_type == "lcb_only":
        return lcb, lcb
    raise ValueError(f"unknown estimate_type {estimate_type!r}")


def compute_partition(
    estimate_type: str,
    ucb: np.ndarray,
    lcb: np.ndarray,
    k: int,
    eps: float,
    rng: np.random.Generator,
    n_restarts: int = 3,
    max_iter: int = 60,
    deviation_cap: int = 64,
) -> Partition:
    """
    Heuristic for

        argmax_pi  min_{|C|<=k, pi'}  gap_C(pi, pi', v_curr, v_dev)

    where (v_curr, v_dev) are picked by ``estimate_type``.  We use repeated
    greedy local search on welfare under ``v_curr`` (which correlates with
    high gap_k), then *score* each candidate by the actual approximated
    gap_k and return the best one.  This is the same routine for every
    algorithm in the experiment, ensuring fair comparison.
    """
    v_curr, v_dev = _select_v(estimate_type, ucb, lcb)
    n = ucb.shape[0]

    candidates: List[Partition] = []
    for _ in range(max(1, n_restarts)):
        candidates.append(_greedy_local_search(n, v_curr, rng, max_iter=max_iter))

    # Always include the all-singletons and all-grand-coalition partitions.
    candidates.append(Partition(np.arange(n, dtype=np.int64)))
    candidates.append(Partition(np.zeros(n, dtype=np.int64)))

    best_p = candidates[0]
    best_score = -np.inf
    for p in candidates:
        # Primary score: approximated min-gap (we *want* this large).
        g = gap_k(p, v_curr, v_dev, k, deviation_cap=deviation_cap)
        # Secondary tiebreak: welfare under v_curr.
        w = total_welfare(p, v_curr) * 1e-6
        score = g + w
        if score > best_score:
            best_score = score
            best_p = p
    return best_p


__all__ = [
    "Partition",
    "random_partition",
    "agent_utility",
    "total_welfare",
    "ASHGEnvironment",
    "gap_C",
    "gap_k",
    "is_stable",
    "compute_partition",
]
