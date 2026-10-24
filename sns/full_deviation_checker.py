"""
sns.full_deviation_checker
==========================

Exact group-deviation gap over the *complete* deviation family of
Definition 2 of the paper, for deviating coalitions of size at most three.

Deviation family (camera-ready definition):

    pi' in Lambda_C(pi)  iff  pi'|_{N \\ C} = pi|_{N \\ C}
                              and pi'(i) != pi(i) (as sets) for every i in C.

Each member of C either joins an existing coalition of pi (after the members
of C have left it) or a new coalition. Several members may share a new
coalition, and different members may form different new coalitions.

This module is a drop-in, call-compatible replacement for
``ashg_lib.gap_k(pi, v_curr, v_dev, k, deviation_cap=...)``. The
``deviation_cap`` argument is accepted for compatibility and ignored. The
implementation's original checker (``ashg_lib.gap_k``) is left untouched and
stays available for comparisons.

Where the original checker is exact
-----------------------------------
For k <= 2 and partitions with at most 7 coalitions (so (K+1)^2 <= 64), the
original checker returns the same value. It can certify an unstable partition when K >= 8, and when k = 3
and K >= 4.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, List, Sequence, Tuple

import numpy as np

MAX_K = 3


def _blocks(cid: np.ndarray) -> List[List[int]]:
    out: dict = {}
    for a, c in enumerate(cid):
        out.setdefault(int(c), []).append(a)
    return [out[c] for c in sorted(out)]


def _assignments(s: int, n_existing: int) -> Iterator[Tuple[int, ...]]:
    """
    All assignments of s ordered deviators to targets.

    Targets 0..n_existing-1 are the remaining existing coalitions. Targets
    n_existing, n_existing+1, ... are new coalitions, labelled in order of
    first use (restricted growth), so each distinct grouping into new
    coalitions is produced exactly once.
    """
    def rec(idx: int, cur: List[int], next_new: int):
        if idx == s:
            yield tuple(cur)
            return
        for tgt in range(n_existing):
            cur.append(tgt)
            yield from rec(idx + 1, cur, next_new)
            cur.pop()
        for tgt in range(n_existing, next_new + 1):
            cur.append(tgt)
            yield from rec(idx + 1, cur, max(next_new, tgt + 1))
            cur.pop()

    yield from rec(0, [], n_existing)


def full_gap_C_min(
    cid: np.ndarray,
    C: Sequence[int],
    v_curr: np.ndarray,
    v_dev: np.ndarray,
    blocks: List[List[int]],
    own: List[frozenset],
    u_curr: np.ndarray,
) -> float:
    """min over pi' in Lambda_C(pi) of max_{i in C}[u_i(pi(i),v_curr) - u_i(pi'(i),v_dev)]."""
    Cset = set(C)
    remaining = []
    for b in blocks:
        r = [a for a in b if a not in Cset]
        if r:
            remaining.append(r)
    R = len(remaining)
    # Utility of each deviator from each remaining coalition (excluding C) under v_dev.
    base = np.array([[sum(v_dev[i, l] for l in rem) for rem in remaining] for i in C]) if R else None
    best = np.inf
    s = len(C)
    for assign in _assignments(s, R):
        worst_loss = -np.inf
        ok = True
        for idx, i in enumerate(C):
            tgt = assign[idx]
            co = [C[m] for m in range(s) if assign[m] == tgt and m != idx]
            if tgt < R:
                new_set = frozenset(remaining[tgt]) | frozenset(co) | {i}
                u_new = base[idx, tgt] + sum(v_dev[i, m] for m in co)
            else:
                new_set = frozenset(co) | {i}
                u_new = sum(v_dev[i, m] for m in co)
            if new_set == own[i]:
                ok = False  # i does not change coalition: not a deviation of C
                break
            loss = u_curr[i] - u_new
            if loss > worst_loss:
                worst_loss = loss
        if ok and worst_loss < best:
            best = worst_loss
    return float(best)


def full_gap_k(pi, v_curr: np.ndarray, v_dev: np.ndarray, k: int, deviation_cap=None) -> float:
    """Exact Gap_k over Lambda_C for all |C| <= k <= 3 (call-compatible with ashg_lib.gap_k)."""
    if k > MAX_K:
        raise ValueError(f"full_gap_k supports k <= {MAX_K}; got k={k}")
    cid = np.asarray(pi.cid if hasattr(pi, "cid") else pi, dtype=np.int64)
    n = cid.shape[0]
    blocks = _blocks(cid)
    own = [None] * n
    for b in blocks:
        fb = frozenset(b)
        for a in b:
            own[a] = fb
    u_curr = np.array([sum(v_curr[i, j] for j in own[i] if j != i) for i in range(n)], dtype=np.float64)
    best = np.inf
    for s in range(1, min(k, n) + 1):
        for C in combinations(range(n), s):
            g = full_gap_C_min(cid, C, v_curr, v_dev, blocks, own, u_curr)
            if g < best:
                best = g
    return float(best)


def full_gap_k_argmin(pi, v_curr: np.ndarray, v_dev: np.ndarray, k: int):
    """
    Exact Gap_k together with a minimising deviation.

    Returns (gap, C, cid_prime): C is the deviating coalition (tuple) and
    cid_prime the label vector of the deviation partition pi' in Lambda_C(pi).
    Ties are broken by the enumeration order: coalitions by size, then
    lexicographically; assignments in restricted-growth order. That order is
    deterministic, so the result is reproducible.
    """
    if k > MAX_K:
        raise ValueError(f"full_gap_k_argmin supports k <= {MAX_K}; got k={k}")
    cid = np.asarray(pi.cid if hasattr(pi, "cid") else pi, dtype=np.int64)
    n = cid.shape[0]
    blocks = _blocks(cid)
    own = [None] * n
    for b in blocks:
        fb = frozenset(b)
        for a in b:
            own[a] = fb
    u_curr = np.array([sum(v_curr[i, j] for j in own[i] if j != i) for i in range(n)], dtype=np.float64)
    best, best_C, best_cid = np.inf, None, None
    for s in range(1, min(k, n) + 1):
        for C in combinations(range(n), s):
            Cset = set(C)
            remaining, rem_label = [], []
            for b in blocks:
                r = [a for a in b if a not in Cset]
                if r:
                    remaining.append(r)
                    rem_label.append(int(cid[r[0]]))
            R = len(remaining)
            for assign in _assignments(s, R):
                worst = -np.inf
                ok = True
                for idx, i in enumerate(C):
                    tgt = assign[idx]
                    co = [C[m] for m in range(s) if assign[m] == tgt and m != idx]
                    if tgt < R:
                        new_set = frozenset(remaining[tgt]) | frozenset(co) | {i}
                        u_new = sum(v_dev[i, l] for l in remaining[tgt]) + sum(v_dev[i, m] for m in co)
                    else:
                        new_set = frozenset(co) | {i}
                        u_new = sum(v_dev[i, m] for m in co)
                    if new_set == own[i]:
                        ok = False
                        break
                    worst = max(worst, u_curr[i] - u_new)
                if ok and worst < best:
                    new_cid = cid.copy()
                    fresh = int(cid.max()) + 1
                    for idx, i in enumerate(C):
                        tgt = assign[idx]
                        new_cid[i] = rem_label[tgt] if tgt < R else fresh + (tgt - R)
                    best, best_C, best_cid = worst, C, new_cid
    return float(best), best_C, best_cid


def full_is_stable(pi, v_curr, v_dev, k: int, eps: float, deviation_cap=None) -> bool:
    return full_gap_k(pi, v_curr, v_dev, k) >= eps


def all_partitions(n: int) -> List[Tuple[int, ...]]:
    """Restricted growth strings of length n (one per set partition)."""
    out: List[Tuple[int, ...]] = []

    def rec(i, cur, m):
        if i == n:
            out.append(tuple(cur))
            return
        for c in range(m + 1):
            cur.append(c)
            rec(i + 1, cur, max(m, c + 1))
            cur.pop()

    rec(0, [], 0)
    return out


def exact_existence(v: np.ndarray, k: int):
    """
    Exhaustive evaluation: returns (max_gap, argmax_partition), where
    max_gap = max_pi Gap_k(pi, v) over all partitions (exact).

    Label for threshold eps (the paper's metric definitions):
        'YES' if max_gap >= eps          (SNS_{eps,k} non-empty)
        'NO'  if max_gap <  0            (SNS_{0,k} empty)
        'Z'   otherwise                  (intermediate region)
    """
    n = v.shape[0]
    best, arg = -np.inf, None
    for p in all_partitions(n):
        g = full_gap_k(np.array(p), v, v, k)
        if g > best:
            best, arg = g, p
    return float(best), arg


def label(max_gap: float, eps: float) -> str:
    if max_gap >= eps:
        return "YES"
    if max_gap < 0:
        return "NO"
    return "Z"


__all__ = ["full_gap_k", "full_gap_k_argmin", "full_is_stable", "exact_existence", "all_partitions", "label", "MAX_K"]
