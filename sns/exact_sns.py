"""
sns.exact_sns -- exact, exhaustive (eps,k)-strong-Nash-stability existence test.

For every partition pi of N (restricted-growth strings) we compute Gap_k(pi, v)
exactly under three deviation families:

  "L"  (Lambda)  : Definition 2 / Bilo et al.  C with 1<=|C|<=k deviates to pi'
                   such that pi'|_{N\\C} == pi|_{N\\C} (as partitions of N\\C) and
                   pi'(i) != pi(i) (as SETS) for every i in C.  Deviators may join
                   any non-empty block of pi|_{N\\C} or form any number of new
                   coalitions among themselves.
  "Lp" (Lambda') : as Lambda, without the pi'(i) != pi(i) condition (only
                   pi' != pi is required).
  "D"            : core/ashg_lib.py::_enumerate_deviations (block moves of C to
                   an existing label or ONE fresh label; plus, if |C|>=2 and
                   (K+1)^|C| <= cap, every independent assignment of members to
                   {existing labels} U {one fresh label}); pi' == pi dropped.

u_i(C, v) = sum_{j in C, j != i} v[i, j]  (row i; valid for asymmetric v).
Gap_C = max_{i in C} [u_i(pi(i)) - u_i(pi'(i))];  Gap_k = min_C min_pi' Gap_C.

Implementation: all partitions are processed at once (vectorised in numpy), per
deviating set C.  For C we enumerate "label patterns": every member a of C picks
a label t in {0, ..., n+|C|-1}.  Label t carries the base set
S_t = {j notin C : cid[j] == t} (empty for labels >= K or for coalitions inside C).
The new coalition of member a is S_{t_a} U {b in C : t_b == t_a}.
  * Lambda'/Lambda: every pattern with labels <= K+|C|-1 (so up to |C| fresh,
    empty labels are available: all groupings of deviators into new coalitions
    and all choices of existing blocks are covered).
  * D: labels <= K (one fresh label); if (K+1)^|C| > cap only block patterns
    (all members on one label).
Unchanged member a: S_{t_a} == pi(i_a) \\ C and the same-label set of a in C
equals pi(i_a) n C.  pi' == pi  iff all members unchanged.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations, product
from typing import Dict, List

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(REPO, "core") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "core"))

BELL = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570]

_PART_CACHE: Dict[int, np.ndarray] = {}


def all_partitions(n: int) -> np.ndarray:
    """All restricted-growth strings of length n (B_n x n, int8)."""
    if n in _PART_CACHE:
        return _PART_CACHE[n]
    rows = [[0]]
    for _ in range(1, n):
        new = []
        for r in rows:
            m = max(r)
            for t in range(m + 2):
                new.append(r + [t])
        rows = new
    arr = np.array(rows, dtype=np.int8).reshape(len(rows), n)
    _PART_CACHE[n] = arr
    return arr


_PAT_CACHE: Dict[tuple, np.ndarray] = {}


def patterns(s: int, L: int) -> np.ndarray:
    key = (s, L)
    if key not in _PAT_CACHE:
        _PAT_CACHE[key] = np.array(list(product(range(L), repeat=s)), dtype=np.int16)
    return _PAT_CACHE[key]


def gap_all(v: np.ndarray, k: int, families=("L", "Lp", "D"), cap: int = 64,
            chunk: int = 20000, return_argmin: bool = False):
    """
    Return dict fam -> array (B_n,) of exact Gap_k(pi, v) for every partition pi
    (row order of all_partitions(n)).  If return_argmin, also fam+"_arg" ->
    (C, pattern-labels) achieving the min for each partition (slow path, small n).
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    P = all_partitions(n).astype(np.int64)
    Bn = P.shape[0]
    out = {f: np.full(Bn, np.inf) for f in families}
    arg = {f: [None] * Bn for f in families} if return_argmin else None
    for c0 in range(0, Bn, chunk):
        cid = P[c0:c0 + chunk]
        B = cid.shape[0]
        K = cid.max(axis=1) + 1  # number of coalitions
        same = cid[:, :, None] == cid[:, None, :]  # B x n x n
        U =(same * v[None]).sum(axis=2) - np.diag(v)[None, :]
        for s in range(1, k + 1):
            L = n + s
            pat = patterns(s, L)  # p x s
            npat = pat.shape[0]
            sameP = pat[:, :, None] == pat[:, None, :]  # p x s x s
            is_block = np.all(pat == pat[:, :1], axis=1)  # p
            maxlab = pat.max(axis=1)  # p
            for C in combinations(range(n), s):
                C = list(C)
                inC = np.zeros(n, bool)
                inC[C] = True
                nonC = np.where(~inC)[0]
                onehot = (cid[:, nonC, None] == np.arange(L)[None, None, :])  # B x (n-s) x L
                W = np.einsum("aj,bjt->bat", v[np.ix_(C, nonC)], onehot.astype(np.float64))  # B x s x L
                Ssize = onehot.sum(axis=1)  # B x L
                emptyS = Ssize == 0
                vC = v[np.ix_(C, C)]
                Q = np.einsum("pab,ab->pa", sameP, vC)  # p x s  (diag of vC is 0)
                cidC = cid[:, C]  # B x s
                samePi = cidC[:, :, None] == cidC[:, None, :]  # B x s x s
                restNE = Ssize[np.arange(B)[:, None], cidC] > 0  # B x s : pi(i)\C nonempty
                loss = np.empty((B, npat, s))
                unchanged = np.empty((B, npat, s), bool)
                for a in range(s):
                    ta = pat[:, a]  # p
                    loss[:, :, a] = U[:, C[a]][:, None] - W[:, a, :][:, ta] - Q[None, :, a]
                    condS = np.where(restNE[:, a][:, None],
                                     ta[None, :] == cidC[:, a][:, None],
                                     emptyS[:, ta])
                    condC = np.all(sameP[None, :, a, :] == samePi[:, None, a, :], axis=2)
                    unchanged[:, :, a] = condS & condC
                worst = loss.max(axis=2)  # B x p
                ident = unchanged.all(axis=2)
                anyun = unchanged.any(axis=2)
                for f in families:
                    if f in ("L", "Lp"):
                        valid = maxlab[None, :] <= (K[:, None] + s - 1)
                        valid = valid & ~ident
                        if f == "L":
                            valid = valid & ~anyun
                    elif f == "D":
                        valid = maxlab[None, :] <= K[:, None]
                        if s >= 2:
                            big = (K + 1) ** s > cap
                            valid = valid & (~big[:, None] | is_block[None, :])
                        valid = valid & ~ident
                    else:
                        raise ValueError(f)
                    g = np.where(valid, worst, np.inf)
                    gm = g.min(axis=1)
                    if return_argmin:
                        am = g.argmin(axis=1)
                        better = gm < out[f][c0:c0 + B]
                        for b in np.where(better)[0]:
                            arg[f][c0 + b] = (tuple(C), tuple(int(x) for x in pat[am[b]]))
                    np.minimum(out[f][c0:c0 + B], gm, out=out[f][c0:c0 + B])
    if return_argmin:
        for f in families:
            out[f + "_arg"] = arg[f]
    return out


def summarize(v, k, families=("L", "Lp", "D"), cap=64):
    """max_pi Gap_k and an argmax partition per family."""
    g = gap_all(v, k, families, cap)
    P = all_partitions(v.shape[0])
    res = {}
    for f in families:
        i = int(np.argmax(g[f]))
        res[f] = (float(g[f][i]), [int(x) for x in P[i]], g[f])
    return res


# ---------------------------------------------------------------------------
# Brute-force reference (definition applied literally) for validation
# ---------------------------------------------------------------------------

def _blocks(cid):
    d = {}
    for i, c in enumerate(cid):
        d.setdefault(int(c), set()).add(i)
    return d


def _coal(cid, i):
    return frozenset(j for j in range(len(cid)) if cid[j] == cid[i])


def _restrict(cid, keep):
    return frozenset(frozenset(j for j in b if j in keep) for b in _blocks(cid).values()
                     if any(j in keep for j in b))


def brute_gap(v, k, require_all_change=True):
    """Gap_k for every partition by testing every (pi, pi', C) triple literally."""
    n = v.shape[0]
    P = all_partitions(n)
    coals = [[_coal(p, i) for i in range(n)] for p in P]
    util = np.array([[sum(v[i, j] for j in coals[a][i] if j != i) for i in range(n)]
                     for a in range(len(P))])
    out = np.full(len(P), np.inf)
    Cs = [C for s in range(1, k + 1) for C in combinations(range(n), s)]
    restr = {}
    for a, p in enumerate(P):
        for C in Cs:
            keep = set(range(n)) - set(C)
            restr[(a, C)] = _restrict(p, keep)
    for a in range(len(P)):
        for b in range(len(P)):
            if a == b:
                continue
            for C in Cs:
                if restr[(a, C)] != restr[(b, C)]:
                    continue
                if require_all_change and any(coals[a][i] == coals[b][i] for i in C):
                    continue
                g = max(util[a, i] - util[b, i] for i in C)
                if g < out[a]:
                    out[a] = g
    return out
