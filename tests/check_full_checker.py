#!/usr/bin/env python3
"""
Validation of sns.full_deviation_checker (exact deviation checker, k <= 3).

    python3 tests/check_full_checker.py

Reference: brute force straight from Definition 2. For every partition pi and
every partition pi', a coalition C deviates from pi to pi' iff pi' restricted
to N\\C equals pi restricted to N\\C, and every i in C ends in a different
coalition (as a set). This shares no code with the checker under test.
"""

from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sns.full_deviation_checker import all_partitions, exact_existence, full_gap_k  # noqa: E402
from sns.paths import ensure_core_on_path  # noqa: E402

ensure_core_on_path()
from ashg_lib import Partition, gap_k as old_gap_k  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------- reference
def canon(labels):
    m, out = {}, []
    for c in labels:
        m.setdefault(c, len(m))
        out.append(m[c])
    return tuple(out)


def coal(p, i):
    return frozenset(a for a in range(len(p)) if p[a] == p[i])


def brute_gap(v_curr, v_dev, k, p):
    n = len(p)
    P = all_partitions(n)
    best = np.inf
    for s in range(1, k + 1):
        for C in combinations(range(n), s):
            rest = [a for a in range(n) if a not in C]
            rp = canon([p[a] for a in rest])
            for q in P:
                if canon([q[a] for a in rest]) != rp:
                    continue
                if any(coal(q, i) == coal(p, i) for i in C):
                    continue
                g = max(sum(v_curr[i, j] for j in coal(p, i) if j != i)
                        - sum(v_dev[i, j] for j in coal(q, i) if j != i) for i in C)
                best = min(best, g)
    return best


def rand_v(n, rng, sym=True, lo=-20, hi=21):
    v = rng.integers(lo, hi, (n, n)).astype(float)
    if sym:
        v = np.triu(v, 1)
        v = v + v.T
    np.fill_diagonal(v, 0.0)
    return v


rng = np.random.default_rng(2026)

# ------------------------------------------------ A. exhaustive brute force
print("\nA. exhaustive comparison with the brute-force definition")
for n, k, reps in [(3, 1, 6), (3, 2, 6), (3, 3, 6), (4, 1, 6), (4, 2, 8), (4, 3, 8), (5, 2, 4), (5, 3, 4)]:
    mism = tot = 0
    for r in range(reps):
        sym = r % 2 == 0
        vc = rand_v(n, rng, sym)
        # alternate: true-utility pairs (vc, vc) and pessimistic-style pairs (x <= y)
        vd = vc if r % 3 else vc + np.abs(rand_v(n, rng, sym, 0, 6))
        np.fill_diagonal(vd, 0.0)
        for p in all_partitions(n):
            a = full_gap_k(np.array(p), vc, vd, k)
            b = brute_gap(vc, vd, k, p)
            tot += 1
            mism += (a != b)
    check(f"n={n} k={k}: full checker == brute force", mism == 0, f"{tot} partitions, {mism} mismatches")

# ------------------------------------------------ B. targeted cases
print("\nB. targeted deviation types")


def V(n, entries, default=0.0, sym=True):
    v = np.full((n, n), default)
    np.fill_diagonal(v, 0.0)
    for (i, j), x in entries.items():
        v[i, j] = x
        if sym:
            v[j, i] = x
    return v


# unilateral: agent 0 alone, likes 1 -> gap = 0 - 5 = -5
v = V(3, {(0, 1): 5, (0, 2): -1, (1, 2): -1})
check("unilateral move to an existing coalition", full_gap_k(np.array([0, 1, 2]), v, v, 1) == -5)
# coordinated pair forming a new coalition that neither can reach alone
v = V(4, {(0, 1): 3, (0, 2): 2, (1, 3): 2, (2, 3): -20, (0, 3): -20, (1, 2): -20})
p = np.array([0, 1, 0, 1])  # {0,2},{1,3}: each gets 2; together {0,1}: each gets 3
check("pair deviation to a new joint coalition", full_gap_k(p, v, v, 2) == -1 and full_gap_k(p, v, v, 1) >= 0,
      f"k=2 {full_gap_k(p, v, v, 2)}, k=1 {full_gap_k(p, v, v, 1)}")
# deviators joining different existing coalitions (swap)
v = V(4, {(0, 1): 1, (2, 3): 1, (0, 3): 5, (1, 2): 5, (0, 2): -20, (1, 3): -20})
p = np.array([0, 0, 1, 1])  # swap 0<->2: 0 joins {3}, 2 joins {1}
check("swap between existing coalitions", full_gap_k(p, v, v, 2) == -4, f"gap {full_gap_k(p, v, v, 2)}")
# deviators splitting into two different new coalitions (k=3): {0,1,2} together;
# 0 and 1 leave to separate singletons is dominated; check split into a new pair + singleton
v = V(5, {(0, 1): 4, (0, 2): -6, (1, 2): -6, (2, 3): 1, (2, 4): 1, (3, 4): 9,
          (0, 3): -9, (0, 4): -9, (1, 3): -9, (1, 4): -9})
p = np.array([0, 0, 0, 1, 1])  # 0,1 in {0,1,2}: u=4-6=-2 ; new {0,1} gives 4 ; 2 alone gives 0 > -12
check("split into several new coalitions", full_gap_k(p, v, v, 3) <= -2, f"gap {full_gap_k(p, v, v, 3)}")
check("  ... agrees with brute force", full_gap_k(p, v, v, 3) == brute_gap(v, v, 3, tuple(p)))
# asymmetric utilities
v = V(2, {(0, 1): 3}, sym=False)
v[1, 0] = -1
check("asymmetric: grand coalition gap = min(3,-1)", full_gap_k(np.array([0, 0]), v, v, 1) == -1)
# destination coalition that empties out (it consisted only of deviators)
v = V(3, {(0, 1): 2, (0, 2): -5, (1, 2): -5})
p = np.array([0, 1, 2])
check("destination emptied by deviators", full_gap_k(p, v, v, 2) == brute_gap(v, v, 2, tuple(p)))

# ------------------------------------------------ C. known checker counterexample
print("\nC. the restricted checker's counterexample (n=12, K=8)")
n = 12
v = np.full((n, n), -100.0)
np.fill_diagonal(v, 0.0)


def s(a, b, x):
    v[a, b] = v[b, a] = x


s(0, 1, 2); s(0, 2, 2); s(1, 2, 10); s(3, 4, 2); s(3, 5, 2); s(4, 5, 10)
s(0, 3, -30); s(0, 4, 3); s(0, 5, 3); s(3, 1, 3); s(3, 2, 3)
for a in (1, 2):
    for b in (4, 5):
        s(a, b, -5)
cid = np.array([0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7])
old = old_gap_k(Partition(cid), v, v, 2, deviation_cap=64)
new = full_gap_k(cid, v, v, 2)
check("restricted checker 2.0, exact checker -2.0", old == 2.0 and new == -2.0, f"old {old}, full {new}")

# ------------------------------------------------ D. agreement where the old checker is exact
print("\nD. agreement with the restricted checker where it is provably exact (k<=2, K<=7)")
mism = tot = 0
for r in range(60):
    n = int(rng.integers(3, 8))
    sym = r % 3 != 0
    vc = rand_v(n, rng, sym, -100, 101)
    vd = vc + (np.abs(rand_v(n, rng, sym, 0, 30)) if r % 2 else 0)
    np.fill_diagonal(vd, 0.0)
    for _ in range(6):
        p = rng.integers(0, int(rng.integers(1, n + 1)), n)
        P = Partition(p)
        if P.num_coalitions > 7:
            continue
        for k in (1, 2):
            tot += 1
            mism += (old_gap_k(P, vc, vd, k, deviation_cap=64) != full_gap_k(P.cid, vc, vd, k))
check("old == full on random (n<=7, k<=2) incl. asymmetric and x<=y pairs", mism == 0, f"{tot} checks")

# ------------------------------------------------ E. exact existence evaluator
print("\nE. exact existence")
bell = [1, 1, 2, 5, 15, 52, 203, 877]
check("partition enumeration matches Bell numbers B1..B7",
      all(len(all_partitions(m)) == bell[m] for m in range(1, 8)))
v = V(2, {(0, 1): 1})
mg, arg = exact_existence(v, 1)
check("two-agent v=1: max gap 1 at the grand coalition", mg == 1 and arg == (0, 0))

# ------------------------------------------------ F. runtime
print("\nF. runtime of one Gap_k evaluation (single core)")
for n, k in [(6, 2), (6, 3), (7, 2), (7, 3), (10, 2), (12, 2)]:
    vv = rand_v(n, rng, True, -100, 101)
    parts = [rng.integers(0, max(1, n // 2), n) for _ in range(5)] + [np.arange(n)]
    t0 = time.perf_counter()
    for p in parts:
        full_gap_k(p, vv, vv, k)
    tf = (time.perf_counter() - t0) / len(parts)
    t0 = time.perf_counter()
    for p in parts:
        old_gap_k(Partition(p), vv, vv, k, deviation_cap=64)
    to = (time.perf_counter() - t0) / len(parts)
    print(f"     n={n:>2} k={k}: full {1e3*tf:8.2f} ms   restricted {1e3*to:8.2f} ms   ratio {tf/to:5.2f}")

n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
sys.exit(1 if n_fail else 0)
