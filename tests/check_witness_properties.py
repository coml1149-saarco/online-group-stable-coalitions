#!/usr/bin/env python3
"""
Property-based tests of UCB-LCB-GS with witness probing (deviation probe),
checked round by round against the pseudocode (Algorithm 1 of the paper).

    python3 tests/check_witness_properties.py

Random small instances (n in {3,4,5}, k <= min(3, n), symmetric and asymmetric
utilities, Uniform and Gaussian models, several noise levels and beta). On every
round of every trajectory:

  P1  d_t = 1 only if the exact pessimistic gap of the played partition is >= eps
  P2  uncovered rounds play the grand coalition with d_t = 0
  P3  at covered uncertified rounds, the recorded minimiser (C, pi°)
      (a) is evaluated on the optimistic candidate actually computed,
      (b) attains the exact pessimistic gap (value equals full_gap_k),
      (c) has the value of Gap_C recomputed from the definition,
      (d) lies in Lambda_C(pi_opt) (checked from the definition),
      and both candidates fail the exact pessimistic test
  P4  the played partition is pi_opt if the witness pair (recomputed
      independently) is co-located in pi_opt, and pi° otherwise; either way it
      co-locates the witness pair
  P5  witness lemma: Go(pi_opt) - Gp(pi_opt) <= kappa_n * b_{e*}
  P6  after the update, the count of e* increased by exactly 1 (q = 1)
  P7  determinism: an identical rerun gives identical partitions and decisions
plus targeted tests of the safeguard branch and of checker dispatch.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sns.paths import ensure_core_on_path  # noqa: E402

ensure_core_on_path()
import ashg_lib  # noqa: E402
from sns import candidate_algorithms as CA  # noqa: E402
from sns.environments import RobustASHGEnvironment  # noqa: E402
from sns.full_deviation_checker import full_gap_k  # noqa: E402

ashg_lib.gap_k = full_gap_k                     # as in every experiment worker
from ashg_lib import ASHGEnvironment, Partition  # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def blocks(cid):
    out = {}
    for a, c in enumerate(cid):
        out.setdefault(int(c), set()).add(a)
    return out


def coal(cid, i):
    return {a for a in range(len(cid)) if cid[a] == cid[i]}


def in_lambda(cid, cid2, C):
    n = len(cid)
    rest = [a for a in range(n) if a not in C]
    for a, b in itertools.combinations(rest, 2):
        if (cid[a] == cid[b]) != (cid2[a] == cid2[b]):
            return False
    return all(coal(cid, i) != coal(cid2, i) for i in C)


def gap_C_def(cid, cid2, C, x, y):
    return max(sum(x[i, j] for j in coal(cid, i) if j != i) - sum(y[i, j] for j in coal(cid2, i) if j != i) for i in C)


def same(p, q):
    return np.array_equal(Partition(p).cid, Partition(q).cid)


def run_traj(n, k, setting, model, sigma, beta, eps, T, seed, record=True):
    if setting == "symmetric":
        env = ASHGEnvironment(n=n, model=model, noise_std=sigma, seed=seed)
    else:
        env = RobustASHGEnvironment(n=n, model=model, noise_std=sigma, seed=seed, utility_mode="asymmetric")
    alg = CA.UCBLCBGSWitness(n=n, sigma=sigma, beta=beta, k=k, eps=eps, rng=np.random.default_rng(seed + 10_000),
                             n_restarts=3, max_iter=60, deviation_cap=64, max_random_coalitions=4,
                             directed=(setting == "asymmetric"))
    rec = {}
    orig_argmin, orig_solve = CA.full_gap_k_argmin, alg._solve

    def argmin_wrap(pi, a, b, kk):
        out = orig_argmin(pi, a, b, kk)
        rec["argmin_in"], rec["argmin_out"] = Partition(pi.cid), out
        return out

    def solve_wrap(etype, ucb, lcb):
        p = orig_solve(etype, ucb, lcb)
        rec[etype] = Partition(p.cid)
        return p

    CA.full_gap_k_argmin = argmin_wrap
    alg._solve = solve_wrap
    kappa = 2 * (2 * n - 3)
    hist, stats = [], dict(rounds=0, certified=0, uncovered=0, optimistic=0, probes=0, safeguard=0)
    try:
        for t in range(1, T + 1):
            rec.clear()
            Tb = alg.estimator.T.copy()
            covered = bool(np.all(Tb[~np.eye(n, dtype=bool)] > 0))
            pi, d = alg.act(t)
            ucb, lcb = alg._bounds(t)
            hist.append((tuple(pi.cid), d))
            stats["rounds"] += 1
            tag = f"[n={n} k={k} {setting} {model} s={sigma} b={beta} seed={seed} t={t}]"
            if d == 1:                                                        # P1
                stats["certified"] += 1
                check(full_gap_k(pi, lcb, ucb, k) >= eps, f"P1 false declaration {tag}")
                if "argmin_in" in rec:
                    stats["safeguard"] += 1
                    check(same(pi.cid, rec["argmin_in"].cid), f"P1 safeguard plays pi_opt {tag}")
            elif not covered:                                                 # P2
                stats["uncovered"] += 1
                check(np.all(pi.cid == 0), f"P2 uncovered round not grand {tag}")
            else:
                po = rec["optimistic"]
                g, C, cid_dev = rec["argmin_out"]
                check(same(rec["argmin_in"].cid, po.cid), f"P3a argmin not on pi_opt {tag}")
                check(abs(g - full_gap_k(po, lcb, ucb, k)) < 1e-9, f"P3b argmin value {tag}")
                check(abs(g - gap_C_def(po.cid, cid_dev, C, lcb, ucb)) < 1e-9, f"P3c Gap_C definition {tag}")
                check(in_lambda(po.cid, cid_dev, C), f"P3d not in Lambda_C {tag}")
                check(1 <= len(C) <= k, f"P3d |C| {tag}")
                check(g < eps and full_gap_k(rec["pessimistic"], lcb, ucb, k) < eps, f"P3 candidates pass {tag}")
                rad = (ucb - lcb) / 2.0
                best, es = -np.inf, None
                for i in C:
                    for j in range(n):
                        if j != i and (j in coal(po.cid, i) or j in coal(cid_dev, i)) and rad[i, j] > best:
                            best, es = rad[i, j], (i, j)
                i, j = es
                if po.cid[i] == po.cid[j]:                                    # P4
                    stats["optimistic"] += 1
                    check(same(pi.cid, po.cid), f"P4 should play pi_opt {tag}")
                else:
                    stats["probes"] += 1
                    check(same(pi.cid, cid_dev), f"P4 should play deviation {tag}")
                check(pi.cid[i] == pi.cid[j], f"P4 witness pair not co-located {tag}")
                Go = full_gap_k(po, ucb, lcb, k)                              # P5
                check(Go - g <= kappa * rad[i, j] + 1e-7, f"P5 witness lemma {tag}")
            obs = env.observe(pi)
            alg.update(obs)
            if d == 0 and covered:                                            # P6
                check(alg.estimator.T[i, j] == Tb[i, j] + 1, f"P6 count of e* {tag}")
    finally:
        CA.full_gap_k_argmin = orig_argmin
    return hist, stats


def test_trajectories():
    rng = np.random.default_rng(2026)
    tot = dict(rounds=0, certified=0, uncovered=0, optimistic=0, probes=0, safeguard=0)
    n_traj = 0
    for setting in ("symmetric", "asymmetric"):
        for n in (3, 4, 5):
            for k in range(1, min(3, n) + 1):
                for rep in range(4):
                    model = ("uniform", "gaussian")[rep % 2]
                    sigma = float(rng.choice([3.0, 10.0, 30.0]))
                    beta = float(rng.choice([1.0, 3.0]))
                    seed = int(rng.integers(0, 10_000))
                    _, st = run_traj(n, k, setting, model, sigma, beta, 1.0, 40, seed)
                    n_traj += 1
                    for a in tot:
                        tot[a] += st[a]
    print(f"  trajectories: {n_traj}; rounds by kind: {tot}")
    check(tot["probes"] > 50 and tot["optimistic"] > 20 and tot["certified"] > 100,
          "coverage of branches too small to be informative")


def test_determinism():
    h1, _ = run_traj(5, 2, "asymmetric", "uniform", 10.0, 3.0, 1.0, 40, 7)
    h2, _ = run_traj(5, 2, "asymmetric", "uniform", 10.0, 3.0, 1.0, 40, 7)
    check(h1 == h2, "P7 determinism")


def test_safeguard():
    n, k, eps = 4, 2, 1.0
    v = np.full((n, n), 50.0); np.fill_diagonal(v, 0.0)
    alg = CA.UCBLCBGSWitness(n=n, sigma=1.0, beta=3.0, k=k, eps=eps, rng=np.random.default_rng(0))
    for _ in range(200):                                  # well-estimated utilities
        alg.update({(i, j): v[i, j] for i in range(n) for j in range(i + 1, n)})
    single = Partition(np.arange(n))
    grand = Partition(np.zeros(n, dtype=np.int64))
    orig = alg._solve
    alg._solve = lambda e, u, l: single if e == "pessimistic" else grand      # heuristic misses the optimum
    pi, d = alg.act(300)
    ucb, lcb = alg._bounds(300)
    check(d == 1 and same(pi.cid, grand.cid) and full_gap_k(grand, lcb, ucb, k) >= eps, "safeguard certifies pi_opt")
    alg._solve = lambda e, u, l: single                                       # neither candidate passes
    pi, d = alg.act(301)
    check(d == 0, "no declaration when both candidates fail")
    alg._solve = orig


def test_dispatch():
    calls = {"n": 0}
    real = ashg_lib.gap_k

    def counting(*a, **kw):
        calls["n"] += 1
        return full_gap_k(*a[:4])
    ashg_lib.gap_k = counting
    try:
        v = np.random.default_rng(1).integers(-100, 101, (5, 5)).astype(float); v = (v + v.T) / 2
        np.fill_diagonal(v, 0)
        ashg_lib.compute_partition("ucb_only", v, v, k=2, eps=1.0, rng=np.random.default_rng(0))
        a = calls["n"]
        ashg_lib.is_stable(Partition(np.zeros(5, dtype=np.int64)), v, v, 2, 1.0)
        check(a > 0 and calls["n"] > a, "solver and stability test dispatch to the installed checker")
    finally:
        ashg_lib.gap_k = real
    check(ashg_lib.gap_k is full_gap_k, "exact checker installed")


def main():
    for f in (test_trajectories, test_determinism, test_safeguard, test_dispatch):
        before = len(FAIL)
        f()
        print(f"{'PASS' if len(FAIL) == before else 'FAIL'}  {f.__name__}")
    if FAIL:
        print(f"{len(FAIL)} failures; first: {FAIL[:5]}")
        sys.exit(1)
    print("all witness property tests passed")


if __name__ == "__main__":
    main()
