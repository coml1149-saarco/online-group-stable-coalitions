#!/usr/bin/env python3
"""
Tests of the coalition-level (per-agent) bandit-feedback implementation.

    python3 tests/check_bandit.py

  B1  round-robin cycle: every unordered pair in exactly one matching,
      r = n-1 (even n) or n (odd n), coalitions of size <= 2
  B2  feedback: only agents in two-agent coalitions produce observations, each
      equal to v_ij + that agent's noise (checked with zero noise and against
      the per-agent reports); larger coalitions and singletons produce none
  B3  no privileged access: along trajectories of every bandit arm, every
      estimator update is exactly the feedback of the played partition, and
      the counts equal the number of such observations (q = 2 symmetric,
      q = 1 per ordered pair asymmetric)
  B4  witness probing, every round: no false declaration (exact pessimistic
      test); uncovered rounds play the next matching; the witness pair is a
      two-agent coalition of the played partition; its count increases by q;
      witness lemma Go - Gp <= kappa_n b_{e*}
  B5  round-robin coverage: probes follow the cycle without reset
  B6  Random under bandit feedback plays exactly the semi-bandit Random
      trajectory (justifies reusing its semi-bandit trajectories)
  B7  determinism
  B8  two-agent counterexample (v = -M): witness probing certifies
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
from sns import bandit_algorithms as BA  # noqa: E402
from sns.environments import RobustASHGEnvironment  # noqa: E402
from sns.full_deviation_checker import full_gap_k  # noqa: E402
from sns.runner import FixedUtilityASHG  # noqa: E402

ashg_lib.gap_k = full_gap_k
from ashg_lib import ASHGEnvironment, Partition  # noqa: E402
from algorithms import RandomBaseline  # noqa: E402

FAIL = []
SOUND = {"bandit_witness", "bandit_rr_L1", "bandit_rr_L2", "pessimistic", "shared_pessimistic"}


def check(c, m):
    if not c:
        FAIL.append(m)


def make_learner(arm, n, k, sigma, beta, seed, directed=False, eps=1.0):
    return BA.BANDIT_ALGORITHMS[arm](n=n, sigma=sigma, beta=beta, k=k, eps=eps, rng=np.random.default_rng(seed + 10_000),
                                     n_restarts=3, max_iter=60, deviation_cap=64, max_random_coalitions=4,
                                     directed=directed)


def env_for(n, model, sigma, seed, setting):
    if setting == "symmetric":
        e = ASHGEnvironment(n=n, model=model, noise_std=sigma, seed=seed)
    else:
        e = RobustASHGEnvironment(n=n, model=model, noise_std=sigma, seed=seed, utility_mode="asymmetric")
    return BA.BanditFeedback(e)


def test_cycle():
    for n in range(2, 10):
        cyc = BA.round_robin(n)
        check(len(cyc) == (n - 1 if n % 2 == 0 else n), f"B1 r for n={n}")
        seen = []
        for p in cyc:
            for b in set(p.cid.tolist()):
                mem = [a for a in range(n) if p.cid[a] == b]
                check(len(mem) <= 2, f"B1 coalition size n={n}")
                if len(mem) == 2:
                    seen.append(tuple(mem))
        check(sorted(seen) == list(itertools.combinations(range(n), 2)), f"B1 pairs n={n}")


def test_feedback():
    v = np.array([[0, 5, -3, 2], [5, 0, 1, 4], [-3, 1, 0, 7], [2, 4, 7, 0]], float)
    fb = BA.BanditFeedback(FixedUtilityASHG(v, noise_std=0.0, seed=0))
    obs = fb.observe(Partition([0, 0, 1, 1]))
    check(obs == {(0, 1): 5.0, (1, 0): 5.0, (2, 3): 7.0, (3, 2): 7.0}, "B2 pairs observed exactly")
    check(fb.observe(Partition([0, 0, 0, 1])) == {}, "B2 triple and singleton give nothing")
    fb2 = BA.BanditFeedback(FixedUtilityASHG(v, noise_std=2.0, seed=3))
    st = fb2.env.rng.bit_generator.state
    R = fb2.reports(Partition([0, 0, 1, 2]))
    fb2.env.rng.bit_generator.state = st
    obs = fb2.observe(Partition([0, 0, 1, 2]))
    check(obs == {(0, 1): R[0], (1, 0): R[1]}, "B2 observation equals the agent's report")


def traj(arm, n, k, setting, model, sigma, beta, seed, T, props=False):
    env = env_for(n, model, sigma, seed, setting)
    directed = setting == "asymmetric"
    alg = make_learner(arm, n, k, sigma, beta, seed, directed)
    q = 1 if directed else 2
    kappa = 2 * (2 * n - 3)
    hist, kinds = [], {"init": 0, "probe": 0, "opt": 0, "cert": 0}
    for t in range(1, T + 1):
        Tb = alg.estimator.T.copy()
        covered = bool(np.all(Tb[~np.eye(n, dtype=bool)] > 0))
        cb = alg.counters() if hasattr(alg, "counters") else None
        pi, d = alg.act(t)
        ucb, lcb = alg._bounds(t)
        hist.append((tuple(pi.cid), d))
        tag = f"[{arm} n={n} k={k} {setting} seed={seed} t={t}]"
        if d == 1:
            kinds["cert"] += 1
            if arm in SOUND:          # arms that declare only through the pessimistic certificate
                check(full_gap_k(pi, lcb, ucb, k) >= 1.0, f"B4 false declaration {tag}")
        obs = env.observe(pi)
        alg.update(obs)
        # B3: estimator increments are exactly the feedback
        dT = alg.estimator.T - Tb
        expect = np.zeros_like(dT)
        for (i, j) in obs:
            expect[i, j] += 1
            if not directed:
                expect[j, i] += 1
        check(np.array_equal(dT, expect), f"B3 estimator update {tag}")
        for (i, j) in obs:
            check(int(np.sum(pi.cid == pi.cid[i])) == 2 and pi.cid[i] == pi.cid[j], f"B3 non-pair observed {tag}")
        if props and cb is not None and d == 0:
            ca = alg.counters()
            if not covered:
                kinds["init"] += 1
                check(ca["n_init"] == cb["n_init"] + 1 and all(np.sum(pi.cid == c) <= 2 for c in set(pi.cid.tolist())),
                      f"B4 uncovered round is a matching {tag}")
            else:
                kinds["probe" if ca["n_probes"] > cb["n_probes"] else "opt"] += 1
                # recompute the witness pair independently
                po = alg._last_opt if hasattr(alg, "_last_opt") else None
                if po is not None:
                    pass
                rad = (ucb - lcb) / 2.0
                cnt = alg.estimator.T - Tb
                # the witness pair is the observed two-agent coalition with the largest pre-round radius among
                # increments; check that some pair was observed with +q and that it forms a two-agent coalition
                inc = [(i, j) for i in range(n) for j in range(n) if i != j and cnt[i, j] == q]
                check(len(inc) > 0, f"B4 no pair observed at an uncertified round {tag}")
    return hist, kinds


def test_witness_props():
    tot = {"init": 0, "probe": 0, "opt": 0, "cert": 0}
    rng = np.random.default_rng(5)
    for setting in ("symmetric", "asymmetric"):
        for n in (3, 4, 5, 6):
            for k in (1, 2, 3):
                if k > n:
                    continue
                seed = int(rng.integers(0, 10_000))
                _, kd = traj("bandit_witness", n, k, setting, ("uniform", "gaussian")[seed % 2],
                             float(rng.choice([3.0, 10.0])), float(rng.choice([1.0, 3.0])), seed, 60, props=True)
                for a in tot:
                    tot[a] += kd[a]
    print(f"  witness rounds by kind: {tot}")
    check(tot["probe"] > 20 and tot["cert"] > 50, "B4 branch coverage")


def test_witness_pair_isolated():
    """B4 directly: instrument the minimiser and verify the witness pair is a two-agent coalition."""
    from sns import bandit_algorithms as M
    real = M.full_gap_k_argmin
    rec = {}

    def wrap(pi, a, b, k):
        out = real(pi, a, b, k)
        rec["in"], rec["out"] = Partition(pi.cid), out
        return out
    M.full_gap_k_argmin = wrap
    try:
        for setting in ("symmetric", "asymmetric"):
            for seed in range(6):
                n, k = 5, 2
                env = env_for(n, "uniform", 10.0, seed, setting)
                directed = setting == "asymmetric"
                alg = make_learner("bandit_witness", n, k, 10.0, 3.0, seed, directed)
                q = 1 if directed else 2
                kappa = 2 * (2 * n - 3)
                for t in range(1, 80):
                    rec.clear()
                    pi, d = alg.act(t)
                    ucb, lcb = alg._bounds(t)
                    if d == 0 and "out" in rec:
                        po, (g, C, cdev) = rec["in"], rec["out"]
                        rad = (ucb - lcb) / 2.0
                        best, es = -np.inf, None
                        for i in C:
                            for j in range(n):
                                if j != i and (po.cid[j] == po.cid[i] or cdev[j] == cdev[i]) and rad[i, j] > best:
                                    best, es = rad[i, j], (i, j)
                        i, j = es
                        check(pi.cid[i] == pi.cid[j] and int(np.sum(pi.cid == pi.cid[i])) == 2,
                              f"B4 witness pair not isolated {setting} s={seed} t={t}")
                        check(full_gap_k(po, ucb, lcb, k) - g <= kappa * rad[i, j] + 1e-7, f"B4 witness lemma t={t}")
                        Tb = alg.estimator.T[i, j]
                        alg.update(env.observe(pi))
                        check(alg.estimator.T[i, j] == Tb + q, f"B4 count of e* +q {setting} t={t}")
                    else:
                        alg.update(env.observe(pi))
    finally:
        M.full_gap_k_argmin = real


def test_others_no_privilege():
    for arm in ["bandit_rr_L1", "bandit_rr_L2", "pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed",
                "shared_mixed", "shared_optimistic", "shared_pessimistic"]:
        for setting in ("symmetric", "asymmetric"):
            traj(arm, 5, 2, setting, "uniform", 3.0, 3.0, 11, 40)


def test_rr_cycle():
    alg = make_learner("bandit_rr_L1", 6, 2, 30.0, 3.0, 2)
    env = env_for(6, "uniform", 30.0, 2, "symmetric")
    cyc = BA.round_robin(6)
    pos = 0
    for t in range(1, 60):
        cb = alg.counters()
        pi, d = alg.act(t)
        ca = alg.counters()
        if ca["matching_rounds"] > cb["matching_rounds"]:
            check(np.array_equal(pi.cid, cyc[pos % len(cyc)].cid), f"B5 cycle order t={t}")
            pos += 1
        alg.update(env.observe(pi))
    check(alg.counters()["completed_cycles"] == pos // len(cyc), "B5 completed cycles")


def test_random_equivalence():
    for seed in range(5):
        a = RandomBaseline(n=6, sigma=3.0, beta=1.0, k=2, eps=1.0, rng=np.random.default_rng(seed + 10_000))
        b = make_learner("random", 6, 2, 3.0, 1.0, seed)
        envb = env_for(6, "uniform", 3.0, seed, "symmetric")
        for t in range(1, 31):
            pa, da = a.act(t); pb, db = b.act(t)
            check(np.array_equal(pa.cid, pb.cid) and da == db, f"B6 random differs seed={seed} t={t}")
            b.update(envb.observe(pb))


def test_determinism():
    h1, _ = traj("bandit_witness", 5, 2, "symmetric", "gaussian", 10.0, 3.0, 9, 50)
    h2, _ = traj("bandit_witness", 5, 2, "symmetric", "gaussian", 10.0, 3.0, 9, 50)
    check(h1 == h2, "B7 determinism")


def test_two_agent():
    M = 10.0
    v = np.array([[0, -M], [-M, 0]])
    d1 = []
    for seed in range(20):
        env = BA.BanditFeedback(FixedUtilityASHG(v, noise_std=3.0, seed=seed))
        alg = make_learner("bandit_witness", 2, 1, 3.0, 3.0, seed)
        n1 = 0
        for t in range(1, 301):
            pi, d = alg.act(t)
            n1 += d
            alg.update(env.observe(pi))
        d1.append(300 - n1)
    print(f"  two-agent v=-M: mean uncertified rounds over T=300: {np.mean(d1):.1f}")
    check(np.mean(d1) < 30, "B8 witness probing certifies on the two-agent counterexample")


def main():
    for f in (test_cycle, test_feedback, test_witness_props, test_witness_pair_isolated, test_others_no_privilege,
              test_rr_cycle, test_random_equivalence, test_determinism, test_two_agent):
        b = len(FAIL)
        f()
        print(f"{'PASS' if len(FAIL) == b else 'FAIL'}  {f.__name__}")
    if FAIL:
        print(len(FAIL), "failures; first:", FAIL[:6])
        sys.exit(1)
    print("all bandit tests passed")


if __name__ == "__main__":
    main()
