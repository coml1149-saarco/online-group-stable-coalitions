"""
Unit tests on small instances whose answers can be checked by hand or by an
independent brute-force implementation written here from the definitions.

    python3 -m unittest discover -s tests -v

The property-based checks in tests/check_*.py are run by tests/test_properties.py.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sns.paths import ensure_core_on_path  # noqa: E402

ensure_core_on_path()
import ashg_lib  # noqa: E402
from ashg_lib import ASHGEnvironment, Partition, agent_utility  # noqa: E402
from sns.full_deviation_checker import exact_existence, full_gap_k, label  # noqa: E402
from sns.environments import RobustASHGEnvironment  # noqa: E402
from sns.bandit_algorithms import BANDIT_ALGORITHMS, BanditFeedback, round_robin  # noqa: E402
from sns.candidate_algorithms import CANDIDATE_ALGORITHMS  # noqa: E402

ashg_lib.gap_k = full_gap_k        # as in every experiment worker
DATA = ROOT / "data" / "benchmarks" / "raw"


# ---------------------------------------------------------------- reference implementation (definitions only)

def set_partitions(items):
    """All partitions of a list into blocks (frozensets)."""
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [p[i] | {first}] + p[i + 1:]
        yield p + [frozenset([first])]


def blocks_of(cid):
    out = {}
    for a, c in enumerate(cid):
        out.setdefault(c, set()).add(a)
    return [frozenset(b) for b in out.values()]


def coalition(blocks, i):
    return next(b for b in blocks if i in b)


def deviations(cid, C):
    """Lambda_C(pi): partitions pi' that agree with pi outside C and move every member of C to a different set."""
    n = len(cid)
    pi = blocks_of(cid)
    rest = set(range(n)) - set(C)
    pi_rest = {b - set(C) for b in pi if b - set(C)}
    out = []
    for q in set_partitions(range(n)):
        if {b - set(C) for b in q if b - set(C)} != pi_rest:
            continue
        if all(coalition(q, i) != coalition(pi, i) for i in C) and rest | set(C) == set(range(n)):
            out.append(q)
    return out


def u(i, block, v):
    return sum(v[i, j] for j in block if j != i)


def brute_gap(cid, v_curr, v_dev, k):
    n = len(cid)
    pi = blocks_of(cid)
    best = np.inf
    for s in range(1, k + 1):
        for C in itertools.combinations(range(n), s):
            for q in deviations(cid, C):
                g = max(u(i, coalition(pi, i), v_curr) - u(i, coalition(q, i), v_dev) for i in C)
                best = min(best, g)
    return best


def rgs(n):
    """All partitions of range(n) as restricted-growth label vectors."""
    for p in set_partitions(range(n)):
        cid = [0] * n
        for c, b in enumerate(sorted(p, key=min)):
            for a in b:
                cid[a] = c
        yield cid


def rand_v(n, rng, sym=True):
    v = rng.integers(-20, 21, size=(n, n)).astype(float)
    if sym:
        v = np.triu(v, 1); v = v + v.T
    np.fill_diagonal(v, 0.0)
    return v


# ---------------------------------------------------------------- tests

class Utilities(unittest.TestCase):
    def test_symmetric_generators(self):
        for model in ("uniform", "gaussian"):
            v = ASHGEnvironment(n=6, model=model, noise_std=3.0, seed=0).v
            self.assertTrue(np.array_equal(v, v.T))
            self.assertTrue(np.all(np.diag(v) == 0))
            self.assertTrue(np.all(v == np.round(v)))
        v = ASHGEnvironment(n=6, model="uniform", noise_std=3.0, seed=0).v
        self.assertTrue(np.all((v >= -100) & (v <= 100)))

    def test_asymmetric_generator(self):
        v = RobustASHGEnvironment(n=6, model="uniform", noise_std=3.0, seed=0, utility_mode="asymmetric").v
        self.assertFalse(np.array_equal(v, v.T))
        self.assertTrue(np.all(np.diag(v) == 0))

    def test_agent_utility_by_hand(self):
        v = np.array([[0, 4, -1], [4, 0, 2], [-1, 2, 0]], float)
        pi = Partition([0, 0, 1])
        self.assertEqual(agent_utility(0, pi, v), 4.0)
        self.assertEqual(agent_utility(2, pi, v), 0.0)
        self.assertEqual(agent_utility(1, Partition([0, 0, 0]), v), 6.0)


class DeviationsAndGap(unittest.TestCase):
    def test_deviation_counts_by_hand(self):
        # n = 3, singletons: agent 0 can join {1} or {2}; {0, 1} can form {0, 1} or join {2} together.
        self.assertEqual(len(deviations([0, 1, 2], (0,))), 2)
        self.assertEqual(len(deviations([0, 1, 2], (0, 1))), 2)
        # grand coalition of 3: agent 0 can only leave alone; {0, 1} can leave together, separately, or with one
        # of them forming a pair with agent 2 and the other alone (2 ways).
        self.assertEqual(len(deviations([0, 0, 0], (0,))), 1)
        self.assertEqual(len(deviations([0, 0, 0], (0, 1))), 4)

    def test_gap_by_hand(self):
        v = np.array([[0, 5], [5, 0]], float)
        self.assertEqual(full_gap_k(Partition([0, 0]), v, v, 1), 5.0)    # leaving loses 5
        self.assertEqual(full_gap_k(Partition([0, 0]), v, v, 2), 5.0)
        self.assertEqual(full_gap_k(Partition([0, 1]), v, v, 1), -5.0)   # joining gains 5
        w = np.array([[0, 3], [-2, 0]], float)                            # asymmetric
        self.assertEqual(full_gap_k(Partition([0, 0]), w, w, 1), -2.0)   # agent 1 gains 2 by leaving

    def test_exact_gap_against_brute_force(self):
        rng = np.random.default_rng(1)
        for n, k, sym in ((3, 1, True), (3, 3, True), (4, 2, True), (4, 3, False), (5, 2, True), (5, 2, False)):
            v = rand_v(n, rng, sym)
            for cid in rgs(n):
                self.assertEqual(full_gap_k(np.array(cid), v, v, k), brute_gap(cid, v, v, k), (n, k, sym, cid))

    def test_gap_monotone_in_k(self):
        rng = np.random.default_rng(2)
        v = rand_v(5, rng)
        for cid in rgs(5):
            g = [full_gap_k(np.array(cid), v, v, k) for k in (1, 2, 3)]
            self.assertTrue(g[0] >= g[1] >= g[2])

    def test_optimistic_and_pessimistic_estimates_bracket_the_gap(self):
        rng = np.random.default_rng(3)
        for sym in (True, False):
            v = rand_v(5, rng, sym)
            r = rng.uniform(0, 4, size=(5, 5))
            if sym:
                r = (r + r.T) / 2
            np.fill_diagonal(r, 0.0)
            ucb, lcb = v + r, v - r
            kappa = 2 * (2 * 5 - 3)
            for cid in rgs(5):
                p = np.array(cid)
                pes, true, opt = full_gap_k(p, lcb, ucb, 2), full_gap_k(p, v, v, 2), full_gap_k(p, ucb, lcb, 2)
                self.assertLessEqual(pes, true + 1e-9)
                self.assertLessEqual(true, opt + 1e-9)
                self.assertLessEqual(opt - pes, kappa * r.max() + 1e-9)


class Feedback(unittest.TestCase):
    def test_semi_bandit_observations(self):
        env = ASHGEnvironment(n=6, model="uniform", noise_std=0.0, seed=4)
        pi = Partition([0, 0, 0, 1, 1, 2])
        obs = env.observe(pi)
        self.assertEqual(set(obs), {(0, 1), (0, 2), (1, 2), (3, 4)})
        for (i, j), x in obs.items():
            self.assertEqual(x, env.v[i, j])

    def test_bandit_observations(self):
        env = ASHGEnvironment(n=6, model="uniform", noise_std=0.0, seed=5)
        fb = BanditFeedback(env)
        pi = Partition([0, 0, 0, 1, 1, 2])
        R = fb.reports(pi)
        self.assertEqual(R[0], env.v[0, 1] + env.v[0, 2])
        self.assertEqual(R[5], 0.0)
        obs = fb.observe(pi)
        self.assertEqual(set(obs), {(3, 4), (4, 3)})          # only the two-agent coalition is informative
        self.assertEqual(obs[(3, 4)], env.v[3, 4])

    def test_round_robin_covers_every_pair_once(self):
        for n in range(3, 9):
            P = round_robin(n)
            self.assertEqual(len(P), n - 1 if n % 2 == 0 else n)
            seen = []
            for p in P:
                for b in p.coalitions():
                    self.assertLessEqual(len(b), 2)
                    if len(b) == 2:
                        seen.append(tuple(sorted(b)))
            self.assertEqual(sorted(seen), sorted(itertools.combinations(range(n), 2)))


def make(arm, n, seed, feedback="semi", sigma=3.0, beta=1.0, k=2, eps=1.0):
    reg = CANDIDATE_ALGORITHMS if feedback == "semi" else BANDIT_ALGORITHMS
    return reg[arm](n=n, sigma=sigma, beta=beta, k=k, eps=eps, rng=np.random.default_rng(seed + 10_000),
                    n_restarts=3, max_iter=60, deviation_cap=64, max_random_coalitions=4, directed=False)


class Learners(unittest.TestCase):
    def test_semi_bandit_initialization(self):
        alg = make("ucb_lcb_gs_witness_dev", 5, 0)
        pi, d = alg.act(1)
        self.assertEqual(d, 0)
        self.assertEqual(pi.num_coalitions, 1)

    def test_bandit_initialization(self):
        env = BanditFeedback(ASHGEnvironment(n=5, model="uniform", noise_std=3.0, seed=0))
        alg = make("bandit_witness", 5, 0, feedback="bandit")
        expected = round_robin(5)
        for t in range(1, len(expected) + 1):
            pi, d = alg.act(t)
            self.assertEqual(d, 0)
            self.assertEqual(pi, expected[t - 1])
            alg.update(env.observe(pi))
        self.assertTrue(np.all(alg.estimator.T[~np.eye(5, dtype=bool)] == 2))

    def test_no_declaration_on_a_no_instance(self):
        # symmetric Gaussian instance n=5, seed 2: no exactly 2-strongly Nash stable partition
        env = ASHGEnvironment(n=5, model="gaussian", noise_std=3.0, seed=2)
        g, _ = exact_existence(env.v, 2)
        self.assertEqual(label(g, 1.0), "NO")
        for arm, fb in (("ucb_lcb_gs_witness_dev", "semi"), ("bandit_witness", "bandit")):
            e = ASHGEnvironment(n=5, model="gaussian", noise_std=3.0, seed=2)
            e = BanditFeedback(e) if fb == "bandit" else e
            alg = make(arm, 5, 2, feedback=fb, beta=3.0)
            for t in range(1, 31):
                pi, d = alg.act(t)
                self.assertEqual(d, 0)
                alg.update(e.observe(pi))

    def test_declared_partitions_are_stable_with_small_noise(self):
        env = ASHGEnvironment(n=5, model="uniform", noise_std=0.01, seed=0)
        g, _ = exact_existence(env.v, 2)
        self.assertGreaterEqual(g, 1.0)
        alg = make("ucb_lcb_gs_witness_dev", 5, 0, sigma=0.01, beta=3.0)
        declared = 0
        for t in range(1, 31):
            pi, d = alg.act(t)
            if d == 1:
                declared += 1
                self.assertGreaterEqual(full_gap_k(pi, env.v, env.v, 2), 1.0)
            alg.update(env.observe(pi))
        self.assertGreater(declared, 0)


class LabelsMetricsAndReproducibility(unittest.TestCase):
    def test_saved_exact_labels(self):
        with (ROOT / "results/saved/labels/exact_labels.csv").open() as fh:
            rows = [r for r in csv.DictReader(fh) if r["setting"] == "symmetric" and r["n"] == "5"][:4]
        for r in rows:
            v = ASHGEnvironment(n=5, model=r["model"], noise_std=3.0, seed=int(r["seed"])).v
            g, _ = exact_existence(v, int(r["k"]))
            self.assertEqual(g, float(r["max_gap"]))

    def _task(self, **kw):
        import run_grid as R
        lab = R.load_labels()
        return R.task("test", kw.get("arm", "ucb_lcb_gs_witness_dev"), "uniform", 6, 30, 2, 3.0, 1.0, 1.0,
                      kw.get("seed", 0), lab, feedback=kw.get("feedback", "semi"))

    def test_trial_metrics_and_reproducibility(self):
        import run_grid as R
        R._init_worker()
        saved = {}
        with (ROOT / "results/saved/synthetic/trials.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                if (r["arm"], r["model"], r["n"], r["T"], r["k"], r["sigma"], r["beta"], r["eps"], r["seed"]) == \
                        ("ucb_lcb_gs_witness_dev", "uniform", 6, 30, 2, 3.0, 1.0, 1.0, 0):
                    saved = r
                    break
        for arm in ("ucb_lcb_gs_witness_dev", "pessimistic"):
            t = self._task(arm=arm)
            a, b = R.run_one(t), R.run_one(t)
            self.assertEqual((a["regret"], a["d_seq"]), (b["regret"], b["d_seq"]))     # same seed, same trial
            self.assertAlmostEqual(a["regret"], sum(a["regret_seq"]), places=4)
            self.assertEqual(a["n_d1"], a["d_seq"].count("1"))
            self.assertEqual(a["n_d0"], 30 - a["n_d1"])                              # existence error on a YES instance
            self.assertTrue(all(x >= 0 for x in a["regret_seq"]))
        a = R.run_one(self._task())
        self.assertEqual((a["regret"], a["d_seq"]), (saved["regret"], saved["d_seq"]))   # saved record reproduced

    @unittest.skipUnless((DATA / "uci_iris" / "iris.data").exists(), "benchmark data not fetched")
    def test_benchmark_trial_reproduces_saved_record(self):
        # The game is regenerated from the downloaded data; the trial must reproduce the published record, and
        # every recorded gap must equal the exact gap of the partition formed in that round.
        import benchmark_games as B
        import run_benchmark as RB
        import run_grid as R
        R._init_worker()
        with (ROOT / "results/saved/benchmark/labels.jsonl").open() as fh:
            rec = next(r for r in map(json.loads, fh) if r["dataset"] == "karate" and r["n"] == 8 and r["domain"] == "AF")
        _, vs, meta = B.game(rec["dataset"], rec["sampler"], rec["n"], rec["seed"])
        v = vs[rec["domain"]]
        task = dict(suite="bench_games", arm="ucb_lcb_gs_witness_dev", feedback="semi", utility_mode="symmetric",
                    noise_model="gaussian", model=f"bm_{rec['dataset']}_{rec['sampler']}_{rec['domain']}", n=rec["n"],
                    seed=rec["seed"], env_seed=meta["env_seed"], alg_seed=meta["alg_seed"], label=rec["label"],
                    max_gap=rec["max_gap"], v=v.astype(int).tolist(), variant=rec["domain"], **RB.KCFG)
        out = RB.run_one(task)
        with (ROOT / "results/saved/benchmark/trials.jsonl").open() as fh:
            saved = next(t for t in map(json.loads, fh) if t["task_id"] == out["task_id"])
        self.assertEqual((out["d_seq"], out["regret_seq"], out["gap_seq"]), (saved["d_seq"], saved["regret_seq"], saved["gap_seq"]))
        for part, g, r in zip(out["partitions"], out["gap_seq"], out["regret_seq"]):
            cid = np.array([ord(c) - 48 for c in part])
            self.assertEqual(full_gap_k(cid, v, v, 2), g)
            self.assertAlmostEqual(max(0.0, 1.0 - g), r, places=6)


class BenchmarkGames(unittest.TestCase):
    def test_types_and_scaling(self):
        import benchmark_games as B
        self.assertEqual(B.domain_type("AF", 8), (8, -1))
        self.assertEqual(B.domain_type("AE", 8), (1, -8))
        self.assertEqual(B.domain_type("B", 8), (1, -1))
        self.assertEqual(B.scale("AF", 8), 12)
        self.assertEqual(B.scale("AE", 12), 8)
        self.assertEqual(B.scale("B", 10), 100)

    def test_relations_and_utilities_by_hand(self):
        import benchmark_games as B
        D = np.array([[0, 1, 5, 10], [1, 0, 4, 9], [5, 4, 0, 5], [10, 9, 5, 0]], float)
        R = B.relations(D, 10.0, 0.25, 0.35)        # friend if d <= 2.5, enemy if d > 3.5
        self.assertEqual(R.tolist(), [[0, 1, -1, -1], [1, 0, -1, -1], [-1, -1, 0, -1], [-1, -1, -1, 0]])
        v = B.utilities(R, "AF")                     # (n, -1) = (4, -1), c = 25
        self.assertEqual(v[0, 1], 100.0)
        self.assertEqual(v[0, 2], -25.0)

    @unittest.skipUnless((DATA / "uci_iris" / "iris.data").exists(), "benchmark data not fetched")
    def test_saved_games_regenerate(self):
        # The published game records carry SHA-256 digests of the sampled agents and of the utility matrix.
        import hashlib
        import benchmark_games as B

        def digest(x):
            return hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()

        with (ROOT / "results/saved/benchmark/labels.jsonl").open() as fh:
            recs = [json.loads(line) for line in fh]
        for ds, sp in B.CONFIGS:
            for n in B.SIZES:
                r = next(x for x in recs if x["dataset"] == ds and x["sampler"] == sp and x["n"] == n)
                _, vs, meta = B.game(ds, sp, n, r["seed"])
                self.assertEqual(digest(meta["agents"]), r["agents_sha256"])
                self.assertEqual(digest(vs[r["domain"]].astype(int).tolist()), r["v_sha256"])


if __name__ == "__main__":
    unittest.main()
