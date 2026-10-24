"""Checks that config/experiments.json agrees with config/final_mapping.csv and with the constants in the code."""

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CFG = json.loads((ROOT / "config/experiments.json").read_text())


def mapping():
    with (ROOT / "config/final_mapping.csv").open() as fh:
        return list(csv.DictReader(fh))


class Config(unittest.TestCase):
    def test_synthetic_grid_matches_mapping(self):
        g = CFG["synthetic_grid"]
        expected = set()
        for exp in ("E1", "E2"):
            base = dict(g["default"])
            if exp == "E2":
                base["beta"] = g["E2"]["default_beta"]
            for p, values in g[exp]["vary_one_at_a_time"].items():
                for x in values:
                    c = dict(base, **{p: x})
                    expected.add((exp, c["n"], c["T"], c["k"], c["sigma"], c["beta"], c["eps"]))
        for T in g["E3"]["T"]:
            for b in g["E3"]["beta"]:
                c = dict(g["default"], T=T, beta=b)
                expected.add(("E3", c["n"], c["T"], c["k"], c["sigma"], c["beta"], c["eps"]))
        found = {(r["experiment"], int(r["n"]), int(r["T"]), int(r["k"]), float(r["sigma"]), float(r["beta"]),
                  float(r["eps"])) for r in mapping()}
        self.assertEqual(found, expected)

    def test_methods_and_seeds(self):
        import run_grid as R
        import run_benchmark as RB
        arms = {r["arm"] for r in mapping()}
        self.assertEqual(arms, set(CFG["methods"]["semi_bandit"]) | set(CFG["methods"]["not_reported_in_paper"]))
        self.assertEqual(RB.SEMI_ARMS, CFG["methods"]["semi_bandit"])
        self.assertEqual(RB.BANDIT_ARMS, CFG["methods"]["coalition_level_bandit"])
        self.assertEqual({r["seeds"] for r in mapping()}, {"0-99"})
        lab = R.load_labels()
        t = R.task("x", "pessimistic", "uniform", 6, 30, 2, 3.0, 1.0, 1.0, 7, lab)
        self.assertEqual((t["env_seed"], t["alg_seed"]), (7, 10_007))

    def test_heuristic_and_learning_parameters(self):
        import run_grid as R
        import run_benchmark as RB
        h = CFG["heuristic"]
        self.assertEqual(R.OPT, dict(n_restarts=h["restarts_R"], optimizer_max_iter=h["sweeps_M"],
                                     deviation_cap=h["deviation_cap"], max_random_coalitions=h["random_fallback_coalitions"]))
        from sns.paths import ensure_core_on_path
        ensure_core_on_path()
        from algorithms import PairwiseEstimator
        self.assertEqual(PairwiseEstimator(3, 1.0, 1.0).init_bonus, h["unseen_pair_radius_B0"])
        self.assertEqual(RB.KCFG, CFG["datasets"]["learning_parameters"])

    def test_bandit_and_robustness_tasks(self):
        import run_grid as R
        lab = R.load_labels()
        T = R.bandit_grid_tasks(lab)
        shared = {t["arm"] for t in T if t["arm"].startswith("shared_")}
        self.assertEqual(shared, set(CFG["methods"]["shared_exploration_control"]))
        c = CFG["coalition_level_bandit"]["shared_exploration_control"]
        self.assertEqual({(t["T"], t["beta"]) for t in T if t["arm"].startswith("shared_")},
                         {(x, b) for x in c["T"] for b in c["beta"]})
        rob = R.robust_tasks(lab, ["ucb_lcb_gs_witness_dev"], range(100), "robust")
        p = CFG["robustness"]["parameters"]
        self.assertEqual({(t["n"], t["k"], t["T"], t["sigma"], t["beta"], t["eps"]) for t in rob},
                         {(p["n"], p["k"], p["T"], p["sigma"], p["beta"], p["eps"])})
        self.assertEqual({(t["utility_mode"], t["noise_model"]) for t in rob},
                         {("asymmetric", "gaussian"), ("symmetric", "student_t5")})

    def test_dataset_construction(self):
        import benchmark_games as B
        d = CFG["datasets"]
        self.assertEqual(B.THRESHOLDS[B.PRIMARY_THRESHOLD], tuple(d["thresholds"]["primary"]))
        self.assertEqual({v for k, v in B.THRESHOLDS.items() if k != B.PRIMARY_THRESHOLD},
                         {tuple(x) for x in d["thresholds"]["label_sensitivity"]})
        self.assertEqual([list(c) for c in B.CONFIGS], d["configurations"])
        self.assertEqual(list(B.SIZES), d["sizes"])
        self.assertEqual(set(B.DOMAINS), set(d["encodings"]) - {"scale"})


if __name__ == "__main__":
    unittest.main()
