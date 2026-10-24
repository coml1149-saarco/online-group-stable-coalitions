#!/usr/bin/env python3
"""
Exact existence labels for the synthetic grid and the robustness study.

    python3 scripts/compute_exact_labels.py [--workers 8]

For every distinct (setting, model, n, seed, k) this computes, by exhaustive
search over all partitions,

    max_gap = max_pi Gap_k(pi, v)

with the exact deviation checker (sns.full_deviation_checker).
The three-way label for a threshold eps then follows the paper's metric
definitions:

    YES  max_gap >= eps   (SNS_{eps,k} non-empty: counts towards E^exists and regret)
    NO   max_gap <  0     (SNS_{0,k} empty: counts towards E^notexists)
    Z    otherwise        (intermediate region: counted by neither metric)

Instances are regenerated with the experiments' own seeds (env seed = s), so
the labels refer to exactly the utility matrices used in the experiments.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "rerun" / "labels" / "exact_labels.csv"


def instance(setting: str, model: str, n: int, seed: int):
    from sns.paths import ensure_core_on_path
    ensure_core_on_path()
    from ashg_lib import ASHGEnvironment
    from sns.environments import RobustASHGEnvironment
    if setting == "symmetric":
        return ASHGEnvironment(n=n, model=model, noise_std=3.0, seed=seed).v
    return RobustASHGEnvironment(n=n, model=model, noise_std=3.0, seed=seed,
                                 utility_mode="asymmetric", noise_model="gaussian").v


def work(task):
    from sns.full_deviation_checker import exact_existence
    setting, model, n, seed, k = task
    t0 = time.perf_counter()
    mg, arg = exact_existence(instance(setting, model, n, seed), k)
    return dict(setting=setting, model=model, n=n, seed=seed, k=k, max_gap=mg,
                argmax="".join(map(str, arg)), seconds=round(time.perf_counter() - t0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = []
    for model in ("uniform", "gaussian"):
        for s in range(100):
            for n in (4, 5, 6, 7):
                tasks.append(("symmetric", model, n, s, 2))
            for k in (1, 3):
                tasks.append(("symmetric", model, 6, s, k))
            tasks.append(("asymmetric", model, 6, s, 2))
    tasks.sort(key=lambda x: -(x[2] ** 2) * x[4])  # longest first
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        rows = pool.map(work, tasks, chunksize=4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["setting"], r["model"], r["n"], r["k"], r["seed"]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} instances in {(time.perf_counter()-t0)/60:.1f} min -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
