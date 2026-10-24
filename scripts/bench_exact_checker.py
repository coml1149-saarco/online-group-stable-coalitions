#!/usr/bin/env python3
"""
Single-process scaling benchmark of exact gap evaluation and of one SOLVE call
(computational-cost appendix). Run on an otherwise idle machine.

    python3 scripts/bench_exact_checker.py

For k in {2, 3} and increasing n: random symmetric integer utilities in
[-100, 100] and three partition shapes (all singletons; coalitions of size 3;
coalitions of size n/2). Reports the time of one exact evaluation of
Gap_k(pi, v, v) (median of repeats) and the number of deviations enumerated,
and the time of one SOLVE call (R = 3 restarts, M = 60 sweeps) with the exact
checker installed. A size is skipped once a single evaluation exceeds 60 s.
Output: results/rerun/scalability/bench_exact_checker.json
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sns.paths import ensure_core_on_path  # noqa: E402

ensure_core_on_path()
import ashg_lib  # noqa: E402
from sns.full_deviation_checker import full_gap_k  # noqa: E402

ashg_lib.gap_k = full_gap_k
from ashg_lib import Partition  # noqa: E402

LIMIT = 60.0


def shapes(n):
    return {"singletons": np.arange(n), "triples": np.arange(n) // 3, "halves": (np.arange(n) >= n // 2).astype(int)}


def main():
    res = []
    rng = np.random.default_rng(0)
    for k in (2, 3):
        stop = False
        for n in (6, 8, 10, 12, 16, 20, 24, 32):
            if stop:
                break
            v = rng.integers(-100, 101, (n, n)).astype(float); v = np.triu(v, 1); v = v + v.T
            row = dict(k=k, n=n, gap_seconds={})
            for name, cid in shapes(n).items():
                p = Partition(cid)
                t0 = time.perf_counter(); full_gap_k(p, v, v, k); dt = time.perf_counter() - t0
                reps = [dt]
                while sum(reps) < 2.0 and len(reps) < 5 and dt < 5:
                    t0 = time.perf_counter(); full_gap_k(p, v, v, k); reps.append(time.perf_counter() - t0)
                row["gap_seconds"][name] = float(np.median(reps))
                if max(reps) > LIMIT:
                    stop = True
            if not stop and max(row["gap_seconds"].values()) < 2.0:
                t0 = time.perf_counter()
                ashg_lib.compute_partition("optimistic", v + 3, v - 3, k=k, eps=1.0, rng=np.random.default_rng(1),
                                           n_restarts=3, max_iter=60, deviation_cap=64)
                row["solve_seconds"] = time.perf_counter() - t0
            res.append(row)
            print(json.dumps(row), flush=True)
    out = ROOT / "results/rerun/scalability/bench_exact_checker.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(machine=platform.platform() + ", single process", results=res), indent=1))


if __name__ == "__main__":
    main()
