#!/usr/bin/env python3
"""
Analysis of the robustness study.

    python3 scripts/analyze_robustness.py

UCB-LCB-GS (witness probing) from robust/trials.jsonl; the six baselines from
results/saved/robust/historical_trials.csv (trajectories of an earlier run whose
deviation checker is exact at n=6, k=2; scripts/run_grid.py --suite robust_smoke
reruns a seed sample). The symmetric/Gaussian setting is the default cell of the
synthetic grid. Labels: exact (asymmetric instances: their own exact labels;
Student-t5 instances share the symmetric utilities and labels).

Trial records are read from $SNS_RUNS (default: results/saved).
Outputs: output/summaries/robustness.json and robustness.md
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results/saved"
RUNS = Path(os.environ.get("SNS_RUNS", SAVED))
OUT = ROOT / "output/summaries"
ARMS = ["ucb_lcb_gs_witness_dev", "pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed", "random"]


def ms(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return [float("nan"), float("nan"), 0]
    return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan"), int(x.size)]


def summarize(trials, eps=1.0):
    Y = [t for t in trials if t["label"] == "YES"]; N = [t for t in trials if t["label"] == "NO"]
    Z = [t for t in trials if t["label"] == "Z"]
    return dict(n_yes=len(Y), n_no=len(N), n_z=len(Z), regret=ms([t["regret"] for t in Y]),
                eex=ms([t["T"] - t["n_d1"] for t in Y]), enot=ms([t["n_d1"] for t in N]),
                zrate=ms([t["n_d1"] / t["T"] for t in Z]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = {}
    for r in csv.DictReader((SAVED / "labels/exact_labels.csv").open()):
        lab[(r["setting"], r["model"], int(r["n"]), int(r["k"]), int(r["seed"]))] = float(r["max_gap"])
    L = lambda g: "YES" if g >= 1.0 else ("NO" if g < 0 else "Z")  # noqa: E731
    # ---------------- robustness
    R = defaultdict(list)
    for r in csv.DictReader(open(SAVED / "robust/historical_trials.csv")):
        if r["algorithm"] == "ucb_lcb_gs":
            continue
        setting = "asymmetric" if r["utility_mode"] == "asymmetric" else "symmetric"
        g = lab[(setting, r["model"], 6, 2, int(r["master_seed"]))]
        R[(r["arm"], r["model"], r["algorithm"])].append(dict(
            regret=float(r["final_cumulative_regret"]), n_d1=int(round(float(r["declared_existence_rate"]) * 30)),
            T=30, label=L(g)))
    for line in (RUNS / "robust/trials.jsonl").open():
        r = json.loads(line)
        arm = "asymmetric" if r["utility_mode"] == "asymmetric" else "student_t5"
        R[(arm, r["model"], r["arm"])].append(dict(regret=r["regret"], n_d1=r["n_d1"], T=r["T"], label=r["label"]))
    for line in (RUNS / "synthetic/trials.jsonl").open():
        r = json.loads(line)
        if (r["arm"] == "ucb_lcb_gs_witness_dev" and r["n"] == 6 and r["T"] == 30 and r["k"] == 2 and r["sigma"] == 3.0
                and r["beta"] == 1.0 and r["eps"] == 1.0):
            R[("baseline", r["model"], r["arm"])].append(dict(regret=r["regret"], n_d1=r["n_d1"], T=r["T"], label=r["label"]))
    rob = {f"{a}|{m}|{alg}": summarize(v) for (a, m, alg), v in R.items()}
    for k, v in rob.items():
        assert v["n_yes"] + v["n_no"] + v["n_z"] == 100, (k, v)
    (OUT / "robustness.json").write_text(json.dumps(rob, indent=1))
    md = ["# Robustness (n=6, k=2, eps=1, sigma=3, beta=1, T=30, 100 seeds)\n",
          "| arm | model | algorithm | YES/NO/Z | regret (YES) | E^exists (YES) | E^notexists (NO) | Z decl. rate |", "|---|---|---|---|---|---|---|---|"]
    for a in ("baseline", "asymmetric", "student_t5"):
        for m in ("uniform", "gaussian"):
            for alg in ARMS:
                v = rob.get(f"{a}|{m}|{alg}")
                if v:
                    md.append(f"| {a} | {m} | {alg} | {v['n_yes']}/{v['n_no']}/{v['n_z']} | {v['regret'][0]:.0f} ± {v['regret'][1]:.0f} | "
                              f"{v['eex'][0]:.2f} ± {v['eex'][1]:.2f} | {v['enot'][0]:.2f} ± {v['enot'][1]:.2f} | {v['zrate'][0]:.3f} |")
    (OUT / "robustness.md").write_text("\n".join(md) + "\n")

    print((OUT / "robustness.md").read_text())


if __name__ == "__main__":
    main()
