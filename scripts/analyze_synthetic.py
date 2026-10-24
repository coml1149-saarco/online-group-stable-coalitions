#!/usr/bin/env python3
"""
Analysis of the synthetic grid.

    python3 scripts/analyze_synthetic.py

Assembles every (experiment, cell, arm) of config/final_mapping.csv from its source:
  new / rerun   synthetic/trials.jsonl (by parameter set and seed)
  reuse         results/saved/synthetic/historical_trials.csv (E1 baselines, k <= 2) or,
                for Random in E2, the E1 Random trajectory of the same cell
  preserved     results/saved/synthetic/historical_trials.csv (original implementation of
                UCB-LCB-GS, not reported in the paper)
and scores it with the exact labels (YES: max gap >= eps, NO: < 0, Z otherwise).

Metrics (paper definitions): regret and E^exists on YES instances, E^notexists
on NO instances; declaration rate on Z; aggregate error (fraction of rounds with
d_t != 1{YES}) over all instances. Mean ± SEM with the class sample size.

Trial records are read from $SNS_RUNS (default: results/saved), so the analysis
can be pointed at the output of scripts/run_grid.py.
Outputs (output/summaries/): cells.csv, synthetic_tables.md, k3_checker_impact.md,
completeness.md, missing.json
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results/saved"
RUNS = Path(os.environ.get("SNS_RUNS", SAVED))
OUT = ROOT / "output/summaries"
ARMS = ["ucb_lcb_gs_witness_dev", "pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed", "random", "ucb_lcb_gs"]
NAME = {"ucb_lcb_gs_witness_dev": "Witness (primary)", "pessimistic": "Pess.-only", "optimistic": "Opt.-only",
        "ucb_only": "UCB-only", "lcb_only": "LCB-only", "mixed": "Mixed", "random": "Random",
        "ucb_lcb_gs": "Original (hist.)"}


def ms(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return (math.nan, math.nan, 0)
    return (float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else math.nan, int(x.size))


def fmt(m, d=1):
    return "–" if m[0] != m[0] else (f"{m[0]:.{d}f} ± {m[1]:.{d}f}" if m[1] == m[1] else f"{m[0]:.{d}f}")


def pkey(arm, model, n, T, k, sigma, beta, eps):
    return (arm, model, int(n), int(T), int(k), float(sigma), float(beta), float(eps))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = {}
    for r in csv.DictReader((SAVED / "labels/exact_labels.csv").open()):
        if r["setting"] == "symmetric":
            lab[(r["model"], int(r["n"]), int(r["k"]), int(r["seed"]))] = float(r["max_gap"])
    new = {}
    dupes = Counter()
    for line in (RUNS / "synthetic/trials.jsonl").open():
        r = json.loads(line)
        key = pkey(r["arm"], r["model"], r["n"], r["T"], r["k"], r["sigma"], r["beta"], r["eps"]) + (r["seed"],)
        dupes[key] += 1
        new[key] = dict(regret=r["regret"], n_d1=r["n_d1"], T=r["T"], sec=r["learner_seconds"], d_seq=r["d_seq"],
                        kind=r["kind_seq"])
    hist = {}
    for r in csv.DictReader(open(SAVED / "synthetic/historical_trials.csv")):
        T = int(r["cfg_T"])
        hist[(r["algorithm"], r["model"], r["ablation"], float(r["param_value"]), int(r["master_seed"]))] = dict(
            regret=float(r["final_cumulative_regret"]), n_d1=int(round(float(r["declared_existence_rate"]) * T)), T=T,
            sec=float(r["learner_seconds"]))
    rows = list(csv.DictReader((ROOT / "config/final_mapping.csv").open()))
    e1_random = {}
    cells, missing = [], []
    for m in rows:
        arm, model, abl, val = m["arm"], m["model"], m["ablation"], float(m["value"])
        n, T, k, eps = int(m["n"]), int(m["T"]), int(m["k"]), float(m["eps"])
        pk = pkey(arm, model, n, T, k, m["sigma"], m["beta"], eps)
        trials = []
        for s in range(100):
            if m["action"] in ("new", "rerun"):
                t = new.get(pk + (s,)); src = "new"
            elif m["experiment"] == "E1":
                t = hist.get((arm, model, abl, val, s)); src = "historical"
            else:   # E2 Random reuse: E1 Random of the same parameter set except beta
                t = new.get(pkey(arm, model, n, T, k, m["sigma"], 1.0, eps) + (s,)) if k == 3 else None
                if t is None:
                    ha = abl if abl != "beta" else "T"
                    hv = val if abl != "beta" else float(T)
                    t = hist.get((arm, model, ha, hv, s))
                src = "reused"
            if t is None:
                missing.append((m["experiment"], m["original_cell"], arm, s)); continue
            g = lab[(model, n, k, s)]
            label = "YES" if g >= eps else ("NO" if g < 0 else "Z")
            trials.append(dict(t, label=label, seed=s, max_gap=g))
        Y = [t for t in trials if t["label"] == "YES"]; N = [t for t in trials if t["label"] == "NO"]
        Z = [t for t in trials if t["label"] == "Z"]
        agg = [(t["T"] - t["n_d1"] if t["label"] == "YES" else t["n_d1"]) / t["T"] for t in trials]
        cells.append(dict(experiment=m["experiment"], cell=m["original_cell"], model=model, ablation=abl, value=val,
                          n=n, T=T, k=k, sigma=float(m["sigma"]), beta=float(m["beta"]), eps=eps, arm=arm,
                          source=m["action"], n_trials=len(trials), n_yes=len(Y), n_no=len(N), n_z=len(Z),
                          regret=ms([t["regret"] for t in Y]), eex=ms([t["T"] - t["n_d1"] for t in Y]),
                          enot=ms([t["n_d1"] for t in N]), zrate=ms([t["n_d1"] / t["T"] for t in Z]),
                          agg=ms(agg), sec=ms([t["sec"] for t in trials]), _trials=trials))
    # ---------------- cells.csv
    with (OUT / "cells.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "cell", "model", "ablation", "value", "n", "T", "k", "sigma", "beta", "eps", "arm",
                    "source", "n_trials", "n_yes", "n_no", "n_z", "regret_yes", "regret_yes_sem", "eexists_yes",
                    "eexists_yes_sem", "enotexists_no", "enotexists_no_sem", "z_decl_rate", "agg_err", "agg_err_sem",
                    "learner_seconds"])
        for c in cells:
            w.writerow([c[x] for x in ("experiment", "cell", "model", "ablation", "value", "n", "T", "k", "sigma", "beta",
                                       "eps", "arm", "source", "n_trials", "n_yes", "n_no", "n_z")]
                       + [*c["regret"][:2], *c["eex"][:2], *c["enot"][:2], c["zrate"][0], *c["agg"][:2], c["sec"][0]])
    # ---------------- tables
    md = ["# Final synthetic grid: results\n",
          "Each entry is regret / E^∃ on YES instances (mean ± SEM). E^∄ and the Z declaration rate are in the "
          "E^∄ tables and in `cells.csv`. 'Original (hist.)' is the preserved historical run of the original "
          "algorithm (restricted checker; exact for k ≤ 2), shown for reference only.\n"]
    for exp, title in (("E1", "E1: historical grid (β as in the original grid)"),
                       ("E2", "E2: theorem-aligned grid (β = 3)"), ("E3", "E3: long horizon (default cell)")):
        md.append(f"## {title}\n")
        arms = ARMS if exp == "E1" else ARMS[:-1]
        for model in ("uniform", "gaussian"):
            C = [c for c in cells if c["experiment"] == exp and c["model"] == model]
            md.append(f"### {model}\n")
            md.append("| cell | YES/NO/Z | " + " | ".join(NAME[a] for a in arms) + " |")
            md.append("|---|---|" + "---|" * len(arms))
            for cid in dict.fromkeys(c["cell"] for c in C):
                row = {c["arm"]: c for c in C if c["cell"] == cid}
                any_ = next(iter(row.values()))
                md.append(f"| {cid.split('|', 1)[1]} | {any_['n_yes']}/{any_['n_no']}/{any_['n_z']} | " + " | ".join(
                    (f"{fmt(row[a]['regret'], 0)} / {fmt(row[a]['eex'], 1)}" if a in row else "–") for a in arms) + " |")
            md.append("")
    md.append("## E^∄ on NO instances (cells with at least one NO instance)\n")
    md.append("| exp | cell | n_NO | " + " | ".join(NAME[a] for a in ARMS) + " |")
    md.append("|---|---|---|" + "---|" * len(ARMS))
    seen = set()
    for c in cells:
        if c["n_no"] and (c["experiment"], c["cell"]) not in seen:
            seen.add((c["experiment"], c["cell"]))
            row = {x["arm"]: x for x in cells if x["experiment"] == c["experiment"] and x["cell"] == c["cell"]}
            md.append(f"| {c['experiment']} | {c['cell']} | {c['n_no']} | " + " | ".join(
                fmt(row[a]["enot"], 2) if a in row else "–" for a in ARMS) + " |")
    (OUT / "synthetic_tables.md").write_text("\n".join(md) + "\n")
    # ---------------- k = 3 checker impact (E1 baselines: historical restricted vs exact rerun)
    k3 = ["# Impact of the exact checker in the k = 3 cells (E1, β = 1)\n",
          "Historical trajectories (restricted checker) versus the reruns with the exact checker, same seeds.\n",
          "| model | arm | trials with different regret | with different number of declarations | mean regret hist → exact | mean declarations hist → exact |",
          "|---|---|---|---|---|---|"]
    for model in ("uniform", "gaussian"):
        for arm in ["pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed", "random"]:
            H = [hist[(arm, model, "k", 3.0, s)] for s in range(100)]
            X = [new[pkey(arm, model, 6, 30, 3, 3.0, 1.0, 1.0) + (s,)] for s in range(100)]
            dr = sum(abs(h["regret"] - x["regret"]) > 1e-6 for h, x in zip(H, X))
            dd = sum(h["n_d1"] != x["n_d1"] for h, x in zip(H, X))
            k3.append(f"| {model} | {arm} | {dr}/100 | {dd}/100 | {np.mean([h['regret'] for h in H]):.1f} → "
                      f"{np.mean([x['regret'] for x in X]):.1f} | {np.mean([h['n_d1'] for h in H]):.2f} → "
                      f"{np.mean([x['n_d1'] for x in X]):.2f} |")
    (OUT / "k3_checker_impact.md").write_text("\n".join(k3) + "\n")
    # ---------------- completeness
    exp_keys = {pkey(m["arm"], m["model"], m["n"], m["T"], m["k"], m["sigma"], m["beta"], m["eps"])
                for m in rows if m["action"] in ("new", "rerun")}
    expected = {k + (s,) for k in exp_keys for s in range(100)}
    comp = ["# Completeness of the synthetic grid\n",
            f"- expected new/rerun trials (unique parameter sets x 100 seeds): {len(expected)}",
            f"- recorded: {len(new)}; duplicate records: {sum(v - 1 for v in dupes.values())}; "
            f"missing: {len(expected - set(new))}; unexpected: {len(set(new) - expected)}",
            f"- (experiment, cell, arm) entries: {len(cells)}; entries with fewer than 100 trials: "
            f"{sum(c['n_trials'] < 100 for c in cells)}; missing trial records: {len(missing)}"]
    (OUT / "completeness.md").write_text("\n".join(comp) + "\n")
    print("\n".join(comp))
    json.dump({"missing": missing[:50]}, (OUT / "missing.json").open("w"))


if __name__ == "__main__":
    main()
