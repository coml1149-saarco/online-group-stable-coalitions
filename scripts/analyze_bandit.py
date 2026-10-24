#!/usr/bin/env python3
"""
Analysis of the coalition-level (per-agent) bandit-feedback grid.

    python3 scripts/analyze_bandit.py

Bandit trials: bandit_grid/trials.jsonl (the bandit variant of UCB-LCB-GS, five
adapted baselines, shared-exploration variants). Random: reused from the
semi-bandit cells (its play does not depend on feedback). Cells follow
config/final_mapping.csv (E1, E2, E3). Cross-feedback comparison: paired by
instance (same model, parameters, seed) with UCB-LCB-GS under semi-bandit
feedback from synthetic/trials.jsonl.

Trial records are read from $SNS_RUNS (default: results/saved); the semi-bandit
cells from output/summaries/cells.csv (run scripts/analyze_synthetic.py first).
Outputs (output/summaries/): bandit_cells.csv, bandit.md, bandit_cross.json
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
BARMS = ["bandit_witness", "pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed"]


def ms(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan"), int(x.size))


def key(arm, model, n, T, k, sigma, beta, eps):
    return (arm, model, int(n), int(T), int(k), float(sigma), float(beta), float(eps))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    B, dup = {}, 0
    for line in (RUNS / "bandit_grid/trials.jsonl").open():
        r = json.loads(line)
        k = key(r["arm"], r["model"], r["n"], r["T"], r["k"], r["sigma"], r["beta"], r["eps"]) + (r["seed"],)
        dup += k in B
        B[k] = r
    S = {}
    for line in (RUNS / "synthetic/trials.jsonl").open():
        r = json.loads(line)
        if r["arm"] == "ucb_lcb_gs_witness_dev":
            S[key("w", r["model"], r["n"], r["T"], r["k"], r["sigma"], r["beta"], r["eps"]) + (r["seed"],)] = r
    semi_cells = {(r["experiment"], r["cell"], r["arm"]): r for r in csv.DictReader((OUT / "cells.csv").open())}
    rows, cross, missing = [], [], 0
    for m in csv.DictReader((ROOT / "config/final_mapping.csv").open()):
        if m["arm"] != "ucb_lcb_gs_witness_dev":
            continue
        p = (m["model"], m["n"], m["T"], m["k"], m["sigma"], m["beta"], m["eps"])
        for arm in BARMS:
            tr = [B.get(key(arm, *p) + (s,)) for s in range(100)]
            missing += sum(t is None for t in tr)
            tr = [t for t in tr if t is not None]
            Y = [t for t in tr if t["label"] == "YES"]; N = [t for t in tr if t["label"] == "NO"]; Z = [t for t in tr if t["label"] == "Z"]
            rows.append(dict(experiment=m["experiment"], cell=m["original_cell"], model=m["model"], ablation=m["ablation"],
                             value=float(m["value"]), n=int(m["n"]), T=int(m["T"]), k=int(m["k"]), sigma=float(m["sigma"]),
                             beta=float(m["beta"]), eps=float(m["eps"]), arm=arm, n_yes=len(Y), n_no=len(N), n_z=len(Z),
                             regret=ms([t["regret"] for t in Y]), eex=ms([t["n_d0"] for t in Y]),
                             enot=ms([t["n_d1"] for t in N]), zrate=ms([t["n_d1"] / t["T"] for t in Z]),
                             sec=ms([1e3 * t["learner_seconds"] / t["T"] for t in tr])))
        sr = semi_cells[(m["experiment"], m["original_cell"], "random")]
        rows.append(dict(experiment=m["experiment"], cell=m["original_cell"], model=m["model"], ablation=m["ablation"],
                         value=float(m["value"]), n=int(m["n"]), T=int(m["T"]), k=int(m["k"]), sigma=float(m["sigma"]),
                         beta=float(m["beta"]), eps=float(m["eps"]), arm="random", n_yes=int(sr["n_yes"]), n_no=int(sr["n_no"]),
                         n_z=int(sr["n_z"]), regret=(float(sr["regret_yes"]), float(sr["regret_yes_sem"]), int(sr["n_yes"])),
                         eex=(float(sr["eexists_yes"]), float(sr["eexists_yes_sem"]), int(sr["n_yes"])),
                         enot=(float(sr["enotexists_no"]) if sr["enotexists_no"] not in ("", "nan") else float("nan"), float("nan"), int(sr["n_no"])),
                         zrate=(float(sr["z_decl_rate"]) if sr["z_decl_rate"] not in ("", "nan") else float("nan"), float("nan"), int(sr["n_z"])),
                         sec=(0.0, 0.0, 100)))
        dr, de, rs, rb, es, eb = [], [], [], [], [], []
        for s in range(100):
            b = B.get(key("bandit_witness", *p) + (s,)); w = S.get(key("w", *p) + (s,))
            if b and w and b["label"] == "YES":
                dr.append(b["regret"] - w["regret"]); de.append(b["n_d0"] - w["n_d0"])
                rs.append(w["regret"]); rb.append(b["regret"]); es.append(w["n_d0"]); eb.append(b["n_d0"])
        cross.append(dict(experiment=m["experiment"], cell=m["original_cell"], model=m["model"], T=int(m["T"]), beta=float(m["beta"]),
                          n=int(m["n"]), k=int(m["k"]), n_pairs=len(dr), regret_semi=ms(rs), regret_bandit=ms(rb),
                          eex_semi=ms(es), eex_bandit=ms(eb), d_regret=ms(dr), d_eex=ms(de)))
    shared = defaultdict(list)
    for (arm, model, n, T, k, sigma, beta, eps, s), r in B.items():
        if arm.startswith("shared_") and r["label"] == "YES":
            shared[(arm, model, T, beta)].append(r)
    with (OUT / "bandit_cells.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "cell", "model", "ablation", "value", "n", "T", "k", "sigma", "beta", "eps", "arm", "n_yes", "n_no", "n_z",
                    "regret_yes", "regret_yes_sem", "eexists_yes", "eexists_yes_sem", "enotexists_no", "enotexists_no_sem", "z_decl_rate",
                    "learner_ms_per_round"])
        for r in rows:
            w.writerow([r[x] for x in ("experiment", "cell", "model", "ablation", "value", "n", "T", "k", "sigma", "beta", "eps", "arm",
                                       "n_yes", "n_no", "n_z")] + [*r["regret"][:2], *r["eex"][:2], *r["enot"][:2], r["zrate"][0], r["sec"][0]])
    (OUT / "bandit_cross.json").write_text(json.dumps(dict(cross=cross, shared={f"{a}|{m}|{T}|{b}": dict(
        regret=ms([t["regret"] for t in v]), eex=ms([t["n_d0"] for t in v])) for (a, m, T, b), v in shared.items()}), indent=1))
    print(f"bandit trials {len(B)}; duplicates {dup}; missing in mapped cells {missing}; cells {len(rows)}")
    # quick summaries
    allc = [r for r in rows if r["experiment"] in ("E1", "E2")]
    G = defaultdict(dict)
    for r in allc:
        G[(r["experiment"], r["cell"])][r["arm"]] = r
    best = sum(min(v, key=lambda a: v[a]["regret"][0]) == "bandit_witness" for v in G.values())
    sound_best = sum(min(("bandit_witness", "pessimistic", "random"), key=lambda a: v[a]["eex"][0]) == "bandit_witness" for v in G.values())
    fd = [(c, v["bandit_witness"]["enot"][0]) for c, v in G.items() if v["bandit_witness"]["n_no"] and v["bandit_witness"]["enot"][0] > 0]
    print(f"bandit primary lowest regret in {best}/{len(G)} cells; lowest E^exists among sound methods in {sound_best}/{len(G)}; "
          f"cells with false declarations: {fd}")
    worse = [(c["cell"], c["T"], c["beta"], round(c["d_regret"][0]), round(c["d_eex"][0], 1)) for c in cross if c["experiment"] in ("E1", "E3")]
    print("cross-feedback (bandit - semi), default-ish cells:", [w for w in worse if "T=" in w[0] or "default" in w[0]][:12])
    for k2, v in sorted(shared.items()):
        print("shared", k2, round(np.mean([t["regret"] for t in v])), round(np.mean([t["n_d0"] for t in v]), 1))


if __name__ == "__main__":
    main()
