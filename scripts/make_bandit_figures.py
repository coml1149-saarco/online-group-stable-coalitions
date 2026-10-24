#!/usr/bin/env python3
"""
Figures and tables for the coalition-level feedback experiments.

    python3 scripts/make_bandit_figures.py

Inputs: output/summaries/bandit_cells.csv and bandit_cross.json
(analyze_bandit_grid.py). Same palette, markers and conventions as
make_figures.py; the bandit variant of UCB-LCB-GS takes the primary's slot.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as M  # noqa: E402

ROOT = M.ROOT
RES = ROOT / "output/summaries"
ARMS = ["bandit_witness"] + M.ARMS[1:]
NAME = dict(M.NAME, bandit_witness="UCB-LCB-GS")
COLOR = dict(M.COLOR, bandit_witness=M.COLOR[M.ARMS[0]])
MARK = dict(M.MARK, bandit_witness=M.MARK[M.ARMS[0]])
TEX = dict(M.TEXNAME, bandit_witness=r"\textsc{UCB-LCB-GS}")


def load():
    C = defaultdict(dict)
    for r in csv.DictReader((RES / "bandit_cells.csv").open()):
        C[(r["experiment"], r["model"], r["ablation"], float(r["beta"]))].setdefault(float(r["value"]), {})[r["arm"]] = r
    return C


def figure(C, abl):
    plt = M.plt
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.9), sharex="col")
    for col, model in enumerate(("uniform", "gaussian")):
        cells = C[("E1", model, abl, 1.0)] if abl != "beta" else {}
        if abl == "beta":
            for b in (0.25, 0.5, 1.0, 2.0, 4.0):
                cells.update(C[("E1", model, abl, b)])
        xs = sorted(cells)
        for row, (key, sem, lab) in enumerate((("regret_yes", "regret_yes_sem", "regret (YES)"),
                                               ("eexists_yes", "eexists_yes_sem", "existence error (YES)"))):
            ax = axes[row, col]; M.style(ax)
            for a in ARMS:
                y = np.array([float(cells[x][a][key]) for x in xs])
                e = np.array([float(cells[x][a][sem]) if cells[x][a][sem] not in ("", "nan") else 0.0 for x in xs])
                M.band(ax, xs, y, e, a, ARMS[0], color=COLOR[a], mark=MARK[a], label=NAME[a])
            if row == 0:
                ax.set_yscale("symlog", linthresh=10); ax.set_title(model.capitalize() + " model", color=M.INK, fontsize=8.5)
            if col == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel(M.XLAB[abl])
            if abl in ("sigma", "eps", "beta"):
                ax.set_xscale("log")
            ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=7, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(M.FIG / f"bandit_E1_{abl}.pdf"); plt.close(fig)


def pooled(rows, key, sem, cnt):
    parts = [(int(r[cnt]), float(r[key]), float(r[sem]) if r[sem] not in ("", "nan") else 0.0) for r in rows if int(r[cnt])]
    N = sum(p[0] for p in parts); Mn = sum(p[0] * p[1] for p in parts) / N
    V = sum(p[0] * ((p[2] ** 2) * (p[0] - 1) + (p[1] - Mn) ** 2) for p in parts) / N
    return Mn, math.sqrt(V * N / (N - 1)) / math.sqrt(N)


def tables():
    R = list(csv.DictReader((RES / "bandit_cells.csv").open()))
    sel = lambda exp, T, b, a: [r for r in R if r["experiment"] == exp and int(r["T"]) == T and float(r["beta"]) == b  # noqa: E731
                                and r["arm"] == a and (exp == "E3" or r["ablation"] == "T") and int(r["n"]) == 6 and int(r["k"]) == 2
                                and float(r["sigma"]) == 3.0 and float(r["eps"]) == 1.0]
    rows = []
    for a in ARMS:
        cells = []
        for exp, T, b in (("E1", 30, 1.0), ("E3", 1000, 3.0)):
            rr = sel(exp, T, b, a)
            cells += [M.fmt(*pooled(rr, "regret_yes", "regret_yes_sem", "n_yes")),
                      M.fmt(*pooled(rr, "eexists_yes", "eexists_yes_sem", "n_yes"), d=1)]
        E = [r for r in R if r["experiment"] in ("E1", "E2") and r["arm"] == a]
        nno = sum(int(r["n_no"]) for r in E)
        no = sum(float(r["enotexists_no"]) * int(r["n_no"]) for r in E if int(r["n_no"]) and r["enotexists_no"] not in ("", "nan")) / nno
        rows.append(f"{TEX[a]} & " + " & ".join(cells) + f" & {no:.2f} \\\\")
    t = [r"\begin{table}[h]", r"\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{Coalition-level feedback, default configuration, both utility models pooled, 100 seeds per model (mean $\pm$ standard error over YES instances). The last column is the non-existence error pooled over the NO instances of both grids. \textsc{Random} is identical under both feedback models.}",
         r"\label{tab:bandit-main}", r"\begin{tabular}{lccccc}", r"\toprule",
         r"& \multicolumn{2}{c}{$T=30$, $\beta=1$} & \multicolumn{2}{c}{$T=1000$, $\beta=3$} & non-existence \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Algorithm & regret & existence error & regret & existence error & error (NO) \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M.TAB / "tab_bandit_main.tex").write_text("\n".join(t) + "\n")

    X = json.loads((RES / "bandit_cross.json").read_text())
    cr = [c for c in X["cross"] if c["n"] == 6 and c["k"] == 2 and (c["experiment"] == "E3" or c["cell"].endswith(("|T=15", "|T=30", "|T=45", "|T=60")))
          and (c["experiment"] != "E2")]
    L = [r"\begin{table}[h]", r"\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{\textsc{UCB-LCB-GS} under pairwise and coalition-level feedback on the same instances (default configuration, YES instances; mean $\pm$ standard error of the paired difference).}",
         r"\label{tab:bandit-cross}", r"\begin{tabular}{llcccccc}", r"\toprule",
         r"model & $(T,\beta)$ & \multicolumn{3}{c}{regret} & \multicolumn{3}{c}{existence error} \\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"& & pairwise & coalition-level & difference & pairwise & coalition-level & difference \\", r"\midrule"]
    for c in sorted(cr, key=lambda c: (c["model"] != "uniform", c["beta"], c["T"])):
        L.append(f"{c['model'].capitalize()} & $({c['T']},{c['beta']:g})$ & {c['regret_semi'][0]:.0f} & {c['regret_bandit'][0]:.0f} & "
                 f"{M.fmt(c['d_regret'][0], c['d_regret'][1])} & {c['eex_semi'][0]:.1f} & {c['eex_bandit'][0]:.1f} & {M.fmt(c['d_eex'][0], c['d_eex'][1], 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M.TAB / "tab_bandit_cross.tex").write_text("\n".join(L) + "\n")

    S = X["shared"]
    own = {(r["arm"], r["model"], int(r["T"]), float(r["beta"])): r for r in R
           if r["experiment"] in ("E1", "E2", "E3") and int(r["n"]) == 6 and int(r["k"]) == 2 and float(r["sigma"]) == 3.0
           and float(r["eps"]) == 1.0 and (r["ablation"] == "T" or r["experiment"] == "E3")}
    L = [r"\begin{table}[h]", r"\centering\small\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Coalition-level feedback: baselines with their own exploration and with the shared matching schedule (regret / existence error on YES instances, Uniform model, default configuration). \textsc{UCB-LCB-GS} is shown for reference.}",
         r"\label{tab:bandit-shared}", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Algorithm & $T=30$, $\beta=1$ & $T=30$, $\beta=3$ & $T=300$, $\beta=1$ & $T=300$, $\beta=3$ \\", r"\midrule"]
    def cell(r):
        return f"{float(r['regret_yes']):.0f} / {float(r['eexists_yes']):.1f}"
    L.append(r"\textsc{UCB-LCB-GS} & " + " & ".join(cell(own[("bandit_witness", "uniform", T, b)]) for T, b in ((30, 1.0), (30, 3.0), (300, 1.0), (300, 3.0))) + r" \\")
    for a in ("pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed"):
        L.append(f"{TEX[a]}, own & " + " & ".join(cell(own[(a, "uniform", T, b)]) for T, b in ((30, 1.0), (30, 3.0), (300, 1.0), (300, 3.0))) + r" \\")
        L.append(f"{TEX[a]}, shared & " + " & ".join(
            f"{S[f'shared_{a}|uniform|{T}|{b}']['regret'][0]:.0f} / {S[f'shared_{a}|uniform|{T}|{b}']['eex'][0]:.1f}"
            for T, b in ((30, 1.0), (30, 3.0), (300, 1.0), (300, 3.0))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (M.TAB / "tab_bandit_shared.tex").write_text("\n".join(L) + "\n")


def main():
    C = load()
    for abl in ("T", "n", "k", "sigma", "eps", "beta"):
        figure(C, abl)
    tables()
    print("bandit figures and tables written")


if __name__ == "__main__":
    main()
