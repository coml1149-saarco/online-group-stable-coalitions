#!/usr/bin/env python3
"""
Figures and LaTeX tables for the synthetic experiments; also the shared plotting
conventions imported by the other figure scripts.

    python3 scripts/make_figures.py

Inputs: output/summaries/cells.csv (scripts/analyze_synthetic.py) and
robustness.json (scripts/analyze_robustness.py); the per-round records of the
long-horizon runs (synthetic/trials.jsonl) and the dataset-derived games
(benchmark/trials.jsonl, for the timing table) under $SNS_RUNS (default:
results/saved); results/saved/scalability/bench_exact_checker.json.
Outputs: output/figures/{E1,E2}_<parameter>.pdf, E3_curves_beta{1,3}.pdf and
output/tables/tab_main.tex, tab_noinst.tex, tab_cost.tex, tab_scaling.tex.

Palette: the validated categorical palette (slots 1-7, light mode); every
series also has its own marker, and every figure a legend. The original
implementation of UCB-LCB-GS is not plotted.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results/saved"
RUNS = Path(os.environ.get("SNS_RUNS", SAVED))
SUMM = ROOT / "output/summaries"
FIG = ROOT / "output/figures"
TAB = ROOT / "output/tables"
ARMS = ["ucb_lcb_gs_witness_dev", "pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed", "random"]
NAME = {"ucb_lcb_gs_witness_dev": "UCB-LCB-GS", "pessimistic": "Pessimistic-Only", "optimistic": "Optimistic-Only",
        "ucb_only": "UCB-Only", "lcb_only": "LCB-Only", "mixed": "Mixed", "random": "Random"}
TEXNAME = {a: r"\textsc{" + NAME[a] + "}" for a in ARMS}
COLOR = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
MARK = dict(zip(ARMS, ["o", "s", "^", "v", "D", "P", "X"]))
INK, MUTED, GRID = "#1f1f1f", "#5c5c5c", "#e4e4e2"
XLAB = {"T": r"horizon $T$", "n": r"number of agents $n$", "k": r"deviation size $k$", "sigma": r"noise $\sigma$",
        "eps": r"stability margin $\varepsilon$", "beta": r"confidence parameter $\beta$"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def band(ax, xs, y, e, a, primary, color=None, mark=None, label=None):
    """Mean line with markers and a shaded band of +-1 standard error; the band's
    lower edge is truncated at zero (both plotted metrics are non-negative)."""
    c = color or COLOR[a]
    ax.plot(xs, y, color=c, marker=mark or MARK[a], markersize=4.5 if a == primary else 3.5,
            linewidth=1.9 if a == primary else 1.0, label=label or NAME[a], zorder=3 if a == primary else 2)
    ax.fill_between(xs, np.maximum(y - e, 0.0), y + e, color=c, alpha=0.15, linewidth=0, zorder=1)


def load_cells():
    C = defaultdict(dict)
    for r in csv.DictReader((SUMM / "cells.csv").open()):
        C[(r["experiment"], r["model"], r["ablation"])].setdefault(float(r["value"]), {})[r["arm"]] = r
    return C


def ablation_figure(C, exp, abl):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.9), sharex="col")
    for col, model in enumerate(("uniform", "gaussian")):
        cells = C.get((exp, model, abl))
        if not cells:
            plt.close(fig); return None
        xs = sorted(cells)
        for row, (key, sem, lab) in enumerate((("regret_yes", "regret_yes_sem", "regret (YES)"),
                                               ("eexists_yes", "eexists_yes_sem", r"existence error (YES)"))):
            ax = axes[row, col]; style(ax)
            for a in ARMS:
                y = np.array([float(cells[x][a][key]) for x in xs])
                e = np.array([float(cells[x][a][sem]) if cells[x][a][sem] not in ("", "nan") else 0.0 for x in xs])
                band(ax, xs, y, e, a, ARMS[0])
            if row == 0:
                ax.set_yscale("symlog", linthresh=10); ax.set_title(model.capitalize() + " model", color=INK, fontsize=8.5)
            if col == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel(XLAB[abl])
            if abl in ("sigma", "eps", "beta"):
                ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
            else:
                ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=7, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG / f"{exp}_{abl}.pdf"; fig.savefig(out); plt.close(fig)
    return out


def long_horizon_figure():
    curves = defaultdict(list)
    for line in (RUNS / "synthetic/trials.jsonl").open():
        r = json.loads(line)
        if r["T"] == 1000 and r["label"] == "YES":
            curves[(r["beta"], r["model"], r["arm"])].append((np.cumsum(r["regret_seq"]),
                                                               np.cumsum([1 - int(c) for c in r["d_seq"]])))
    for beta in (1.0, 3.0):
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.9), sharex=True)
        for col, model in enumerate(("uniform", "gaussian")):
            for row in (0, 1):
                ax = axes[row, col]; style(ax)
                for a in ARMS:
                    Y = np.array([c[row] for c in curves[(beta, model, a)]]); t = np.arange(1, Y.shape[1] + 1)
                    m = Y.mean(0); s = Y.std(0, ddof=1) / math.sqrt(Y.shape[0])
                    ax.plot(t, m, color=COLOR[a], linewidth=1.9 if a == ARMS[0] else 1.0, label=NAME[a],
                            marker=MARK[a], markevery=125, markersize=3.5)
                    ax.fill_between(t, m - s, m + s, color=COLOR[a], alpha=0.15, linewidth=0)
                if row == 0:
                    ax.set_yscale("symlog", linthresh=10); ax.set_title(model.capitalize() + " model", color=INK, fontsize=8.5)
                    if col == 0: ax.set_ylabel("cumulative regret")
                else:
                    ax.set_xlabel("round $t$")
                    if col == 0: ax.set_ylabel("rounds with $d_t=0$")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=7, fontsize=7, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(FIG / f"E3_curves_beta{beta:g}.pdf"); plt.close(fig)


def fmt(m, s, d=0):
    if m != m:
        return "--"
    return f"{m:.{d}f}" + (f"\\,$\\pm$\\,{s:.{d}f}" if s == s else "")


def pooled(exp, abl, value, arm, C):
    """Pool the two models: weighted mean and SEM recomputed from per-model mean, SEM and counts."""
    out = {}
    for key, sem, cnt in (("regret_yes", "regret_yes_sem", "n_yes"), ("eexists_yes", "eexists_yes_sem", "n_yes"),
                          ("enotexists_no", "enotexists_no_sem", "n_no")):
        parts = []
        for model in ("uniform", "gaussian"):
            r = C[(exp, model, abl)][value][arm]
            n = int(r[cnt]); m = float(r[key]) if r[key] not in ("", "nan") else float("nan")
            s = float(r[sem]) if r[sem] not in ("", "nan") else float("nan")
            if n > 0 and m == m:
                var = (s * s * n if s == s else 0.0) * (n - 1) / n if n > 1 else 0.0  # population variance estimate
                parts.append((n, m, var))
        if not parts:
            out[key] = (float("nan"), float("nan")); continue
        N = sum(p[0] for p in parts); M = sum(p[0] * p[1] for p in parts) / N
        V = sum(p[0] * (p[2] + (p[1] - M) ** 2) for p in parts) / N
        out[key] = (M, math.sqrt(V * N / (N - 1)) / math.sqrt(N) if N > 1 else float("nan"))
    return out


def main_table(C, rob):
    rows = []
    for a in ARMS:
        e1 = pooled("E1", "T", 30.0, a, C)
        # E3 cells are keyed by T and beta; select beta=3
        e3c = {}
        for key, sem, cnt in (("regret_yes", "regret_yes_sem", "n_yes"), ("eexists_yes", "eexists_yes_sem", "n_yes")):
            parts = []
            for model in ("uniform", "gaussian"):
                r = [x for x in csv.DictReader((SUMM / "cells.csv").open()) if x["experiment"] == "E3" and x["model"] == model
                     and x["arm"] == a and float(x["T"]) == 1000 and float(x["beta"]) == 3.0][0]
                parts.append((int(r[cnt]), float(r[key]), float(r[sem])))
            N = sum(p[0] for p in parts); M = sum(p[0] * p[1] for p in parts) / N
            V = sum(p[0] * ((p[2] ** 2) * (p[0] - 1) + (p[1] - M) ** 2) for p in parts) / N
            e3c[key] = (M, math.sqrt(V * N / (N - 1)) / math.sqrt(N))
        asym = [rob[f"asymmetric|{m}|{a}"]["enot"] for m in ("uniform", "gaussian")]
        nN = sum(x[2] for x in asym); enot = sum(x[0] * x[2] for x in asym) / nN
        rows.append(f"{TEXNAME[a]} & {fmt(*e1['regret_yes'])} & {fmt(*e1['eexists_yes'], d=1)} & "
                    f"{fmt(*e3c['regret_yes'])} & {fmt(*e3c['eexists_yes'], d=1)} & {enot:.2f} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Default configuration ($n=6$, $k=2$, $\sigma=3$, $\varepsilon=1$), both utility models pooled, 100 seeds per model. Regret and existence error are averaged over YES instances (mean $\pm$ standard error). Non-existence error is averaged over the 178 NO instances of the asymmetric-utility robustness study, where false declarations can occur. Lower is better in every column.}
\label{tab:main}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{$T=30$, $\beta=1$} & \multicolumn{2}{c}{$T=1000$, $\beta=3$} & non-existence \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Algorithm & regret & existence error & regret & existence error & error (NO) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "tab_main.tex").write_text(tex)


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    C = load_cells()
    made = [ablation_figure(C, e, a) for e in ("E1", "E2") for a in ("T", "n", "k", "sigma", "eps", "beta")]
    print("figures:", [p.name for p in made if p])
    long_horizon_figure()
    rob = json.loads((SUMM / "robustness.json").read_text())
    main_table(C, rob)
    print("tables written")



# ---------------------------------------------------------------- additional appendix tables

def extra_tables():
    import statistics
    rows = list(csv.DictReader((SUMM / "cells.csv").open()))
    lines = [r"\begin{table}[h]", r"\centering\small\setlength{\tabcolsep}{3pt}",
             r"\caption{Declarations on instances without an $(\varepsilon,k)$-strongly Nash stable partition, pooled over all cells of each grid. ``NO'': mean number of rounds with $d_t=1$ per NO instance (the non-existence error). ``int.'': fraction of rounds with $d_t=1$ on intermediate instances, which neither metric scores. The counts are instance--cell pairs; the same instance appears in several cells.}",
             r"\label{tab:no-instances}", r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule",
             r"& \multicolumn{2}{c}{historical grid} & \multicolumn{2}{c}{$\beta=3$ grid} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Algorithm & NO & int. & NO & int. \\", r"\midrule"]
    cnt = {}
    for a in ARMS:
        vals = []
        for exp in ("E1", "E2"):
            R = [r for r in rows if r["experiment"] == exp and r["arm"] == a]
            nno = sum(int(r["n_no"]) for r in R); nz = sum(int(r["n_z"]) for r in R)
            no = sum(float(r["enotexists_no"]) * int(r["n_no"]) for r in R if int(r["n_no"])) / nno
            z = sum(float(r["z_decl_rate"]) * int(r["n_z"]) for r in R if int(r["n_z"])) / nz
            vals += [f"{no:.2f}", f"{z:.2f}"]; cnt[exp] = (nno, nz)
        lines.append(f"{TEXNAME[a]} & " + " & ".join(vals) + r" \\")
    lines += [r"\midrule", f"instance--cell pairs & {cnt['E1'][0]} & {cnt['E1'][1]} & {cnt['E2'][0]} & {cnt['E2'][1]} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_noinst.tex").write_text("\n".join(lines) + "\n")

    times = defaultdict(list)
    for line in (RUNS / "synthetic/trials.jsonl").open():
        r = json.loads(line)
        if r["arm"] == ARMS[0] and r["T"] <= 60:
            times[(r["n"], r["k"], "synthetic")].append(1e3 * r["learner_seconds"] / r["T"])
    for line in (RUNS / "benchmark/trials.jsonl").open():
        r = json.loads(line)
        if r["arm"] == ARMS[0] and r["feedback"] == "semi":
            times[(r["n"], r["k"], "datasets")].append(1e3 * r["learner_seconds"] / r["T"])
    C = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Measured learner time per round of \textsc{UCB-LCB-GS} with the exact deviation checker (milliseconds; median and 90th percentile over trials, $T\le60$), with $11$ trials running in parallel on the machine described in Appendix~\ref{supp:impl}.}",
         r"\label{tab:cost}", r"\begin{tabular}{lcccc}", r"\toprule", r"instances & $n$ & $k$ & median & 90th percentile \\", r"\midrule"]
    for (n, k, src) in sorted(times, key=lambda x: (x[2] != "synthetic", x[0], x[1])):
        x = times[(n, k, src)]
        C.append(f"{src} & {n} & {k} & {np.median(x):.1f} & {np.percentile(x, 90):.1f} \\\\")
    C += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_cost.tex").write_text("\n".join(C) + "\n")



def scaling_table():
    """Exact-checker scaling table from bench_exact_checker.json (idle machine, single process)."""
    B = json.loads((SAVED / "scalability/bench_exact_checker.json").read_text())["results"]
    L = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Time of one exact evaluation of ${\gap}_k(\pi,\mathbf v)$ for three partition shapes, and of one call of the partition heuristic (three restarts), measured in a single process on an otherwise idle machine. Utilities are random integers in $[-100,100]$. ``--'': not run because a single evaluation took more than two seconds.}",
         r"\label{tab:scaling}", r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$k$ & $n$ & singletons & triples & halves & heuristic call \\", r"\midrule"]
    def s(x):
        return f"{1e3*x:.1f}\\,ms" if x < 1 else f"{x:.1f}\\,s"
    for r in B:
        g = r["gap_seconds"]
        L.append(f"{r['k']} & {r['n']} & {s(g['singletons'])} & {s(g['triples'])} & {s(g['halves'])} & "
                 + (s(r["solve_seconds"]) if "solve_seconds" in r else "--") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_scaling.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
    extra_tables()
    scaling_table()
    print("synthetic tables written")
