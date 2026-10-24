#!/usr/bin/env python3
"""
Figures for the robustness study.

    python3 scripts/make_robustness_figures.py

    robustness_yes_bars.pdf  regret and existence error on YES instances; grouped
                             bars, one colour per setting; one column per utility model
    robustness_no_bars.pdf   declarations on NO instances: horizontal bars per method,
                             one panel per setting with NO instances

Input: output/summaries/robustness.json. Each entry stores [mean, standard error,
number of instances]; the standard errors are computed over the instances of the
relevant class (YES or NO) in each condition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as M  # noqa: E402

plt = M.plt
RES = M.ROOT / "output/summaries"
OUT = M.FIG
SETTINGS = [("baseline", "Symmetric, Gaussian noise"),
            ("asymmetric", "Asymmetric, Gaussian noise"),
            ("student_t5", r"Symmetric, Student-$t_5$ noise")]
SCOL = {"baseline": "#2f6aa8", "asymmetric": "#c4721c", "student_t5": "#7a5aa6"}
MODELS = ("uniform", "gaussian")
PRIMARY = M.ARMS[0]


def load():
    return json.loads((RES / "robustness.json").read_text())


def bars(ax, rob, model, key, log):
    x = np.arange(len(M.ARMS))
    w = 0.26
    ax.axvspan(-0.5, 0.5, color="#f0f0ee", zorder=0)          # locate the proposed method
    base = 50 if log else 0
    for j, (s, lab) in enumerate(SETTINGS):
        ys = np.array([rob[f"{s}|{model}|{a}"][key][0] for a in M.ARMS])
        es = np.array([rob[f"{s}|{model}|{a}"][key][1] for a in M.ARMS])
        lo = np.minimum(es, ys - (base + 1e-9) if log else ys)  # never extend below the axis
        ax.bar(x + (j - 1) * w, ys - base, w * 0.92, bottom=base, color=SCOL[s], label=lab, zorder=2,
               edgecolor="white", linewidth=0.6)
        ax.errorbar(x + (j - 1) * w, ys, yerr=[lo, es], fmt="none", ecolor=M.INK, elinewidth=0.7,
                    capsize=1.6, zorder=3)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(base, 1.2e4)
    ax.set_xticks(x)
    ax.set_xticklabels([M.NAME[a] for a in M.ARMS], rotation=30, ha="right", fontsize=7)
    for t in ax.get_xticklabels()[:1]:
        t.set_fontweight("bold")
    ax.set_xlim(-0.5, len(M.ARMS) - 0.5)
    M.style(ax)
    ax.grid(axis="x", visible=False)


def title(rob, model):
    c = [rob[f"{s}|{model}|{PRIMARY}"]["n_yes"] for s, _ in SETTINGS]
    return f"{model.capitalize()} model (YES instances: {c[0]} / {c[1]} / {c[2]})"


def cand_A(rob, log):
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.4), sharex=True)
    for col, model in enumerate(MODELS):
        bars(axes[0, col], rob, model, "regret", log)
        bars(axes[1, col], rob, model, "eex", False)
        axes[1, col].set_ylim(0, 31)
        axes[0, col].set_title(title(rob, model), color=M.INK, fontsize=8)
    axes[0, 0].set_ylabel("regret" + (" (log scale)" if log else ""))
    axes[1, 0].set_ylabel("existence error (rounds)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "robustness_yes_bars.pdf")
    plt.close(fig)


def fig20_bars(rob):
    panels = []
    for s, _ in SETTINGS:
        for model in MODELS:
            n_no = rob[f"{s}|{model}|{PRIMARY}"]["n_no"]
            if n_no:
                short = {"baseline": "Symmetric", "asymmetric": "Asymmetric", "student_t5": r"Student-$t_5$"}[s]
                panels.append((s, model, f"{short}, {model.capitalize()}\n({n_no} NO instances)"))
    panels.sort(key=lambda p: -rob[f"{p[0]}|{p[1]}|{PRIMARY}"]["n_no"])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.75, 2.7), sharey=True)
    y = np.arange(len(M.ARMS))[::-1]
    for ax, (s, model, ttl) in zip(axes, panels):
        M.style(ax)
        ax.grid(axis="y", visible=False)
        for yi, a in zip(y, M.ARMS):
            m, e, _ = rob[f"{s}|{model}|{a}"]["enot"]
            ax.barh(yi, m, height=0.7, color=M.COLOR[a], zorder=2)
            if e > 0:
                ax.errorbar(m, yi, xerr=[[min(e, m)], [e]], fmt="none", ecolor=M.INK, elinewidth=0.7,
                            capsize=1.6, zorder=3)
            ax.text(min(m + e, 30) + 0.8, yi, "0" if m == 0 else f"{m:.1f}", va="center", ha="left",
                    fontsize=6.8, color=M.INK)
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_title(ttl, color=M.INK, fontsize=7.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([M.NAME[a] for a in M.ARMS], fontsize=7)
    fig.supxlabel("rounds with a false declaration, out of 30", fontsize=7.5, color=M.INK)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_no_bars.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rob = load()
    cand_A(rob, log=False)
    fig20_bars(rob)


if __name__ == "__main__":
    main()
