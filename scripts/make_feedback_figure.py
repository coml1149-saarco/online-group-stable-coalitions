#!/usr/bin/env python3
"""
Figure comparing the two feedback models (feedback_comparison.pdf).

    python3 scripts/make_feedback_figure.py

Input: output/summaries/bandit_cross.json. Conventions as in make_figures.py
(validated palette, markers, legend; shaded +-1 SEM bands).
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


def cross():
    X = json.loads((RES / "bandit_cross.json").read_text())["cross"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    styles = {("uniform", "semi"): ("#2a78d6", "o", "-"), ("uniform", "bandit"): ("#eb6834", "o", "-"),
              ("gaussian", "semi"): ("#2a78d6", "s", "--"), ("gaussian", "bandit"): ("#eb6834", "s", "--")}
    for model in ("uniform", "gaussian"):
        cs = sorted([c for c in X if c["model"] == model and c["n"] == 6 and c["k"] == 2 and c["beta"] == 1.0
                     and (c["experiment"] == "E3" or c["cell"].endswith(("|T=15", "|T=30", "|T=45", "|T=60")))], key=lambda c: c["T"])
        Ts = [c["T"] for c in cs]
        for col, (k1, k2, lab) in enumerate((("regret_semi", "regret_bandit", "regret (YES)"), ("eex_semi", "eex_bandit", "existence error (YES)"))):
            ax = axes[col]; M.style(ax)
            for fb, k in (("semi", k1), ("bandit", k2)):
                c_, m_, ls = styles[(model, fb)]
                y = np.array([c[k][0] for c in cs]); e = np.array([c[k][1] for c in cs])
                ax.plot(Ts, y, color=c_, marker=m_, linestyle=ls, linewidth=1.3, markersize=3.5,
                        label=f"{model.capitalize()}, {'pairwise' if fb == 'semi' else 'coalition-level'}")
                ax.fill_between(Ts, np.maximum(y - e, 0), y + e, color=c_, alpha=0.15, linewidth=0)
            ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts])
            if col == 0: ax.set_yscale("log")
            ax.set_xlabel("horizon $T$"); ax.set_title(lab, color=M.INK, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(M.FIG / "feedback_comparison.pdf"); plt.close(fig)


if __name__ == "__main__":
    M.FIG.mkdir(parents=True, exist_ok=True); cross(); print("feedback comparison figure written")
