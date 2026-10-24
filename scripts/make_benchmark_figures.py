#!/usr/bin/env python3
"""
Figures and tables for the benchmark-derived games (Appendix L), generated from the stored labels and trials.

    python3 scripts/analyze_benchmark.py
    python3 scripts/make_benchmark_figures.py

Selection rule for the example games (fixed in advance): seed 0 with n = 12 in each configuration. A learned
partition is shown only if UCB-LCB-GS, under semi-bandit feedback and appreciation of friends, declared existence
at some round; the partition is the one formed at the last such round, and its exact gap is recomputed here.
The games are regenerated from the downloaded data (scripts/fetch_benchmark_data.sh); the four example trials are
rerun locally, because the published trial records omit the partitions (run scripts/rebuild_benchmark_games.py first).

Figures (output/figures/)
  bench_games_iris_karate.pdf   Iris (two of four measurements; all four determine the distances) and the karate
                                club network, each with its sampled game and the relation matrix of that game
  bench_games_jazz.pdf          the jazz network with a uniform and a snowball sample, and their relation matrices
  bench_{semi,bandit}_{regret,error}.pdf   all seven methods, against n, per configuration (columns) and type (rows)
  bench_results.pdf             UCB-LCB-GS under both feedback models (game size and feedback comparison)
Tables (output/tables/)
  tab_bench_labels.tex, tab_bench_{semi,bandit}_{regret,error}.tex, tab_bench_intermediate.tex
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_games as B  # noqa: E402
import make_figures as M  # noqa: E402

plt = M.plt
FIG = ROOT / "output/figures"
TAB = ROOT / "output/tables"
P16 = Path(os.environ.get("SNS_RUNS", ROOT / "results/saved")) / "benchmark"
SUMMARY = ROOT / "output/summaries/benchmark_summary.json"
GAMES = Path(os.environ.get("SNS_GAMES", ROOT / "results/local")) / "benchmark"
REL = {"friend": "#2f6aa8", "enemy": "#c4721c", "neutral": "#e3e3e3"}
CLASS_COL = ["#1baf7a", "#6250d6", "#eda100"]
CLASS_MARK = ["o", "s", "^"]
CFG = [("iris", "stratified", "Iris"), ("karate", "stratified", "Karate club"), ("jazz", "uniform", "Jazz, uniform"),
       ("jazz", "snowball", "Jazz, snowball")]
TYPES = [("AF", "appreciation of friends"), ("AE", "aversion to enemies"), ("B", "balanced")]
TYPE_SHORT = {"AF": "friends", "AE": "enemies", "B": "balanced"}
PRIM = {"semi": "ucb_lcb_gs_witness_dev", "bandit": "bandit_witness"}
ARMS = M.ARMS   # UCB-LCB-GS, Pessimistic-Only, Optimistic-Only, UCB-Only, LCB-Only, Mixed, Random


def arm_key(fb, a):
    return PRIM[fb] if a == "ucb_lcb_gs_witness_dev" else a


# ------------------------------------------------------------------ example games
def declared(model, n, seed, v):
    from sns.paths import ensure_core_on_path
    ensure_core_on_path()
    from sns.full_deviation_checker import full_gap_k
    for line in (P16 / "trials.jsonl").open():
        t = json.loads(line)
        if t["model"] == model and t["n"] == n and t["seed"] == seed and t["arm"] == "ucb_lcb_gs_witness_dev":
            if "1" not in t["d_seq"]:
                return None, None
            i = len(t["d_seq"]) - 1 - t["d_seq"][::-1].index("1")
            parts = t["partitions"] if "partitions" in t else rerun_partitions(t)
            p = [ord(c) - 48 for c in parts[i]]
            g = full_gap_k(np.array(p), v, v, 2)
            assert g == t["gap_seq"][i]
            return (p, g) if g >= 1.0 else (None, g)
    return None, None


def rerun_partitions(saved):
    """The published trial records omit the per-round partitions. Rerun the trial (deterministic: the game is
    rebuilt from the downloaded data and the seeds are those of the game) and check that it reproduces the
    saved record before using its partitions."""
    import run_benchmark as RB
    import run_grid as R
    R._init_worker()
    task = next(x for x in RB.tasks(GAMES / "labels.jsonl") if RB.tid(x) == saved["task_id"])
    out = RB.run_one(task)
    assert (out["d_seq"], out["regret"], out["gap_seq"]) == (saved["d_seq"], saved["regret"], saved["gap_seq"]), \
        f"rerun of {saved['task_id']} differs from the saved record"
    return out["partitions"]


def spring_layout(n, edges, seed=3, iters=400):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    k = 1.0 / np.sqrt(n)
    A = np.zeros((n, n))
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    t = 0.1
    for _ in range(iters):
        d = pos[:, None] - pos[None]
        dist = np.linalg.norm(d, axis=-1) + 1e-9
        disp = ((k * k / dist ** 2)[..., None] * d - (A * dist / k)[..., None] * d).sum(1)
        ln = np.linalg.norm(disp, axis=1)[:, None] + 1e-9
        pos += disp / ln * np.minimum(ln, t)
        t *= 0.99
    return pos


def matrix(ax, R, order, labels, title):
    from matplotlib.colors import ListedColormap
    cm = ListedColormap([REL["enemy"], REL["neutral"], REL["friend"]]); cm.set_bad("white")
    Rm = R[np.ix_(order, order)].astype(float)
    np.fill_diagonal(Rm, np.nan)
    ax.imshow(np.ma.masked_invalid(Rm), cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if labels is not None:
        lab = np.array(labels)[order]
        for b in np.where(np.diff(lab) != 0)[0]:
            ax.axhline(b + 0.5, color="black", lw=0.6); ax.axvline(b + 0.5, color="black", lw=0.6)
    ax.set_title(title, fontsize=7.5)


def outline(ax, xy, part):
    styles = ["-", "--", ":", "-."]
    for k, c in enumerate(sorted(set(part), key=lambda c: -sum(1 for x in part if x == c))):
        idx = [i for i, x in enumerate(part) if x == c]
        if len(idx) == 1:
            continue
        ax.scatter(xy[idx, 0], xy[idx, 1], s=95, facecolors="none", edgecolors="black", linewidths=0.8,
                   linestyle=styles[k % 4], zorder=5)


def legend_rel(fig, y=-0.01):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=REL[x], label=x) for x in ("friend", "neutral", "enemy")], loc="lower center",
               ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, y))


def iris_karate_figure():
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.6))
    # Iris
    ax = axes[0, 0]
    d = B.iris()
    R, vs, meta = B.game("iris", "stratified", 12, 0)
    S = meta["agents"]; X = d["X"]
    ax.scatter(X[:, 2], X[:, 3], s=5, color="#c8c8c8", zorder=1, linewidths=0)
    xy = X[S][:, 2:4]
    for i in range(12):
        for j in range(i + 1, 12):
            if R[i, j] > 0:
                ax.plot(*xy[[i, j]].T, color=REL["friend"], lw=0.6, alpha=0.7, zorder=2)
    for c in range(3):
        idx = [i for i in range(12) if d["labels"][S[i]] == c]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=24, color=CLASS_COL[c], marker=CLASS_MARK[c], zorder=4,
                   label=d["label_names"][c].replace("Iris-", ""), edgecolors="white", linewidths=0.4)
    p, g = declared("bm_iris_stratified_AF", 12, 0, vs["AF"])
    if p is not None:
        outline(ax, xy, p)
    M.style(ax); ax.set_xlabel("petal length (cm)", fontsize=7); ax.set_ylabel("petal width (cm)", fontsize=7)
    ax.set_title("(a) Iris: sampled flowers and friend pairs", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper left", handletextpad=0.2)
    matrix(axes[1, 0], R, np.argsort(meta["labels"], kind="stable"), meta["labels"], "(c) Iris: relations, by species")
    gaps = {"iris": g}
    # Karate
    ax = axes[0, 1]
    k = B.karate()
    pos = spring_layout(34, [(a - 1, b - 1) for a, b in k["edges"]])
    for a, b in k["edges"]:
        ax.plot(*pos[[a - 1, b - 1]].T, color="#d0d0d0", lw=0.5, zorder=1)
    R, vs, meta = B.game("karate", "stratified", 12, 0)
    S = meta["agents"]
    for i in range(12):
        for j in range(i + 1, 12):
            if R[i, j] > 0:
                ax.plot(*pos[[S[i], S[j]]].T, color=REL["friend"], lw=1.4, zorder=2)
    others = [u for u in range(34) if u not in S]
    ax.scatter(pos[others, 0], pos[others, 1], s=9, color="#c8c8c8", zorder=3, linewidths=0)
    for c in range(2):
        idx = [S[i] for i in range(12) if k["labels"][S[i]] == c]
        ax.scatter(pos[idx, 0], pos[idx, 1], s=28, color=CLASS_COL[c], marker=CLASS_MARK[c], zorder=4,
                   label=k["label_names"][c], edgecolors="white", linewidths=0.4)
    p, g = declared("bm_karate_stratified_AF", 12, 0, vs["AF"])
    if p is not None:
        outline(ax, pos[S], p)
    gaps["karate"] = g
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("(b) Karate club: network and sampled members", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower left", handletextpad=0.2)
    matrix(axes[1, 1], R, np.argsort(meta["labels"], kind="stable"), meta["labels"], "(d) Karate club: relations, by group")
    legend_rel(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "bench_games_iris_karate.pdf", bbox_inches="tight")
    plt.close(fig)
    return gaps


def jazz_figure():
    j = B.jazz()
    N = j["D"].shape[0]
    pos = spring_layout(N, [(a, b) for a in range(N) for b in j["adj"][a] if a < b], seed=5, iters=300)
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.6))
    gaps = {}
    for col, (sp, title) in enumerate((("uniform", "uniform sample"), ("snowball", "snowball sample"))):
        ax = axes[0, col]
        for a in range(N):
            for b in j["adj"][a]:
                if a < b:
                    ax.plot(*pos[[a, b]].T, color="#e6e6e6", lw=0.25, zorder=1)
        R, vs, meta = B.game("jazz", sp, 12, 0)
        S = meta["agents"]
        for x in range(12):
            for y in range(x + 1, 12):
                if R[x, y] > 0:
                    ax.plot(*pos[[S[x], S[y]]].T, color=REL["friend"], lw=1.2, zorder=3)
                elif R[x, y] == 0:
                    ax.plot(*pos[[S[x], S[y]]].T, color="#8c8c8c", lw=0.6, ls=(0, (2, 2)), zorder=2)
        ax.scatter(pos[S, 0], pos[S, 1], s=22, color="#2b2b2b", zorder=4, edgecolors="white", linewidths=0.4)
        p, g = declared(f"bm_jazz_{sp}_AF", 12, 0, vs["AF"])
        gaps[sp] = g
        if p is not None:
            outline(ax, pos[S], p)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(f"({'ab'[col]}) Jazz network, {title}", fontsize=7.5)
        order = np.arange(12) if p is None else np.argsort(p, kind="stable")
        matrix(axes[1, col], R, order, None if p is None else p, f"({'cd'[col]}) relations, {title}")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color="#d0d0d0", lw=0.8, label="network edge"),
                        Line2D([], [], color=REL["friend"], lw=1.2, label="friends (sampled)"),
                        Line2D([], [], color="#8c8c8c", lw=0.8, ls=(0, (2, 2)), label="neutral (sampled)"),
                        Patch(color=REL["friend"], label="friend"), Patch(color=REL["neutral"], label="neutral"),
                        Patch(color=REL["enemy"], label="enemy")],
               loc="lower center", ncol=6, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "bench_games_jazz.pdf", bbox_inches="tight")
    plt.close(fig)
    return gaps


# ------------------------------------------------------------------ performance
def perf_figure(fb, key, ylabel, fname, log):
    S = json.loads(SUMMARY.read_text())["cells"]
    fig, axes = plt.subplots(3, 4, figsize=(6.75, 5.4), sharex=True)
    for r, (dom, dname) in enumerate(TYPES):
        for c, (ds, sp, title) in enumerate(CFG):
            ax = axes[r, c]
            for a in ARMS:
                xs, ys, es = [], [], []
                for n in (8, 10, 12):
                    cell = S[f"{ds}|{sp}|{dom}|{n}|{fb}|{arm_key(fb, a)}"][key]
                    if cell["mean"] is not None and cell["n"] >= 5:
                        xs.append(n); ys.append(cell["mean"]); es.append(cell["se"])
                if xs:
                    M.band(ax, np.array(xs), np.array(ys), np.array(es), a, ARMS[0])
            if log:
                ax.set_yscale("symlog", linthresh=10); ax.set_ylim(0, 3e4)
            else:
                ax.set_ylim(-1, 31)
            ax.set_xticks([8, 10, 12])
            M.style(ax)
            if r == 0:
                ax.set_title(title, fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f"{TYPE_SHORT[dom]}\n{ylabel}", fontsize=7)
            if r == 2:
                ax.set_xlabel("$n$", fontsize=7.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


DOM_STY = {"AF": dict(color="#2f6aa8", marker="o", label="friends"), "AE": dict(color="#c4721c", marker="^", label="enemies"),
           "B": dict(color="#6a6a6a", marker="s", label="balanced")}


def feedback_figure():
    S = json.loads(SUMMARY.read_text())["cells"]
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.6), sharex=True)
    for j, (ds, sp, title) in enumerate(CFG):
        for dom, st in DOM_STY.items():
            for fb, ls in (("semi", "-"), ("bandit", "--")):
                for i, key in enumerate(("regret", "existence_error")):
                    xs, ys, es = [], [], []
                    for n in (8, 10, 12):
                        c = S[f"{ds}|{sp}|{dom}|{n}|{fb}|{PRIM[fb]}"][key]
                        if c["mean"] is not None and c["n"] >= 5:
                            xs.append(n); ys.append(c["mean"]); es.append(c["se"])
                    axes[i, j].errorbar(xs, ys, yerr=es, color=st["color"], marker=st["marker"], ls=ls, lw=1.1,
                                        markersize=3.5, capsize=1.5,
                                        label=f"{st['label']}, {'semi-bandit' if fb == 'semi' else 'bandit'}")
        axes[0, j].set_title(title, fontsize=7.5)
        axes[0, j].set_yscale("symlog", linthresh=10)
        axes[0, j].set_ylim(0, 2e4)
        axes[1, j].set_ylim(-1, 31)
        axes[1, j].set_xlabel("$n$", fontsize=7.5)
        axes[1, j].set_xticks([8, 10, 12])
        for i in range(2):
            M.style(axes[i, j])
    axes[0, 0].set_ylabel("regret (YES games)", fontsize=7.5)
    axes[1, 0].set_ylabel("existence error (rounds)", fontsize=7.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(FIG / "bench_results.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ tables
HEAD = {"ucb_lcb_gs_witness_dev": "UCB-LCB-GS", "pessimistic": "\\textsc{Pess.}", "optimistic": "\\textsc{Opt.}",
        "ucb_only": "\\textsc{UCB-O.}", "lcb_only": "\\textsc{LCB-O.}", "mixed": "\\textsc{Mixed}", "random": "\\textsc{Random}"}


def fmt(c, d):
    if c["mean"] is None:
        return "--"
    return f"{c['mean']:,.{d}f}\\,{{\\scriptsize({c['se']:,.{d}f})}}"


def perf_table(fb, key, d, label, caption):
    S = json.loads(SUMMARY.read_text())["cells"]
    rows = []
    for ds, sp, title in CFG:
        for dom, _ in TYPES:
            for n in (8, 10, 12):
                cells = [S[f"{ds}|{sp}|{dom}|{n}|{fb}|{arm_key(fb, a)}"] for a in ARMS]
                yes = cells[0]["yes"]
                vals = [fmt(c[key], d) for c in cells]
                lead = f"{title} & {TYPE_SHORT[dom]}" if n == 8 else " & "
                rows.append(f"{lead} & {n} & {yes} & " + " & ".join(vals) + " \\\\")
            rows.append("\\addlinespace[1pt]")
        rows[-1] = "\\midrule"
    rows = rows[:-1]
    tex = ("\\begin{table}[p]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
           "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{llrr" + "r" * 7 + "}\n\\toprule\n"
           "dataset & type & $n$ & YES & " + " & ".join(HEAD[a] for a in ARMS) + " \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (TAB / f"tab_bench_{fb}_{'regret' if key == 'regret' else 'error'}.tex").write_text(tex)


def intermediate_table():
    S = json.loads(SUMMARY.read_text())["cells"]
    rows = []
    for fb, fname in (("semi", "semi-bandit"), ("bandit", "bandit")):
        for ds, sp, title in CFG:
            for dom, _ in TYPES:
                tot = {a: 0.0 for a in ARMS}
                cnt = 0
                for n in (8, 10, 12):
                    for a in ARMS:
                        c = S[f"{ds}|{sp}|{dom}|{n}|{fb}|{arm_key(fb, a)}"]["declarations_intermediate"]
                        if c["n"]:
                            tot[a] += c["mean"] * c["n"]
                    cnt += S[f"{ds}|{sp}|{dom}|{n}|{fb}|{arm_key(fb, ARMS[0])}"]["intermediate"]
                if cnt == 0:
                    continue
                vals = [f"{tot[a] / cnt:.2f}" if tot[a] / cnt < 1 else f"{tot[a] / cnt:.1f}" for a in ARMS]
                rows.append(f"{fname} & {title} & {TYPE_SHORT[dom]} & {cnt} & " + " & ".join(vals) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    tex = ("\\begin{table}[!htbp]\n\\centering\n\\caption{Declarations on intermediate games: average number of rounds, out of $30$, "
           "in which each method declares existence on a game that has an exactly $2$-strongly Nash stable partition but no "
           "$(\\varepsilon,2)$-strongly Nash stable one, pooled over $n\\in\\{8,10,12\\}$; int.: number of intermediate games. "
           "These declarations are not non-existence errors, which are defined on games without an exactly stable partition. "
           "Rows without intermediate games are omitted; abbreviations as in \\Cref{tab:bench-semi-regret}.}\n\\label{tab:bench-intermediate}\n\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
           "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lllr" + "r" * 7 + "}\n\\toprule\nfeedback & dataset & type & int. & "
           + " & ".join(HEAD[a] for a in ARMS) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (TAB / "tab_bench_intermediate.tex").write_text(tex)


def labels_table():
    S = json.loads(SUMMARY.read_text())
    L, sens, ns = S["labels"], S["sensitivity"], S["neutral_share"]
    rows = []
    for ds, sp, title in CFG:
        for dom, _ in TYPES:
            yes = [L[f"{ds}|{sp}|{n}|{dom}"]["YES"] for n in (8, 10, 12)]
            sv = []
            for t in ("0.2_0.2", "0.4_0.4"):
                keys = [f"{ds}|{sp}|{n}|{dom}" for n in (8, 10, 12)]
                ok = t in sens and all(k in sens[t] and sum(sens[t][k][x] for x in ("YES", "intermediate", "NO")) == 100 for k in keys)
                sv.append(str(sum(sens[t][k]["YES"] for k in keys)) if ok else "--")
            neutral = f"{100 * sum(ns[f'{ds}|{sp}|{n}'] for n in (8, 10, 12)) / 3:.0f}\\%" if dom == "AF" else ""
            rows.append(f"{title if dom == 'AF' else ''} & {neutral} & {TYPE_SHORT[dom]} & " + " & ".join(map(str, yes)) +
                        f" & {sum(yes)} & " + " & ".join(sv) + " \\\\")
        rows.append("\\midrule")
    body = "\n".join(rows[:-1])
    tex = r"""\begin{table}[!htbp]
\centering
\caption{Number of YES instances among the $100$ games of each configuration ($k=2$, $\varepsilon=1$); all other games are intermediate instances, and none is a NO instance. Neutral: average share of neutral pairs under the thresholds $(f,e)=(0.25,0.35)$ used for learning. The last two columns give the number of YES instances among the $300$ games of a row, labelled again under the other two thresholds of \citet{csaji2025clustering}, which have no neutral zone; no learning runs use them.}
\label{tab:bench-labels}
\small
\begin{tabular}{lrl rrr r rr}
\toprule
 & & & \multicolumn{4}{c}{$(f,e)=(0.25,0.35)$} & $(0.2,0.2)$ & $(0.4,0.4)$ \\
\cmidrule(lr){4-7}\cmidrule(lr){8-8}\cmidrule(lr){9-9}
dataset & neutral & type & $n=8$ & $n=10$ & $n=12$ & all & all & all \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB / "tab_bench_labels.tex").write_text(tex)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    g1 = iris_karate_figure()
    g2 = jazz_figure()
    print("verified example gaps:", g1, g2)
    for fb in ("semi", "bandit"):
        perf_figure(fb, "regret", "regret", f"bench_{fb}_regret.pdf", True)
        perf_figure(fb, "existence_error", "existence error", f"bench_{fb}_error.pdf", False)
    feedback_figure()
    cap = {"semi": "semi-bandit feedback", "bandit": "coalition-level bandit feedback"}
    for fb in ("semi", "bandit"):
        perf_table(fb, "regret", 0, f"tab:bench-{fb}-regret",
                   f"Regret on YES games under {cap[fb]} ($T=30$, $k=2$, $\\sigma=3$, $\\beta=1$, $\\varepsilon=1$): mean and, in parentheses, "
                   f"standard error over the YES games among the $100$ games of each row; YES gives their number. "
                   f"Abbreviations: \\textsc{{Pess.}}, \\textsc{{Opt.}}, \\textsc{{UCB-O.}} and \\textsc{{LCB-O.}} denote \\textsc{{Pessimistic-Only}}, "
                   f"\\textsc{{Optimistic-Only}}, \\textsc{{UCB-Only}} and \\textsc{{LCB-Only}}.")
        perf_table(fb, "existence_error", 1, f"tab:bench-{fb}-error",
                   f"Existence error on YES games under {cap[fb]}, in rounds out of $30$; layout and abbreviations as in "
                   f"\\Cref{{tab:bench-{fb}-regret}}.")
    intermediate_table()
    labels_table()
