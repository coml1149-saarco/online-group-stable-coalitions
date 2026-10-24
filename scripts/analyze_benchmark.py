#!/usr/bin/env python3
"""
Analysis of the dataset-derived games (labels and learning runs).

    python3 scripts/analyze_benchmark.py

Inputs: the game records benchmark/labels*.jsonl under $SNS_GAMES (default: results/local, written by
scripts/rebuild_benchmark_games.py from the downloaded data; use results/rerun after scripts/benchmark_labels.py)
and the trial records benchmark/trials.jsonl under $SNS_RUNS (default: results/saved).
The published trial records omit the per-round partitions; for the partition formed at the last declaration
they carry its number of coalitions and its Rand index with respect to the classes, which are used instead.
Output: output/summaries/benchmark_summary.json

Class-specific denominators: regret, existence error and time to the first stable declaration over YES games;
non-existence error over NO games (there are none); declarations on intermediate games reported separately.
Unstable declarations: rounds with d_t = 1 whose formed partition has exact Gap_2 < eps, over all games.
Means with standard errors over games; paired bandit-minus-semi-bandit differences over the same YES games.
The Rand index of a declared partition against the class labels (Iris, Karate) is descriptive only.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P16 = Path(os.environ.get("SNS_RUNS", ROOT / "results/saved")) / "benchmark"
GAMES = Path(os.environ.get("SNS_GAMES", ROOT / "results/local")) / "benchmark"
OUT = ROOT / "output/summaries"
EPS = 1.0
PRIMARY = {"semi": "ucb_lcb_gs_witness_dev", "bandit": "bandit_witness"}
CONFIGS = [("iris", "stratified"), ("karate", "stratified"), ("jazz", "uniform"), ("jazz", "snowball")]
DOMAINS = ("AF", "AE", "B")
SIZES = (8, 10, 12)


def ms(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": None, "se": None, "n": 0}
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0, "n": int(len(x))}


def rand_index(a, b):
    pairs = list(combinations(range(len(a)), 2))
    return sum((a[i] == a[j]) == (b[i] == b[j]) for i, j in pairs) / len(pairs)


def load_labels(path):
    return {(f"bm_{r['dataset']}_{r['sampler']}_{r['domain']}", r["n"], r["seed"]): r for r in map(json.loads, path.open())}


def label_summary(lab):
    out = defaultdict(lambda: {"YES": 0, "intermediate": 0, "NO": 0, "max_gap_type_units": []})
    for r in lab.values():
        k = f"{r['dataset']}|{r['sampler']}|{r['n']}|{r['domain']}"
        out[k][r["label"]] += 1
        if r["label"] == "YES":
            out[k]["max_gap_type_units"].append(r["max_gap"] / r["scale"])
    res = {}
    for k, v in out.items():
        g = v.pop("max_gap_type_units")
        res[k] = {**v, "max_gap_type_units_median": float(np.median(g)) if g else None,
                  "neutral_share": None}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (GAMES / "labels.jsonl").exists():
        raise SystemExit(f"{GAMES / 'labels.jsonl'} not found: run  python3 scripts/rebuild_benchmark_games.py  first")
    lab = load_labels(GAMES / "labels.jsonl")
    for r in lab.values():
        npairs = r["n"] * (r["n"] - 1) // 2
        r["neutral_share"] = r["neutral_pairs"] / npairs
    res = {"labels": label_summary(lab), "sensitivity": {}}
    for t in ("0.2_0.2", "0.4_0.4"):
        p = GAMES / f"labels_{t}.jsonl"
        if p.exists():
            res["sensitivity"][t] = label_summary(load_labels(p))
    # neutral shares per configuration
    ns = defaultdict(list)
    for r in lab.values():
        ns[f"{r['dataset']}|{r['sampler']}|{r['n']}"].append(r["neutral_share"])
    res["neutral_share"] = {k: float(np.mean(v)) for k, v in ns.items()}

    with (P16 / "trials.jsonl").open() as fh:
        trials = [json.loads(l) for l in fh]
    by = defaultdict(list)
    for t in trials:
        L = lab[(t["model"], t["n"], t["seed"])]
        t["_lab"] = L
        by[(L["dataset"], L["sampler"], L["domain"], t["n"], t["feedback"], t["arm"])].append(t)
    cells = {}
    for (ds, sp, dom, n, fb, arm), ts in by.items():
        yes = [t for t in ts if t["label"] == "YES"]
        inter = [t for t in ts if t["label"] == "intermediate"]
        no = [t for t in ts if t["label"] == "NO"]
        unstable = [sum(1 for d, g in zip(t["d_seq"], t["gap_seq"]) if d == "1" and g < EPS) for t in ts]
        first = []
        for t in yes:
            fs = next((i + 1 for i, (d, g) in enumerate(zip(t["d_seq"], t["gap_seq"])) if d == "1" and g >= EPS), None)
            if fs is not None:
                first.append(fs)
        cells[f"{ds}|{sp}|{dom}|{n}|{fb}|{arm}"] = {
            "games": len(ts), "yes": len(yes), "intermediate": len(inter), "no": len(no),
            "regret": ms([t["regret"] for t in yes]), "existence_error": ms([t["n_d0"] for t in yes]),
            "nonexistence_error": ms([t["n_d1"] for t in no]),
            "declarations_intermediate": ms([t["n_d1"] for t in inter]),
            "unstable_declarations": ms(unstable), "games_with_unstable_declaration": int(sum(u > 0 for u in unstable)),
            "first_stable_declaration": ms(first),
            "ms_per_round_median": float(np.median([t["learner_seconds"] / t["T"] * 1000 for t in ts])),
            "ms_per_round_p90": float(np.percentile([t["learner_seconds"] / t["T"] * 1000 for t in ts], 90))}
    res["cells"] = cells
    # paired bandit - semi for the primary method on the same YES games
    paired = {}
    for ds, sp in CONFIGS:
        for dom in DOMAINS:
            for n in SIZES:
                s = {t["seed"]: t for t in by.get((ds, sp, dom, n, "semi", PRIMARY["semi"]), []) if t["label"] == "YES"}
                b = {t["seed"]: t for t in by.get((ds, sp, dom, n, "bandit", PRIMARY["bandit"]), []) if t["label"] == "YES"}
                common = sorted(set(s) & set(b))
                paired[f"{ds}|{sp}|{dom}|{n}"] = {
                    "regret_diff": ms([b[x]["regret"] - s[x]["regret"] for x in common]),
                    "existence_error_diff": ms([b[x]["n_d0"] - s[x]["n_d0"] for x in common])}
    res["paired_bandit_minus_semi"] = paired
    # declared partitions of the primary method (last round with d_t = 1)
    decl = {}
    for ds, sp in CONFIGS:
        for dom in DOMAINS:
            for n in SIZES:
                for fb in ("semi", "bandit"):
                    rows = []
                    for t in by.get((ds, sp, dom, n, fb, PRIMARY[fb]), []):
                        if "1" not in t["d_seq"]:
                            continue
                        i = len(t["d_seq"]) - 1 - t["d_seq"][::-1].index("1")
                        cls = t["_lab"]["labels"]
                        if "partitions" in t:
                            p = [ord(c) - 48 for c in t["partitions"][i]]
                            coal, ri = max(p) + 1, (rand_index(p, cls) if cls is not None else None)
                        else:   # published records: precomputed from the same partition
                            coal, ri = t["last_declared_coalitions"], t["last_declared_rand_index"]
                        rows.append({"gap": t["gap_seq"][i], "coalitions": coal, "rand_index": ri})
                    decl[f"{ds}|{sp}|{dom}|{n}|{fb}"] = {
                        "declared_games": len(rows), "all_declared_stable": all(r["gap"] >= EPS for r in rows),
                        "min_gap": min((r["gap"] for r in rows), default=None),
                        "coalitions": ms([r["coalitions"] for r in rows]),
                        "rand_index": ms([r["rand_index"] for r in rows if r["rand_index"] is not None])}
    res["declared_partitions"] = decl
    (OUT / "benchmark_summary.json").write_text(json.dumps(res, indent=1))
    print(f"{len(trials)} trials; cells {len(cells)}")


if __name__ == "__main__":
    main()
