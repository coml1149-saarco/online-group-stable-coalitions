#!/usr/bin/env python3
"""
Compare rerun trials with the saved trial records.

    python3 scripts/compare_reruns.py --suite synthetic|historical|smoke|robust|robust_smoke|bandit_grid|benchmark
                                      [--rerun results/rerun]

Every trial in <rerun>/<suite>/trials.jsonl is matched with the saved record of the same
configuration and seed:
  synthetic, robust, bandit_grid   results/saved/<suite>/trials.jsonl: final regret and the
                                   per-round declaration sequence must be identical
  benchmark                        results/saved/benchmark/trials.jsonl (by task id): final regret and the
                                   per-round declaration, regret and gap sequences must be identical, and the
                                   number of coalitions of the partition formed at the last declaration
                                   (the published records omit the partitions themselves)
  historical, robust_smoke         results/saved/{synthetic,robust}/historical_trials.csv: final
                                   regret and number of declarations (only these are stored)
  smoke                            baselines and the original implementation against the historical
                                   trials; UCB-LCB-GS against results/saved/synthetic/trials.jsonl
  labels                           results/rerun/labels/exact_labels.csv against results/saved/labels/exact_labels.csv
                                   (maximal gap and maximizing partition of every instance)
  benchmark_labels                 a game-record file written by benchmark_labels.py (--file) against the published
                                   per-game scalars and the digests of the sampled agents and utility matrix
Exit status 1 if any trial or record differs or has no saved counterpart.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "results" / "saved"
CFG = ("arm", "feedback", "utility_mode", "noise_model", "model", "n", "T", "k", "sigma", "beta", "eps", "seed")


def jsonl(p):
    return [json.loads(line) for line in Path(p).open()]


def key(r):
    return tuple(r[k] for k in CFG)


def historical(kind):
    """{(algorithm, utility_mode, noise_model, model, n, T, k, sigma, beta, eps, seed): [(regret, n_d1), ...]}"""
    path = SAVED / ("synthetic" if kind == "synthetic" else "robust") / "historical_trials.csv"
    H = {}
    for r in csv.DictReader(path.open()):
        T = int(r["cfg_T"])
        k = (r["algorithm"], r.get("utility_mode", "symmetric"), r.get("noise_model", "gaussian"), r["model"],
             int(r["cfg_n"]), T, int(r["cfg_k"]), float(r["cfg_sigma"]), float(r["cfg_beta"]), float(r["cfg_eps"]),
             int(r["master_seed"]))
        H.setdefault(k, []).append((float(r["final_cumulative_regret"]), round(float(r["declared_existence_rate"]) * T)))
    return H


def hkey(r):
    return (r["arm"], r["utility_mode"], r["noise_model"], r["model"], r["n"], r["T"], r["k"], float(r["sigma"]),
            float(r["beta"]), float(r["eps"]), r["seed"])


def compare_labels(a):
    import hashlib

    def digest(x):
        return hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()

    if a.suite == "labels":
        def load(p):
            with open(p) as fh:
                return {tuple(r[k] for k in ("setting", "model", "n", "seed", "k")): (r["max_gap"], r["argmax"])
                        for r in csv.DictReader(fh)}
        new, ref = load(Path(a.rerun) / "labels" / "exact_labels.csv"), load(SAVED / "labels" / "exact_labels.csv")
        bad = [k for k in new if ref.get(k) != new[k]]
        n = len(new)
    else:
        key = lambda r: (r["dataset"], r["sampler"], r["n"], r["seed"], r["domain"], r["threshold"])  # noqa: E731
        ref = {}
        for f in ("labels.jsonl", "labels_0.2_0.2.jsonl", "labels_0.4_0.4.jsonl"):
            ref.update({key(r): r for r in jsonl(SAVED / "benchmark" / f)})
        new = jsonl(a.file)
        fields = ("label", "max_gap", "max_gap1", "n_S0", "n_S1", "n_yes_partitions", "exhaustive_max_gap", "scale")
        bad = []
        for r in new:
            q = ref.get(key(r))
            ok = q is not None and all(q.get(f) == r.get(f) for f in fields) \
                and q["agents_sha256"] == digest(r["agents"]) and q["v_sha256"] == digest(r["v"])
            if not ok:
                bad.append(key(r))
        n = len(new)
    print(f"{a.suite}: {n - len(bad)}/{n} records identical to the published ones")
    if bad:
        print("first differences:", bad[:5])
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=["synthetic", "historical", "smoke", "robust", "robust_smoke", "bandit_grid", "benchmark",
                             "labels", "benchmark_labels"])
    ap.add_argument("--file", help="game-record file for --suite benchmark_labels")
    ap.add_argument("--rerun", default=str(ROOT / "results" / "rerun"))
    a = ap.parse_args()
    if a.suite in ("labels", "benchmark_labels"):
        compare_labels(a)
        return
    rows = jsonl(Path(a.rerun) / a.suite / "trials.jsonl")
    n_ok, bad = 0, []
    if a.suite == "benchmark":
        ref = {r["task_id"]: r for r in jsonl(SAVED / "benchmark" / "trials.jsonl")}
        for r in rows:
            q = ref.get(r["task_id"])
            ok = (q is not None and abs(q["regret"] - r["regret"]) <= 1e-9 and q["d_seq"] == r["d_seq"]
                  and q["regret_seq"] == r["regret_seq"] and q["gap_seq"] == r["gap_seq"])
            if ok and "1" in r["d_seq"] and "last_declared_coalitions" in q:
                i = len(r["d_seq"]) - 1 - r["d_seq"][::-1].index("1")
                ok = max(ord(c) - 48 for c in r["partitions"][i]) + 1 == q["last_declared_coalitions"]
            if not ok:
                bad.append(r["task_id"])
            else:
                n_ok += 1
    else:
        saved_runs = {}
        if a.suite in ("synthetic", "robust", "bandit_grid", "smoke"):
            src = "synthetic" if a.suite == "smoke" else a.suite
            saved_runs = {key(r): r for r in jsonl(SAVED / src / "trials.jsonl")}
        hist = {}
        if a.suite in ("historical", "smoke"):
            hist = historical("synthetic")
        elif a.suite == "robust_smoke":
            hist = historical("robust")
        for r in rows:
            if a.suite in ("synthetic", "robust", "bandit_grid") or (a.suite == "smoke" and r["arm"] == "ucb_lcb_gs_witness_dev"):
                q = saved_runs.get(key(r))
                ok = q is not None and abs(q["regret"] - r["regret"]) <= 1e-9 and q["d_seq"] == r["d_seq"]
            else:
                H = hist.get(hkey(r))
                ok = bool(H) and all(abs(h[0] - r["regret"]) <= 1e-6 and h[1] == r["n_d1"] for h in H)
            if ok:
                n_ok += 1
            else:
                bad.append(key(r))
    print(f"{a.suite}: {n_ok}/{len(rows)} rerun trials identical to the saved records")
    if bad:
        print("first differences or missing counterparts:", bad[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()
