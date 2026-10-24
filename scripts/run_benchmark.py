#!/usr/bin/env python3
"""
Learning runs on the dataset-derived games (Appendix L of the paper).

    python3 scripts/run_benchmark.py [--workers 8] [--labels results/local/benchmark/labels.jsonl]
                                     [--seeds 0-4] [--match n=8 --match feedback=semi]

Each labelled game is run (game records rebuilt from the downloaded data by scripts/rebuild_benchmark_games.py,
or produced by scripts/benchmark_labels.py) with the seven semi-bandit arms and the seven
coalition-level bandit arms, with the environment, registries and exact checker of scripts/run_grid.py.
Every round also records the partition formed and its exact gap. Parameters: T=30, k=2, sigma=3, beta=1,
eps=1; the environment and algorithm seeds are those stored with each game. Model name:
bm_<data set>_<sampler>_<domain>. Output: <out>/benchmark/trials.jsonl (default out: results/rerun),
resumable through deterministic task identifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_grid as R  # noqa: E402

KCFG = dict(T=30, k=2, sigma=3.0, beta=1.0, eps=1.0)
SEMI_ARMS = ["ucb_lcb_gs_witness_dev", "pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed", "random"]
BANDIT_ARMS = ["bandit_witness", "pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed", "random"]


def tid(t):
    return f"{t['suite']}|{t['feedback']}|{t['arm']}|{t['model']}|{t['n']}|{t['seed']}"


def tasks(labels_path, suite="bench_games"):
    out = []
    with Path(labels_path).open() as fh:
        records = [json.loads(line) for line in fh]
    for r in records:
        for fb, arms in (("semi", SEMI_ARMS), ("bandit", BANDIT_ARMS)):
            for arm in arms:
                out.append(dict(suite=suite, arm=arm, feedback=fb, utility_mode="symmetric", noise_model="gaussian",
                                model=f"bm_{r['dataset']}_{r['sampler']}_{r['domain']}", n=r["n"], seed=r["seed"],
                                env_seed=r["env_seed"], alg_seed=r["alg_seed"], label=r["label"], max_gap=r["max_gap"],
                                v=r["v"], variant=r["domain"], **KCFG))
    return out


def _canon(cid):
    m, out = {}, []
    for c in cid:
        m.setdefault(int(c), len(m))
        out.append(m[int(c)])
    return out


def run_one(task):
    from sns.full_deviation_checker import full_gap_k
    cfg, env = R.make_env(task)
    alg = R.registry(task["feedback"])[task["arm"]](
        n=cfg.n, sigma=cfg.sigma, beta=cfg.beta, k=cfg.k, eps=cfg.eps, rng=np.random.default_rng(task["alg_seed"]),
        n_restarts=cfg.n_restarts, max_iter=cfg.optimizer_max_iter, deviation_cap=cfg.deviation_cap,
        max_random_coalitions=cfg.max_random_coalitions, directed=False)
    has_c = hasattr(alg, "counters")
    d_seq, r_seq, g_seq, parts, kinds = [], [], [], [], []
    t_learn, t0 = 0.0, time.perf_counter()
    for t in range(1, cfg.T + 1):
        before = alg.counters() if has_c else None
        a0 = time.perf_counter()
        pi, d = alg.act(t)
        t_learn += time.perf_counter() - a0
        if d == 1:
            kinds.append("C")
        elif has_c:
            after = alg.counters()
            kinds.append("P" if after["n_probes"] > before["n_probes"] else
                         ("I" if after.get("n_init", 0) > before.get("n_init", 0) else "O"))
        else:
            kinds.append("F")
        g = full_gap_k(pi, env.v, env.v, cfg.k)
        g_seq.append(float(g))
        r_seq.append(max(0.0, cfg.eps - g))
        d_seq.append(int(d))
        parts.append("".join(chr(48 + int(c)) for c in _canon(pi.cid)))
        obs = env.observe(pi)
        a0 = time.perf_counter()
        alg.update(obs)
        t_learn += time.perf_counter() - a0
    out = {k: task[k] for k in R.KEY}
    out.update({"label": task["label"], "max_gap": task["max_gap"], "variant": task["variant"],
                "regret": float(np.sum(r_seq)), "n_d1": int(np.sum(d_seq)), "n_d0": int(cfg.T - np.sum(d_seq)),
                "first_d1": (int(np.argmax(d_seq)) + 1) if any(d_seq) else -1,
                "learner_seconds": t_learn, "trial_seconds": time.perf_counter() - t0,
                "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "d_seq": "".join(map(str, d_seq)), "kind_seq": "".join(kinds),
                "regret_seq": [round(x, 6) for x in r_seq], "gap_seq": g_seq, "partitions": parts})
    if has_c:
        out.update(alg.counters())
    out["task_id"] = tid(task)
    return out


def run(tasks_, outdir, workers):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "trials.jsonl"
    done = set()
    if path.exists():
        with path.open() as fh:
            for line in fh:
                done.add(json.loads(line)["task_id"])
    todo = [t for t in tasks_ if tid(t) not in done]
    todo.sort(key=lambda t: -t["n"])
    print(f"{len(tasks_)} tasks, {len(done)} done, {len(todo)} to run; workers {workers}", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=R._init_worker) as pool, path.open("a") as fh:
        for i, r in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(r) + "\n")
            if i % 200 == 0 or i == len(todo):
                fh.flush()
                print(f"  {i}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)
    (outdir / "manifest.json").write_text(json.dumps({"tasks": len(tasks_), "wall_minutes": (time.time() - t0) / 60,
                                                      "workers": workers, "kcfg": KCFG}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--labels", default=str(ROOT / "results/local/benchmark/labels.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/rerun"))
    ap.add_argument("--seeds", default="", help="seed range, e.g. 0-4")
    ap.add_argument("--match", action="append", default=[], help="keep tasks with field=value, e.g. n=8")
    a = ap.parse_args()
    if not Path(a.labels).exists():
        raise SystemExit(f"{a.labels} not found: run  python3 scripts/rebuild_benchmark_games.py  first")
    T = R.restrict(tasks(a.labels), a.seeds, a.match)
    if not T:
        raise SystemExit("no tasks selected")
    run(T, Path(a.out) / "benchmark", a.workers)


if __name__ == "__main__":
    main()
