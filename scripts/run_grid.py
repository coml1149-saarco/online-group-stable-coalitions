#!/usr/bin/env python3
"""
Runner for the synthetic, robustness and coalition-level bandit experiments.

    python3 scripts/run_grid.py --suite synthetic   [--workers 8] [--seeds 0-99] [--match n=6 --match T=30]
    python3 scripts/run_grid.py --suite historical  [...]
    python3 scripts/run_grid.py --suite smoke
    python3 scripts/run_grid.py --suite robust | robust_smoke
    python3 scripts/run_grid.py --suite bandit_grid

Trial protocol:
  environment seed = s, algorithm seed = s + 10000, so instances are paired
  across arms; the exact deviation checker is installed in every worker before
  any learner is built (the learners, the pessimistic test and the regret
  evaluation all use it); existence labels are the exact labels of
  results/saved/labels/exact_labels.csv (YES: max gap >= eps; NO: max gap < 0;
  Z, intermediate, otherwise).

Suites:
  synthetic     every parameter set of config/final_mapping.csv with action
                "rerun" or "new" (the trials in results/saved/synthetic/trials.jsonl)
  historical    the parameter sets with action "reuse" or "preserved", whose
                trials the paper takes from results/saved/synthetic/historical_trials.csv
                (an earlier run with a deviation checker that is exact in these
                cells); scripts/check_smoke.py compares the two
  smoke         a reproducibility sample: baselines in cells where the
                historical trajectories apply, and UCB-LCB-GS in the default cell
  robust        UCB-LCB-GS on the asymmetric-utility and Student-t5 arms of the
                robustness study (n=6, T=30, k=2, sigma=3, beta=1, eps=1, seeds 0-99);
                the baselines of this study are the historical trajectories in
                results/saved/robust/historical_trials.csv
  robust_smoke  baselines and the original implementation on seeds 0-2 of the
                robustness study, compared with those historical trajectories
  bandit_grid   every parameter set of the semi-bandit grid under coalition-level
                bandit feedback, for the bandit variant of UCB-LCB-GS and five
                baselines (Random reuses its semi-bandit trajectories), plus the
                shared-exploration control experiment in the default cell

--seeds and --match restrict the task list (for representative reruns). Output:
<out>/<suite>/trials.jsonl (default out: results/rerun), appended as each
trial completes; a run resumes from it. MANIFEST.json records the
configuration, the task-list hash, code hashes and library versions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAVED = ROOT / "results" / "saved"
MAPPING = ROOT / "config" / "final_mapping.csv"
LABELS = SAVED / "labels" / "exact_labels.csv"
OPT = dict(n_restarts=3, optimizer_max_iter=60, deviation_cap=64, max_random_coalitions=4)
CODE = ["core/algorithms.py", "core/ashg_lib.py", "core/experiment.py", "sns/candidate_algorithms.py",
        "sns/coverage_algorithms.py", "sns/full_deviation_checker.py", "sns/environments.py",
        "sns/runner.py", "sns/bandit_algorithms.py", "scripts/run_grid.py"]
KEY = ("suite", "arm", "feedback", "utility_mode", "noise_model", "model", "n", "T", "k", "sigma", "beta", "eps", "seed")
BASE6 = ["pessimistic", "optimistic", "ucb_only", "lcb_only", "mixed", "random"]


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from sns.paths import ensure_core_on_path
    ensure_core_on_path()
    import ashg_lib
    from sns.full_deviation_checker import full_gap_k
    ashg_lib.gap_k = full_gap_k


def registry(feedback):
    from sns.candidate_algorithms import CANDIDATE_ALGORITHMS
    if feedback == "semi":
        return CANDIDATE_ALGORITHMS
    from sns.bandit_algorithms import BANDIT_ALGORITHMS
    return BANDIT_ALGORITHMS


def make_env(task):
    from sns.runner import make_environment, FixedUtilityASHG
    from experiment import TrialConfig
    cfg = TrialConfig(n=task["n"], model=task["model"], sigma=task["sigma"], beta=task["beta"], k=task["k"],
                      eps=task["eps"], T=task["T"], n_restarts=OPT["n_restarts"],
                      optimizer_max_iter=OPT["optimizer_max_iter"], deviation_cap=OPT["deviation_cap"],
                      max_random_coalitions=OPT["max_random_coalitions"])
    if task.get("v") is not None:
        env = FixedUtilityASHG(np.asarray(task["v"], float), noise_std=task["sigma"], seed=task["env_seed"])
    else:
        env = make_environment(cfg, env_seed=task["env_seed"], utility_mode=task["utility_mode"],
                               noise_model=task["noise_model"])
    if task["feedback"] == "bandit":
        from sns.bandit_algorithms import BanditFeedback
        env = BanditFeedback(env, noise_model=task["noise_model"])
    return cfg, env


def run_one(task):
    import ashg_lib
    from sns.full_deviation_checker import full_gap_k
    assert ashg_lib.gap_k is full_gap_k, "exact checker not installed"
    cfg, env = make_env(task)
    asym = task["utility_mode"] == "asymmetric"
    alg = registry(task["feedback"])[task["arm"]](
        n=cfg.n, sigma=cfg.sigma, beta=cfg.beta, k=cfg.k, eps=cfg.eps, rng=np.random.default_rng(task["alg_seed"]),
        n_restarts=cfg.n_restarts, max_iter=cfg.optimizer_max_iter, deviation_cap=cfg.deviation_cap,
        max_random_coalitions=cfg.max_random_coalitions, directed=asym)
    has_c = hasattr(alg, "counters")
    d_seq, r_seq, kinds = [], [], []
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
            kinds.append("O" if task["arm"] == "ucb_lcb_gs" else "F")
        g = full_gap_k(pi, env.v, env.v, cfg.k)
        r_seq.append(max(0.0, cfg.eps - g))
        d_seq.append(int(d))
        obs = env.observe(pi)
        a0 = time.perf_counter()
        alg.update(obs)
        t_learn += time.perf_counter() - a0
    out = {k: task[k] for k in KEY}
    out.update({"label": task["label"], "max_gap": task["max_gap"], "regret": float(np.sum(r_seq)),
                "n_d1": int(np.sum(d_seq)), "n_d0": int(cfg.T - np.sum(d_seq)),
                "first_d1": (int(np.argmax(d_seq)) + 1) if any(d_seq) else -1,
                "learner_seconds": t_learn, "trial_seconds": time.perf_counter() - t0,
                "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "d_seq": "".join(map(str, d_seq)), "kind_seq": "".join(kinds),
                "regret_seq": [round(x, 6) for x in r_seq]})
    if has_c:
        out.update(alg.counters())
    if hasattr(alg.estimator, "T"):
        Tm = alg.estimator.T
        off = ~np.eye(cfg.n, dtype=bool)
        out.update({"min_pair_count": int(Tm[off].min()), "mean_pair_count": float(Tm[off].mean())})
    return out


def load_labels():
    lab = {}
    with LABELS.open() as fh:
        for r in csv.DictReader(fh):
            if r["setting"] in ("symmetric", "asymmetric"):
                lab[(r["setting"], r["model"], int(r["n"]), int(r["k"]), int(r["seed"]))] = float(r["max_gap"])
    return lab


def label_of(g, eps):
    return "YES" if g >= eps else ("NO" if g < 0 else "Z")


def task(suite, arm, model, n, T, k, sigma, beta, eps, seed, lab, feedback="semi", utility_mode="symmetric",
         noise_model="gaussian"):
    g = lab[(utility_mode, model, n, k, seed)]
    return dict(suite=suite, arm=arm, feedback=feedback, utility_mode=utility_mode, noise_model=noise_model,
                model=model, n=n, T=T, k=k, sigma=sigma, beta=beta, eps=eps, seed=seed, env_seed=seed,
                alg_seed=seed + 10_000, label=label_of(g, eps), max_gap=g)


def tasks_from_mapping(lab, actions, suite, experiments=("E1", "E2", "E3")):
    seen, out = set(), []
    with MAPPING.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["action"] not in actions or r["experiment"] not in experiments or r["unique_key"] in seen:
            continue
        seen.add(r["unique_key"])
        for s in range(100):
            out.append(task(suite, r["arm"], r["model"], int(r["n"]), int(r["T"]), int(r["k"]), float(r["sigma"]),
                            float(r["beta"]), float(r["eps"]), s, lab))
    return out


def tasks_synthetic(lab):
    return tasks_from_mapping(lab, ("rerun", "new"), "synthetic")


def tasks_historical(lab):
    """E1 cells whose baseline (and original-implementation) trials are the historical trajectories.
    The E2 Random cells reuse E1 Random trajectories (Random does not depend on beta) and are not rerun."""
    return tasks_from_mapping(lab, ("reuse", "preserved"), "historical", experiments=("E1",))


def tasks_smoke(lab):
    """Historical reproducibility sample, and UCB-LCB-GS (witness probing) in the default cell."""
    T_ = []
    cells = [dict(n=6, T=30, k=2, sigma=3.0, beta=1.0, eps=1.0), dict(n=4, T=30, k=2, sigma=3.0, beta=1.0, eps=1.0),
             dict(n=7, T=30, k=2, sigma=3.0, beta=1.0, eps=1.0), dict(n=6, T=60, k=2, sigma=3.0, beta=1.0, eps=1.0),
             dict(n=6, T=30, k=2, sigma=20.0, beta=1.0, eps=1.0), dict(n=6, T=30, k=2, sigma=3.0, beta=1.0, eps=10.0),
             dict(n=6, T=30, k=2, sigma=3.0, beta=0.25, eps=1.0), dict(n=6, T=30, k=1, sigma=3.0, beta=1.0, eps=1.0)]
    for c in cells:
        for model in ("uniform", "gaussian"):
            for arm in BASE6 + ["ucb_lcb_gs"]:
                for s in range(5):
                    T_.append(task("smoke", arm, model, c["n"], c["T"], c["k"], c["sigma"], c["beta"], c["eps"], s, lab))
    for model in ("uniform", "gaussian"):
        for beta in (1.0, 3.0):
            for s in range(50):
                T_.append(task("smoke", "ucb_lcb_gs_witness_dev", model, 6, 30, 2, 3.0, beta, 1.0, s, lab))
            for s in range(5):
                T_.append(task("smoke", "ucb_lcb_gs_witness_dev", model, 6, 300, 2, 3.0, beta, 1.0, s, lab))
    return T_


def robust_tasks(lab, arms, seeds, suite):
    out = []
    for arm in arms:
        for model in ("uniform", "gaussian"):
            for um, nm in (("asymmetric", "gaussian"), ("symmetric", "student_t5")):
                for s in seeds:
                    out.append(task(suite, arm, model, 6, 30, 2, 3.0, 1.0, 1.0, s, lab, utility_mode=um, noise_model=nm))
    return out


def bandit_grid_tasks(lab, primary="bandit_witness"):
    """Every parameter set of the semi-bandit grid (E1, E2, E3: all cells, 100 seeds) for the bandit
    primary and five baselines (Random reuses its semi-bandit trajectories), plus the shared-exploration
    control experiment in the default cell."""
    psets = {}
    with MAPPING.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["arm"] == "ucb_lcb_gs_witness_dev":
            p = (r["model"], int(r["n"]), int(r["T"]), int(r["k"]), float(r["sigma"]), float(r["beta"]), float(r["eps"]))
            psets[p] = r["experiment"]
    out = []
    for (model, n, T, k, sigma, beta, eps) in sorted(psets):
        for arm in [primary] + BASE6[:5]:
            for s in range(100):
                out.append(task("bandit_grid", arm, model, n, T, k, sigma, beta, eps, s, lab, feedback="bandit"))
    for model in ("uniform", "gaussian"):
        for beta in (1.0, 3.0):
            for T in (30, 300):
                for arm in BASE6[:5]:
                    for s in range(100):
                        out.append(task("bandit_grid", "shared_" + arm, model, 6, T, 2, 3.0, beta, 1.0, s, lab,
                                        feedback="bandit"))
    return out


def file_hash(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def cost(t):
    f = {(4, 2): 0.29, (5, 2): 0.71, (6, 2): 1.0, (7, 2): 1.75, (6, 1): 0.29, (6, 3): 4.8}.get((t["n"], t["k"]), (t["n"] / 6) ** 3)
    return t["T"] * f


def run(tasks, outdir, workers, extra_manifest=None):
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "trials.jsonl"
    key = lambda d: tuple(d[k] for k in KEY)  # noqa: E731
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), n_tasks=len(tasks),
                    task_list_sha256=hashlib.sha256(json.dumps(sorted(map(key, tasks))).encode()).hexdigest(),
                    workers=workers, optimizer=OPT, seeds="env = s, algorithm = s + 10000",
                    checker="sns.full_deviation_checker.full_gap_k (exact, k <= 3)",
                    code_sha256={p: file_hash(p) for p in CODE},
                    python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
                    machine=platform.machine(), cpu_count=os.cpu_count())
    if extra_manifest:
        manifest.update(extra_manifest)
    mf = outdir / "MANIFEST.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        if old["task_list_sha256"] != manifest["task_list_sha256"] or old["code_sha256"] != manifest["code_sha256"]:
            raise SystemExit("MANIFEST mismatch: task list or code changed since this run started; use a new --out")
        manifest["resumed"] = old.get("resumed", 0) + 1
        manifest["created"] = old["created"]
    mf.write_text(json.dumps(manifest, indent=1))
    done = set()
    if out.exists():
        with out.open() as fh:
            for line in fh:
                done.add(key(json.loads(line)))
    todo = sorted((t for t in tasks if key(t) not in done), key=lambda t: -cost(t))
    print(f"{len(tasks)} trials in design; {len(todo)} to run; workers {workers}", flush=True)
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init_worker) as pool, out.open("a") as f:
        for i, res in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            f.write(json.dumps(res) + "\n"); f.flush()
            if i % 1000 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    with out.open() as fh:
        keys = [key(json.loads(l)) for l in fh]
    dup = len(keys) - len(set(keys))
    missing = len({key(t) for t in tasks} - set(keys))
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min; duplicates {dup}; missing {missing}", flush=True)


def restrict(tasks, seeds, matches):
    if seeds:
        a, _, b = seeds.partition("-")
        lo, hi = int(a), int(b or a)
        tasks = [t for t in tasks if lo <= t["seed"] <= hi]
    for m in matches:
        k, _, v = m.partition("=")
        tasks = [t for t in tasks if str(t[k]) == v or (isinstance(t[k], float) and float(v) == t[k])]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=["synthetic", "historical", "smoke", "robust", "robust_smoke", "bandit_grid"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(ROOT / "results" / "rerun"))
    ap.add_argument("--seeds", default="", help="seed range, e.g. 0-4")
    ap.add_argument("--match", action="append", default=[], help="keep tasks with field=value, e.g. n=6")
    a = ap.parse_args()
    lab = load_labels()
    build = {"synthetic": tasks_synthetic, "historical": tasks_historical, "smoke": tasks_smoke,
             "robust": lambda l: robust_tasks(l, ["ucb_lcb_gs_witness_dev"], range(100), "robust"),
             "robust_smoke": lambda l: robust_tasks(l, BASE6 + ["ucb_lcb_gs"], range(3), "robust_smoke"),
             "bandit_grid": bandit_grid_tasks}[a.suite]
    tasks = restrict(build(lab), a.seeds, a.match)
    if not tasks:
        raise SystemExit("no tasks selected")
    run(tasks, Path(a.out) / a.suite, a.workers)


if __name__ == "__main__":
    main()
