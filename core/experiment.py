"""
experiment.py
=============

Simulation runner, metrics logger, and ablation harness.

Public entry points:

  * ``run_single(...)``       - run one (algorithm, env, seed) trial.
  * ``run_ablation(...)``     - sweep ONE parameter, average over seeds.
  * ``save_results``/``load_results``  - JSON-friendly persistence.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ashg_lib import (
    ASHGEnvironment,
    Partition,
    gap_k,
    is_stable,
)
from algorithms import ALGORITHMS, BaseAlgorithm


# ============================================================
# Configuration dataclass
# ============================================================


@dataclass
class TrialConfig:
    n: int = 6
    model: str = "uniform"          # "uniform" | "gaussian"
    sigma: float = 5.0              # sub-Gaussian noise std
    beta: float = 1.0               # confidence parameter
    k: int = 2                      # max coalition size for stability
    eps: float = 1.0                # stability margin
    T: int = 200                    # rounds
    n_restarts: int = 3             # optimizer restarts
    optimizer_max_iter: int = 60    # local search iterations
    deviation_cap: int = 64
    max_random_coalitions: int = 4  # for random fallback partitions

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Single-trial runner
# ============================================================


def _make_algorithm(name: str, cfg: TrialConfig, rng: np.random.Generator) -> BaseAlgorithm:
    cls = ALGORITHMS[name]
    return cls(
        n=cfg.n,
        sigma=cfg.sigma,
        beta=cfg.beta,
        k=cfg.k,
        eps=cfg.eps,
        rng=rng,
        n_restarts=cfg.n_restarts,
        max_iter=cfg.optimizer_max_iter,
        deviation_cap=cfg.deviation_cap,
        max_random_coalitions=cfg.max_random_coalitions,
    )


def run_single(
    algorithm_name: str,
    cfg: TrialConfig,
    seed: int,
    true_exists: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run one trial of ``algorithm_name`` against a freshly drawn ASHG.

    Returns a dict with cumulative regret, per-round gaps, declared existence,
    and the existence error rate.

    ``true_exists`` is an optional ground-truth flag for whether an
    (eps, k)-stable partition exists; if ``None`` we approximate it once with
    the true utilities using the same heuristic optimizer.
    """
    env_seed = seed
    alg_seed = seed + 10_000

    env = ASHGEnvironment(
        n=cfg.n, model=cfg.model, noise_std=cfg.sigma, seed=env_seed
    )
    alg_rng = np.random.default_rng(alg_seed)
    alg = _make_algorithm(algorithm_name, cfg, alg_rng)

    # Ground-truth existence: approximate by running compute_partition with
    # the true utilities and checking is_stable.  This is heuristic but the
    # same heuristic for all algorithms, keeping comparisons fair.
    if true_exists is None:
        from ashg_lib import compute_partition

        oracle_rng = np.random.default_rng(seed + 20_000)
        pi_oracle = compute_partition(
            "ucb_only",
            env.v,
            env.v,
            k=cfg.k,
            eps=cfg.eps,
            rng=oracle_rng,
            n_restarts=max(cfg.n_restarts, 4),
            max_iter=cfg.optimizer_max_iter,
            deviation_cap=cfg.deviation_cap,
        )
        true_exists = bool(
            is_stable(pi_oracle, env.v, env.v, cfg.k, cfg.eps, deviation_cap=cfg.deviation_cap)
        )

    regret_per_round: List[float] = []
    gap_per_round: List[float] = []
    declared_existence: List[int] = []
    existence_errors: List[int] = []
    cum_regret: List[float] = []

    running = 0.0
    for t in range(1, cfg.T + 1):
        pi_t, d_t = alg.act(t)
        # Evaluate against the *true* utilities.
        g = gap_k(pi_t, env.v, env.v, cfg.k, deviation_cap=cfg.deviation_cap)
        r = max(0.0, cfg.eps - g)
        running += r
        regret_per_round.append(r)
        gap_per_round.append(g)
        cum_regret.append(running)
        declared_existence.append(int(d_t))
        existence_errors.append(int(d_t != int(true_exists)))

        # Update estimates with observations from the played partition.
        obs = env.observe(pi_t)
        alg.update(obs)

    return {
        "algorithm": algorithm_name,
        "seed": seed,
        "true_exists": int(true_exists),
        "regret_per_round": regret_per_round,
        "gap_per_round": gap_per_round,
        "declared_existence": declared_existence,
        "existence_errors": existence_errors,
        "cumulative_regret": cum_regret,
        "config": cfg.as_dict(),
    }


# ============================================================
# Ablation harness
# ============================================================


def run_ablation(
    param_name: str,
    param_values: Iterable[Any],
    base_cfg: TrialConfig,
    algorithms: Iterable[str],
    seeds: Iterable[int],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ``algorithms`` over the cross-product of ``param_values`` and ``seeds``.

    Returns a nested dict::

        {
          "param_name": param_name,
          "param_values": [...],
          "algorithms": [...],
          "config": base_cfg.as_dict(),
          "trials": {
              algo: { value: [trial_dict, trial_dict, ...] }, ...
          },
        }
    """
    algorithms = list(algorithms)
    seeds = list(seeds)
    param_values = list(param_values)

    out = {
        "param_name": param_name,
        "param_values": [_jsonable(v) for v in param_values],
        "algorithms": algorithms,
        "config": base_cfg.as_dict(),
        "trials": {a: {} for a in algorithms},
    }

    t0 = time.time()
    for v in param_values:
        cfg_dict = base_cfg.as_dict()
        cfg_dict[param_name] = v
        cfg = TrialConfig(**cfg_dict)
        for algo in algorithms:
            results = []
            for s in seeds:
                tr = run_single(algo, cfg, seed=int(s))
                results.append(tr)
            out["trials"][algo][_jsonable(v)] = results
            if verbose:
                avg_reg = np.mean([r["cumulative_regret"][-1] for r in results])
                err_rate = np.mean([np.mean(r["existence_errors"]) for r in results])
                print(
                    f"  [{param_name}={v}] {algo:>20s} | "
                    f"avg cum-regret = {avg_reg:8.2f} | "
                    f"existence-err rate = {err_rate:.3f}"
                )
        if verbose:
            print(f"  >>> finished {param_name}={v}  ({time.time()-t0:.1f}s elapsed)")
    if verbose:
        print(f"Ablation '{param_name}' done in {time.time()-t0:.1f}s.")
    return out


# ============================================================
# Persistence
# ============================================================


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def save_results(results: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(results, f, default=_jsonable, indent=2)
    else:
        with open(path, "wb") as f:
            pickle.dump(results, f)


def load_results(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Aggregation helpers (used by plotting)
# ============================================================


def aggregate_final_regret(results: Dict[str, Any]) -> Dict[str, Tuple[List[Any], np.ndarray, np.ndarray]]:
    """For each algorithm, return (param_values, mean_final_regret, stderr)."""
    out: Dict[str, Tuple[List[Any], np.ndarray, np.ndarray]] = {}
    for algo, by_v in results["trials"].items():
        xs, means, stds = [], [], []
        for v in results["param_values"]:
            trials = by_v[v] if v in by_v else by_v[str(v)]
            finals = np.array([t["cumulative_regret"][-1] for t in trials])
            xs.append(v)
            means.append(finals.mean())
            stds.append(finals.std(ddof=1) / np.sqrt(max(1, len(finals))))
        out[algo] = (xs, np.array(means), np.array(stds))
    return out


def aggregate_existence_error(results: Dict[str, Any]) -> Dict[str, Tuple[List[Any], np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[List[Any], np.ndarray, np.ndarray]] = {}
    for algo, by_v in results["trials"].items():
        xs, means, stds = [], [], []
        for v in results["param_values"]:
            trials = by_v[v] if v in by_v else by_v[str(v)]
            rates = np.array([np.mean(t["existence_errors"]) for t in trials])
            xs.append(v)
            means.append(rates.mean())
            stds.append(rates.std(ddof=1) / np.sqrt(max(1, len(rates))))
        out[algo] = (xs, np.array(means), np.array(stds))
    return out


def aggregate_cumulative_curve(
    results: Dict[str, Any], param_value: Any
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return per-algorithm cumulative-regret curves for a given x value."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for algo, by_v in results["trials"].items():
        trials = by_v[param_value] if param_value in by_v else by_v[str(param_value)]
        curves = np.array([t["cumulative_regret"] for t in trials])
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(max(1, curves.shape[0]))
        out[algo] = (mean, sem)
    return out


__all__ = [
    "TrialConfig",
    "run_single",
    "run_ablation",
    "save_results",
    "load_results",
    "aggregate_final_regret",
    "aggregate_existence_error",
    "aggregate_cumulative_curve",
]
