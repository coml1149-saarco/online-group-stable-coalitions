"""
sns.runner
==========

Single-trial helpers shared by the experiment scripts.

``run_trial`` transcribes ``core/experiment.py::run_single`` with two
additions: the true utility matrix may be supplied from outside (used for the
dataset-derived games), and wall-clock time is recorded (learner time per
round and total trial time). No algorithmic behaviour is changed.

``make_environment`` and ``FixedUtilityASHG`` build the environments used by
``scripts/run_grid.py`` and ``scripts/run_benchmark.py``.

Everything substantive - the partition heuristic, the deviation checker,
``gap_k``, ``is_stable`` and the algorithms - is imported from ``core/`` and
executed unmodified.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .paths import ensure_core_on_path

ensure_core_on_path()

from ashg_lib import (  # noqa: E402  (import after sys.path fix)
    ASHGEnvironment,
    compute_partition,
    gap_k,
    is_stable,
)
from experiment import TrialConfig, _make_algorithm  # noqa: E402
from algorithms import ALGORITHMS  # noqa: E402

from .environments import RobustASHGEnvironment  # noqa: E402


class FixedUtilityASHG(ASHGEnvironment):
    """
    An ``ASHGEnvironment`` whose true utility matrix is supplied rather than
    drawn.  ``observe`` (the noisy partial-feedback model) is inherited
    unchanged, so the learner sees exactly the same feedback process as in the
    synthetic experiments:  ``V^t_{i,j} = v_{i,j} + eps^t_{i,j}`` for every
    same-coalition pair, with i.i.d. ``N(0, sigma^2)`` noise.
    """

    def __init__(self, v: np.ndarray, noise_std: float, seed: int, model: str = "fixed"):
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != v.shape[1]:
            raise ValueError(f"utility matrix must be square, got shape {v.shape}")
        if not np.allclose(v, v.T, atol=1e-9):
            raise ValueError("utility matrix must be symmetric (paper assumes symmetric ASHGs)")
        if not np.allclose(np.diag(v), 0.0):
            raise ValueError("utility matrix must have a zero diagonal (v_ii = 0)")
        # Deliberately does NOT call super().__init__: we skip the random draw.
        self.n = int(v.shape[0])
        self.model = model
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.v = v.copy()


def make_algorithm(algorithm_name: str, cfg: TrialConfig, rng: np.random.Generator,
                   directed: bool = False):
    """
    Construct an algorithm.  With ``directed=False`` this is exactly
    ``experiment._make_algorithm``; ``directed=True`` additionally tells the
    shared estimator not to mirror (i, j) into (j, i), which is required when
    utilities are asymmetric.
    """
    if not directed:
        return _make_algorithm(algorithm_name, cfg, rng)
    cls = ALGORITHMS[algorithm_name]
    return cls(
        n=cfg.n, sigma=cfg.sigma, beta=cfg.beta, k=cfg.k, eps=cfg.eps, rng=rng,
        n_restarts=cfg.n_restarts, max_iter=cfg.optimizer_max_iter,
        deviation_cap=cfg.deviation_cap,
        max_random_coalitions=cfg.max_random_coalitions,
        directed=True,
    )


def make_environment(
    cfg: TrialConfig,
    env_seed: int,
    v: Optional[np.ndarray] = None,
    utility_mode: str = "symmetric",
    noise_model: str = "gaussian",
) -> ASHGEnvironment:
    """Synthetic draw when ``v is None``, otherwise a fixed-utility ASHG."""
    if v is None:
        if utility_mode == "symmetric" and noise_model == "gaussian":
            # The paper's exact environment - identical RNG consumption.
            return ASHGEnvironment(
                n=cfg.n, model=cfg.model, noise_std=cfg.sigma, seed=env_seed
            )
        return RobustASHGEnvironment(
            n=cfg.n, model=cfg.model, noise_std=cfg.sigma, seed=env_seed,
            utility_mode=utility_mode, noise_model=noise_model,
        )
    if v.shape[0] != cfg.n:
        raise ValueError(
            f"utility matrix has n={v.shape[0]} but config says n={cfg.n}"
        )
    return FixedUtilityASHG(v, noise_std=cfg.sigma, seed=env_seed, model=cfg.model)


def ground_truth_existence(
    env: ASHGEnvironment, cfg: TrialConfig, oracle_seed: int
) -> bool:
    """
    Heuristic ground-truth label ``d*`` for whether an (eps, k)-SNS partition
    exists, computed exactly as in ``run_single``: one independent run of the
    shared optimizer on the *true* utilities, followed by the shared stability
    test.  Identical for every algorithm, so any bias cancels in comparisons
    (Appendix H, "Ground-truth existence label").
    """
    oracle_rng = np.random.default_rng(oracle_seed)
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
    return bool(
        is_stable(
            pi_oracle, env.v, env.v, cfg.k, cfg.eps, deviation_cap=cfg.deviation_cap
        )
    )


def run_trial(
    algorithm_name: str,
    cfg: TrialConfig,
    env_seed: int,
    algorithm_seed: int,
    oracle_seed: int,
    v: Optional[np.ndarray] = None,
    true_exists: Optional[bool] = None,
    keep_curves: bool = True,
    utility_mode: str = "symmetric",
    noise_model: str = "gaussian",
) -> Dict[str, Any]:
    """
    Run one (algorithm, instance, seed) trial.

    Returns a JSON-friendly dict with per-round regret / gap / declaration
    curves, the summary scalars, and timings.
    """
    t_trial0 = time.perf_counter()

    env = make_environment(cfg, env_seed=env_seed, v=v,
                           utility_mode=utility_mode, noise_model=noise_model)
    alg_rng = np.random.default_rng(algorithm_seed)
    alg = make_algorithm(algorithm_name, cfg, alg_rng,
                         directed=(utility_mode == "asymmetric"))

    if true_exists is None:
        true_exists = ground_truth_existence(env, cfg, oracle_seed)

    regret_per_round = []
    gap_per_round = []
    declared_existence = []
    existence_errors = []
    cum_regret = []

    running = 0.0
    learner_seconds = 0.0
    last_partition = None
    for t in range(1, cfg.T + 1):
        t_act0 = time.perf_counter()
        pi_t, d_t = alg.act(t)
        learner_seconds += time.perf_counter() - t_act0
        last_partition = pi_t

        # Evaluate against the *true* utilities (regret is never observed by
        # the learner; it only ever sees noisy same-coalition feedback).
        g = gap_k(pi_t, env.v, env.v, cfg.k, deviation_cap=cfg.deviation_cap)
        r = max(0.0, cfg.eps - g)
        running += r
        regret_per_round.append(float(r))
        gap_per_round.append(float(g))
        cum_regret.append(float(running))
        declared_existence.append(int(d_t))
        existence_errors.append(int(d_t != int(true_exists)))

        obs = env.observe(pi_t)
        t_upd0 = time.perf_counter()
        alg.update(obs)
        learner_seconds += time.perf_counter() - t_upd0

    trial_seconds = time.perf_counter() - t_trial0

    out: Dict[str, Any] = {
        "algorithm": algorithm_name,
        "utility_mode": utility_mode,
        "noise_model": noise_model,
        "true_exists": int(true_exists),
        # --- headline scalars -------------------------------------------
        "final_cumulative_regret": float(running),
        "mean_regret_per_round": float(running / cfg.T),
        "existence_error_rate": float(np.mean(existence_errors)),
        "existence_error_count": int(np.sum(existence_errors)),
        "mean_gap": float(np.mean(gap_per_round)),
        "best_gap": float(np.max(gap_per_round)),
        "frac_rounds_stable": float(np.mean([g >= cfg.eps for g in gap_per_round])),
        "declared_existence_rate": float(np.mean(declared_existence)),
        # --- final state / stopping behaviour ----------------------------
        # Added for the exact-vs-approximate deviation-checker comparison;
        # purely observational, no effect on the trajectory.
        "final_partition": (
            [int(x) for x in last_partition.cid] if last_partition is not None else []
        ),
        "final_num_coalitions": (
            int(last_partition.cid.max()) + 1 if last_partition is not None else 0
        ),
        "first_certified_round": (
            int(np.argmax(declared_existence)) + 1 if any(declared_existence) else -1
        ),
        "n_certified_rounds": int(np.sum(declared_existence)),
        # --- timing ------------------------------------------------------
        "learner_seconds": float(learner_seconds),
        "learner_ms_per_round": float(1e3 * learner_seconds / cfg.T),
        "trial_seconds": float(trial_seconds),
        "trial_ms_per_round": float(1e3 * trial_seconds / cfg.T),
        "config": cfg.as_dict(),
    }
    if keep_curves:
        out.update(
            {
                "regret_per_round": regret_per_round,
                "gap_per_round": gap_per_round,
                "declared_existence": declared_existence,
                "existence_errors": existence_errors,
                "cumulative_regret": cum_regret,
            }
        )
    return out


__all__ = [
    "FixedUtilityASHG",
    "make_environment",
    "ground_truth_existence",
    "run_trial",
    "TrialConfig",
]
