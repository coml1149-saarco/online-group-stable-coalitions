"""
sns.environments
================

Environments for the robustness study:

  1. **Asymmetric utilities** - ``v_ij`` and ``v_ji`` drawn independently
     (``v_ii = 0``), with the same marginals as the symmetric generators.
  2. **Student-t(5) observation noise** - the utility model is unchanged;
     only the feedback noise is replaced, rescaled to the same variance as
     the Gaussian noise it replaces.

The paper's analysis extends to asymmetric utilities. Student-t(5) noise is
not sub-Gaussian (it is heavy-tailed, with moments only up to order < 5), so
the theorems do not cover it.

Exact-reproduction guarantee
----------------------------
In the baseline configuration (``symmetric`` + ``gaussian``) this class
delegates verbatim to ``ASHGEnvironment``, so the baseline arm of each
robustness experiment consumes the same random stream and produces the same
instances and feedback as the main synthetic runs.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .paths import ensure_core_on_path

ensure_core_on_path()

from ashg_lib import ASHGEnvironment, Partition  # noqa: E402

UTILITY_MODES = ("symmetric", "asymmetric")
NOISE_MODELS = ("gaussian", "student_t5")

# Student-t with df degrees of freedom has variance df / (df - 2).
_STUDENT_T_DF = 5.0
_STUDENT_T_SCALE = math.sqrt((_STUDENT_T_DF - 2.0) / _STUDENT_T_DF)  # = sqrt(3/5)


class RobustASHGEnvironment(ASHGEnvironment):
    """
    ``ASHGEnvironment`` with a configurable utility symmetry mode and a
    configurable observation-noise family.

    Feedback model
    --------------
    *Symmetric* (the paper's model): one observation per **unordered**
    co-located pair, ``V^t_{ij} = v_ij + eps``.

    *Asymmetric*: agent ``i``'s valuation of ``j`` is its own quantity, so the
    learner receives one observation per **ordered** co-located pair -
    ``(i, j)`` carrying ``v_ij`` and ``(j, i)`` carrying ``v_ji``, with
    independent noise draws.  The estimator must be run with
    ``directed=True`` so that the two directions are not averaged together
    (see ``PairwiseEstimator`` in ``core/algorithms.py``).
    """

    def __init__(
        self,
        n: int,
        model: str = "uniform",
        noise_std: float = 1.0,
        seed: int | None = None,
        utility_mode: str = "symmetric",
        noise_model: str = "gaussian",
    ):
        if utility_mode not in UTILITY_MODES:
            raise ValueError(f"utility_mode must be one of {UTILITY_MODES}, got {utility_mode!r}")
        if noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}, got {noise_model!r}")
        # Must be set before super().__init__, which calls _draw_utilities().
        self.utility_mode = utility_mode
        self.noise_model = noise_model
        super().__init__(n=n, model=model, noise_std=noise_std, seed=seed)

    # ---- utilities -------------------------------------------------------
    def _draw_utilities(self) -> np.ndarray:
        if self.utility_mode == "symmetric":
            # Delegate verbatim: identical draw and identical RNG consumption.
            return super()._draw_utilities()

        n = self.n
        v = np.zeros((n, n), dtype=np.float64)
        if self.model == "uniform":
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    v[i, j] = float(self.rng.integers(-100, 101))
        elif self.model == "gaussian":
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    mu = float(self.rng.integers(-100, 101))
                    v[i, j] = float(round(self.rng.normal(mu, 10.0)))
        else:
            raise ValueError(f"unknown model {self.model!r}")
        np.fill_diagonal(v, 0.0)
        return v

    # ---- noise -----------------------------------------------------------
    def _noise(self) -> float:
        """One centred noise draw with standard deviation ``self.noise_std``."""
        if self.noise_model == "gaussian":
            return float(self.rng.normal(0.0, self.noise_std))
        # Student-t(5) rescaled to variance noise_std**2:
        #   Var(t_5) = 5/3, so multiply by sqrt(3/5).
        return float(self.noise_std * _STUDENT_T_SCALE * self.rng.standard_t(_STUDENT_T_DF))

    # ---- feedback --------------------------------------------------------
    def observe(self, partition: Partition) -> Dict[Tuple[int, int], float]:
        if self.utility_mode == "symmetric" and self.noise_model == "gaussian":
            # Delegate verbatim to the paper's implementation.
            return super().observe(partition)

        cid = partition.cid
        out: Dict[Tuple[int, int], float] = {}
        if self.utility_mode == "symmetric":
            # Same unordered-pair feedback as the paper, non-Gaussian noise.
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if cid[i] == cid[j]:
                        out[(i, j)] = self.v[i, j] + self._noise()
        else:
            # Directed feedback: one observation per ordered co-located pair.
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and cid[i] == cid[j]:
                        out[(i, j)] = self.v[i, j] + self._noise()
        return out


def noise_variance_check(
    noise_std: float, noise_model: str, n_draws: int = 400_000, seed: int = 0
) -> Dict[str, float]:
    """
    Empirical sanity check that ``student_t5`` really carries the same variance
    as the Gaussian noise it replaces.  Used by the robustness runner's
    self-test; cheap enough to run before every sweep.
    """
    env = RobustASHGEnvironment(
        n=2, model="uniform", noise_std=noise_std, seed=seed, noise_model=noise_model
    )
    draws = np.array([env._noise() for _ in range(n_draws)])
    return {
        "noise_model": noise_model,
        "target_std": float(noise_std),
        "empirical_mean": float(draws.mean()),
        "empirical_std": float(draws.std(ddof=1)),
        "excess_kurtosis": float(
            ((draws - draws.mean()) ** 4).mean() / (draws.std() ** 4) - 3.0
        ),
    }


__all__ = [
    "RobustASHGEnvironment",
    "UTILITY_MODES",
    "NOISE_MODELS",
    "noise_variance_check",
]
