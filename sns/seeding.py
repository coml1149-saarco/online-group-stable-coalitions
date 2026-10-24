"""
sns.seeding
===========

Reproducible seed management.

A single integer ``master_seed`` deterministically controls every source of
randomness in a trial. The dataset-derived games use the streams
``<tag>|subset`` (which data points are sampled as agents), ``<tag>|env``
(the per-round feedback noise) and ``<tag>|algorithm`` (random restarts,
tie-breaking and random partitions of the learners).

Two derivation schemes are provided:

``legacy_offsets``
    Reproduces the offsets hard-coded in ``core/experiment.py``
    (``env = s``, ``algorithm = s + 10_000``, ``oracle = s + 20_000``), as
    used by the synthetic experiments.

``hashed``
    ``sub_seed = SHA256(master_seed || stream_name)`` truncated to 63 bits.
    Used for the dataset-derived games. Hashing means that adding or removing
    a stream never shifts the others.

Both schemes are pure functions of ``(master_seed, stream)``: no global state
and no ``np.random`` module-level calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

# Default stream names.
STREAMS = ("subset", "bootstrap", "env", "algorithm", "oracle")

# Offsets baked into core/experiment.py::run_single.
_LEGACY_OFFSETS: Dict[str, int] = {
    "env": 0,
    "algorithm": 10_000,
    "oracle": 20_000,
}

_MAX_SEED = (1 << 63) - 1


def _hash_seed(master_seed: int, stream: str) -> int:
    """SHA-256 based sub-seed; stable across machines, Python versions and runs."""
    payload = f"{int(master_seed)}::{stream}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & _MAX_SEED


@dataclass(frozen=True)
class SeedBundle:
    """All sub-seeds for one trial, derived from a single ``master_seed``."""

    master_seed: int
    scheme: str
    seeds: Dict[str, int] = field(default_factory=dict)

    def __getitem__(self, stream: str) -> int:
        if stream not in self.seeds:
            raise KeyError(
                f"stream {stream!r} not derived for master_seed={self.master_seed}; "
                f"available: {sorted(self.seeds)}"
            )
        return self.seeds[stream]

    def rng(self, stream: str) -> np.random.Generator:
        """A fresh ``numpy`` Generator for ``stream`` (independent of call order)."""
        return np.random.default_rng(self[stream])

    def as_dict(self) -> Dict[str, int]:
        out: Dict[str, int] = {"master_seed": int(self.master_seed)}
        out.update({f"seed_{k}": int(v) for k, v in sorted(self.seeds.items())})
        return out


def derive_seeds(
    master_seed: int,
    streams: Iterable[str] = STREAMS,
    scheme: str = "hashed",
) -> SeedBundle:
    """
    Derive one sub-seed per stream from ``master_seed``.

    Parameters
    ----------
    master_seed:
        The only number that needs to be logged to reproduce a trial.
    streams:
        Stream names to derive.  Unknown names are allowed under ``hashed``
        (they simply get their own hash); under ``legacy_offsets`` only the
        three streams that ``run_single`` knows about are legal.
    scheme:
        ``"hashed"`` or ``"legacy_offsets"``.
    """
    master_seed = int(master_seed)
    streams = list(streams)

    if scheme == "legacy_offsets":
        unknown = [s for s in streams if s not in _LEGACY_OFFSETS]
        if unknown:
            raise ValueError(
                "legacy_offsets only defines "
                f"{sorted(_LEGACY_OFFSETS)}; cannot derive {unknown}. "
                "Use scheme='hashed' for other streams."
            )
        seeds = {s: master_seed + _LEGACY_OFFSETS[s] for s in streams}
    elif scheme == "hashed":
        seeds = {s: _hash_seed(master_seed, s) for s in streams}
    else:
        raise ValueError(f"unknown seeding scheme {scheme!r}")

    return SeedBundle(master_seed=master_seed, scheme=scheme, seeds=seeds)


def seed_list(n_seeds: int, base: int = 0) -> List[int]:
    """The canonical list of master seeds for a condition."""
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")
    return [int(base) + i for i in range(int(n_seeds))]


__all__ = ["STREAMS", "SeedBundle", "derive_seeds", "seed_list"]
