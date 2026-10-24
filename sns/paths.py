"""
sns.paths
=========

Locate the repository root and put the original implementation (``core/``)
on ``sys.path`` so that ``ashg_lib``, ``algorithms`` and ``experiment`` are
importable unmodified.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "core"


def ensure_core_on_path() -> str:
    """Prepend ``core/`` to ``sys.path`` (idempotent)."""
    p = str(CORE_DIR)
    if not CORE_DIR.is_dir():
        raise RuntimeError(f"cannot find the implementation at {CORE_DIR!s}")
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def ensure_dir(path) -> Path:
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path


__all__ = ["REPO_ROOT", "CORE_DIR", "ensure_core_on_path", "ensure_dir"]
