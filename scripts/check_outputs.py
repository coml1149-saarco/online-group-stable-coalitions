#!/usr/bin/env python3
"""
Compare the regenerated outputs with the reference copies shipped in results/.

    python3 scripts/check_outputs.py

  output/summaries/<f>   byte-identical to results/summaries/<f>
  output/tables/<f>      byte-identical to results/reference/tables/<f> (the tables of the paper)
  output/figures/<f>     identical to results/reference/figures/<f> (the figures of the paper)
                         after removing the PDF creation date, which matplotlib sets at write time
Exit status 1 if any reference file is missing from output/ or differs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = re.compile(rb"/CreationDate \(D:[^)]*\)")


def same(a: Path, b: Path, pdf: bool) -> bool:
    if not b.exists():
        return False
    x, y = a.read_bytes(), b.read_bytes()
    return DATE.sub(b"", x) == DATE.sub(b"", y) if pdf else x == y


def main():
    groups = [("summaries", ROOT / "results/summaries", ROOT / "output/summaries", False),
              ("tables", ROOT / "results/reference/tables", ROOT / "output/tables", False),
              ("figures", ROOT / "results/reference/figures", ROOT / "output/figures", True)]
    failed = False
    for name, ref, out, pdf in groups:
        files = sorted(p for p in ref.iterdir() if p.is_file())
        bad = [p.name for p in files if not same(p, out / p.name, pdf)]
        print(f"{name}: {len(files) - len(bad)}/{len(files)} identical" + (f"; differ or missing: {bad}" if bad else ""))
        failed |= bool(bad)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
