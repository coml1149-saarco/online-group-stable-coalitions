#!/usr/bin/env python3
"""
Rebuild the dataset-derived games of Appendix L locally from the downloaded data.

    sh scripts/fetch_benchmark_data.sh            # once
    python3 scripts/rebuild_benchmark_games.py

The public release stores only per-game scalars (label, maximal gap, labelling statistics) in
results/saved/benchmark/labels*.jsonl, together with SHA-256 digests of the sampled agents and of the
utility matrix. This script regenerates every game with scripts/benchmark_games.py (same data files, sampling
procedures, seed derivation, thresholds, encodings and scaling as in the paper), checks both digests, and
writes the complete records (sampled agents, their classes, relation counts, seeds, utility matrix) to
results/local/benchmark/labels*.jsonl, which the analysis, the figures and the benchmark runner read.
The exact labels are not recomputed here (scripts/benchmark_labels.py does that).
Exit status 1 if any regenerated game differs from the digests.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_games as B  # noqa: E402

SAVED = ROOT / "results/saved/benchmark"
LOCAL = ROOT / "results/local/benchmark"
FILES = ("labels.jsonl", "labels_0.2_0.2.jsonl", "labels_0.4_0.4.jsonl")


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()


def main():
    if not (ROOT / "data/benchmarks/raw/uci_iris/iris.data").exists():
        raise SystemExit("benchmark data not found: run  sh scripts/fetch_benchmark_data.sh  first")
    LOCAL.mkdir(parents=True, exist_ok=True)
    cache, bad, total = {}, [], 0
    for name in FILES:
        out = []
        with (SAVED / name).open() as fh:
            records = [json.loads(line) for line in fh]
        for r in records:
            key = (r["dataset"], r["sampler"], r["n"], r["seed"], r["threshold"])
            if key not in cache:
                cache[key] = B.game(*key)
            _, vs, meta = cache[key]
            v = vs[r["domain"]].astype(int).tolist()
            ok = (digest(meta["agents"]) == r["agents_sha256"] and digest(v) == r["v_sha256"]
                  and B.scale(r["domain"], r["n"]) == r["scale"])
            if not ok:
                bad.append((name,) + key + (r["domain"],))
            full = dict(r)
            for k in ("agents_sha256", "v_sha256"):
                full.pop(k)
            full.update(meta, domain=r["domain"], v=v)
            out.append(json.dumps(full))
            total += 1
        (LOCAL / name).write_text("\n".join(out) + "\n")
    print(f"{total} game records rebuilt into {LOCAL.relative_to(ROOT)}; {total - len(bad)} match the published digests")
    if bad:
        print("mismatches:", bad[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()
