#!/usr/bin/env python3
"""
Exact (eps, k)-strong Nash stability labels of the dataset-derived games (k = 2, eps = 1).

    python3 scripts/benchmark_labels.py labels [--seeds 100] [--workers 11] [--threshold 0.25,0.35]

Labels (Appendix "More Implementation Details" of the paper): YES if max_pi Gap_2 >= eps; intermediate if
0 <= max_pi Gap_2 < eps; NO if max_pi Gap_2 < 0.  All labels are exact:
  * YES test: every partition with Gap_2 >= eps has Gap_1 >= eps (single-agent deviations belong to the
    deviation family of Gap_2), so the maximum of Gap_2 over S_1 = {pi : Gap_1(pi) >= eps} decides the question,
    and equals max_pi Gap_2 when it is at least eps.  Gap_1 is computed for every
    partition, Gap_2 for every partition of S_1, with the exhaustive checker sns.exact_sns.gap_all.
  * If not YES: the same argument with threshold 0 decides between intermediate (some pi in
    S_0 = {Gap_1 >= 0} has Gap_2 >= 0; the search stops at the first one) and NO (none has).
  * The utilities are integers, so every gap is an integer.
  * At n = 8, the full exhaustive maximum of Gap_2 over all partitions is also computed and must agree.
The witness partition of every YES game is rechecked with sns.full_deviation_checker.full_gap_k.
Output: results/rerun/benchmark/labels[_<threshold>].jsonl (one record per game and domain; --out to change)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_games as B  # noqa: E402
from sns import exact_sns as E  # noqa: E402

OUTDIR = ROOT / "results/rerun/benchmark"
EPS, K = 1.0, 2


def gaps_on(v, k, P, chunk=20_000):
    n = v.shape[0]
    out = np.empty(P.shape[0])
    saved = E._PART_CACHE.get(n)
    try:
        for c in range(0, P.shape[0], chunk):
            E._PART_CACHE[n] = np.ascontiguousarray(P[c:c + chunk], dtype=np.int8)
            out[c:c + chunk] = E.gap_all(v, k, ("L",))["L"]
    finally:
        if saved is None:
            E._PART_CACHE.pop(n, None)
        else:
            E._PART_CACHE[n] = saved
    return out


def label_game(v):
    from sns.paths import ensure_core_on_path
    ensure_core_on_path()
    from sns.full_deviation_checker import full_gap_k
    n = v.shape[0]
    P = E.all_partitions(n)
    g1 = gaps_on(v, 1, P, chunk=200_000)
    rec = {"max_gap1": float(g1.max()), "n_S1": int((g1 >= EPS).sum()), "n_S0": int((g1 >= 0).sum())}
    S1 = np.where(g1 >= EPS)[0]
    if len(S1):
        g2 = gaps_on(v, K, P[S1])
        if g2.max() >= EPS:
            w = S1[int(np.argmax(g2))]
            chk = full_gap_k(P[w].astype(np.int64), v, v, K)
            assert chk == g2.max(), (chk, g2.max())
            rec.update(label="YES", max_gap=float(g2.max()), witness=P[w].tolist(), n_yes_partitions=int((g2 >= EPS).sum()))
            return rec
    # not YES: max Gap_2 < eps, hence <= 0 (integers).  Intermediate iff some pi in S_0 has Gap_2 >= 0.
    S0 = np.where(g1 >= 0)[0]
    for c in range(0, len(S0), 20_000):
        idx = S0[c:c + 20_000]
        g2 = gaps_on(v, K, P[idx])
        if (g2 >= 0).any():
            w = idx[int(np.argmax(g2 >= 0))]
            assert full_gap_k(P[w].astype(np.int64), v, v, K) >= 0
            rec.update(label="intermediate", max_gap=0.0, witness=P[w].tolist())
            return rec
    rec.update(label="NO", max_gap=None, witness=None)
    return rec


def task(args):
    dataset, sampler, n, seed, threshold = args
    R, vs, meta = B.game(dataset, sampler, n, seed, threshold)
    out = []
    for dom, v in vs.items():
        t0 = time.perf_counter()
        rec = label_game(v)
        if n <= 8:   # full exhaustive cross-check
            mx = float(E.gap_all(v, K, ("L",))["L"].max())
            lab = "YES" if mx >= EPS else ("intermediate" if mx >= 0 else "NO")
            assert lab == rec["label"] and (rec["max_gap"] is None or rec["max_gap"] == mx or lab != "YES"), (meta, dom, mx, rec)
            rec["exhaustive_max_gap"] = mx
        rec.update(meta, domain=dom, scale=B.scale(dom, n), v=v.astype(int).tolist(), seconds=time.perf_counter() - t0)
        out.append(rec)
    return out


def _init():
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--threshold", default=B.PRIMARY_THRESHOLD)
    ap.add_argument("--sizes", default="8,10,12")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sizes = tuple(int(x) for x in a.sizes.split(","))
    suffix = "" if a.threshold == B.PRIMARY_THRESHOLD else "_" + a.threshold.replace(",", "_")
    path = Path(a.out) if a.out else OUTDIR / f"labels{suffix}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            done.add((r["dataset"], r["sampler"], r["n"], r["seed"]))
    jobs = [(d, s, n, seed, a.threshold) for n in sorted(sizes, reverse=True) for (d, s) in B.CONFIGS
            for seed in range(a.seed0, a.seed0 + a.seeds) if (d, s, n, seed) not in done]
    print(f"{len(jobs)} games to label -> {path}", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool, path.open("a") as fh:
        for i, recs in enumerate(pool.imap_unordered(task, jobs), 1):
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
