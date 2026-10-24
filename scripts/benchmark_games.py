#!/usr/bin/env python3
"""
Hedonic games derived from clustering and community-detection benchmarks, following
Csaji, Gundert, Rothe and Schlotter, "Clustering via Hedonic Games: New Concepts and Algorithms" (NeurIPS 2025).

Construction (Section 5 of Csaji et al.).  For data points x, y of a data set with distance d(x, y)
(Euclidean distance between feature vectors; number of edges of a shortest path in a network) and the
diameter diam = max d over the whole data set,
    y is a friend of x  iff  d(x, y) <= f * diam,
    y is an enemy of x  iff  d(x, y) >  e * diam   (and y is not a friend),
    otherwise y is neutral.
A FEN game of type (a_f, a_e) has v_xy = a_f for friends, a_e for enemies and 0 for neutrals; the three
preference domains are AF = (n, -1), AE = (1, -n) and B = (1, -1), where n is the number of agents.
A sampled game is the restriction of the full-data-set game to the sampled agents: distances and the
diameter are those of the full data set (graph distances are taken in the full network), so the relation
between two agents does not depend on which other agents are sampled.  Utilities are the type multiplied by
c = floor(100 / max(|a_f|, |a_e|)), the largest integer for which every utility lies in [-100, 100], the
range of the synthetic Uniform model; scaling by c > 0 changes no preference and no stability label
(Gap scales by c and labels use eps = 1 with integer type gaps), only the signal-to-noise ratio.

No learning algorithm, utility generator of the completed experiments, or stored result is modified.
"""

from __future__ import annotations

import re
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/benchmarks/raw"
UCI_IRIS = RAW / "uci_iris/iris.data"   # fetched by scripts/fetch_benchmark_data.sh

THRESHOLDS = {"0.25,0.35": (0.25, 0.35), "0.2,0.2": (0.2, 0.2), "0.4,0.4": (0.4, 0.4)}
PRIMARY_THRESHOLD = "0.25,0.35"
DOMAINS = ("AF", "AE", "B")
# Fisher's (1936) values for the two UCI records documented as erroneous in the UCI repository's notes
# (1-based records 35 and 38); scikit-learn's load_iris, used by Csaji et al., ships these corrected values.
IRIS_CORRECTIONS = {34: [4.9, 3.1, 1.5, 0.2], 37: [4.9, 3.6, 1.4, 0.1]}


def domain_type(domain: str, n: int):
    return {"AF": (n, -1), "AE": (1, -n), "B": (1, -1)}[domain]


def scale(domain: str, n: int) -> int:
    af, ae = domain_type(domain, n)
    return 100 // max(abs(af), abs(ae))


# ---------------------------------------------------------------- data sets
@lru_cache(None)
def iris():
    X, y = [], []
    for line in UCI_IRIS.open():
        p = line.strip().split(",")
        if len(p) == 5:
            X.append([float(t) for t in p[:4]]); y.append(p[4])
    X = np.array(X)
    for i, row in IRIS_CORRECTIONS.items():
        X[i] = row
    classes = sorted(set(y))
    labels = np.array([classes.index(t) for t in y])
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return {"name": "iris", "X": X, "labels": labels, "label_names": classes, "D": D, "kind": "features"}


def _bfs_distances(n, adj):
    D = np.full((n, n), np.inf)
    for s in range(n):
        dist = {s: 0}; dq = deque([s])
        while dq:
            x = dq.popleft()
            for w in adj[x]:
                if w not in dist:
                    dist[w] = dist[x] + 1; dq.append(w)
        for w, d in dist.items():
            D[s, w] = d
    return D


def _read_konect(path):
    edges = set()
    for line in path.open():
        if line.startswith("%") or not line.strip():
            continue
        a, b = map(int, line.split()[:2])
        if a != b:
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


@lru_cache(None)
def karate():
    edges = _read_konect(RAW / "konect_zachary/ucidata-zachary/out.ucidata-zachary")
    n = 34
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a - 1].add(b - 1); adj[b - 1].add(a - 1)
    # club membership after the split, as recorded in NetworkX's karate_club_graph (parsed as text, not executed)
    src = (RAW / "networkx/social.py").read_text()
    club1 = set(map(int, re.search(r"club1 = \{([0-9, ]+)\}", src).group(1).split(",")))
    block = re.search(r'zacharydat = """\\\n(.*?)"""', src, re.S).group(1).strip().split("\n")
    W = np.array([[int(t) for t in row.split()] for row in block[:n]])
    nx_edges = {(i + 1, j + 1) for i in range(n) for j in range(i + 1, n) if W[i, j] > 0 or W[j, i] > 0}  # as NetworkX adds them
    assert nx_edges == set(edges), "KONECT and NetworkX edge lists differ"
    labels = np.array([0 if i in club1 else 1 for i in range(n)])
    D = _bfs_distances(n, adj)
    assert np.isfinite(D).all()
    return {"name": "karate", "edges": edges, "adj": adj, "labels": labels, "label_names": ["Mr. Hi", "Officer"],
            "D": D, "kind": "graph"}


@lru_cache(None)
def jazz():
    edges = _read_konect(RAW / "konect_jazz/arenas-jazz/out.arenas-jazz")
    nodes = sorted({u for e in edges for u in e})
    idx = {u: k for k, u in enumerate(nodes)}
    n = len(nodes)
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[idx[a]].add(idx[b]); adj[idx[b]].add(idx[a])
    D = _bfs_distances(n, adj)
    assert np.isfinite(D).all()
    return {"name": "jazz", "edges": edges, "adj": adj, "labels": None, "label_names": None, "D": D, "kind": "graph"}


DATASETS = {"iris": iris, "karate": karate, "jazz": jazz}


# ---------------------------------------------------------------- relations and utilities
def relations(D, diam, f, e):
    """+1 friend, -1 enemy, 0 neutral (published rule; zero diagonal)."""
    F = D <= f * diam
    E = (D > e * diam) & ~F
    R = F.astype(int) - E.astype(int)
    np.fill_diagonal(R, 0)
    return R


def utilities(R, domain):
    n = R.shape[0]
    af, ae = domain_type(domain, n)
    c = scale(domain, n)
    v = np.where(R > 0, c * af, np.where(R < 0, c * ae, 0)).astype(float)
    np.fill_diagonal(v, 0.0)
    return v


# ---------------------------------------------------------------- sampling
def sample(dataset, sampler, n, rng):
    d = DATASETS[dataset]()
    N = d["D"].shape[0]
    if sampler == "stratified":
        labels = d["labels"]
        groups = [np.where(labels == c)[0] for c in range(labels.max() + 1)]
        g = len(groups)
        counts = np.full(g, n // g)
        counts[rng.choice(g, size=n % g, replace=False)] += 1
        S = np.concatenate([rng.choice(grp, size=k, replace=False) for grp, k in zip(groups, counts)])
    elif sampler == "uniform":
        S = rng.choice(N, size=n, replace=False)
    elif sampler == "snowball":
        adj = d["adj"]
        s = int(rng.integers(N)); seen = [s]; dq = deque([s])
        while len(seen) < n:
            if not dq:   # component exhausted (cannot happen for the connected jazz network)
                rest = [u for u in range(N) if u not in seen]; s = int(rng.choice(rest)); seen.append(s); dq.append(s)
                continue
            x = dq.popleft()
            nb = sorted(adj[x]); rng.shuffle(nb)
            for w in nb:
                if w not in seen and len(seen) < n:
                    seen.append(w); dq.append(w)
        S = np.array(seen)
    else:
        raise ValueError(sampler)
    return np.sort(S.astype(int))


CONFIGS = [("iris", "stratified"), ("karate", "stratified"), ("jazz", "uniform"), ("jazz", "snowball")]
SIZES = (8, 10, 12)


def game(dataset, sampler, n, seed, threshold=PRIMARY_THRESHOLD):
    """Agents, relation matrix and the three utility matrices of one game, plus its seeds."""
    from sns.seeding import derive_seeds
    d = DATASETS[dataset]()
    tag = f"bm|{dataset}|{sampler}|n{n}"
    b = derive_seeds(seed, streams=(tag + "|subset", tag + "|env", tag + "|algorithm"), scheme="hashed")
    S = sample(dataset, sampler, n, b.rng(tag + "|subset"))
    f, e = THRESHOLDS[threshold]
    diam = float(d["D"].max())
    Ds = d["D"][np.ix_(S, S)]
    ties = bool(np.any(np.isclose(Ds[np.triu_indices(n, 1)], f * diam)) or np.any(np.isclose(Ds[np.triu_indices(n, 1)], e * diam)))
    R = relations(Ds, diam, f, e)
    vs = {dom: utilities(R, dom) for dom in DOMAINS}
    meta = {"dataset": dataset, "sampler": sampler, "n": n, "seed": seed, "threshold": threshold, "agents": S.tolist(),
            "labels": None if d["labels"] is None else d["labels"][S].tolist(), "diameter": diam, "ties_at_threshold": ties,
            "env_seed": b[tag + "|env"], "alg_seed": b[tag + "|algorithm"],
            "friend_pairs": int((R > 0).sum() // 2), "enemy_pairs": int((R < 0).sum() // 2),
            "neutral_pairs": int(n * (n - 1) // 2 - (R > 0).sum() // 2 - (R < 0).sum() // 2)}
    return R, vs, meta
