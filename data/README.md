# Data

## Synthetic instances

The synthetic utility matrices are not stored. They are regenerated from the seeds, by `core/ashg_lib.py` (`ASHGEnvironment`: Uniform and Gaussian models) and `sns/environments.py` (asymmetric utilities, Student-t5 noise), with environment seed `s` for seed `s`. Their exact labels are in `results/saved/labels/exact_labels.csv` (`scripts/compute_exact_labels.py`).

## Benchmark datasets (Appendix L of the paper)

The three datasets are **not redistributed** with this code. `scripts/fetch_benchmark_data.sh` downloads them into `data/benchmarks/raw/` and checks them against `data/benchmarks/SHA256SUMS`, the checksums of the files used for the paper. Only the files listed there are read.

| dataset | file read | downloaded from | cite | licence |
|---|---|---|---|---|
| Iris | `uci_iris/iris.data` | UCI Machine Learning Repository, `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data` | R. A. Fisher (1936), *Annals of Eugenics* 7(2):179–188; Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository, doi:10.24432/C56C76 | CC BY 4.0 (stated on the UCI page) |
| Zachary's karate club | `konect_zachary/ucidata-zachary/out.ucidata-zachary` | KONECT, `http://konect.cc/files/download.tsv.ucidata-zachary.tar.bz2` | W. W. Zachary (1977), *Journal of Anthropological Research* 33(4):452–473; J. Kunegis (2013), KONECT, WWW Companion, 1343–1350 | no licence stated by KONECT |
| karate club split | `networkx/social.py` (parsed as text, not executed) | NetworkX 3.3, `https://raw.githubusercontent.com/networkx/networkx/networkx-3.3/networkx/generators/social.py` | NetworkX | the NetworkX code is BSD-3-Clause; the club memberships are Zachary's data |
| jazz (network of bands) | `konect_jazz/arenas-jazz/out.arenas-jazz` | KONECT, `http://konect.cc/files/download.tsv.arenas-jazz.tar.bz2` | P. M. Gleiser and L. Danon (2003), *Advances in Complex Systems* 6(4):565–573; J. Kunegis (2013), KONECT | no licence stated by KONECT |

Notes:

- **Iris.** The UCI file contains two records that the UCI notes document as erroneous (records 35 and 38). `scripts/benchmark_games.py` replaces them with Fisher's values (`IRIS_CORRECTIONS`), which are also the values shipped by scikit-learn's `load_iris`.
- **Karate club.** The KONECT edge list has 78 edges, and the script checks that it equals the edge set of NetworkX's `karate_club_graph`. The two clubs after the split (from NetworkX) define the strata of the stratified sampler.
- **Jazz.** The network is the network of bands of Gleiser and Danon (2003). Its 198 nodes are bands, and two bands are adjacent if they have at least one musician in common; its mean distance, 2.24, agrees with the 2.26 they report for that network. KONECT's description of the nodes as musicians is inconsistent with this: the musicians network has 1,275 nodes. The construction uses only graph distances.
- Nodes in the two networks are identified only by numbers.

## Records of the dataset-derived games

The release does not distribute records from which the sampled games, and through them the structure of the karate-club and jazz networks, could be reconstructed. The same reduced format is used for Iris.

- **`results/saved/benchmark/labels*.jsonl`** (one record per game, encoding and threshold setting). Per-game scalars only:
  - the exact label and maximal gap, and the statistics of the labelling search;
  - SHA-256 digests of the sampled node indices and of the utility matrix.

  Omitted: the sampled node indices, their classes or clubs, the utility matrix, the friend/enemy/neutral counts, the witness partition and the derived seeds.
- **`results/saved/benchmark/trials.jsonl`** (one record per learning trial). Per-round declarations, regret and exact gap of the partition formed. The partitions themselves are omitted. For the partition formed at the last declaration, two scalars are stored: its number of coalitions and its Rand index with respect to the classes.
- **Local reconstruction.** `scripts/rebuild_benchmark_games.py` regenerates every game from the downloaded data with `scripts/benchmark_games.py`, checks both digests and writes the complete records to `results/local/benchmark/` (not distributed). The rebuilt records are identical to those used for the paper: sampled nodes, classes, seeds, relation counts and utility matrices, for all 10,800 records. Exact labels are recomputed by `scripts/benchmark_labels.py`. Trials with partitions are recomputed by `scripts/run_benchmark.py`.
- **Figures.** The two example-game figures of Appendix L (`bench_games_*.pdf`) are the figures published in the paper. They are kept in `results/reference/figures/` for checking.
