# Licence of the experimental results

The code, tests, configuration and documentation of this repository are released under the MIT licence (`LICENSE`). This file sets out the licence of the experimental results in `results/`.

## Released under CC BY 4.0

The following files are licensed under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), © 2026 Saar Cohen and Michael Wooldridge. They contain original numerical results of our experiments, aggregate statistics computed from them, and the plots and tables made from them. When you use them, please cite the paper (see `CITATION.cff`).

| path | content |
|---|---|
| `results/saved/synthetic/` | trial records of the synthetic experiments, including the historical baseline trials |
| `results/saved/robust/` | trial records of the robustness study, including the historical baseline trials |
| `results/saved/bandit_grid/` | trial records of the coalition-level bandit experiments |
| `results/saved/labels/exact_labels.csv` | exact existence labels of the synthetic instances |
| `results/saved/scalability/bench_exact_checker.json` | timings of the exact deviation checker on random instances |
| `results/summaries/` | aggregate statistics (`cells.csv`, `bandit_cells.csv`, `bandit_cross.json`, `robustness.json`, `benchmark_summary.json`) |
| `results/reference/tables/` | the 12 tables of the paper |
| `results/reference/figures/` | the figures of the paper, **except** the two listed under "Not covered" below |

`benchmark_summary.json`, the `tab_bench_*` tables, the dataset rows of `tab_cost.tex` and the figures `bench_{semi,bandit}_{regret,error}.pdf` and `bench_results.pdf` contain only aggregate statistics of our learning runs on the games derived from the datasets. They contain no data from the datasets themselves.

## Not covered by CC BY 4.0

**Third-party data.** None of the datasets is included in this repository, and no licence in this repository applies to them:
- Iris (UCI Machine Learning Repository, CC BY 4.0);
- Zachary's karate club and the jazz network of Gleiser and Danon (KONECT; no licence stated);
- the NetworkX file used for the karate-club split (BSD-3-Clause).

`scripts/fetch_benchmark_data.sh` downloads them from their sources, and their terms apply. Sources and citations are in `data/README.md`.

**Figures that depict third-party data.**
- `results/reference/figures/bench_games_iris_karate.pdf` shows Iris measurements and part of the karate-club network.
- `results/reference/figures/bench_games_jazz.pdf` shows part of the jazz network.

They are figures of the paper, included only so that `scripts/check_outputs.py` can verify the regenerated figures. The Iris part is subject to the attribution requirement of CC BY 4.0.

**Per-game and per-trial records of the dataset-derived games (unresolved).** These records are provided for reproducibility. They are not placed under CC BY 4.0 while the licensing question for the karate-club and jazz networks, from which some of the games are derived, remains open:
- `results/saved/benchmark/labels.jsonl`, `labels_0.2_0.2.jsonl` and `labels_0.4_0.4.jsonl`: per-game labels, gaps, labelling statistics and digests;
- `results/saved/benchmark/trials.jsonl`: per-trial declarations, regret and gaps.

They contain no node identifiers, classes, utility matrices, relations or partitions, and the games themselves must be rebuilt from the downloaded data.
