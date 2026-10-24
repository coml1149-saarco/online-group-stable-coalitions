# Online Learning of Group-Stable Coalition Structures: experiment code

Code, configurations and saved results for the experiments of

> Saar Cohen and Michael Wooldridge. *Online Learning of Group-Stable Coalition Structures.* NeurIPS 2026.

Authors: Saar Cohen (`saar.cohen@cs.ox.ac.uk`) and Michael Wooldridge (`michael.wooldridge@cs.ox.ac.uk`), Department of Computer Science, University of Oxford.

## The problem and the algorithm

Agents with unknown pairwise utilities (an additively separable hedonic game) are partitioned into coalitions over $T$ rounds.
- **Semi-bandit feedback:** after each round, the learner observes a noisy utility for every pair of agents that share a coalition.
- **Per-agent coalition-level bandit feedback:** each agent reports only a noisy total utility for its coalition.

The learner has two objectives:
- form partitions close to $(\varepsilon,k)$-strong Nash stability, measured by regret;
- decide in every round whether a $k$-strongly Nash stable partition exists, measured by the existence and non-existence errors.

**UCB-LCB-GS** works as follows:
- It keeps upper and lower confidence bounds on the pairwise utilities.
- It declares existence only when a partition passes a pessimistic test.
- Otherwise it explores by *witness probing*: it observes the most uncertain pair in the deviation that most threatens its optimistic candidate.

Six baselines isolate the roles of optimism and pessimism: Pessimistic-Only, Optimistic-Only, UCB-Only, LCB-Only, Mixed and Random.

## Repository structure

| path | content |
|---|---|
| `core/` | original implementation: utility generators, partitions, the greedy local-search heuristic (`compute_partition`), the confidence-bound estimator and the six baselines |
| `sns/` | exact group-deviation checker (`full_deviation_checker.py`, `exact_sns.py`), UCB-LCB-GS with witness probing (`candidate_algorithms.py`), coalition-level bandit feedback and its variant of UCB-LCB-GS (`bandit_algorithms.py`), robustness environments, seeding, trial helpers |
| `scripts/` | experiment runners, exact labelling, game reconstruction, analysis, figures and tables, data download, checks |
| `config/` | `experiments.json` (all parameters), `final_mapping.csv` (every synthetic cell and method, with the source of its trials) |
| `data/` | dataset provenance and licences (`README.md`) and checksums; the datasets are downloaded into `data/benchmarks/raw/` |
| `results/saved/` | saved records of every reported experiment (see "What is stored") |
| `results/summaries/` | the summaries computed from them (reference copies) |
| `results/reference/` | the 26 figures and 12 tables of the paper (reference copies) |
| `tests/` | unit tests and property-based checks |
| `MANIFEST.md` | every file, and the code and data behind each figure and table of the paper |
| `LICENSE`, `RESULTS_LICENSE.md` | MIT licence of the code; licence of the results |

Generated files go to three ignored directories (`.gitignore`):
- `output/`: summaries, figures and tables;
- `results/local/`: games rebuilt from the downloaded data;
- `results/rerun/`: new trial records.

## What is stored

| data | stored here | how to obtain it otherwise |
|---|---|---|
| third-party datasets (Iris, karate club, jazz) | no | `scripts/fetch_benchmark_data.sh` downloads them and checks SHA-256 checksums |
| synthetic instances | no (generated from seeds) | `core/ashg_lib.py`, `sns/environments.py` |
| synthetic, robustness and bandit trial records, exact labels of the synthetic instances | yes, complete (`results/saved/{synthetic,robust,bandit_grid,labels}`) | `scripts/run_grid.py`, `scripts/compute_exact_labels.py` |
| dataset-derived games (sampled agents, their classes, utility matrices, relation counts) | **no**: only per-game scalars (label, maximal gap, labelling statistics) and SHA-256 digests of the sampled agents and of the utility matrix (`results/saved/benchmark/labels*.jsonl`) | `scripts/rebuild_benchmark_games.py` rebuilds every game from the downloaded data and checks the digests |
| trial records on these games | yes, without the per-round partitions (`results/saved/benchmark/trials.jsonl`); for the partition formed at the last declaration, its number of coalitions and its Rand index with respect to the classes are stored | `scripts/run_benchmark.py` (full records, including partitions) |
| aggregate summaries, figures, tables | yes | `scripts/reproduce_figures.sh` |

The karate-club and jazz networks carry no stated licence, so records from which their structure could be reconstructed are not distributed. The same reduced format is used for Iris.

## Requirements and installation

- **Python:** 3.9. The release was tested with Python 3.9.6 on macOS (Apple silicon); the experiments were run with NumPy 2.0.2 and Matplotlib 3.9.4.
- **Packages:** `requirements.txt` pins NumPy and Matplotlib, the only third-party packages.
- **System tools:** `curl`, `tar` (with bzip2) and `shasum` are needed to download the benchmark data.
- **Disk:** about 190 MB for the saved records.
- **Memory:** about 30 MB per worker process for the learning runs; up to 1 GB per worker for the exact labelling of dataset games with 12 agents.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
sh scripts/fetch_benchmark_data.sh            # Iris, karate club, jazz; verifies SHA-256 checksums
python3 scripts/rebuild_benchmark_games.py    # rebuilds the 10,800 game records (3 threshold settings); about 10 s
```

All commands below are run from this directory. `--workers` sets the number of parallel processes; it defaults to the number of CPUs minus one. Runtimes are wall-clock times on a 16-core Apple M4 Max.

## 1. Quick smoke test (about 1 minute)

```sh
python3 -m unittest discover -s tests                 # 29 tests, about 30 s
python3 scripts/run_grid.py --suite smoke --workers 4 # 780 trials, about 30 s
python3 scripts/compare_reruns.py --suite smoke       # compares them with the saved records
```

The smoke suite covers two groups of trials:
- the baselines in the cells whose trials come from the earlier historical run;
- UCB-LCB-GS in the default configuration.

`compare_reruns.py` exits with status 0 only if every trial is identical to its saved counterpart. Two tests need the downloaded data and are skipped without it. **The smoke test is a reproducibility check on a sample, not a rerun of the experiments.**

## 2. Regenerating every figure and table of the paper from the saved records (about 1 minute)

```sh
sh scripts/reproduce_figures.sh
```

The script requires the downloaded data and runs the following steps:
1. **Rebuild the games.** It rebuilds the dataset-derived games (`rebuild_benchmark_games.py`).
2. **Recompute the summaries.** It recomputes every summary from the saved records (`analyze_*.py`).
3. **Draw and write.** It draws every figure and writes every table (`make_*.py`) into `output/figures/` and `output/tables/`. The two example-game figures of Appendix L show the partition that UCB-LCB-GS formed in four trials. Those four trials are rerun locally, because the stored trial records omit partitions; each rerun must reproduce its saved record.
4. **Compare.** It runs `scripts/check_outputs.py`, which compares the results with the reference copies.

The comparisons are:
- summaries and tables must be byte-identical;
- figure PDFs must be identical apart from their creation date.

`MANIFEST.md` lists the command and inputs of each figure and table. No other learning run is repeated.

## 3. Regenerating the dataset-derived games from the downloaded data

| step | command | runtime |
|---|---|---|
| download, with checksum verification | `sh scripts/fetch_benchmark_data.sh` | seconds |
| rebuild all games and check them against the published digests | `python3 scripts/rebuild_benchmark_games.py` | about 10 s |
| recompute the exact labels (primary thresholds) | `python3 scripts/benchmark_labels.py labels` | about 40 min, up to 1 GB per worker |
| label sensitivity to the thresholds | `python3 scripts/benchmark_labels.py labels --threshold 0.2,0.2` and `--threshold 0.4,0.4` | as above, each |

`benchmark_labels.py` writes complete records, including the witness partitions, to `results/rerun/benchmark/`. To analyse them, set `SNS_GAMES=results/rerun`.

A short labelling check, on the 8-agent games of seeds 0–2, with a comparison against the published labels and digests:

```sh
python3 scripts/benchmark_labels.py labels --sizes 8 --seeds 3 --out results/rerun/benchmark/labels_check.jsonl
python3 scripts/compare_reruns.py --suite benchmark_labels --file results/rerun/benchmark/labels_check.jsonl
```

## 4. Full experimental rerun

Every runner writes `results/rerun/<suite>/trials.jsonl` and resumes an interrupted run. Two options select a representative subset:
- `--seeds A-B` restricts the seeds;
- `--match field=value` restricts any task field, for example `n=6`, `T=30` or `arm=pessimistic`.

`scripts/compare_reruns.py --suite <suite>` checks that every rerun trial is identical to the saved record. To analyse rerun records instead of the saved ones, set `SNS_RUNS=results/rerun` before `reproduce_figures.sh`. A partial rerun leaves cells incomplete, and the analysis reports this.

The full reruns, with runtimes using 11 workers:

| experiment | command | trials | runtime |
|---|---|---|---|
| synthetic grid, semi-bandit feedback (Section 5; Appendix K): ablations over $T,n,k,\sigma,\varepsilon,\beta$ with $\beta=1$ and $\beta=3$, long horizons | `python3 scripts/run_grid.py --suite synthetic` | 31,400 | 32 min |
| baselines whose trials come from the earlier run (E1 cells, see below) | `python3 scripts/run_grid.py --suite historical` | 28,200 | about 20 min (estimated from the times recorded in the earlier run) |
| coalition-level bandit feedback (Section 4; Appendix F.7) | `python3 scripts/run_grid.py --suite bandit_grid` | 54,400 | 38 min |
| robustness: asymmetric utilities, Student-$t_5$ noise (Appendix M) | `python3 scripts/run_grid.py --suite robust` and `--suite robust_smoke` | 400 + 84 | 1 min |
| dataset-derived games (Appendix L), after step 3 | `python3 scripts/run_benchmark.py` | 50,400 | 57 min |
| exact labels of the synthetic instances (compare with `compare_reruns.py --suite labels`) | `python3 scripts/compute_exact_labels.py` | 1,400 instances | 1 min |
| scalability of the exact checker (Appendix N) | `python3 scripts/bench_exact_checker.py` (single process, idle machine) | — | about 15 min |

Representative subsets, as run for the release check:

```sh
python3 scripts/run_grid.py --suite synthetic --match n=6 --match T=30 --match k=2 --seeds 0-4
python3 scripts/compare_reruns.py --suite synthetic
python3 scripts/run_grid.py --suite historical --match n=6 --match T=30 --seeds 0-4
python3 scripts/compare_reruns.py --suite historical
python3 scripts/run_grid.py --suite bandit_grid --match n=6 --match T=30 --match k=2 --seeds 0-4
python3 scripts/compare_reruns.py --suite bandit_grid
python3 scripts/run_grid.py --suite robust --seeds 0-9
python3 scripts/compare_reruns.py --suite robust
python3 scripts/run_grid.py --suite robust_smoke
python3 scripts/compare_reruns.py --suite robust_smoke
python3 scripts/run_benchmark.py --match n=8 --seeds 0-1
python3 scripts/compare_reruns.py --suite benchmark
```

**Historical trials.** In the E1 grid, the paper takes the trajectories of the six baselines in the cells with $k\le2$ from an earlier run. Those trajectories are stored in `results/saved/synthetic/historical_trials.csv`. The same applies to the robustness baselines, stored in `results/saved/robust/historical_trials.csv`.

The earlier run used a deviation checker that coincides with the exact checker in these cells. Its records store only final regret and number of declarations. Three suites rerun these trials with the released code: `historical`, `smoke` and `robust_smoke`. `compare_reruns.py` compares them with the stored values.

## Outputs

| location | content |
|---|---|
| `results/local/benchmark/labels*.jsonl` | rebuilt dataset-derived games |
| `results/rerun/<suite>/trials.jsonl` | one record per trial: configuration, label, regret, declarations per round, per-round regret, timings; benchmark trials also record the partition formed and its exact gap per round |
| `results/rerun/labels/`, `results/rerun/benchmark/labels*.jsonl` | exact labels |
| `results/rerun/scalability/bench_exact_checker.json` | timings |
| `output/summaries/` | `cells.csv`, `bandit_cells.csv`, `bandit_cross.json`, `robustness.json`, `benchmark_summary.json`, plus Markdown tables |
| `output/figures/`, `output/tables/` | figures (PDF) and LaTeX tables |

## Determinism

All randomness comes from seeded NumPy generators, and every trial is a deterministic function of its seeds. Reruns reproduced the saved trial records exactly. Several things are nonetheless not deterministic:
- The learner and trial times, and the memory fields, depend on the machine and its load. So do the two timing tables (`tab_cost.tex`, `tab_scaling.tex`), which `check_outputs.py` compares only as regenerated from the saved timings.
- The order of lines in a `trials.jsonl` file depends on process scheduling. The analysis does not depend on it.
- Floating-point results could differ on another platform or NumPy version. The utilities are integers, so the gaps are integers, and the tests check exact values.

## Datasets

The games of Appendix L are built from three datasets:
- Iris (Fisher, 1936; UCI Machine Learning Repository, CC BY 4.0);
- Zachary's karate club (Zachary, 1977), through KONECT (Kunegis, 2013);
- the jazz network of Gleiser and Danon (2003), whose nodes are bands, through KONECT.

None of them is redistributed here. Sources, citations, licences and the preprocessing are in `data/README.md`.

## Theoretical oracle and implemented heuristic

The guarantees of the paper assume that the optimistic max-min problem is solved exactly. Theorems 3.2, 3.3 and 4.1(b, c) depend on this assumption. Soundness (Theorems 3.1 and 4.1(a)) needs only an exact pessimistic test.

The code works as follows:
- **Pessimistic test.** It is evaluated exactly, with the complete deviation family and the exact checker (`sns/full_deviation_checker.py`, $k\le3$).
- **Candidate partitions.** They are computed by the greedy local-search heuristic `compute_partition` (`core/ashg_lib.py`): 3 restarts and 60 sweeps, with local optima rescored by the exact checker. All methods share this heuristic.

The heuristic has no approximation guarantee. Appendix G of the paper bounds the existence error and regret by the number of rounds in which the heuristic's optimistic candidate is insufficient. The logarithmic bounds therefore apply to the experiments only if such rounds are rare, and the experiments do not certify this.

## Known limitations

- **Instance size.** Exact deviation checking and exact labelling limit the experiments to $n\le7$ synthetic agents and $n\le12$ agents in the dataset-derived games, with $k\le3$. Checking grows combinatorially in $n$ and $k$ (Appendix N).
- **Parameter range.** The synthetic grid uses $\beta=1$ in its first part, outside the range $\beta>2$ covered by the theorems. The $\beta=3$ grid repeats the ablations inside it.
- **Historical trials.** They are stored only as final regret and number of declarations, without per-round records.
- **Benchmark records.**
  - The dataset-derived results need the downloaded data. Without it, only the synthetic, robustness and bandit results can be regenerated.
  - Two statistics of the declared partitions in the dataset games come from stored per-trial values: the number of coalitions and the Rand index with respect to the classes. They can be checked independently by rerunning those trials with `run_benchmark.py`.
- **Timings.** They are machine-specific.
- **Unreported variants.** The registries contain variants not reported in the paper: the original implementation (`ucb_lcb_gs`), coverage-probe variants, the grand-coalition witness probe and the round-robin bandit variants. They are kept because the reported methods are built on the same classes, and the original implementation appears in the smoke checks.

## Licence

- **Code, tests, configuration and documentation:** MIT licence (`LICENSE`), © 2026 Saar Cohen and Michael Wooldridge.
- **Experimental results in `results/`:** CC BY 4.0, except the material excluded in `RESULTS_LICENSE.md`. That file lists the two figures that depict third-party data and the per-game and per-trial records of the dataset-derived games.
- **Third-party datasets:** they are not part of this repository, and their own terms apply (`data/README.md`).

## Citation

If you use this code or these results, please cite the paper. `CITATION.cff` gives the citation metadata.
