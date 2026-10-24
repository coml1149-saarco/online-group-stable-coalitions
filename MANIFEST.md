# Manifest: experiment-reproduction

## Figures and tables of the paper

Commands are run from this directory (`python3 scripts/<script>`); `sh scripts/reproduce_figures.sh` runs the whole chain from the saved records. Each file is written to `output/figures/` or `output/tables/` and compared with its reference copy in `results/reference/` by `scripts/check_outputs.py`.

| paper | file | regenerate from saved records | inputs | full rerun |
|---|---|---|---|---|
| Figure 1 (p. 32) | `feedback_comparison.pdf` | `analyze_bandit.py` → `make_feedback_figure.py` | `results/saved/bandit_grid/trials.jsonl`, `results/saved/synthetic/trials.jsonl` | `run_grid.py --suite bandit_grid`, `--suite synthetic` |
| Figure 2 (p. 34) | `bandit_E1_T.pdf` | `analyze_synthetic.py` → `analyze_bandit.py` → `make_bandit_figures.py` | `results/saved/bandit_grid/trials.jsonl`, `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite bandit_grid` |
| Figure 3 (p. 34) | `bandit_E1_n.pdf` | `analyze_synthetic.py` → `analyze_bandit.py` → `make_bandit_figures.py` | `results/saved/bandit_grid/trials.jsonl`, `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite bandit_grid` |
| Figure 4 (p. 35) | `bandit_E1_sigma.pdf` | `analyze_synthetic.py` → `analyze_bandit.py` → `make_bandit_figures.py` | `results/saved/bandit_grid/trials.jsonl`, `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite bandit_grid` |
| Figure 5 (p. 46) | `E1_T.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` (and `--suite historical` for the E1 baselines with $k\le2$) |
| Figure 6 (p. 47) | `E2_T.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` |
| Figure 7 (p. 48) | `E1_eps.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` (and `--suite historical` for the E1 baselines with $k\le2$) |
| Figure 8 (p. 49) | `E2_eps.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` |
| Figure 9 (p. 50) | `E1_n.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` (and `--suite historical` for the E1 baselines with $k\le2$) |
| Figure 10 (p. 51) | `E2_n.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` |
| Figure 11 (p. 52) | `E1_k.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` (and `--suite historical` for the E1 baselines with $k\le2$) |
| Figure 12 (p. 52) | `E2_k.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` |
| Figure 13 (p. 53) | `E1_sigma.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` (and `--suite historical` for the E1 baselines with $k\le2$) |
| Figure 14 (p. 54) | `E2_sigma.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` |
| Figure 15 (p. 55) | `E1_beta.pdf` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic` (and `--suite historical` for the E1 baselines with $k\le2$) |
| Figure 16 (p. 56) | `E3_curves_beta1.pdf` | `make_figures.py` | `results/saved/synthetic/trials.jsonl` (per-round records, $T=1000$) | `run_grid.py --suite synthetic` |
| Figure 17 (p. 57) | `E3_curves_beta3.pdf` | `make_figures.py` | `results/saved/synthetic/trials.jsonl` (per-round records, $T=1000$) | `run_grid.py --suite synthetic` |
| Figure 18 (p. 62) | `bench_games_iris_karate.pdf` | `rebuild_benchmark_games.py` → `make_benchmark_figures.py` (reruns the four example trials) | downloaded data (`data/benchmarks/raw/`), `results/saved/benchmark/labels.jsonl` (digests), `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels`, `run_benchmark.py` (the games themselves: `rebuild_benchmark_games.py`) |
| Figure 19 (p. 63) | `bench_games_jazz.pdf` | `rebuild_benchmark_games.py` → `make_benchmark_figures.py` (reruns the four example trials) | downloaded data (`data/benchmarks/raw/`), `results/saved/benchmark/labels.jsonl` (digests), `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels`, `run_benchmark.py` (the games themselves: `rebuild_benchmark_games.py`) |
| Figure 20 (p. 64) | `bench_semi_regret.pdf` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (also `--threshold 0.2,0.2` and `0.4,0.4`), `run_benchmark.py` |
| Figure 21 (p. 65) | `bench_semi_error.pdf` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (also `--threshold 0.2,0.2` and `0.4,0.4`), `run_benchmark.py` |
| Figure 22 (p. 67) | `bench_bandit_regret.pdf` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (also `--threshold 0.2,0.2` and `0.4,0.4`), `run_benchmark.py` |
| Figure 23 (p. 68) | `bench_bandit_error.pdf` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (also `--threshold 0.2,0.2` and `0.4,0.4`), `run_benchmark.py` |
| Figure 24 (p. 69) | `bench_results.pdf` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (also `--threshold 0.2,0.2` and `0.4,0.4`), `run_benchmark.py` |
| Figure 25 (p. 75) | `robustness_yes_bars.pdf` | `analyze_robustness.py` → `make_robustness_figures.py` | `results/saved/robust/trials.jsonl`, `results/saved/robust/historical_trials.csv`, `results/saved/synthetic/trials.jsonl`, `results/saved/labels/exact_labels.csv` | `run_grid.py --suite robust` (baselines: `--suite robust_smoke` reruns a sample) |
| Figure 26 (p. 76) | `robustness_no_bars.pdf` | `analyze_robustness.py` → `make_robustness_figures.py` | `results/saved/robust/trials.jsonl`, `results/saved/robust/historical_trials.csv`, `results/saved/synthetic/trials.jsonl`, `results/saved/labels/exact_labels.csv` | `run_grid.py --suite robust` (baselines: `--suite robust_smoke` reruns a sample) |
| Table 1 (p. 32) | `tab_bandit_main.tex` | `analyze_bandit.py` → `make_bandit_figures.py` | `results/saved/bandit_grid/trials.jsonl`, `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite bandit_grid` |
| Table 2 (p. 32) | `tab_bandit_shared.tex` | `analyze_bandit.py` → `make_bandit_figures.py` | `results/saved/bandit_grid/trials.jsonl`, `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite bandit_grid` |
| Table 3 (p. 46) | `tab_main.tex` | `analyze_synthetic.py`, `analyze_robustness.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv`, `results/saved/robust/*` | `run_grid.py --suite synthetic`, `--suite historical`, `--suite robust` |
| Table 4 (p. 58) | `tab_noinst.tex` | `analyze_synthetic.py` → `make_figures.py` | `results/saved/synthetic/trials.jsonl`, `results/saved/synthetic/historical_trials.csv`, `results/saved/labels/exact_labels.csv`, `config/final_mapping.csv` | `run_grid.py --suite synthetic`, `--suite historical` |
| Table 5 (p. 61) | `tab_bench_labels.tex` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (all three thresholds), `run_benchmark.py` |
| Table 6 (p. 65) | `tab_bench_intermediate.tex` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (all three thresholds), `run_benchmark.py` |
| Table 7 (p. 70) | `tab_bench_semi_regret.tex` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (all three thresholds), `run_benchmark.py` |
| Table 8 (p. 71) | `tab_bench_semi_error.tex` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (all three thresholds), `run_benchmark.py` |
| Table 9 (p. 72) | `tab_bench_bandit_regret.tex` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (all three thresholds), `run_benchmark.py` |
| Table 10 (p. 73) | `tab_bench_bandit_error.tex` | `rebuild_benchmark_games.py` → `analyze_benchmark.py` → `make_benchmark_figures.py` | downloaded data, `results/saved/benchmark/labels*.jsonl`, `results/saved/benchmark/trials.jsonl` | `benchmark_labels.py labels` (all three thresholds), `run_benchmark.py` |
| Table 11 (p. 75) | `tab_cost.tex` | `make_figures.py` | learner times in `results/saved/synthetic/trials.jsonl` and `results/saved/benchmark/trials.jsonl` | `run_grid.py --suite synthetic`, `run_benchmark.py` (machine-specific) |
| Table 12 (p. 76) | `tab_scaling.tex` | `make_figures.py` | `results/saved/scalability/bench_exact_checker.json` (from `bench_exact_checker.py`) | `bench_exact_checker.py` (machine-specific) |

Written by the scripts but not included in the paper: `E2_beta.pdf` (the $\beta=3$ grid has a single $\beta$), `bandit_E1_{beta,eps,k}.pdf`, `tab_bandit_cross.tex`, and the Markdown summaries in `output/summaries/`.

## Files

| file | size | SHA-256 |
|---|---|---|
| `.gitignore` | 300 B | `43c64bf7a1dd2b9098112fbb16ea07cd78315f2428a4a539866e0a00d5d18a8d` |
| `CITATION.cff` | 1.0 KB | `e228475ec4988f8c23792837138b4ef58afee6ad4f9a13a1786e31a3fdbba8ea` |
| `LICENSE` | 1.1 KB | `d3785f5223d37c7695a207b5479be70eba37d2745de5b13fe309f3877c0093bc` |
| `README.md` | 16.5 KB | `0aa64fc811ef96be63b2ee5503648ab2bf0e619434cf9738822c137f5b19c9d7` |
| `RESULTS_LICENSE.md` | 3.4 KB | `061168767756637d8232f0c6286ed29898573bfe2b197870ff2e0e83399a46c7` |
| `config/experiments.json` | 6.1 KB | `b9facde2af89c36ddd5a967f294aac495f7ec233319bfebac0e45e828046c0c8` |
| `config/final_mapping.csv` | 149.6 KB | `a51c342558b0d58977864c9b6aac04ab598da6c06c9a66f87376eca385d7c8a7` |
| `core/algorithms.py` | 9.7 KB | `5dabc99bc588883e5f7ed34e00939f5730e9a707cd590a72487fe359b14f5ec9` |
| `core/ashg_lib.py` | 13.5 KB | `3129cbb23fe212f712ac69fe0462dbb75cc0895ec1d2f11bdc64807f94f1fcbf` |
| `core/experiment.py` | 10.0 KB | `65c8addf62946848d8030c88027277d0947358e80b72d0123f9bdcfc0a3b0c86` |
| `data/README.md` | 4.6 KB | `95c7b434ad2ce2fc8840e4686f6af5721697eb7eb28a465af0613c927737dc0f` |
| `data/benchmarks/SHA256SUMS` | 393 B | `583b437bc16edc6c357f9168cbb9460e69189112de4c3087368f55b1d31c14a3` |
| `requirements.txt` | 89 B | `38db9b61217ed5b95154ad35ef2574c1338a8272049db2e6b595c1abaa9af396` |
| `results/reference/figures/E1_T.pdf` | 29.1 KB | `478908e10d34b723d0080748bca9524dfcea073347b44d3c31cb6549e741e1d2` |
| `results/reference/figures/E1_beta.pdf` | 29.7 KB | `35c9504bcfaef18f1bd9af9d97544f3d8d6ed09b0630f5752a5b160827caecda` |
| `results/reference/figures/E1_eps.pdf` | 30.2 KB | `5b07dbd1d55c94d9c4f65979a6b35e84b8efe1b05d26b6b5683d3c10969b71bb` |
| `results/reference/figures/E1_k.pdf` | 27.9 KB | `7f0efa7a5003aa07d1fd12c3d856df63b812d02090f3744d3dccac2a9e6c7458` |
| `results/reference/figures/E1_n.pdf` | 29.3 KB | `749bce66a0e42e2b3b6e6ee0a9356044fa3731569ed6c0199a0cc144d8304286` |
| `results/reference/figures/E1_sigma.pdf` | 29.5 KB | `b2760011329162058c73eccf0087c43adc748a8f94173aba265edcf9cb0f97f2` |
| `results/reference/figures/E2_T.pdf` | 29.1 KB | `9f34e2b556018bd79c77d8e753ad960525f43890c35ef8b3b92c4232150d3a34` |
| `results/reference/figures/E2_eps.pdf` | 30.2 KB | `01d45b8ad527195cd5996ced988dee285b7c6ad8dc8f29a611fa75c04ebf5ae4` |
| `results/reference/figures/E2_k.pdf` | 27.9 KB | `64eaeba34cd3f5d9a418c45cd104b79dedd8ebe05b86f8a0ce8c384e1adef473` |
| `results/reference/figures/E2_n.pdf` | 29.3 KB | `4cafdf99c5f5d774dfacfcea6c533b06a8fc105fd6eee9490c41e8939466a434` |
| `results/reference/figures/E2_sigma.pdf` | 29.5 KB | `1e8052648c0c4aa5af25a6cee3cb9cadb237608c8d787f1fb25ad9154ef3274b` |
| `results/reference/figures/E3_curves_beta1.pdf` | 477.8 KB | `71b430682b27bb269622be14359eb4441d77d333e1c6355c6f07a8fe9246f132` |
| `results/reference/figures/E3_curves_beta3.pdf` | 481.0 KB | `02976c0d3783ffe8b7fe96d34c20efe81bed5476973d936e85b4f7a1ef2ed9d7` |
| `results/reference/figures/bandit_E1_T.pdf` | 29.2 KB | `9a1f197cdfddbaff1437078ab4d525fb17745bfa1fb243a4b1956b9dc55cfc84` |
| `results/reference/figures/bandit_E1_n.pdf` | 29.5 KB | `4ff86d00c88e1c76cc7bd7cf5923bd55d6338dbb2e11c104672526b1ccffd4c8` |
| `results/reference/figures/bandit_E1_sigma.pdf` | 30.0 KB | `fcf8697f5e0a7c200900b2c3ad2b544179c225925cd32d2aa6bc579f0464bc60` |
| `results/reference/figures/bench_bandit_error.pdf` | 46.0 KB | `1321060ca473c611c7c959e92c00dd37de0a3c6a5e14b840a747675b6617f069` |
| `results/reference/figures/bench_bandit_regret.pdf` | 48.6 KB | `9e2eba827c194b459dbb11a77c2581a0d54cb3f9227c3471af55bdcef6932e38` |
| `results/reference/figures/bench_games_iris_karate.pdf` | 21.7 KB | `d2ad5ea09319d1ae6db99fd32bd6a59b829252ebb4171d85e30981315896f72a` |
| `results/reference/figures/bench_games_jazz.pdf` | 48.9 KB | `89ed4435be6caae5990919caeb2ccd59b04f9308815c15383dca6aa2a651d640` |
| `results/reference/figures/bench_results.pdf` | 37.2 KB | `41d1b548a26e7d894cfb2f0a0d48ad83a463f7dc77764f7dbdb9ae0fd589ffa7` |
| `results/reference/figures/bench_semi_error.pdf` | 46.3 KB | `78dc5e51920805a9fe79ecc7e3779ddc7c422ce8cfc7494060875933dfebe4ba` |
| `results/reference/figures/bench_semi_regret.pdf` | 49.0 KB | `ffb6f9e2d88a5d67629d13812f0592d006e2b1579d56a011e92a55eec72e7a02` |
| `results/reference/figures/feedback_comparison.pdf` | 21.4 KB | `e93349e9eed2c61a3823b65e8315e38d4f5cbf6411d8c8a037d3d0c978d053ab` |
| `results/reference/figures/robustness_no_bars.pdf` | 23.6 KB | `1cbd293b16be39402111c2eff20e2e7bfa1bf46f5c34704ecb0dfba3e72b2832` |
| `results/reference/figures/robustness_yes_bars.pdf` | 37.7 KB | `2e90474d59feae21b87503e3263a36bf159c7c16f69c54acc5fc0956f6d8a9e1` |
| `results/reference/tables/tab_bandit_main.tex` | 1.4 KB | `8766be0981ac21c8d74547fb945462f2e7107c7d48afe9a852cc18794f53249b` |
| `results/reference/tables/tab_bandit_shared.tex` | 1.4 KB | `9b51c6dd3e2cd31c4bd6760551eb0cc19738d4912e03e41d4dccbd49933d8388` |
| `results/reference/tables/tab_bench_bandit_error.tex` | 8.0 KB | `785c00306eb63e33eabe62e56f694da01089623f1ce5c932d8edabb071d4aecb` |
| `results/reference/tables/tab_bench_bandit_regret.tex` | 8.4 KB | `7ab6fbb11416b5aaa2b79c094c8384bda508f6dbc7d28716f62ac08922dabdea` |
| `results/reference/tables/tab_bench_intermediate.tex` | 2.5 KB | `772cd147e5d9ba4ba59ee0bc73ff48313f5ffb779bca28b7584e990f9d92f073` |
| `results/reference/tables/tab_bench_labels.tex` | 1.5 KB | `09b5b0d38a6b5d874c563e9ae122e6df830d60888b68346b4000589bd2ff8c28` |
| `results/reference/tables/tab_bench_semi_error.tex` | 8.0 KB | `0153908c81018e8c49a033d080a9079632cb6887d17c4356fa29dde420035bc5` |
| `results/reference/tables/tab_bench_semi_regret.tex` | 8.1 KB | `82842f0036d309f509d72db37935a2bc5cbf4ee83825d0501b477dfd8b21681a` |
| `results/reference/tables/tab_cost.tex` | 743 B | `2eeb166417613a341604906ff16b68303c1522558d56c34e9ae43093251a8cfc` |
| `results/reference/tables/tab_main.tex` | 1.5 KB | `f9afc5cf87a3c57f620c979e90087007d2442a9e45fcd4bc728f8131e0c4fe08` |
| `results/reference/tables/tab_noinst.tex` | 1.1 KB | `eec2d0ba12217c91641bd1530353e3dc9bfafb0fbe9ba8c9c119e6273e474f37` |
| `results/reference/tables/tab_scaling.tex` | 1.3 KB | `6d573de5c3ba213dfd22f1851879a658ae3b13e8bf6fa74c6ae1939e752bd655` |
| `results/saved/bandit_grid/trials.jsonl` | 71.4 MB | `bf135392e52cc3a6a372c4705eefaba676098e6514bba962572b5d4952933974` |
| `results/saved/benchmark/labels.jsonl` | 1.4 MB | `98812c2923c0ad670f2bb215f61a63887a085d591607ee0de60eda27da36c2a5` |
| `results/saved/benchmark/labels_0.2_0.2.jsonl` | 1.4 MB | `8199a51ff13113772226aa7a84157b6f4c28d01c391a50fca35f902ece2dda78` |
| `results/saved/benchmark/labels_0.4_0.4.jsonl` | 1.4 MB | `bd012b7d73ef1e0623aa1ba2ba47a40b64086f332d7817bbab3ffa00a79a5f77` |
| `results/saved/benchmark/trials.jsonl` | 54.9 MB | `93cdba0d71b7979272b220b8ef7fe6f0039e61809c7d328ab00737cae4292fe1` |
| `results/saved/labels/exact_labels.csv` | 60.2 KB | `04057b588d366d67b61f83ba894463716149ba4e8a53bc1337a6b29bc1758bca` |
| `results/saved/robust/historical_trials.csv` | 947.7 KB | `776a7577f2df38a05c3836705f2aa151cbf4462f5a344b744ab5bc9d7dac3cd9` |
| `results/saved/robust/trials.jsonl` | 332.4 KB | `a06e667e5dc29cb8ba98e8a048fab9f45cfc27263ea4b46f3f221617aec2751b` |
| `results/saved/scalability/bench_exact_checker.json` | 3.1 KB | `07171b80f54fd49a076e1f05371ca196168c4745af7d4ce80df479e26c8904b6` |
| `results/saved/synthetic/historical_trials.csv` | 7.7 MB | `eb8f963e368ac71414f1c24a8f9e55d16ad26f24bf6988847c2fa10f8dcc2322` |
| `results/saved/synthetic/trials.jsonl` | 48.4 MB | `e05437b2dbfe0b417e13a3103725f1cc167ff7b301c4097830538fb3cf15dd4a` |
| `results/summaries/bandit_cells.csv` | 132.1 KB | `8820f48f4c3a384e09c7313f86839c9b138fedef9dd41868cdf54b093244e4f9` |
| `results/summaries/bandit_cross.json` | 70.8 KB | `f2f39c287e9fe76bb0bc55375874ae53116d83457e3613b20c74253f0bed689a` |
| `results/summaries/benchmark_summary.json` | 424.4 KB | `0e28c44e041e61e92cc54acb99200495c7cef7de64dd63b3c75f4a36f30b3b09` |
| `results/summaries/cells.csv` | 172.2 KB | `f6f65c850679d5dc268a5971f5a7b8fdec7d98527f1ee3f9e376db24b1549aa7` |
| `results/summaries/robustness.json` | 12.1 KB | `77a44e48e6f87c139fd25cab7f976a250af4082956bf28240bdb9b99e28765a1` |
| `scripts/analyze_bandit.py` | 7.6 KB | `4edd90532b69f90dfb2f427d018ecee5a46f53ba3ceb7aad7cf1fd948fcadc5d` |
| `scripts/analyze_benchmark.py` | 7.8 KB | `ea03b8583f4490beda6b2690dd3659d3652c4a3656ca6b6e19cacebf1d9cbe18` |
| `scripts/analyze_robustness.py` | 4.3 KB | `3d2b42791afaaf3a35da99b738bf405c7f905caf08ef118c5236e662beb8b5e2` |
| `scripts/analyze_synthetic.py` | 10.7 KB | `427320e07b6b16ac1bd2ccf6ec06bc2b0b03dc176afe466e9c52c616eae36d54` |
| `scripts/bench_exact_checker.py` | 2.8 KB | `5194129bbd70f466b9b398c16bad9b06962f09f19582033254a2b81ab37bdea7` |
| `scripts/benchmark_games.py` | 8.6 KB | `bde4503ec4f8221be338aa48fd532578308f8be1b442b97f3221a3366947bb83` |
| `scripts/benchmark_labels.py` | 5.9 KB | `905aaa6828ed1646f85f90a371582f95ccbe9405fb60c2e9bacb2654922547f9` |
| `scripts/check_outputs.py` | 1.6 KB | `4c18fda4a77e8a0225faa7f49ac3d432c1ccda5528a534341d0081975506d500` |
| `scripts/compare_reruns.py` | 7.1 KB | `bb1e468a029e91dd880e8ab25bdf46af831a5e4ef6d59b2a0b95e2b10deddefe` |
| `scripts/compute_exact_labels.py` | 3.0 KB | `036fba0c75f34a6d32302eff4330b982f8ae7884713026146ecf65641701f357` |
| `scripts/fetch_benchmark_data.sh` | 1.2 KB | `ab9ba369ee4f026e48c19e233fd16ab7da75ea41955348b10fb34271df8326cd` |
| `scripts/make_bandit_figures.py` | 8.3 KB | `8d2df8c67cbf97883643afd29f285668494eda99d26bd2dbf7ba03646cff901b` |
| `scripts/make_benchmark_figures.py` | 21.4 KB | `241b84c5f8f3250b96d6830b15647747d38b664d724278051fd5227c3569f316` |
| `scripts/make_feedback_figure.py` | 2.4 KB | `9de03126e28b30f0bbe066419527f65f983e13877c373270accae0d2805686e7` |
| `scripts/make_figures.py` | 14.7 KB | `c589e36cf7eb99399ae13f9f6e61f562f544ce9ea90336f3fc7463bef3410409` |
| `scripts/make_robustness_figures.py` | 4.9 KB | `9dc37c4c4cc5eb28d93f18d6152c439c4d797d30e71acbc26f80148b48734469` |
| `scripts/rebuild_benchmark_games.py` | 2.9 KB | `80b1f7044c16523c5466216f47eb7266dd583d227a6df251c8fad5208651eab8` |
| `scripts/reproduce_figures.sh` | 899 B | `9ac376bedbbe1c96a569a2288d488bb5cc2a50cdc87fdd8f78e36e74097c0e4f` |
| `scripts/run_benchmark.py` | 6.5 KB | `6c7ff4baaa89ce6b60b38edb6a0ffd7ae2b45cea9adf7dc621e3d49c356333e7` |
| `scripts/run_grid.py` | 16.0 KB | `f67366dd8baf10f6e46f20e7faf78cb828d6d39b94ddbe2a25c4e8089c572c50` |
| `sns/__init__.py` | 373 B | `0799c4f2298f20b80b9ef9b5e5644645fb89b2f9eeb66203d290233ddb1b1296` |
| `sns/bandit_algorithms.py` | 11.4 KB | `105e16c6616607b89e9f68f83d7b8aca2fb91404b2b7c7ca473b75ab4dfc2ff2` |
| `sns/candidate_algorithms.py` | 5.8 KB | `e9e7990f6a25499ee6f7aff0a5f4cf2f74ec005fe3aa58ed1766b6a49c3b5486` |
| `sns/coverage_algorithms.py` | 4.7 KB | `92fc2f9f9b258ef788015b5121ff2f0c6c9d7e2126a306dbeb1b34ed17fcc6d6` |
| `sns/environments.py` | 6.3 KB | `d470d6677e5f15bd66fad2ebe699dc473806760c3ceb9ef62411ad45772856cc` |
| `sns/exact_sns.py` | 8.9 KB | `a5e0e6b7a0ce4e685391818a054a372cd35229a948161abdcefcbf5bd8724e2a` |
| `sns/full_deviation_checker.py` | 8.6 KB | `adad55f0c871bb8aefe19f3f8b7857938cd522fbedacc7006ad4e9caf38bb44e` |
| `sns/paths.py` | 843 B | `7514ede585eb7953e72574629145e5bb46d9650c6b53b98e2424a3f5f1e378fd` |
| `sns/runner.py` | 9.2 KB | `7e8e0eab1cc7a967d47c0214e6bec3fb54d946e382cdbfdc9ff8155948c65fe8` |
| `sns/seeding.py` | 4.2 KB | `784c0ed88f1e1e9281fb9d26dfde30282a69da14fd82b22348120e6d13c6d770` |
| `tests/check_bandit.py` | 12.0 KB | `8a7c15bf2c05792950b174488775ad7c12b5c590aa7a7ae4604a2950eb51f6d6` |
| `tests/check_full_checker.py` | 8.0 KB | `62a08b0b212e51cf426b2ce4ab83c830af9421e2b236c212bc28e26778cdc7f9` |
| `tests/check_witness_properties.py` | 10.0 KB | `9c3cb11f2c651b3e70674476c0bf9160e06a794dc9a8a195885cbdb8b166a242` |
| `tests/test_config.py` | 4.3 KB | `01f905089f6f6155ebf7df4f119ae9d76c157459cf6ae578861314b9d08fb8e8` |
| `tests/test_properties.py` | 728 B | `42a70a81e13ba3e0ccc54a6a2612c4fd0673e7056112e35cb6446d084d6d31d6` |
| `tests/test_release.py` | 16.3 KB | `7e80687a3d024e188eaf1a36dfe1e7a07a329bf84ca96c65e78ecbc783cf7152` |
