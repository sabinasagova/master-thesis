# Genetic Programming Hyper-heuristics for the Multi-Mode Resource-Constrained Project Scheduling Problem

## Overview

This repository builds on the implementation of "Learning Heuristics via
Genetic Programming for Multi-Mode Resource-Constrained Project Scheduling"
by Yuan Tian, Yi Mei, and Mengjie Zhang (CEC 2024) -- see `yuantian/gphh_solver.py`
and the citation/acknowledgements below. Everything in `yuantian/custom_ea.py`,
`yuantian/modifications.py`, `yuantian/experiment_runner.py`,
`yuantian/run_evaluation.py`, `yuantian/plot_comparison.py`,
`yuantian/manual_heuristic_baseline.py`, `yuantian/param_sensitivity.py`, and
`yuantian/plot_sensitivity.py` is this thesis's own contribution: a set of GP
terminal, operator, and decoding extensions (collectively DMGE) proposed and
evaluated against the Tian et al. baseline, plus the supporting evaluation
harnesses, in Chapters 4-5.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH   # required before running anything below
```

`discrete_optimization/` is a local, modified copy bundled in this directory
(not the PyPI package of the same name) -- nothing to install separately.

## Reproducing the thesis's Chapter 5 results

**Baseline GPHH vs. custom EA (DMGE), head-to-head** (Table 5.3, Figures
5.2-5.4, PSPLIB J20 + MMLIB+ NR50):

```bash
python -m yuantian.experiment_runner --pop 60 --gen 20 -n 10 --seed 42 \
    --datasets PSPLIB_J20 --train 16 --val 8 --test 16 --out results/two_way_j20_v2
python -m yuantian.manual_heuristic_baseline --datasets PSPLIB_J20 \
    --test 16 --out results/two_way_j20_v2
python -m yuantian.plot_comparison --in results/two_way_j20_v2 \
    --out ../thesis-en-master/img --title "PSPLIB J20 (10 seeds, pop=60, gen=20)" \
    --configs baseline_gphh,custom_ea --format pdf

python -m yuantian.experiment_runner --pop 50 --gen 12 -n 15 --seed 42 \
    --datasets MMLIBPLUS_NR_50 --train 16 --val 8 --test 16 --out results/two_way_mmlib50_v2
python -m yuantian.manual_heuristic_baseline --datasets MMLIBPLUS_NR_50 \
    --test 16 --out results/two_way_mmlib50_v2
python -m yuantian.plot_comparison --in results/two_way_mmlib50_v2 \
    --out ../thesis-en-master/img --title "MMLIB+ NR50 (15 seeds, pop=50, gen=12)" \
    --configs baseline_gphh,custom_ea --format pdf
```

(after both: rename `convergence.pdf`/`boxplot_dev_feas.pdf`/`boxplot_feasible.pdf`
in `../thesis-en-master/img` to the dataset-specific names chap05.tex's
`\includegraphics` calls expect, e.g. `convergence-j20.pdf`,
`boxplot-j20-ard.pdf`, `boxplot-j20-feasibility.pdf` -- `plot_comparison.py`
always writes the generic names so they don't collide between the two runs.)

The manual-heuristic-baseline step is deterministic (no training, no seeds)
and adds the EST/GRPW/LSTLFT priority-rule comparison discussed alongside
Table 5.3.

**Baseline vs. DMGE (+ ablations), MMLIB+ NR50 ablation study** (Section
5.6's per-graft attribution data, the 15-seed result already in
`results/nr50_powered/`, re-runnable from scratch with):

```bash
python -m yuantian.run_evaluation --pop 40 --gen 8 -n 15 --seed 42 \
    --datasets MMLIBPLUS_NR_50 --out results/nr50_powered
```

**Combination/screening study** (Table 5.2, Section 5.4: driver x
modification-flag comparison — baseline, DMGE, TDRE, Lexicase, with/without
modifications):

```bash
python -m yuantian.run_evaluation --pop 20 --gen 4 -n 5 --seed 42 \
    --datasets MMLIBPLUS_NR_50 \
    --configs baseline,dmge_full,tdre,tdre_mods,lexicase,lexicase_mods \
    --out results/combo_screen
```

**Parameter sensitivity sweep** (Table 5.1, Figure 5.1, Section 5.3:
population-size and generation-count one-factor-at-a-time sweep for DMGE on
PSPLIB J20):

```bash
python -m yuantian.param_sensitivity --datasets PSPLIB_J20 -n 5 \
    --out results/param_sensitivity
python -m yuantian.plot_sensitivity --in results/param_sensitivity \
    --out ../thesis-en-master/img/sensitivity.pdf
```

Both `run_evaluation.py` and `experiment_runner.py` parallelise across seeds
and configurations with `ProcessPoolExecutor` (`--workers N`, defaults to all
CPU cores) and write `raw.csv` (per-seed) and `summary.csv` (aggregated, with
paired Wilcoxon p-values) to `--out`.

`multi_sgs`/`multi_sgs_mods` are defined in `run_evaluation.py`'s `CONFIGS`
but are **not** run by default -- they have a known unresolved hang on
>= 4 MMLIB+ NR instances with population >= 8 (documented in
`custom_ea.py`'s module docstring). Pass `--configs` explicitly without them.

## Plotting

`yuantian/plot_comparison.py` reads any `raw.csv` produced by either harness
above (plus `convergence/*.json` when produced by `experiment_runner.py`) and
writes convergence and boxplot figures:

```bash
python -m yuantian.plot_comparison --in results/<run_dir> --out <dir> \
    --title "<plot title>" --configs baseline,dmge_full --format pdf
```

## Characterization tests

`yuantian/_characterize.py` is a before/after regression oracle for changes
to `custom_ea.py`/`modifications.py` (not part of the experimental pipeline):

```bash
python -m yuantian._characterize snapshot before.json
# ... make changes ...
python -m yuantian._characterize snapshot after.json
python -m yuantian._characterize diff before.json after.json
```

## Tian et al. baseline -- original usage

The unmodified baseline can still be run directly via its own entry point:

```bash
python yuantian/gphh_solver.py -s serial -d activity_first --default --dataset MMLIB50
```

- `-s`: schedule generation scheme, `serial` or `parallel` (default `serial`).
- `-d`: decision type, `activity_first`, `mode_first`, or `simultaneous`.
- `--default`: use the paper's parameters (otherwise a minimal working example).
- `--dataset`: `MMLIB50`, `MMLIB100`, `MMLIBPLUS_50`, `MMLIBPLUS_100`.
- `-n`, `--seed`, `--start_index`: run count, RNG seed, starting instance index.
- `--log`: write detailed per-generation logs to `logs/`.
- `--multi_process`: parallelise fitness evaluation across instances.

## Citation

```bibtex
@INPROCEEDINGS{10612172,
  author={Tian, Yuan and Mei, Yi and Zhang, Mengjie},
  booktitle={2024 IEEE Congress on Evolutionary Computation (CEC)},
  title={Learning Heuristics via Genetic Programming for Multi-Mode Resource-Constrained Project Scheduling},
  year={2024},
  pages={01-08},
  doi={10.1109/CEC60901.2024.10612172}}
```

## Acknowledgements

Built on:
- [DEAP](https://github.com/DEAP/deap)
- [discrete-optimization](https://github.com/airbus/discrete-optimization)

MMLIB is from the [Operations Research and Scheduling (OR&S) Research group](https://www.projectmanagement.ugent.be/research/project_scheduling/mmrcpsp)
at Ghent University. PSPLIB is from [Kolisch and Sprecher (1997)](http://129.187.106.231/psplib/).
