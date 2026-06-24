# Evolutionary Algorithms for Project Scheduling — Master Thesis

This repository contains the code, data, and LaTeX source for a master's thesis at Charles University, Prague, on solving the **Multi-Mode Resource-Constrained Project Scheduling Problem (MRCPSP)** using evolutionary algorithms.

---

## Thesis overview

The thesis designs and evaluates a custom evolutionary algorithm for MRCPSP and compares it against a published baseline: the **Genetic Programming Hyper-Heuristic (GPHH)** by Tian, Mei, and Zhang (CEC 2024). Experiments are run on the **MMLIB50** and **MMLIB100** benchmark datasets.

---

## Repository structure

```text
master-thesis/
│
├── README.md
│
├── GP_MRCPSP_CEC2024/           # GPHH baseline (Tian et al.) + extensions
│   ├── readme.md                # Usage instructions and experiment log
│   ├── requirements.txt
│   ├── discrete_optimization/   # Airbus discrete-optimization library (vendored)
│   └── yuantian/                # Core algorithm code
│       ├── gphh_solver.py           # Main GPHH solver
│       ├── gp_algorithms.py         # GP loop (standard_gp, varOr, elitism)
│       ├── multitreegp.py           # Multi-tree GP individual (activity + mode trees)
│       ├── rcpsp_simulation.py      # SGS simulators, feature set, decision types
│       ├── rcpsp_dataset.py         # MMLIB dataset loader and data providers
│       ├── cp_propagation.py        # Critical-path propagation terminals (CP_FORWARD etc.)
│       ├── hybrid_gp.py             # Epsilon-lexicase selection + memetic GP loop
│       ├── local_search.py          # Critical-path repair local search for elites
│       ├── heuristic_rule_evaluator.py
│       ├── instance_indicator.py
│       ├── utils.py
│       └── experiments/             # Scripts running the comparison experiments
│           ├── cp_propagation_experiment.py
│           ├── lexicase_local_search_experiment.py
│           ├── full_mmlib_experiment.py
│           ├── serial_lexicase_power_followup.py
│           └── results/             # Raw per-run JSON and convergence plots, one folder per experiment
│
├── thesis-en-master/            # LaTeX thesis source (Charles University template)
│   ├── thesis.tex               # Main file
│   ├── preface.tex              # Preface
│   ├── chap01.tex               # Chapter 1 – Introduction
│   ├── chap02.tex               # Chapter 2 – Overview of current approaches
│   ├── chap03.tex               # Chapter 3 – Project scheduling dataset overview
│   ├── chap04.tex               # Chapter 4 – Proposed approach and novelty
│   ├── chap05.tex               # Chapter 5 – Models' experimentation and comparison
│   ├── epilog.tex               # Chapter 6 – Conclusions
│   └── bibliography.bib         # References
│
├── data/
│   └── data-explanation/        # PSPLIB/MMLIB instance documentation and notes
│
├── notebooks/                   # Scratch notebooks
│
└── docs/
    ├── literature-review/       # Literature review notes and summaries
    ├── references/              # PDFs of cited papers
    └── meeting-minutes/         # Supervisor meeting notes
```

---

## Baseline algorithm

The baseline is the **GPHH** method (Tian, Mei, Zhang, CEC 2024):

> *Learning Heuristics via Genetic Programming for Multi-Mode Resource-Constrained Project Scheduling.*
> IEEE Congress on Evolutionary Computation 2024. DOI: 10.1109/CEC60901.2024.10612172

Original source: <https://github.com/TianYuanSX/GP_MRCPSP_CEC2024>

GPHH evolves scheduling priority rules as symbolic GP trees (one activity-selection tree, one mode-selection tree) and applies them inside a Serial SGS to construct schedules. Fitness is the mean relative deviation from the CPM lower bound across a training set of MMLIB instances.

---

## Extensions to the baseline

Two extensions on top of the baseline, both described in more detail (with results) in [`GP_MRCPSP_CEC2024/readme.md`](GP_MRCPSP_CEC2024/readme.md):

| # | Extension | Location | Status |
|---|-----------|----------|--------|
| 1 | Critical-path propagation terminals (`CP_FORWARD`, `CP_BACKWARD`, `CP_SLACK_SCORE`, `CP_PROB`) | `cp_propagation.py` | Tested over 10 seeds: faster convergence, no significant quality gain (likely redundant with existing ES/EF/LS/LF terminals) |
| 2 | Epsilon-lexicase selection with critical-path local search on elites | `hybrid_gp.py`, `local_search.py` | Tested at small scale (significant) and on full MMLIB50 (significant on training fitness; trending but not yet significant on held-out test fitness, additional seeds in progress) |

Both extensions are opt-in flags on the baseline; the GP representation itself (trees, primitives) is unchanged.

---

## Running the code

### Setup

```bash
cd GP_MRCPSP_CEC2024
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Run the baseline

```bash
python yuantian/gphh_solver.py --default --dataset MMLIB50
```

### Run the baseline with CP propagation terminals enabled

```bash
python yuantian/gphh_solver.py --default --dataset MMLIB50 --cp_propagation
```

### Run the comparison experiments (lexicase, local search, full MMLIB sweep)

These are standalone scripts under `yuantian/experiments/`, not CLI flags on `gphh_solver.py`. Each script runs multiple seeds, performs the train/val/test split, and prints a Wilcoxon test at the end. Run with `-O` to suppress debug output from the parallel SGS:

```bash
python -O yuantian/experiments/cp_propagation_experiment.py
python -O yuantian/experiments/lexicase_local_search_experiment.py
python -O yuantian/experiments/full_mmlib_experiment.py
```

Results (raw per-run JSON and a convergence plot) are written to `yuantian/experiments/results/<script_name>/`.

### Key CLI flags for `gphh_solver.py`

| Flag | Default | Description |
|------|---------|-------------|
| `-s` | `serial` | SGS type: `serial` / `parallel` |
| `-d` | `activity_first` | Decision type: `activity_first` / `mode_first` / `simultaneous` |
| `--default` | off | Full paper parameters (pop=1000, gen=50) |
| `--dataset` | small | `MMLIB50` / `MMLIB100` / `MMLIBPLUS_50` / `MMLIBPLUS_100` |
| `-n` | 1 | Number of independent runs |
| `--seed` | 1 | Starting random seed |
| `--cp_propagation` | off | Add the critical-path propagation terminals |
| `--dynamic` | off | Use dynamic CPM terminals |
| `--split` | off | Split training set across generations |
| `--multiprocess` | off | Parallel fitness evaluation |
| `--log` | `./results/` | Output directory for result JSON files |

---

## Benchmark datasets

Experiments use the **MMLIB** benchmark library (Van Peteghem & Vanhoucke, 2014) for MRCPSP, hosted by the OR&S Research Group at Ghent University:
<https://www.projectmanagement.ugent.be/research/project_scheduling/mmrcpsp>
