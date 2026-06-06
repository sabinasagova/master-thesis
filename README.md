# Evolutionary Algorithms for Project Scheduling – Master Thesis

This repository contains the code, data, and LaTeX source for my master's thesis at Charles University, Prague, on solving the **Multi-Mode Resource-Constrained Project Scheduling Problem (MRCPSP)** using evolutionary algorithms.

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
├── GP_Algorithm/               # GPHH baseline + proposed modifications
│   ├── readme.md               # Original usage instructions
│   ├── requirements.txt
│   ├── discrete_optimization/  # Airbus discrete-optimization library (vendored)
│   └── yuantian/               # Core algorithm code
│       ├── gphh_solver.py          # Main GPHH solver (baseline + modifications flag)
│       ├── modifications.py        # All proposed modifications to the baseline
│       ├── run_comparison.py       # Runs baseline and modified side-by-side
│       ├── compare_results.py      # Summarises and diffs result JSON files
│       ├── gp_algorithms.py        # GP loop (standard_gp, varOr, elitism)
│       ├── multitreegp.py          # Multi-tree GP individual (activity + mode trees)
│       ├── rcpsp_simulation.py     # SGS simulators, feature set, decision types
│       ├── rcpsp_dataset.py        # MMLIB dataset loader and data providers
│       ├── heuristic_rule_evaluator.py
│       ├── instance_indicator.py
│       └── utils.py
│
├── thesis-en-master/           # LaTeX thesis source (Charles University template)
│   ├── thesis.tex              # Main file
│   ├── preface.tex             # Preface
│   ├── chap01.tex              # Chapter 1 – Introduction
│   ├── chap02.tex              # Chapter 2 – Overview of current approaches
│   ├── chap03.tex              # Chapter 3 – Project scheduling dataset overview
│   ├── chap04.tex              # Chapter 4 – Proposed approach and novelty
│   ├── chap05.tex              # Chapter 5 – Models' experimentation and comparison
│   ├── epilog.tex              # Chapter 6 – Conclusions
│   └── bibliography.bib        # References
│
├── data/
│   └── data-explanation/       # PSPLIB/MMLIB instance documentation and notes
│
└── docs/
    ├── literature-review/      # Literature review notes and summaries
    └── meeting-minutes/        # Supervisor meeting notes
```

---

## Baseline algorithm

The baseline is the **GPHH** method (Tian, Mei, Zhang — CEC 2024):

> *Learning Heuristics via Genetic Programming for Multi-Mode Resource-Constrained Project Scheduling.*
> IEEE Congress on Evolutionary Computation 2024. DOI: 10.1109/CEC60901.2024.10612172

Original source: <https://github.com/TianYuanSX/GP_MRCPSP_CEC2024>

GPHH evolves scheduling priority rules as symbolic GP trees (one activity-selection tree, one mode-selection tree) and applies them inside a Serial SGS to construct schedules. Fitness is the mean relative deviation from the CPM lower bound across a training set of MMLIB instances.

---

## Proposed modifications

All modifications to the baseline live in [`GP_Algorithm/yuantian/modifications.py`](GP_Algorithm/yuantian/modifications.py). Each entry is documented with the original code, the replacement, and the rationale.

**Current modifications:**

| # | Primitive | Change | Effect |
|---|-----------|--------|--------|
| 1 | `if_then_else` | Lazy evaluation + explicit `> 0` threshold | Enables `IS_ON_CRITICAL_PATH` to act as a Boolean gate; GP can evolve critical-path-aware branching rules |

More modifications will be added here as the proposed approach is developed.

---

## Running the code

### Setup

```bash
cd GP_Algorithm
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Run the baseline only

```bash
python yuantian/gphh_solver.py --default --dataset MMLIB50
```

### Run with proposed modifications

```bash
python yuantian/gphh_solver.py --default --dataset MMLIB50 --modifications
```

### Run both and compare (recommended for experiments)

```bash
# Quick smoke-test
python yuantian/run_comparison.py

# Full experiment – 5 runs on MMLIB50
python yuantian/run_comparison.py --default --dataset MMLIB50 -n 5

# Print comparison table
python yuantian/compare_results.py \
    --baseline results/baseline \
    --modified  results/modified
```

### Key CLI flags for `gphh_solver.py`

| Flag | Default | Description |
|------|---------|-------------|
| `-s` | `serial` | SGS type: `serial` / `parallel` |
| `-d` | `activity_first` | Decision type: `activity_first` / `mode_first` / `simultaneous` |
| `--default` | off | Full paper parameters (pop=1000, gen=50) |
| `--medium` | off | Medium parameters (pop=50, gen=10) — fast runs with per-generation ETA logging |
| `--dataset` | small | `MMLIB50` / `MMLIB100` / `MMLIBPLUS_50` / `MMLIBPLUS_100` |
| `-n` | 1 | Number of independent runs |
| `--seed` | 1 | Starting random seed |
| `--modifications` | off | Enable modifications from `modifications.py` |
| `--dynamic` | off | Use dynamic CPM terminals |
| `--multiprocess` | off | Parallel fitness evaluation |
| `--log` | `./results/` | Output directory for result JSON files |

---

## Adding a new modification

1. Add your function to [`modifications.py`](GP_Algorithm/yuantian/modifications.py) with `ORIGINAL / MODIFIED / RATIONALE` comments.
2. Register it in `ACTIVE_MODIFICATIONS` at the bottom of that file using the primitive name as the key (e.g. `"add"`, `"if_else"`).
3. Run `run_comparison.py` to measure its effect — no other files need to change.

---

## Benchmark datasets

Experiments use the **MMLIB** benchmark library (Van Peteghem & Vanhoucke, 2014) for MRCPSP, hosted by the OR&S Research Group at Ghent University:
<https://www.projectmanagement.ugent.be/research/project_scheduling/mmrcpsp>
