# Evolutionary Algorithms for Project Scheduling

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

## Running the code

### Setup

Create and activate a virtual environment first (this repo uses a `.env`
folder at the repo root for it, not a dotenv config file):

```bash
python3 -m venv .env
source .env/bin/activate
```

Then install dependencies and set up the path:

```bash
cd GP_MRCPSP_CEC2024
pip install -r requirements.txt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Run the baseline

```bash
python yuantian/gphh_solver.py --default --dataset MMLIB50
```

### Key CLI flags for `gphh_solver.py`

| Flag | Default | Description |
|------|---------|-------------|
| `-s` | `serial` | SGS type: `serial` / `parallel` |
| `-d` | `activity_first` | Decision type: `activity_first` / `mode_first` / `simultaneous` |
| `--default` | off | Full paper parameters (pop=1000, gen=50) |
| `--dataset` | small | `MMLIB50` / `MMLIB100` / `MMLIBPLUS_50` / `MMLIBPLUS_100` |
| `-n` | 1 | Number of independent runs |
| `--seed` | 1 | Starting random seed |
| `--multiprocess` | off | Parallel fitness evaluation |
| `--log` | `./results/` | Output directory for result JSON files |

---

## Benchmark datasets

Experiments use the **MMLIB** benchmark library (Van Peteghem & Vanhoucke, 2014) for MRCPSP, hosted by the OR&S Research Group at Ghent University:
<https://www.projectmanagement.ugent.be/research/project_scheduling/mmrcpsp>

---

## License

This repository is licensed under the **MIT License**: see [`LICENSE`](LICENSE).

The vendored code under [`GP_MRCPSP_CEC2024/`](GP_MRCPSP_CEC2024) (the GPHH baseline and the `discrete_optimization` library) carries its own MIT license from the original authors: see [`GP_MRCPSP_CEC2024/LICENSE`](GP_MRCPSP_CEC2024/LICENSE). The matrix experiment scripts under `yuantian/experiments/`, the NR terminal extensions, and other additions on top of the original GPHH baseline are original contributions by the thesis author, not part of the vendored upstream code.
