"""
Same baseline vs NR-terminals comparison as nr_terminals_experiment.py,
but on MMLIB+ instead of MMLIB50 -- does the result (NR terminals
significantly better, p=0.003 at n=18 on MMLIB50, see readme.md extension
#3) hold up on a different benchmark family?

MMLIB+ uses "Jall{class}_{case}.mm" naming, class 1-324 are 50-activity
instances and 325-648 are 100-activity (see rcpsp_dataset.py). Sticking to
the 50-activity range here so we're only changing one thing (dataset)
instead of two (dataset and instance size).

Just two conditions, no cp_propagation arm this time:
  - baseline: normal terminal set, NR resources exist but invisible to GP
  - baseline+nr: baseline + NR_STOCK_RATIO, NR_MODE_DEMAND_RATIO,
    NR_BUDGET_PRESSURE

Reuses nr_terminals_experiment.py's build_params/run_single so the
feasibility-filtered analysis lines up with that experiment's.

Run with (from the repo root):
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 -O \\
        yuantian/experiments/nr_terminals_mmlib_plus_experiment.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.experiments.nr_terminals_experiment import (
    N_GEN,
    N_SEEDS,
    POP_SIZE,
    rank_biserial_effect_size,
    run_single,
)
from yuantian.gphh_solver import read_instances
from yuantian.rcpsp_dataset import RCPSPDatabase

SEED_BASE = 11000
CONDITIONS = ["baseline", "baseline+nr"]

# 5 classes spread across the 50-activity MMLIB+ range (class_id 1-324),
# case "_1" for train / "_2" for test -- mirrors nr_terminals_experiment.py's
# convention of same classes, different case, for MMLIB50.
TRAIN_CLASSES = [1, 70, 140, 210, 280]
TRAIN_FILES = [f"Jall{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"Jall{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "nr_terminals_mmlib_plus_experiment"


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = read_instances(
        [str(RCPSPDatabase.MMLIB_PLUS_DIR + f) for f in TRAIN_FILES], keep_non_renewable=True
    )
    test = read_instances(
        [str(RCPSPDatabase.MMLIB_PLUS_DIR + f) for f in TEST_FILES], keep_non_renewable=True
    )
    print(f"Training instances: {TRAIN_FILES}")
    print(f"Test instances: {TEST_FILES}")
    print(f"pop_size={POP_SIZE}, n_gen={N_GEN}, n_seeds={N_SEEDS}")

    all_results = []
    for condition in CONDITIONS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            t0 = time.time()
            out_path = OUTPUT_DIR / f"{condition.replace('+', '_')}_seed{seed}.json"
            record = run_single(seed, condition, training, test, out_path)
            all_results.append(record)
            print(
                f"[{condition}] seed={seed} best={record['best_fitness']:.4f} "
                f"train_feasible={record['train_feasible']} "
                f"test={record['test_fitness']:.4f} test_feasible={record['test_feasible']} "
                f"({time.time() - t0:.1f}s)"
            )

    with open(OUTPUT_DIR / "all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    analyze(all_results)


def analyze(all_results: list):
    by_condition = {
        c: sorted([r for r in all_results if r["condition"] == c], key=lambda r: r["seed"])
        for c in CONDITIONS
    }
    baseline = by_condition["baseline"]
    nr = by_condition["baseline+nr"]
    seeds = [r["seed"] for r in baseline]
    assert seeds == [r["seed"] for r in nr]

    print(f"\n{'=' * 80}\nRESULTS TABLE (MMLIB+, n={len(seeds)} seeds)\n{'=' * 80}")
    for condition in CONDITIONS:
        records = by_condition[condition]
        feas_vals = np.array([r["best_fitness"] for r in records if r["train_feasible"]])
        n_feas = len(feas_vals)
        print(
            f"{condition:<18} n_feasible={n_feas}/{len(records)} "
            f"mean={feas_vals.mean():.4f} min={feas_vals.min():.4f} std={feas_vals.std():.4f}"
        )

    print(f"\n{'=' * 80}\nSTATISTICAL SIGNIFICANCE (paired Wilcoxon, dropping pairs infeasible on either side)\n{'=' * 80}")
    b_vals = np.array([r["best_fitness"] for r in baseline])
    n_vals = np.array([r["best_fitness"] for r in nr])
    b_feas = np.array([r["train_feasible"] for r in baseline])
    n_feas = np.array([r["train_feasible"] for r in nr])
    mask = b_feas & n_feas
    if (~mask).sum():
        print(f"dropping {(~mask).sum()}/{len(seeds)} infeasible-either-side pair(s)")
    b_f, n_f = b_vals[mask], n_vals[mask]
    diffs = b_f - n_f
    stat, p_value = wilcoxon(b_f, n_f)
    effect_size = rank_biserial_effect_size(diffs)
    direction = "baseline+nr better" if diffs.mean() > 0 else "baseline better"
    significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
    print(
        f"baseline+nr: n={len(diffs)} W={stat:.4f} p={p_value:.6f} r={effect_size:.4f} "
        f"mean(baseline-baseline+nr)={diffs.mean():.4f} ({direction}) => {significance}"
    )


if __name__ == "__main__":
    main()
