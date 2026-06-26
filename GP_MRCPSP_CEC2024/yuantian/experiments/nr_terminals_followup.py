"""
8 more seeds (7010-7017) on top of nr_terminals_experiment.py's run, same
instances/settings, this time with the fixed harness so case_feasible is
available instead of having to guess from the fitness magnitude.

Why: the first 10-seed run was "significant" at p=0.049, but that was
because one baseline seed (7009) was actually an infeasible-schedule
sentinel, not a real fitness value -- once dropped it falls to p=0.098.
Same idea as serial_lexicase_power_followup.py: add seeds instead of
rerunning everything, then look at the combined picture.

Run with (from the repo root):
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 \\
        yuantian/experiments/nr_terminals_followup.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from yuantian.experiments.nr_terminals_experiment import (
    CONDITIONS,
    OUTPUT_DIR,
    TEST_FILES,
    TRAIN_FILES,
    rank_biserial_effect_size,
    run_single,
)
from yuantian.gphh_solver import read_instances
from yuantian.rcpsp_dataset import RCPSPDatabase

FOLLOWUP_SEEDS = list(range(7010, 7018))  # 8 additional seeds


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = read_instances(
        [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TRAIN_FILES], keep_non_renewable=True
    )
    test = read_instances(
        [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES], keep_non_renewable=True
    )

    new_results = []
    for condition in CONDITIONS:
        for seed in FOLLOWUP_SEEDS:
            t0 = time.time()
            out_path = OUTPUT_DIR / f"{condition.replace('+', '_')}_seed{seed}.json"
            record = run_single(seed, condition, training, test, out_path)
            new_results.append(record)
            print(
                f"[{condition}] seed={seed} best={record['best_fitness']:.4f} "
                f"train_feasible={record['train_feasible']} "
                f"test={record['test_fitness']:.4f} test_feasible={record['test_feasible']} "
                f"({time.time() - t0:.1f}s)"
            )

    with open(OUTPUT_DIR / "followup_runs.json", "w") as f:
        json.dump(new_results, f, indent=2)

    # Original run predates case_feasible, so backfill it manually -- we
    # already know seed 7009's baseline training run was the only
    # contaminated point (everything else was under 200, nowhere near
    # sentinel scale).
    with open(OUTPUT_DIR / "all_runs.json") as f:
        original_results = json.load(f)
    for r in original_results:
        r.setdefault("train_feasible", not (r["condition"] == "baseline" and r["seed"] == 7009))
        r.setdefault("test_feasible", r["test_fitness"] < 1000.0)

    combined = original_results + new_results
    with open(OUTPUT_DIR / "all_runs_combined.json", "w") as f:
        json.dump(combined, f, indent=2)

    analyze(combined)


def analyze(all_results: list):
    by_condition = {
        c: sorted([r for r in all_results if r["condition"] == c], key=lambda r: r["seed"])
        for c in CONDITIONS
    }
    baseline = by_condition["baseline"]
    seeds = [r["seed"] for r in baseline]
    for c in CONDITIONS:
        assert [r["seed"] for r in by_condition[c]] == seeds, f"seed mismatch in {c}"

    n = len(seeds)
    print(f"\n{'=' * 80}\nCOMBINED RESULTS, n={n} seeds (10 original + {len(FOLLOWUP_SEEDS)} follow-up)\n{'=' * 80}")
    print(f"{'Condition':<18}{'n_feas':>8}{'Mean':>12}{'Min':>10}{'Std':>10}")
    for condition in CONDITIONS:
        records = by_condition[condition]
        feas_vals = np.array([r["best_fitness"] for r in records if r["train_feasible"]])
        n_feas = len(feas_vals)
        print(
            f"{condition:<18}{n_feas:>5}/{len(records):<2}{feas_vals.mean():>12.4f}"
            f"{feas_vals.min():>10.4f}{feas_vals.std():>10.4f}"
        )

    print(f"\n{'=' * 80}\nSTATISTICAL SIGNIFICANCE (paired Wilcoxon, combined n={n}, "
          f"dropping pairs infeasible on either side)\n{'=' * 80}")
    best_baseline = np.array([r["best_fitness"] for r in baseline])
    baseline_feasible = np.array([r["train_feasible"] for r in baseline])
    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        records = by_condition[condition]
        vals = np.array([r["best_fitness"] for r in records])
        cond_feasible = np.array([r["train_feasible"] for r in records])
        mask = baseline_feasible & cond_feasible
        b_f, v_f = best_baseline[mask], vals[mask]
        diffs = b_f - v_f
        if (~mask).sum():
            print(f"{condition}: dropping {(~mask).sum()}/{len(records)} infeasible-either-side pair(s)")
        stat, p_value = wilcoxon(b_f, v_f)
        effect_size = rank_biserial_effect_size(diffs)
        direction = f"{condition} better" if diffs.mean() > 0 else "baseline better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(
            f"{condition}: n={len(diffs)} W={stat:.4f} p={p_value:.6f} r={effect_size:.4f} "
            f"mean(baseline-{condition})={diffs.mean():.4f} ({direction}) => {significance}"
        )


if __name__ == "__main__":
    main()
