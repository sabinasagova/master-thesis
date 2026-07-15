"""
Targeted power follow-up: serial SGS, baseline vs lexicase only.

Reuses `full_mmlib_experiment.run_single` (identical settings) to add
N_EXTRA_SEEDS new seeds to the serial/baseline and serial/lexicase cells,
pools them with the existing records, and reports the updated paired
Wilcoxon test. Run from the repo root with:

    PYTHONPATH=$(pwd) python -O yuantian/experiments/serial_lexicase_power_followup.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# gphh_solver.GPHH.init_model() does a bare `import multitreegp`, which
# requires yuantian/ itself (not this experiments/ subfolder) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yuantian.experiments.full_mmlib_experiment as base
from yuantian.experiments.full_mmlib_experiment import (
    METHOD_LABELS,
    STRATIFIED_CLASSES,
    TEST_FILES,
    TRAIN_FILES,
    VAL_FILES,
    RCPSPDatabase,
    rank_biserial_effect_size,
    read_instances,
    run_single,
)
from scipy.stats import wilcoxon

EXISTING_RESULTS = base.OUTPUT_DIR / "serial_lexicase_power_followup.json"
OUTPUT_PATH = base.OUTPUT_DIR / "serial_lexicase_power_followup.json"

N_EXTRA_SEEDS = 10
SGS_TYPE = "serial"
METHODS_OF_INTEREST = ("baseline", "lexicase")


def main():
    warnings.filterwarnings("ignore")

    with open(EXISTING_RESULTS) as f:
        existing = json.load(f)
    existing_serial = [
        r for r in existing if r["sgs"] == SGS_TYPE and r["method"] in METHODS_OF_INTEREST
    ]
    used_seeds = sorted({r["seed"] for r in existing_serial})
    print(f"Existing seeds for serial/baseline & serial/lexicase: {used_seeds}")
    next_seed_base = max(used_seeds) + 1
    new_seeds = list(range(next_seed_base, next_seed_base + N_EXTRA_SEEDS))
    print(f"Adding {N_EXTRA_SEEDS} new seeds: {new_seeds}")

    train_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TRAIN_FILES]
    val_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in VAL_FILES]
    test_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES]
    training = read_instances(train_paths)
    validation = read_instances(val_paths)
    test = read_instances(test_paths)
    print(f"Stratified classes: {STRATIFIED_CLASSES}, sizes: "
          f"{len(training)}/{len(validation)}/{len(test)}")

    new_records = list(existing_serial)
    for method in METHODS_OF_INTEREST:
        for seed in new_seeds:
            t0 = time.time()
            record = run_single(seed, method, SGS_TYPE, training, validation, test)
            new_records.append(record)
            print(
                f"[{SGS_TYPE}/{method}] seed={seed} "
                f"train_best={record['best_fitness_train']:.4f} "
                f"test={record['test_fitness']:.4f} "
                f"({time.time() - t0:.1f}s)"
            )
            with open(OUTPUT_PATH, "w") as f:
                json.dump(new_records, f, indent=2)

    analyze(new_records)


def analyze(records: list):
    baseline = sorted(
        [r for r in records if r["method"] == "baseline"], key=lambda r: r["seed"]
    )
    lexicase = sorted(
        [r for r in records if r["method"] == "lexicase"], key=lambda r: r["seed"]
    )
    seeds_b = [r["seed"] for r in baseline]
    seeds_l = [r["seed"] for r in lexicase]
    assert seeds_b == seeds_l, "seeds must match pairwise for a paired test"
    n = len(seeds_b)

    print("\n" + "=" * 78)
    print(f"POOLED RESULT: serial SGS, baseline vs lexicase, n={n} seeds")
    print("=" * 78)

    for metric_key, metric_label in [
        ("test_fitness", "held-out test_fitness [PRIMARY]"),
        ("best_fitness_train", "training best_fitness [secondary]"),
    ]:
        vals_b = np.array([r[metric_key] for r in baseline])
        vals_l = np.array([r[metric_key] for r in lexicase])
        diffs = vals_b - vals_l  # positive => lexicase better
        print(f"\n--- {metric_label} ---")
        print(f"{METHOD_LABELS['baseline']}: mean={vals_b.mean():.4f}, std={vals_b.std():.4f}")
        print(f"{METHOD_LABELS['lexicase']}: mean={vals_l.mean():.4f}, std={vals_l.std():.4f}")
        if np.all(diffs == 0):
            print("All differences zero, Wilcoxon not informative")
            continue
        stat, p_value = wilcoxon(vals_b, vals_l)
        effect = rank_biserial_effect_size(diffs)
        direction = "lexicase better" if diffs.mean() > 0 else "baseline better"
        sig = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "not significant"
        print(
            f"Wilcoxon: W={stat:.3f}, p={p_value:.6f}, r={effect:.4f}, "
            f"mean diff={diffs.mean():.4f} ({direction}) -> {sig}"
        )


if __name__ == "__main__":
    main()
