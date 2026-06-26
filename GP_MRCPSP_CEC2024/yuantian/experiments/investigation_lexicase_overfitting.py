"""
Investigation: why does epsilon-lexicase selection significantly improve
training fitness (p=0.0002, r=0.72 at n=31) but not held-out test fitness
(p=0.325, r=0.21) in the serial-SGS GPHH comparison?

This script does NOT re-run evolution. It reconstructs the GP trees already
stored as strings in the experiment result JSONs (`best_tree`, the
final-generation hall-of-fame winner) and re-evaluates them directly on the
train/validation/test instance sets to get per-instance ("case") fitness,
exact tree size/depth, and terminal usage -- none of which were persisted
by the original experiment scripts.

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python -O yuantian/experiments/investigation_lexicase_overfitting.py
"""
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.experiments.full_mmlib_experiment import (
    RCPSPDatabase,
    TEST_FILES,
    TRAIN_FILES,
    VAL_FILES,
    build_params,
    read_instances,
)
from yuantian.gphh_solver import GPHH, StaticDatasetProvider, evaluate_heuristic
from yuantian.multitreegp import MultiPrimitiveTree, TerminalTypeEnum
from yuantian.rcpsp_simulation import FeatureEnum

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "investigation_lexicase_overfitting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FOLLOWUP_JSON = EXPERIMENTS_DIR / "results" / "full_mmlib_experiment" / "serial_lexicase_power_followup.json"
ALL_RUNS_JSON = EXPERIMENTS_DIR / "results" / "full_mmlib_experiment" / "all_runs.json"

TERMINAL_NAMES = {f.value for f in FeatureEnum}


def build_solver_and_instances():
    """One GPHH solver per process gives us toolbox.compile / pset / simulator
    / decision_type, matching exactly what full_mmlib_experiment.run_single
    uses for serial SGS (the cell under investigation)."""
    params = build_params("serial")
    train_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TRAIN_FILES]
    val_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in VAL_FILES]
    test_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES]
    training = read_instances(train_paths)
    validation = read_instances(val_paths)
    test = read_instances(test_paths)
    solver = GPHH(
        training_set_provider=StaticDatasetProvider(training),
        validation_set_provider=StaticDatasetProvider(validation),
        params_gphh=params,
    )
    solver.init_model()
    return solver, training, validation, test


def reconstruct(tree_str: str, pset) -> MultiPrimitiveTree:
    return MultiPrimitiveTree.from_string(tree_str, pset)


def eval_on(individual, instances, solver):
    fit = evaluate_heuristic(
        individual=individual,
        domains=instances,
        compile_func=solver.toolbox.compile,
        pset=solver.pset,
        decision_type=solver.decision_type,
        simulator=solver.simulator,
    )
    return float(fit[0]), [float(v) for v in individual.case_fitness]


def tree_stats(individual: MultiPrimitiveTree) -> dict:
    out = {}
    total_size = 0
    max_depth = 0
    for ttype in (TerminalTypeEnum.ACTIVITY.value, TerminalTypeEnum.MODE.value):
        subtree = individual[ttype]
        out[f"{ttype}_size"] = len(subtree)
        out[f"{ttype}_depth"] = subtree.height
        total_size += len(subtree)
        max_depth = max(max_depth, subtree.height)
    out["total_size"] = total_size
    out["max_depth"] = max_depth
    return out


def terminal_usage(individual: MultiPrimitiveTree) -> Counter:
    counts = Counter()
    for ttype in (TerminalTypeEnum.ACTIVITY.value, TerminalTypeEnum.MODE.value):
        for node in individual[ttype]:
            name = getattr(node, "name", None) or getattr(node, "value", None)
            if name in TERMINAL_NAMES:
                counts[str(name)] += 1
    return counts


def enrich_record(record: dict, solver, training, validation, test) -> dict:
    tree = reconstruct(record["best_tree"], solver.pset)
    train_fit, train_cases = eval_on(tree, training, solver)
    val_fit, val_cases = eval_on(tree, validation, solver)
    test_fit, test_cases = eval_on(tree, test, solver)
    stats = tree_stats(tree)
    usage = terminal_usage(tree)
    enriched = dict(record)
    enriched.update(
        {
            "recon_train_fitness": train_fit,
            "recon_val_fitness": val_fit,
            "recon_test_fitness": test_fit,
            "train_case_fitness": train_cases,
            "val_case_fitness": val_cases,
            "test_case_fitness": test_cases,
            "tree_stats": stats,
            "terminal_usage": dict(usage),
        }
    )
    return enriched


def main():
    warnings.filterwarnings("ignore")
    solver, training, validation, test = build_solver_and_instances()
    print(f"Train/Val/Test instance counts: {len(training)}/{len(validation)}/{len(test)}")

    print("\n=== Enriching serial_lexicase_power_followup.json (n=31 per method) ===")
    followup = json.load(open(FOLLOWUP_JSON))
    enriched_followup = []
    for i, record in enumerate(followup):
        enriched = enrich_record(record, solver, training, validation, test)
        enriched_followup.append(enriched)
        gap_check = abs(enriched["recon_test_fitness"] - record["test_fitness"])
        print(
            f"[{i+1}/{len(followup)}] {record['method']} seed={record['seed']} "
            f"recon_test={enriched['recon_test_fitness']:.3f} "
            f"(recorded={record['test_fitness']:.3f}, diff={gap_check:.3f}) "
            f"recon_val={enriched['recon_val_fitness']:.3f} "
            f"tree_size={enriched['tree_stats']['total_size']}"
        )
    with open(RESULTS_DIR / "enriched_followup.json", "w") as f:
        json.dump(enriched_followup, f, indent=2)

    print("\n=== Enriching full_mmlib_experiment/all_runs.json (serial SGS only, n=10/method incl. hybrid) ===")
    all_runs = json.load(open(ALL_RUNS_JSON))
    serial_runs = [r for r in all_runs if r["sgs"] == "serial"]
    enriched_all_runs = []
    for i, record in enumerate(serial_runs):
        enriched = enrich_record(record, solver, training, validation, test)
        enriched_all_runs.append(enriched)
        print(
            f"[{i+1}/{len(serial_runs)}] {record['method']} seed={record['seed']} "
            f"recon_test={enriched['recon_test_fitness']:.3f} "
            f"(recorded={record['test_fitness']:.3f}) "
            f"recon_train={enriched['recon_train_fitness']:.3f} "
            f"(recorded best_fitness_train={record['best_fitness_train']:.3f})"
        )
    with open(RESULTS_DIR / "enriched_all_runs_serial.json", "w") as f:
        json.dump(enriched_all_runs, f, indent=2)

    print("\nDone. Enriched datasets written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
