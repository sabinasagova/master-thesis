"""Cheap full-108-class re-evaluation of the matrix experiment's champions.

The matrix experiment (matrix_runner.py) trained and tested every cell on only
n_classes=10 stratified classes out of MMLIB's 108 (compute budget). Its stored
test fitness is therefore a mean over just 10 held-out instances. This script
does the cheap half of "scale up to 108": it takes each cell's already-evolved
best heuristic (best_heuristic_validation.tree, the deployed rule) and simply
*re-evaluates* it, with no further evolution, on the full 108-class held-out
test set (case 5 of every class 1..108). Training is left untouched at 10
classes; only the test set grows to the paper's full 108, so the numbers say
how well the 10-class-trained rules generalise across the whole class
distribution.

Per condition the evaluation reproduces matrix_runner.py's own conventions:
  * keep_non_renewable = condition in {nr, baseline_nr}  (NR-preserving
    instances); every other condition uses the paper's renewable-only
    conversion.
  * nr_terminals_feature = (condition == "nr").
  * decision type / SGS from the cell's strategy / sgs.

Output: results/full108_evaluation/all_runs.json  (one record per cell)

Run (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python -O yuantian/experiments/full108_eval.py
"""
import glob
import json
import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deap import gp

from yuantian.gphh_solver import ParametersGPHH, evaluate_heuristic, read_instances
from yuantian.rcpsp_dataset import RCPSPDatabase
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum

MATRIX_DIR = Path(__file__).resolve().parent / "results" / "matrix"
OUT_DIR = Path(__file__).resolve().parent / "results" / "full108_evaluation"

STRATEGY_TO_DECISION = {
    "AF": DecisionTypeEnum.ACTIVITY_THEN_MODE,
    "MF": DecisionTypeEnum.MODE_THEN_ACTIVITY,
    "S": DecisionTypeEnum.SIMULTANEOUS,
}
SGS_TO_SIMULATOR = {
    "serial": SimulatorTypeEnum.SERIAL_SGS,
    "parallel": SimulatorTypeEnum.PARALLEL_SGS,
}
NR_CONDITIONS = ("nr", "baseline_nr")
TEST_FILE_GETTER = {
    "MMLIB50": RCPSPDatabase.get_some_MMLIB_50_each_class_files,
    "MMLIB100": RCPSPDatabase.get_some_MMLIB_100_each_class_files,
}


class IndDict(dict):
    """Two-tree individual (activity + mode) that also carries eval attrs."""


class IndStr(str):
    """Single integrated-tree individual (simultaneous) with eval attrs."""


@lru_cache(maxsize=None)
def _params(sgs, strategy, condition):
    return ParametersGPHH.fast(
        simulator_type=SGS_TO_SIMULATOR[sgs],
        decision_type=STRATEGY_TO_DECISION[strategy],
        nr_terminals_feature=(condition == "nr"),
    )


@lru_cache(maxsize=None)
def _test_set(dataset, keep_nr):
    files = TEST_FILE_GETTER[dataset](5, 6)  # case 5 of every class 1..108
    return read_instances(files, keep_non_renewable=keep_nr)


def _wrap_individual(tree_str):
    """Stored tree is either a dict-repr (two trees) or a bare integrated
    expression string (simultaneous)."""
    if tree_str.lstrip().startswith("{"):
        return IndDict(eval(tree_str))
    return IndStr(tree_str)


def evaluate_file(path):
    d = json.load(open(path))
    cell = d["matrix_cell"]
    cond, strat, sgs, ds = (cell["condition"], cell["strategy"],
                            cell["sgs"], cell["dataset"])
    keep_nr = cond in NR_CONDITIONS
    params = _params(sgs, strat, cond)
    test = _test_set(ds, keep_nr)
    decision = STRATEGY_TO_DECISION[strat]

    ind = _wrap_individual(d["best_heuristic_validation"]["tree"])
    evaluate_heuristic(individual=ind, domains=test, compile_func=gp.compile,
                       pset=params.set_primitves, decision_type=decision,
                       simulator=params.simulator)

    feas = ind.case_feasible
    vals = ind.case_fitness
    feas_vals = [v for v, ok in zip(vals, feas) if ok]

    # 10-class figures, recomputed feasible-only from the stored records so the
    # comparison is apples-to-apples (the stored mean can be sentinel-inflated).
    recs = d["best_heuristic_validation"].get("test_case_records", [])
    ten_vals = [r["fitness"] for r in recs if r.get("feasible")]

    return {
        "dataset": ds, "sgs": sgs, "strategy": strat,
        "condition": cond, "seed": cell["seed"],
        "test_ard_108": (sum(feas_vals) / len(feas_vals)) if feas_vals else None,
        "feasibility_108": sum(feas) / len(feas),
        "n_feasible_108": sum(feas),
        "test_ard_10": (sum(ten_vals) / len(ten_vals)) if ten_vals else None,
        "feasibility_10": (sum(1 for r in recs if r.get("feasible")) / len(recs))
        if recs else None,
    }


N_WORKERS = int(os.environ.get("FULL108_WORKERS", "4"))


def main():
    files = sorted(glob.glob(str(MATRIX_DIR / "*.json")))
    n = len(files)
    print(f"re-evaluating {n} cells on the full 108-class test set "
          f"({N_WORKERS} workers)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "all_runs.json"
    progress = OUT_DIR / "progress.txt"

    results = []
    with Pool(processes=N_WORKERS) as pool:
        for i, rec in enumerate(pool.imap_unordered(evaluate_file, files, chunksize=1), 1):
            results.append(rec)
            if i % 25 == 0 or i == n:
                print(f"  {i}/{n}", flush=True)
                progress.write_text(f"{i}/{n}\n")
                # checkpoint so a crash/kill keeps completed work
                json.dump(results, open(out, "w"))

    json.dump(results, open(out, "w"))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
