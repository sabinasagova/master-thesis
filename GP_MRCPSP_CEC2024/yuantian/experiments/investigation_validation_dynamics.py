"""
Investigation 2: per-generation train/validation/test fitness dynamics for
baseline vs lexicase (serial SGS).

Neither full_mmlib_experiment.py nor serial_lexicase_power_followup.py
persists the per-generation validation_fitness that gp_algorithms.standard_gp
already computes internally (it's in log.chapters["generation_best"] but
only the final value ever reaches the saved JSON). This script re-runs a
modest number of fresh seeds, captures the full logbook, and additionally
reconstructs each generation's hall-of-fame tree to evaluate it on the test
set too (cheap: ~20 extra small evaluations per run, no extra evolution).

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python -O yuantian/experiments/investigation_validation_dynamics.py
"""
import json
import random
import sys
import time
import warnings
from functools import partial
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
from yuantian.gp_algorithms import standard_gp
from yuantian.gphh_solver import GPHH, StaticDatasetProvider, RefreshHallOfFame, evaluate_heuristic
from yuantian.hybrid_gp import epsilon_lexicase_selection
from yuantian.multitreegp import MultiPrimitiveTree
from yuantian.utils import PopulationArchive
from deap import tools

SEED_BASE = 6000
N_SEEDS = 6
METHODS = ("baseline", "lexicase")

OUTPUT_DIR = Path(__file__).parent / "results" / "investigation_validation_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "curves.json"


def _make_mstats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def run_with_curves(seed, method, training, validation, test):
    random.seed(seed)
    np.random.seed(seed)
    params = build_params("serial")
    solver = GPHH(
        training_set_provider=StaticDatasetProvider(training),
        validation_set_provider=StaticDatasetProvider(validation),
        params_gphh=params,
    )
    solver.init_model()
    if method == "lexicase":
        solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)

    mstats = _make_mstats()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = RefreshHallOfFame(1)
    pop_archive = PopulationArchive()

    _, log = standard_gp(
        pop,
        solver.toolbox,
        cxpb=params.crossover_rate,
        mutpb=params.mutation_rate,
        n_elite=params.n_elite,
        ngen=params.n_gen,
        training_data_provider=StaticDatasetProvider(training),
        validation_data_provider=StaticDatasetProvider(validation),
        stats=mstats,
        halloffame=hof,
        pop_archive=pop_archive,
        verbose=False,
    )

    gen_best = log.chapters["generation_best"]
    train_curve = [g["fitness"] for g in gen_best]
    val_curve = [g["validation_fitness"] for g in gen_best]
    test_curve = []
    for g in gen_best:
        tree = MultiPrimitiveTree.from_string(g["tree"], solver.pset)
        fit = evaluate_heuristic(
            individual=tree,
            domains=test,
            compile_func=solver.toolbox.compile,
            pset=solver.pset,
            decision_type=solver.decision_type,
            simulator=solver.simulator,
        )
        test_curve.append(float(fit[0]))

    return {
        "seed": seed,
        "method": method,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "test_curve": test_curve,
    }


def main():
    warnings.filterwarnings("ignore")
    train_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TRAIN_FILES]
    val_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in VAL_FILES]
    test_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES]
    training = read_instances(train_paths)
    validation = read_instances(val_paths)
    test = read_instances(test_paths)

    results = []
    if OUTPUT_PATH.exists():
        results = json.load(open(OUTPUT_PATH))
        print(f"Resuming, {len(results)} runs already done")
    done = {(r["seed"], r["method"]) for r in results}

    for method in METHODS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            if (seed, method) in done:
                continue
            t0 = time.time()
            record = run_with_curves(seed, method, training, validation, test)
            results.append(record)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(
                f"[{method}] seed={seed} "
                f"train[-1]={record['train_curve'][-1]:.3f} "
                f"val[-1]={record['val_curve'][-1]:.3f} "
                f"test[-1]={record['test_curve'][-1]:.3f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

    print("Done.")


if __name__ == "__main__":
    main()
