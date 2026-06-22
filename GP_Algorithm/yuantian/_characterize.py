"""Characterization harness for custom_ea.py / modifications.py (dev tool, not
part of the thesis pipeline).

Runs every driver in EA_REGISTRY for a couple of generations on a tiny dataset
with a fixed seed, and snapshots fitness values + best-individual string +
logbook into JSON. Used as a before/after oracle while refactoring: if the
snapshot is byte-identical after a change, the change was behaviour-preserving.

Usage:
    python _characterize.py snapshot before.json
    python _characterize.py snapshot after.json
    python _characterize.py diff before.json after.json
"""
import json
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from deap import tools

from yuantian.gphh_solver import GPHH, ParametersGPHH, read_instances_with_nr
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum
from yuantian.custom_ea import EA_REGISTRY
from utils import PopulationArchive

DT, SIM = DecisionTypeEnum.ACTIVITY_THEN_MODE, SimulatorTypeEnum.SERIAL_SGS
SEED = 12345
POP, GEN = 8, 2
DRIVER_KWARGS = {
    "mod_integrated": {"enabled_grafts": ("NR", "CP", "RENEWABLE")},
}


def make_stats():
    sf = tools.Statistics(lambda ind: ind.fitness.values)
    ms = tools.MultiStatistics(fitness=sf, size=tools.Statistics(len))
    for name, fn in [("avg", np.mean), ("min", np.min)]:
        ms.register(name, fn)
    return ms


def tiny_dataset():
    files = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(1, 2)[:4]
    instances = read_instances_with_nr(files)
    return instances[:2], instances[2:3], instances[3:4]


def make_params(all_mods: bool) -> ParametersGPHH:
    p = ParametersGPHH.medium(
        decision_type=DT, simulator_type=SIM, cpus=1,
        use_modifications=all_mods, use_nr_terminals=all_mods,
        use_scheduling_state_terminals=all_mods, use_cp_mutation=all_mods,
    )
    p.pop_size, p.n_gen = POP, GEN
    return p


def run_one(algorithm: str, all_mods: bool, train, val) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    train_p, val_p = StaticDatasetProvider(train), StaticDatasetProvider(val)
    params = make_params(all_mods)
    solver = GPHH(training_set_provider=train_p, validation_set_provider=val_p,
                  test_set_provider=val_p, params_gphh=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.init_model()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = tools.HallOfFame(1)
    kwargs = DRIVER_KWARGS.get(algorithm, {})
    final_pop, logbook = EA_REGISTRY[algorithm](
        pop, solver.toolbox, cxpb=params.crossover_rate, mutpb=params.mutation_rate,
        n_elite=params.n_elite, ngen=params.n_gen,
        training_data_provider=train_p, validation_data_provider=val_p,
        stats=make_stats(), halloffame=hof, pop_archive=PopulationArchive(),
        **kwargs,
    )
    fits = sorted(round(ind.fitness.values[0], 6) for ind in final_pop)
    return {
        "best_fitness": round(hof[0].fitness.values[0], 6),
        "best_tree": str(hof[0]),
        "final_pop_fitnesses": fits,
        "n_records": len(logbook),
    }


SKIP = {"multi_sgs"}  # documented unresolved hang, see custom_ea.py's module docstring -- a
                       # try/except around run_one() can't catch a hang, so skip it outright


def snapshot(out_path: str):
    train, val, _test = tiny_dataset()
    results = {}
    for algo in EA_REGISTRY:
        if algo in SKIP:
            continue
        for all_mods in (False, True):
            key = f"{algo}__mods={all_mods}"
            print(f"running {key} ...", flush=True)
            try:
                results[key] = run_one(algo, all_mods, train, val)
            except Exception as exc:
                results[key] = {"error": f"{type(exc).__name__}: {exc}"}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")


def diff(path_a: str, path_b: str):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    keys = sorted(set(a) | set(b))
    n_diff = 0
    for k in keys:
        if a.get(k) != b.get(k):
            n_diff += 1
            print(f"DIFF  {k}")
            print(f"  before: {a.get(k)}")
            print(f"  after:  {b.get(k)}")
    if n_diff == 0:
        print("IDENTICAL — no behavioural change detected.")
    else:
        print(f"\n{n_diff}/{len(keys)} configs differ.")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "snapshot":
        snapshot(sys.argv[2])
    elif cmd == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
