"""
Per-instance hardness analysis (Category A of the niche-discovery plan):
re-evaluates trained baseline/DMGE individuals on the held-out test set,
one row per (config, seed, test instance), tagged with that instance's Order
Strength, nonrenewable Resource Factor, and mean nonrenewable Resource
Strength (Van Peteghem & Vanhoucke, 2014 -- see instance_indicator.py).

This does NOT exist anywhere in the current pipeline: experiment_runner.py's
evaluate_on_test() only returns the test-set AGGREGATE (mean dev, feasible
fraction), not a per-instance breakdown, and no run has persisted its best
individual. This script re-trains (same seeds, same params as the thesis's
head-to-head run) and additionally dumps the per-instance table that
hardness_plots.py consumes.

Usage
-----
    python -m yuantian.instance_hardness --datasets PSPLIB_J20 -n 10 \
        --pop 60 --gen 20 --train 16 --val 8 --test 16 --out results/hardness_j20
"""
import csv
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from optparse import OptionParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from yuantian.experiment_runner import (
    CONFIGS, _cached_dataset, make_params, select_best_on_validation,
)
from yuantian.gphh_solver import GPHH, RefreshHallOfFame
from yuantian.rcpsp_dataset import StaticDatasetProvider
from yuantian.custom_ea import EA_REGISTRY, _build_heuristic
from yuantian.instance_indicator import OrderStrength, ResourceFactor_NR, ResourceStrength_NR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PopulationArchive  # noqa: E402


def instance_hardness(domain) -> dict:
    """Scalar hardness triple for one RCPSPModel instance. ResourceStrength_NR
    returns one value per nonrenewable resource; mean() collapses that to a
    single scalar for plotting (instances here have 1-2 NR resource types)."""
    rs_nr = ResourceStrength_NR(domain)
    return {
        "order_strength": OrderStrength(domain),
        "resource_factor_nr": ResourceFactor_NR(domain),
        "resource_strength_nr": float(np.mean(list(rs_nr.values()))) if rs_nr else float("nan"),
    }


def per_instance_eval(individual, toolbox, test_domains, label, seed):
    simulator, heuristic = _build_heuristic(individual, toolbox)
    rows = []
    for domain in test_domains:
        sol = simulator.buildSolution(domain=domain, choose=heuristic)
        mk = sol.get_end_time(domain.sink_task)
        dev = (mk - domain.cpm_esd) * 100 / domain.cpm_esd
        feasible = bool(getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7)
        row = {"config": label, "seed": seed,
               "instance_id": os.path.basename(getattr(domain, "file_path", "?")),
               "dev": dev, "is_feasible": feasible}
        row.update(instance_hardness(domain))
        rows.append(row)
    return rows


def run_one(label, algo, params, kwargs, train, val, test, seed):
    random.seed(seed)
    np.random.seed(seed)
    train_p, val_p = StaticDatasetProvider(train), StaticDatasetProvider(val)
    solver = GPHH(training_set_provider=train_p, validation_set_provider=val_p,
                  test_set_provider=val_p, params_gphh=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.init_model()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = RefreshHallOfFame(1)
    final_pop, _logbook = EA_REGISTRY[algo](
        pop, solver.toolbox, cxpb=params.crossover_rate, mutpb=params.mutation_rate,
        n_elite=params.n_elite, ngen=params.n_gen,
        training_data_provider=train_p, validation_data_provider=val_p,
        stats=None, halloffame=hof, pop_archive=PopulationArchive(), **kwargs)
    best = select_best_on_validation(final_pop, solver.toolbox, val)
    return per_instance_eval(best, solver.toolbox, test, label, seed)


def _task(args):
    (dataset, label, algo, all_mods, kwargs, seed, pop, gen,
     n_train, n_val, n_test) = args
    try:
        train, val, test = _cached_dataset(dataset, n_train, n_val, n_test, False)
        params = make_params(all_mods, pop, gen)
        rows = run_one(label, algo, params, kwargs, train, val, test, seed)
        for row in rows:
            row["dataset"] = dataset
        return rows
    except Exception as exc:
        print(f"!! FAILED {dataset}/{label}/seed{seed}: {exc}", flush=True)
        return []


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--datasets", dest="datasets", default="PSPLIB_J20")
    parse.add_option("--pop", dest="pop", type="int", default=60)
    parse.add_option("--gen", dest="gen", type="int", default=20)
    parse.add_option("-n", dest="n_seeds", type="int", default=10)
    parse.add_option("--seed", dest="seed", type="int", default=42)
    parse.add_option("--train", dest="n_train", type="int", default=16)
    parse.add_option("--val", dest="n_val", type="int", default=8)
    parse.add_option("--test", dest="n_test", type="int", default=16)
    parse.add_option("--out", dest="out", default="results/hardness")
    parse.add_option("--workers", dest="workers", type="int", default=0)
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    datasets = [d.strip() for d in opt.datasets.split(",") if d.strip()]
    seeds = [opt.seed + 100 * s for s in range(opt.n_seeds)]
    workers = opt.workers or os.cpu_count()

    tasks = [(d, label, algo, all_mods, kwargs, sd, opt.pop, opt.gen,
              opt.n_train, opt.n_val, opt.n_test)
             for d in datasets for sd in seeds
             for (label, algo, all_mods, kwargs) in CONFIGS]
    print(f"Datasets={datasets}  configs={[c[0] for c in CONFIGS]}  "
          f"seeds={opt.n_seeds}  pop/gen={opt.pop}/{opt.gen}  "
          f"{len(tasks)} runs on {workers} workers")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            rows.extend(result)
            elapsed = time.time() - t0
            eta = elapsed / k * (len(tasks) - k)
            n_feas = sum(r["is_feasible"] for r in result)
            print(f"[{k:>3}/{len(tasks)}] done ({n_feas}/{len(result)} feasible test "
                  f"instances)  elapsed={elapsed:>5.0f}s ETA={eta:>5.0f}s", flush=True)

    out_path = os.path.join(opt.out, "per_instance.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "config", "seed", "instance_id",
                                          "dev", "is_feasible", "order_strength",
                                          "resource_factor_nr", "resource_strength_nr"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows, {time.time()-t0:.0f}s total)")
