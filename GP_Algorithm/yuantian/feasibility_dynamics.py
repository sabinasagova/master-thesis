"""
Category C (Feasibility Dynamics) data collection: per-generation population
feasibility fraction, baseline_gphh vs custom_ea, both datasets.

No core-code changes were needed for this: modification_integrated_gp already
sets ind.infeas_frac per individual (just never persisted past a stdout
print), and DEAP's tools.Statistics/_record machinery already persists
whatever a supplied Statistics object computes into the per-generation
logbook -- so a feasibility-aware Statistics object, supplied from here,
is enough. standard_gp's baseline driver does not set ind.infeas_frac (it
has no per-individual feasibility concept at all), so for that driver this
falls back to a fitness-magnitude heuristic: any individual whose mean
training fitness is in the sentinel-contaminated range (>= 1e6, several
orders of magnitude above any feasible schedule's ARD%) is treated as
"this individual's training evaluation hit at least one infeasible
instance." This is an approximation (mean-based, not per-instance), stated
explicitly here and in the resulting plots' captions.

Usage
-----
    python -m yuantian.feasibility_dynamics --datasets PSPLIB_J20 -n 10 \
        --pop 60 --gen 20 --train 16 --val 8 --test 16 --out results/feas_dynamics_j20
"""
import json
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from optparse import OptionParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from deap import tools

from yuantian.experiment_runner import CONFIGS, _cached_dataset, make_params
from yuantian.gphh_solver import GPHH, RefreshHallOfFame
from yuantian.rcpsp_dataset import StaticDatasetProvider
from yuantian.custom_ea import EA_REGISTRY

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PopulationArchive  # noqa: E402

SENTINEL_FITNESS_THRESHOLD = 1e6  # see module docstring's baseline-fallback caveat


def _is_feasible_individual(ind) -> float:
    infeas_frac = getattr(ind, "infeas_frac", None)
    if infeas_frac is not None:
        return 1.0 if infeas_frac == 0 else 0.0
    return 1.0 if ind.fitness.values[0] < SENTINEL_FITNESS_THRESHOLD else 0.0


def make_feasibility_stats():
    return tools.Statistics(_is_feasible_individual)


def run_one(algo, params, kwargs, train, val, seed):
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
    stats = make_feasibility_stats()
    stats.register("frac", np.mean)
    _final_pop, logbook = EA_REGISTRY[algo](
        pop, solver.toolbox, cxpb=params.crossover_rate, mutpb=params.mutation_rate,
        n_elite=params.n_elite, ngen=params.n_gen,
        training_data_provider=train_p, validation_data_provider=val_p,
        stats=stats, halloffame=hof, pop_archive=PopulationArchive(), **kwargs)
    return logbook.select("frac")


def _task(args):
    (dataset, label, algo, all_mods, kwargs, seed, pop, gen,
     n_train, n_val, n_test) = args
    try:
        train, val, _test = _cached_dataset(dataset, n_train, n_val, n_test, False)
        params = make_params(all_mods, pop, gen)
        trace = run_one(algo, params, kwargs, train, val, seed)
        return dataset, label, seed, [float(x) for x in trace]
    except Exception as exc:
        print(f"!! FAILED {dataset}/{label}/seed{seed}: {exc}", flush=True)
        return dataset, label, seed, []


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
    parse.add_option("--out", dest="out", default="results/feas_dynamics")
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

    traces = {}  # (dataset, label) -> list of per-seed traces
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            dataset, label, seed, trace = fut.result()
            traces.setdefault((dataset, label), []).append(trace)
            elapsed = time.time() - t0
            eta = elapsed / k * (len(tasks) - k)
            print(f"[{k:>3}/{len(tasks)}] {dataset} {label} seed{seed}  "
                  f"final_feas_frac={trace[-1] if trace else float('nan'):.2f}  "
                  f"elapsed={elapsed:>5.0f}s ETA={eta:>5.0f}s", flush=True)

    for (dataset, label), seed_traces in traces.items():
        out_path = os.path.join(opt.out, f"{dataset}__{label}__feasibility_traces.json")
        with open(out_path, "w") as f:
            json.dump(seed_traces, f)
        print(f"Wrote {out_path}")
    print(f"Done ({time.time()-t0:.0f}s total)")
