"""
Head-to-head benchmark: baseline GPHH (Tian, Mei & Zhang, CEC 2024) vs. our
custom EA (DMGE, modification_integrated_gp). Both arms are hyper-heuristics:
each TRAINS one evolved rule on a training set, then applies it (cheaply) to
every held-out test instance. The per-seed result is the mean ARD% over that
same held-out test set, so the two arms are directly, fairly comparable via
paired statistics across seeds.

Metrics (held-out TEST set, per seed -> aggregated)
  dev_all       mean % deviation from CPM lower bound over all test instances
                (NR-infeasible schedules carry the SGS sentinel makespan --
                this can dominate the mean; dev_feas below is the metric to
                report when feasibility isn't ~100%)
  feasible      fraction of test instances scheduled NR-feasibly
  dev_feas      mean % deviation over feasible test instances only
  time_s        wall-clock seconds for this arm's whole per-seed run
                (training + test evaluation)

Convergence curves (best fitness per generation) are written per
(dataset, label, seed) to <out>/convergence/ for plotting (see
plot_comparison.py); not part of the aggregated scalar table.

Significance: paired Wilcoxon signed-rank across seeds (custom_ea vs
baseline_gphh).

Usage
-----
    python yuantian/experiment_runner.py --pop 50 --gen 10 -n 10 \
        --datasets PSPLIB_J20 --train 16 --test 16 --out results/two_way

Quick smoke test:
    python yuantian/experiment_runner.py --pop 10 --gen 3 -n 2 \
        --datasets PSPLIB_J20 --train 4 --test 4
"""

import csv
import functools
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
from scipy import stats

from yuantian.gphh_solver import GPHH, ParametersGPHH, RefreshHallOfFame, read_instances_with_nr, read_instances
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum
from yuantian.custom_ea import EA_REGISTRY, _build_heuristic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PopulationArchive  # noqa: E402

DT, SIM = DecisionTypeEnum.ACTIVITY_THEN_MODE, SimulatorTypeEnum.SERIAL_SGS

# (label, algo, all_mods, driver_kwargs)
CONFIGS = [
    ("baseline_gphh", "standard", False, {}),
    ("custom_ea", "mod_integrated", True, {"enabled_grafts": ("NR", "CP", "RENEWABLE")}),
]
FLAGSHIP = "custom_ea"

DATASET_FETCHERS = {
    "PSPLIB_J20": RCPSPDatabase.get_psplib_j20_files,
    "MMLIBPLUS_NR_50": RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files,
    "MMLIBPLUS_NR_100": RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files,
}

# (train_range, val_range, test_range) of within-class instance indices, a
# 60/20/20 split matching Tian et al. (2024)'s protocol -- MMLIB+ has 5 cases
# per class (3/1/1), PSPLIB J20 has 10 instances per class (6/2/2).
DATASET_SPLITS = {
    "PSPLIB_J20": ((1, 7), (7, 9), (9, 11)),
    "MMLIBPLUS_NR_50": ((1, 4), (4, 5), (5, 6)),
    "MMLIBPLUS_NR_100": ((1, 4), (4, 5), (5, 6)),
}


def make_params(all_mods, pop, gen, use_opportunity=False):
    p = ParametersGPHH.medium(
        decision_type=DT, simulator_type=SIM, cpus=1,
        use_modifications=all_mods, use_nr_terminals=all_mods,
        use_scheduling_state_terminals=all_mods, use_cp_mutation=all_mods,
        use_opportunity_terminals=use_opportunity)
    p.pop_size, p.n_gen = pop, gen
    return p


def make_stats():
    sf = tools.Statistics(lambda ind: ind.fitness.values)
    ms = tools.MultiStatistics(fitness=sf, size=tools.Statistics(len))
    for name, fn in [("avg", np.mean), ("min", np.min)]:
        ms.register(name, fn)
    return ms


def evenly_sampled(files, n):
    if n >= len(files):
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in dict.fromkeys(idx)]


def load_dataset(name, n_train, n_val, n_test, renewable=False):
    fetch = DATASET_FETCHERS[name]
    loader = read_instances if renewable else read_instances_with_nr
    train_range, val_range, test_range = DATASET_SPLITS[name]
    train = loader(evenly_sampled(fetch(*train_range), n_train))
    val = loader(evenly_sampled(fetch(*val_range), n_val))
    test = loader(evenly_sampled(fetch(*test_range), n_test))
    return train, val, test


@functools.lru_cache(maxsize=None)
def _cached_dataset(name, n_train, n_val, n_test, renewable):
    return load_dataset(name, n_train, n_val, n_test, renewable)


# ── GPHH-family arm ──────────────────────────────────────────────────────────

def evaluate_on_test(individual, toolbox, test_domains):
    simulator, heuristic = _build_heuristic(individual, toolbox)
    devs, feas_devs, n_feasible = [], [], 0
    for domain in test_domains:
        sol = simulator.buildSolution(domain=domain, choose=heuristic)
        mk = sol.get_end_time(domain.sink_task)
        dev = (mk - domain.cpm_esd) * 100 / domain.cpm_esd
        devs.append(dev)
        if getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7:
            n_feasible += 1
            feas_devs.append(dev)
    return (float(np.mean(devs)), n_feasible / len(test_domains),
            float(np.mean(feas_devs)) if feas_devs else float("nan"))


def select_best_on_validation(final_pop, toolbox, val_domains):
    """Tian et al. (2024), Sec. III-A: the final individual is the one with
    the best VALIDATION-set fitness among the whole final population, not
    the training-fitness-best HallOfFame entry. See run_evaluation.py's
    copy of this function for the full rationale."""
    best_ind, best_val = None, float("inf")
    for ind in final_pop:
        simulator, heuristic = _build_heuristic(ind, toolbox)
        devs = []
        for domain in val_domains:
            sol = simulator.buildSolution(domain=domain, choose=heuristic)
            mk = sol.get_end_time(domain.sink_task)
            devs.append((mk - domain.cpm_esd) * 100 / domain.cpm_esd)
        val_fitness = float(np.mean(devs))
        if val_fitness < best_val:
            best_val, best_ind = val_fitness, ind
    return best_ind


def run_gphh(algorithm, params, kwargs, train, val, test, seed, convergence_path):
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
    t0 = time.perf_counter()
    final_pop, logbook = EA_REGISTRY[algorithm](
        pop, solver.toolbox, cxpb=params.crossover_rate, mutpb=params.mutation_rate,
        n_elite=params.n_elite, ngen=params.n_gen,
        training_data_provider=train_p, validation_data_provider=val_p,
        stats=make_stats(), halloffame=hof, pop_archive=PopulationArchive(), **kwargs)
    best = select_best_on_validation(final_pop, solver.toolbox, val)
    dev_all, feasible, dev_feas = evaluate_on_test(best, solver.toolbox, test)
    elapsed = time.perf_counter() - t0
    try:
        history = logbook.chapters["fitness"].select("min")
    except (KeyError, AttributeError):
        history = []
    _write_convergence(convergence_path, history)
    return dev_all, feasible, dev_feas, elapsed


def _write_convergence(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([float(x) for x in history], f)


# ── task dispatch ────────────────────────────────────────────────────────────

def _task(args):
    (dataset, label, algo, all_mods, kwargs, seed, pop, gen,
     n_train, n_val, n_test, renewable, out_dir) = args
    conv_path = os.path.join(out_dir, "convergence", f"{dataset}__{label}__seed{seed}.json")
    try:
        train, val, test = _cached_dataset(dataset, n_train, n_val, n_test, renewable)
        res = run_gphh(algo, make_params(all_mods, pop, gen), kwargs, train, val, test, seed, conv_path)
    except Exception as exc:
        print(f"!! FAILED {dataset}/{label}/seed{seed}: {exc}", flush=True)
        res = (float("nan"), float("nan"), float("nan"), float("nan"))
    return dataset, label, seed, res


# ── statistics ───────────────────────────────────────────────────────────────

def wilcoxon(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return float("nan"), 1.0
    try:
        s, p = stats.wilcoxon(a, b, alternative="less", zero_method="zsplit")
        return float(s), float(p)
    except ValueError:
        return float("nan"), float("nan")


def star(p):
    if np.isnan(p):
        return " "
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--datasets", dest="datasets", default="PSPLIB_J20",
                      help="comma list: PSPLIB_J20,MMLIBPLUS_NR_50,MMLIBPLUS_NR_100")
    parse.add_option("--pop", dest="pop", type="int", default=50, help="GPHH-arm population size")
    parse.add_option("--gen", dest="gen", type="int", default=10, help="GPHH-arm generations")
    parse.add_option("-n", dest="n_seeds", type="int", default=10)
    parse.add_option("--seed", dest="seed", type="int", default=42)
    parse.add_option("--train", dest="n_train", type="int", default=16,
                      help="training-set size for both arms")
    parse.add_option("--val", dest="n_val", type="int", default=8,
                      help="validation-set size (used for final-individual selection, Tian et al.'s protocol)")
    parse.add_option("--test", dest="n_test", type="int", default=16)
    parse.add_option("--out", dest="out", default="results/two_way")
    parse.add_option("--algos", dest="algos", default="",
                      help="comma list to run a subset, e.g. custom_ea (default: both)")
    parse.add_option("--workers", dest="workers", type="int", default=0)
    parse.add_option("--renewable", action="store_true", dest="renewable", default=False,
                      help="strip nonrenewable resources (renewable-only benchmark)")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    datasets = [d.strip() for d in opt.datasets.split(",") if d.strip()]
    configs = CONFIGS
    if opt.algos:
        wanted = [c.strip() for c in opt.algos.split(",") if c.strip()]
        configs = [c for c in CONFIGS if c[0] in wanted]
        assert configs, f"no known configs in {wanted}"

    seeds = [opt.seed + 100 * s for s in range(opt.n_seeds)]
    workers = opt.workers or os.cpu_count()
    tasks = [(d, label, algo, all_mods, kwargs, sd, opt.pop, opt.gen,
              opt.n_train, opt.n_val, opt.n_test, opt.renewable, opt.out)
             for d in datasets for sd in seeds
             for (label, algo, all_mods, kwargs) in configs]
    print(f"Datasets={datasets}  configs={[c[0] for c in configs]}  "
          f"seeds={opt.n_seeds}  pop/gen={opt.pop}/{opt.gen}  "
          f"renewable_only={opt.renewable}  {len(tasks)} runs on {workers} workers")

    raw_rows = []
    raw = {d: {label: {} for label, *_ in configs} for d in datasets}

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            dataset, label, seed, res = fut.result()
            raw[dataset][label][seed] = res
            raw_rows.append([dataset, label, seed, *res])
            elapsed = time.time() - t_start
            eta = elapsed / k * (len(tasks) - k)
            print(f"[{k:>3}/{len(tasks)}] {dataset:15s} {label:14s} seed {seed}  "
                  f"dev_all={res[0]:>9.2f} feas={res[1] * 100:>3.0f}% time={res[3]:>7.1f}s  "
                  f"elapsed={elapsed:>5.0f}s ETA={eta:>5.0f}s", flush=True)

    with open(os.path.join(opt.out, "raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "seed", "dev_all", "feasible", "dev_feas", "time_s"])
        w.writerows(raw_rows)

    summary_rows = []
    for dataset in datasets:
        print(f"\n{'=' * 100}\n{dataset}\n{'=' * 100}")
        print(f"{'config':16s} {'dev_all mean±std':>22s} {'feas%':>7s} {'dev_feas':>10s} "
              f"{'time_s':>9s} {'vs custom_ea':>12s}")
        print("-" * 100)

        def col(lbl, i):
            return np.array([raw[dataset][lbl][sd][i] for sd in seeds])

        dev_all = {lbl: col(lbl, 0) for lbl, *_ in configs}
        feas = {lbl: col(lbl, 1) for lbl, *_ in configs}
        dev_feas = {lbl: col(lbl, 2) for lbl, *_ in configs}
        time_s = {lbl: col(lbl, 3) for lbl, *_ in configs}

        have_flagship = FLAGSHIP in dev_all
        for label, *_ in configs:
            da, fe, df, ts = dev_all[label], feas[label], dev_feas[label], time_s[label]
            dfv = df[~np.isnan(df)]
            p_vs_flagship = (wilcoxon(dev_all[FLAGSHIP], da)[1]
                              if have_flagship and label != FLAGSHIP else float("nan"))
            vs_flagship = "—" if label == FLAGSHIP or not have_flagship else star(p_vs_flagship)
            dfv_str = f"{dfv.mean():.2f}" if dfv.size else "n/a"
            print(f"{label:16s} {da.mean():>11.2f} ± {da.std():<8.2f} "
                  f"{fe.mean() * 100:>5.0f}% {dfv_str:>10s} {ts.mean():>9.1f} {vs_flagship:>12s}")
            summary_rows.append([dataset, label, da.mean(), da.std(), fe.mean(),
                                  (dfv.mean() if dfv.size else float("nan")),
                                  ts.mean(), p_vs_flagship])

        try:
            cols = [dev_all[lbl] for lbl, *_ in configs]
            chi, p_fried = stats.friedmanchisquare(*cols)
            print(f"\nFriedman across configs (dev_all): chi2={chi:.2f}  p={p_fried:.4f}")
        except ValueError as e:
            print(f"\nFriedman test undefined: {e}")

    with open(os.path.join(opt.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "dev_all_mean", "dev_all_std", "feasible",
                    "dev_feas_mean", "time_s_mean", "p_vs_custom_ea"])
        w.writerows(summary_rows)

    print(f"\nWrote {opt.out}/raw.csv, {opt.out}/summary.csv, "
          f"{opt.out}/convergence/*.json  ({time.time() - t_start:.0f}s total)")
