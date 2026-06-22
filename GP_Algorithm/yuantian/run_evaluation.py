"""
Evaluation harness for DMGE (modification_integrated_gp) on NR-preserving MRCPSP.

Produces thesis-grade evidence: multiple seeds, multiple datasets, control
configurations that isolate the operator from the terminals, leave-one-out graft
ablations, and paired significance tests.

Configurations
--------------
  baseline        original GPHH (standard driver, no modifications)
  mods_standard   all modifications/terminals, STANDARD driver
                  → the key control: same terminals, no diagnostic-graft operator
  dmge_full       DMGE with all grafts (NR + CP + RENEWABLE)
  dmge_no_nr      DMGE ablation: NR graft disabled
  dmge_no_cp      DMGE ablation: CP graft disabled
  dmge_no_renew   DMGE ablation: RENEWABLE graft disabled

All configs share identical per-seed RNG and the same subsampled instance set,
so every test metric is paired across seeds.

Metrics (held-out TEST set, per seed → aggregated)
  dev_all    mean % deviation from CPM lower bound over all test instances
             (NR-infeasible schedules carry the SGS sentinel makespan)
  feasible   fraction of test instances scheduled NR-feasibly
  dev_feas   mean % deviation over feasible test instances only

Significance
  Paired Wilcoxon signed-rank on dev_all across seeds:
    dmge_full vs baseline       (does the method help at all?)
    dmge_full vs mods_standard  (does the OPERATOR help, terminals held fixed?)
    dmge_full vs each ablation  (does each graft contribute?)
  Friedman test across all configs as an omnibus check.
  Results saved to <out>/raw.csv and <out>/summary.csv for the thesis.

Usage
-----
    python yuantian/run_evaluation.py --pop 50 --gen 10 -n 10 \
        --datasets MMLIBPLUS_NR_50,MMLIBPLUS_NR_100 \
        --train 16 --val 8 --test 16 --out results/thesis_eval

Quick smoke test:
    python yuantian/run_evaluation.py --pop 20 --gen 4 -n 3 --train 8 --val 4 --test 8
"""

import os
import sys
import csv
import time
import random
import warnings
import functools
from optparse import OptionParser
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from deap import tools

from yuantian.gphh_solver import (GPHH, ParametersGPHH, read_instances_with_nr,
                                  read_instances, RefreshHallOfFame)
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum
from yuantian.custom_ea import (EA_REGISTRY, _build_heuristic,
                                evaluate_on_test_multi_sgs)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils import PopulationArchive  # noqa: E402

DT, SIM = DecisionTypeEnum.ACTIVITY_THEN_MODE, SimulatorTypeEnum.SERIAL_SGS


# ── configurations ───────────────────────────────────────────────────────────
# (label, algorithm, all_mods?, driver_kwargs)
CONFIGS = [
    ("baseline",      "standard",       False, {}),
    ("mods_standard", "standard",       True,  {}),
    ("dmge_full",     "mod_integrated", True,  {"enabled_grafts": ("NR", "CP", "RENEWABLE")}),
    ("dmge_no_nr",    "mod_integrated", True,  {"enabled_grafts": ("CP", "RENEWABLE")}),
    ("dmge_no_cp",    "mod_integrated", True,  {"enabled_grafts": ("NR", "RENEWABLE")}),
    ("dmge_no_renew", "mod_integrated", True,  {"enabled_grafts": ("NR", "CP")}),
    ("dmge_nograft",  "mod_integrated", True,  {"enabled_grafts": ()}),  # grafts OFF — isolates graft effect
    # TDRE: NR-feasibility-directed variation, no graft operator
    ("tdre",          "trace_directed", False, {}),
    ("tdre_mods",     "trace_directed", True,  {}),
    # Lexicase family: epsilon-lexicase selection + ERCs + mini-batch (La Cava et al. 2016)
    ("lexicase",      "lexicase",       False, {}),                       # lexicase improvements, no mods
    ("lexicase_mods", "lexicase",       True,  {}),                       # lexicase + all modifications
]

# Multi-SGS family (serial + parallel + backward SGS, best per instance) is
# DELIBERATELY EXCLUDED from CONFIGS: multi_sgs_gp has an unresolved
# pathological hang (>=2 generations on >=4 MMLIB+ NR instances, population
# >= 8 -- see custom_ea.py's module docstring). Add it back explicitly via
# --configs only if you have first fixed or worked around that hang.
EXCLUDED_HANGING_CONFIGS = [
    ("multi_sgs",      "multi_sgs", False, {}),
    ("multi_sgs_mods", "multi_sgs", True,  {}),
]
FLAGSHIP = "dmge_full"
CONTROL = "mods_standard"


def make_params(all_mods, pop, gen):
    p = ParametersGPHH.medium(
        decision_type=DT, simulator_type=SIM, cpus=1,
        use_modifications=all_mods, use_nr_terminals=all_mods,
        use_scheduling_state_terminals=all_mods, use_cp_mutation=all_mods)
    p.pop_size, p.n_gen = pop, gen
    return p


def make_stats():
    sf = tools.Statistics(lambda ind: ind.fitness.values)
    ms = tools.MultiStatistics(fitness=sf, size=tools.Statistics(len))
    for name, fn in [("avg", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
        ms.register(name, fn)
    return ms


def evenly_sampled(files, n):
    if n >= len(files):
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in dict.fromkeys(idx)]


def load_dataset(name, n_train, n_val, n_test, renewable=False):
    fetch = (RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files
             if name.endswith("100")
             else RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files)
    loader = read_instances if renewable else read_instances_with_nr  # NR-strip vs NR-preserve
    train = loader(evenly_sampled(fetch(1, 4), n_train))
    val = loader(evenly_sampled(fetch(4, 5), n_val))
    test = loader(evenly_sampled(fetch(5, 6), n_test))
    return train, val, test


def evaluate_on_test(individual, toolbox, test_domains, multi_sgs=False):
    """Evaluate best individual on the held-out test set.

    When ``multi_sgs=True`` runs serial + parallel + backward SGS and takes
    the best feasible makespan per instance, matching the training fitness
    landscape used by the ``multi_sgs`` driver.
    """
    if multi_sgs:
        return evaluate_on_test_multi_sgs(individual, toolbox, test_domains)
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
    """Tian et al. (2024), Sec. III-A: 'all individuals in the final
    population are evaluated on a validation set. The best-performing
    individual on the validation set is selected as the final output.'
    DEAP's HallOfFame tracks the training-fitness best instead (see
    gp_algorithms.py: halloffame.update() ranks by the training-set
    .fitness DEAP sets, validation is only logged, never used for
    selection) -- this replicates the paper's actual protocol."""
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


def run_once(algorithm, params, kwargs, train, val, test, seed):
    random.seed(seed)
    np.random.seed(seed)
    train_p, val_p, test_p = (StaticDatasetProvider(train),
                              StaticDatasetProvider(val), StaticDatasetProvider(test))
    solver = GPHH(training_set_provider=train_p, validation_set_provider=val_p,
                  test_set_provider=test_p, params_gphh=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.init_model()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = RefreshHallOfFame(1)
    final_pop, _logbook = EA_REGISTRY[algorithm](
        pop, solver.toolbox, cxpb=params.crossover_rate, mutpb=params.mutation_rate,
        n_elite=params.n_elite, ngen=params.n_gen,
        training_data_provider=train_p, validation_data_provider=val_p,
        stats=make_stats(), halloffame=hof, pop_archive=PopulationArchive(), **kwargs)
    use_triple = (algorithm == "multi_sgs")
    # multi_sgs's own training landscape is best-of-three-SGS, not the plain
    # serial decode select_best_on_validation uses -- keep the training-
    # fitness HOF for that one driver rather than mismatch the two.
    best = hof[0] if use_triple else select_best_on_validation(final_pop, solver.toolbox, val)
    return evaluate_on_test(best, solver.toolbox, test, multi_sgs=use_triple)


@functools.lru_cache(maxsize=None)
def _cached_dataset(name, n_train, n_val, n_test, renewable):
    """Per-worker dataset cache: each process loads a given dataset once."""
    return load_dataset(name, n_train, n_val, n_test, renewable)


def _task(args):
    """One (dataset, config, seed) run — executed in a worker process.  Fully
    in-process GP, so DEAP individuals never cross a process boundary."""
    dataset, label, algo, all_mods, kwargs, seed, pop, gen, counts, renewable = args
    try:
        train, val, test = _cached_dataset(dataset, *counts, renewable)
        res = run_once(algo, make_params(all_mods, pop, gen), kwargs,
                       train, val, test, seed)
    except Exception as exc:                       # keep the matrix alive on failure
        print(f"!! FAILED {dataset}/{label}/seed{seed}: {exc}", flush=True)
        res = (float("nan"), float("nan"), float("nan"))
    return dataset, label, seed, res


def wilcoxon(a, b):
    """Paired one-sided Wilcoxon: is `a` < `b` (a better, lower dev)?  Returns
    (statistic, p) or (nan, nan) when the test is undefined."""
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
    parse.add_option("--datasets", dest="datasets", default="MMLIBPLUS_NR_50",
                     help="comma list: MMLIBPLUS_NR_50,MMLIBPLUS_NR_100")
    parse.add_option("--pop", dest="pop", type="int", default=50)
    parse.add_option("--gen", dest="gen", type="int", default=10)
    parse.add_option("-n", dest="n_seeds", type="int", default=10)
    parse.add_option("--seed", dest="seed", type="int", default=42)
    parse.add_option("--train", dest="n_train", type="int", default=16)
    parse.add_option("--val", dest="n_val", type="int", default=8)
    parse.add_option("--test", dest="n_test", type="int", default=16)
    parse.add_option("--out", dest="out", default="results/thesis_eval")
    parse.add_option("--configs", dest="configs", default="",
                     help="comma list to run a subset, e.g. baseline,dmge_full "
                          "(default: all six)")
    parse.add_option("--workers", dest="workers", type="int", default=0,
                     help="parallel worker processes (default: all CPU cores)")
    parse.add_option("--renewable", action="store_true", dest="renewable", default=False,
                     help="strip nonrenewable resources (renewable-only benchmark)")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    datasets = [d.strip() for d in opt.datasets.split(",") if d.strip()]
    if opt.configs:
        wanted = [c.strip() for c in opt.configs.split(",") if c.strip()]
        CONFIGS = [c for c in CONFIGS if c[0] in wanted]
        assert CONFIGS, f"no known configs in {wanted}"

    seeds = [opt.seed + 100 * s for s in range(opt.n_seeds)]
    workers = opt.workers or os.cpu_count()
    counts = (opt.n_train, opt.n_val, opt.n_test)
    tasks = [(d, label, algo, all_mods, kwargs, sd, opt.pop, opt.gen, counts, opt.renewable)
             for d in datasets for sd in seeds
             for (label, algo, all_mods, kwargs) in CONFIGS]
    print(f"Datasets={datasets}  configs={[c[0] for c in CONFIGS]}  "
          f"seeds={opt.n_seeds}  pop={opt.pop} gen={opt.gen}  "
          f"renewable_only={opt.renewable}  {len(tasks)} runs on {workers} workers")

    # raw[dataset][label][seed] = (dev_all, feasible, dev_feas) — seed-keyed for pairing
    raw_rows = []
    raw = {d: {label: {} for label, *_ in CONFIGS} for d in datasets}

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
                  f"dev_all={res[0]:>12.2f} feas={res[1]*100:>3.0f}%  "
                  f"elapsed={elapsed:>5.0f}s ETA={eta:>5.0f}s", flush=True)

    # ── write raw ────────────────────────────────────────────────────────────
    with open(os.path.join(opt.out, "raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "seed", "dev_all", "feasible", "dev_feas"])
        w.writerows(raw_rows)

    # ── summary + significance ───────────────────────────────────────────────
    summary_rows = []
    for dataset in datasets:
        print(f"\n{'='*94}\n{dataset}\n{'='*94}")
        print(f"{'config':14s} {'dev_all mean±std':>22s} {'feas%':>7s} {'dev_feas':>10s} "
              f"{'vs base':>9s} {'vs ctrl':>9s}")
        print("-" * 94)
        # index 0/1/2 = dev_all/feasible/dev_feas, read in a fixed seed order so
        # the columns stay paired across configs for the Wilcoxon/Friedman tests
        def col(lbl, i):
            return np.array([raw[dataset][lbl][sd][i] for sd in seeds])
        dev_all = {lbl: col(lbl, 0) for lbl, *_ in CONFIGS}
        feas = {lbl: col(lbl, 1) for lbl, *_ in CONFIGS}
        dev_feas = {lbl: col(lbl, 2) for lbl, *_ in CONFIGS}

        have_base = "baseline" in dev_all
        have_ctrl = CONTROL in dev_all
        for label, *_ in CONFIGS:
            da, fe, df = dev_all[label], feas[label], dev_feas[label]
            dfv = df[~np.isnan(df)]
            p_base = wilcoxon(da, dev_all["baseline"])[1] if have_base else float("nan")
            p_ctrl = wilcoxon(da, dev_all[CONTROL])[1] if have_ctrl else float("nan")
            vs_base = "—" if label == "baseline" or not have_base else star(p_base)
            vs_ctrl = "—" if label == CONTROL or not have_ctrl else star(p_ctrl)
            dfv_str = f"{dfv.mean():.2f}" if dfv.size else "n/a"
            print(f"{label:14s} {da.mean():>11.2f} ± {da.std():<8.2f} "
                  f"{fe.mean()*100:>5.0f}% {dfv_str:>10s} {vs_base:>9s} {vs_ctrl:>9s}")
            summary_rows.append([
                dataset, label, da.mean(), da.std(), fe.mean(),
                (dfv.mean() if dfv.size else float("nan")), p_base, p_ctrl])

        # omnibus Friedman across configs (dev_all, paired by seed)
        try:
            cols = [dev_all[lbl] for lbl, *_ in CONFIGS]
            chi, p_fried = stats.friedmanchisquare(*cols)
            print(f"\nFriedman across configs (dev_all): chi2={chi:.2f}  p={p_fried:.4f}")
        except ValueError as e:
            print(f"\nFriedman test undefined: {e}")
        if FLAGSHIP in dev_all and have_base:
            ref = CONTROL if have_ctrl else "baseline"
            label_txt = ("operator effect, terminals fixed" if have_ctrl
                         else "vs baseline")
            print(f"\nKey test — {FLAGSHIP} vs {ref} ({label_txt}):")
            _, p = wilcoxon(dev_all[FLAGSHIP], dev_all[ref])
            print(f"  Wilcoxon one-sided p={p:.4f} {star(p)}   "
                  f"(feas {feas[FLAGSHIP].mean()*100:.0f}% vs {feas[ref].mean()*100:.0f}%)")

    with open(os.path.join(opt.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "dev_all_mean", "dev_all_std", "feasible_mean",
                    "dev_feas_mean", "p_vs_baseline", "p_vs_control"])
        w.writerows(summary_rows)

    print(f"\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
    print(f"Saved: {opt.out}/raw.csv , {opt.out}/summary.csv")
