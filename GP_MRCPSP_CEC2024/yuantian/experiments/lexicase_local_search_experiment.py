"""
Experiment: epsilon-lexicase selection + critical-path local search on elites
vs the original baseline GPHH (tournament selection, no local search).

Both conditions share: population size, generations, crossover/mutation
rates, elitism count, tournament size (unused by the proposed condition,
kept identical for clarity), training/test instances, and the same
mate/mutate/compile/evaluate operators. Only the selection operator and the
elite local-search step differ.

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python yuantian/experiments/lexicase_local_search_experiment.py
"""
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from deap import tools
from scipy.stats import wilcoxon

# gphh_solver.GPHH.init_model() does a bare `import multitreegp`, which
# requires yuantian/ itself (not this experiments/ subfolder) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.gp_algorithms import standard_gp
from yuantian.gphh_solver import GPHH, ParametersGPHH, RefreshHallOfFame, read_instances
from yuantian.hybrid_gp import epsilon_lexicase_selection, lexicase_memetic_gp
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum
from yuantian.utils import PopulationArchive

# ---------------------------------------------------------------------------
# Experiment configuration (shared by both conditions; only selection + local
# search differ)
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_BASE = 2000
POP_SIZE = 60
N_GEN = 25
N_ELITE = 3
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE
SIMULATOR_TYPE = SimulatorTypeEnum.SERIAL_SGS

# local-search-specific settings (Part 2)
ELITE_FRACTION = 0.08  # top 5-10% of the population, per spec
LOCAL_SEARCH_ITERS = 10

TRAIN_CLASSES = [1, 20, 40, 60, 80]
TRAIN_FILES = [f"J50{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"J50{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "lexicase_local_search_experiment"


def build_params() -> ParametersGPHH:
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        cp_propagation_feature=False,  # unchanged GP representation/terminal set
    )
    params.pop_size = POP_SIZE
    params.n_gen = N_GEN
    params.n_elite = N_ELITE
    params.tournament_size = TOURNAMENT_SIZE
    params.crossover_rate = CROSSOVER_RATE
    params.mutation_rate = MUTATION_RATE
    params.deap_verbose = False
    return params


def _make_mstats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def run_single(seed: int, condition: str, training, test) -> dict:
    assert condition in ("baseline", "proposed")
    random.seed(seed)
    np.random.seed(seed)
    params = build_params()
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()

    mstats = _make_mstats()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = RefreshHallOfFame(1)
    pop_archive = PopulationArchive()

    if condition == "proposed":
        solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)
        _, log = lexicase_memetic_gp(
            pop,
            solver.toolbox,
            cxpb=params.crossover_rate,
            mutpb=params.mutation_rate,
            n_elite=params.n_elite,
            ngen=params.n_gen,
            training_data_provider=StaticDatasetProvider(training),
            validation_data_provider=None,
            decision_type=params.decision_type,
            simulator=solver.simulator,
            pset=solver.pset,
            elite_fraction=ELITE_FRACTION,
            local_search_iters=LOCAL_SEARCH_ITERS,
            stats=mstats,
            halloffame=hof,
            pop_archive=pop_archive,
            rng=random,
            verbose=False,
        )
    else:
        _, log = standard_gp(
            pop,
            solver.toolbox,
            cxpb=params.crossover_rate,
            mutpb=params.mutation_rate,
            n_elite=params.n_elite,
            ngen=params.n_gen,
            training_data_provider=StaticDatasetProvider(training),
            validation_data_provider=None,
            stats=mstats,
            halloffame=hof,
            pop_archive=pop_archive,
            verbose=False,
        )

    test_fitness = solver.toolbox.evaluate(individual=hof[0], domains=test)[0]
    gen_best = log.chapters["generation_best"]
    fitness_curve = [g["fitness"] for g in gen_best]
    best_fitness = fitness_curve[-1]
    convergence_gen = next(
        i for i, f in enumerate(fitness_curve) if f <= best_fitness + 1e-9
    )
    final_pop_stats = log.chapters["fitness"][-1]
    fitness_std_per_gen = [rec["std"] for rec in log.chapters["fitness"]]
    unique_trees_per_gen = [
        len(set(str(ind) for ind in gen_pop)) for gen_pop in pop_archive
    ]

    return {
        "seed": seed,
        "condition": condition,
        "best_fitness": best_fitness,
        "mean_final_pop": float(final_pop_stats["avg"]),
        "std_final_pop": float(final_pop_stats["std"]),
        "convergence_gen": convergence_gen,
        "test_fitness": test_fitness,
        "avg_fitness_std_over_run": float(np.mean(fitness_std_per_gen)),
        "avg_unique_trees_over_run": float(np.mean(unique_trees_per_gen)),
        "best_tree": gen_best[-1]["tree"],
    }


def rank_biserial_effect_size(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    return float((r_plus - r_minus) / ranks.sum())


def main():
    warnings.filterwarnings("ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TRAIN_FILES])
    test = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES])
    print(f"Training instances: {TRAIN_FILES}")
    print(f"Test instances: {TEST_FILES}")
    print(
        f"pop_size={POP_SIZE}, n_gen={N_GEN}, n_elite={N_ELITE}, "
        f"cx={CROSSOVER_RATE}, mut={MUTATION_RATE}, "
        f"elite_fraction={ELITE_FRACTION}, local_search_iters={LOCAL_SEARCH_ITERS}"
    )

    all_results = []
    for condition in ("baseline", "proposed"):
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            t0 = time.time()
            record = run_single(seed, condition, training, test)
            all_results.append(record)
            print(
                f"[{condition}] seed={seed} best={record['best_fitness']:.4f} "
                f"test={record['test_fitness']:.4f} conv_gen={record['convergence_gen']} "
                f"uniq_trees={record['avg_unique_trees_over_run']:.1f} "
                f"({time.time() - t0:.1f}s)"
            )

    with open(OUTPUT_DIR / "all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    analyze(all_results)


def analyze(all_results: list):
    baseline = sorted(
        [r for r in all_results if r["condition"] == "baseline"], key=lambda r: r["seed"]
    )
    proposed = sorted(
        [r for r in all_results if r["condition"] == "proposed"], key=lambda r: r["seed"]
    )
    assert [r["seed"] - SEED_BASE for r in baseline] == [
        r["seed"] - SEED_BASE for r in proposed
    ], "Seeds must match pairwise between conditions for a paired test"

    print("\n" + "=" * 78)
    print("RESULTS TABLE (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 78)
    print(f"{'Method':<12}{'Mean':>10}{'Best':>10}{'Std':>10}{'ConvGen':>10}{'TestMean':>12}")
    for label, records in [("Baseline", baseline), ("Proposed", proposed)]:
        best_vals = np.array([r["best_fitness"] for r in records], dtype=float)
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        test_vals = np.array([r["test_fitness"] for r in records], dtype=float)
        print(
            f"{label:<12}{best_vals.mean():>10.4f}{best_vals.min():>10.4f}"
            f"{best_vals.std():>10.4f}{conv_vals.mean():>10.2f}{test_vals.mean():>12.4f}"
        )

    best_baseline = np.array([r["best_fitness"] for r in baseline], dtype=float)
    best_proposed = np.array([r["best_fitness"] for r in proposed], dtype=float)
    diffs = best_baseline - best_proposed  # positive => proposed better

    print("\n" + "=" * 78)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test on best_fitness)")
    print("=" * 78)
    if np.all(diffs == 0):
        print("All paired differences are zero; Wilcoxon test is not informative.")
    else:
        stat, p_value = wilcoxon(best_baseline, best_proposed)
        effect_size = rank_biserial_effect_size(diffs)
        print(f"W statistic = {stat:.4f}")
        print(f"p-value     = {p_value:.6f}")
        print(f"effect size (matched-pairs rank-biserial r) = {effect_size:.4f}")
        direction = "Proposed better" if diffs.mean() > 0 else "Baseline better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(f"Mean(best_baseline - best_proposed) = {diffs.mean():.4f} ({direction})")
        print(f"=> {significance}")

    print("\n" + "=" * 78)
    print("DIAGNOSTICS: diversity and convergence character")
    print("=" * 78)
    for label, records in [("Baseline", baseline), ("Proposed", proposed)]:
        fit_std = np.array([r["avg_fitness_std_over_run"] for r in records], dtype=float)
        uniq = np.array([r["avg_unique_trees_over_run"] for r in records], dtype=float)
        print(
            f"{label:<12} avg population fitness std={fit_std.mean():.4f}, "
            f"avg unique trees/gen={uniq.mean():.1f} (out of {POP_SIZE})"
        )


if __name__ == "__main__":
    main()
