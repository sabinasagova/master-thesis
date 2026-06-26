"""
Experiment: isolating the contribution of the CPM-aware resource_shift move
within the elite local-search step, against a one-shot CPM construction and
against no refinement at all.

Four conditions, all under epsilon-lexicase selection (held fixed across all
four so only the elite-refinement strategy varies -- selection is not the
thing being measured here; see lexicase_local_search_experiment.py and
full_mmlib_experiment.py for that comparison):
  - baseline: no refinement applied to elites (RefinementStrategyEnum.BASELINE)
  - local_search_no_cp: hill-climbing with mode + swap moves only
  - critical_path_only: one-shot construction from CPM slack, no search loop
  - local_search_with_cp: hill-climbing with mode + swap + resource_shift
    (the original behavior, unchanged)

All four share population size, generations, crossover/mutation rates,
elitism count, the elite fraction eligible for refinement, and (for the two
hill-climbing conditions) the same max_iters budget, so the comparison
isolates "does the CPM-aware move help" rather than "did one condition get
more search budget."

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 -O yuantian/experiments/local_search_variants_experiment.py
"""
import json
import random
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from deap import tools
from scipy.stats import wilcoxon

# gphh_solver.GPHH.init_model() does a bare `import multitreegp`, which
# requires yuantian/ itself (not this experiments/ subfolder) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.gphh_solver import GPHH, ParametersGPHH, RefreshHallOfFame, read_instances
from yuantian.hybrid_gp import epsilon_lexicase_selection, lexicase_memetic_gp
from yuantian.local_search import RefinementStrategyEnum
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum
from yuantian.utils import PopulationArchive

# ---------------------------------------------------------------------------
# Experiment configuration (matches lexicase_local_search_experiment.py /
# full_mmlib_experiment.py's scale so results sit alongside them)
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_BASE = 3000
POP_SIZE = 60
N_GEN = 25
N_ELITE = 3
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE
SIMULATOR_TYPE = SimulatorTypeEnum.SERIAL_SGS

ELITE_FRACTION = 0.08
LOCAL_SEARCH_ITERS = 10  # same max_iters budget for both hill-climbing conditions

CONDITIONS = [
    RefinementStrategyEnum.BASELINE,
    RefinementStrategyEnum.LOCAL_SEARCH_NO_CP,
    RefinementStrategyEnum.CRITICAL_PATH_ONLY,
    RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP,
]
CONDITION_LABELS = {
    RefinementStrategyEnum.BASELINE: "baseline",
    RefinementStrategyEnum.LOCAL_SEARCH_NO_CP: "local_search_no_cp",
    RefinementStrategyEnum.CRITICAL_PATH_ONLY: "critical_path_only",
    RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP: "local_search_with_cp",
}

TRAIN_CLASSES = [1, 20, 40, 60, 80]
TRAIN_FILES = [f"J50{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"J50{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "local_search_variants_experiment"


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


def run_single(seed: int, condition: RefinementStrategyEnum, training, test) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    params = build_params()
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    # Selection held fixed at epsilon-lexicase across all four conditions:
    # only the elite-refinement strategy should vary.
    solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)

    mstats = _make_mstats()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = RefreshHallOfFame(1)
    pop_archive = PopulationArchive()
    move_stats = {"attempted": 0, "accepted": 0, "construct_failures": 0}

    t0 = time.time()
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
        refinement_strategy=condition,
        stats=mstats,
        halloffame=hof,
        pop_archive=pop_archive,
        move_stats=move_stats,
        rng=random,
        verbose=False,
    )
    elapsed = time.time() - t0

    test_fitness = solver.toolbox.evaluate(individual=hof[0], domains=test)[0]
    gen_best = log.chapters["generation_best"]
    fitness_curve = [g["fitness"] for g in gen_best]
    best_fitness = fitness_curve[-1]
    convergence_gen = next(
        i for i, f in enumerate(fitness_curve) if f <= best_fitness + 1e-9
    )
    final_pop_stats = log.chapters["fitness"][-1]

    return {
        "seed": seed,
        "condition": CONDITION_LABELS[condition],
        "best_fitness": best_fitness,
        "mean_final_pop": float(final_pop_stats["avg"]),
        "std_final_pop": float(final_pop_stats["std"]),
        "convergence_gen": convergence_gen,
        "test_fitness": test_fitness,
        "elapsed_sec": elapsed,
        "moves_attempted": move_stats["attempted"],
        "moves_accepted": move_stats["accepted"],
        "construct_failures": move_stats["construct_failures"],
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
        f"pop_size={POP_SIZE}, n_gen={N_GEN}, n_elite={N_ELITE}, cx={CROSSOVER_RATE}, "
        f"mut={MUTATION_RATE}, elite_fraction={ELITE_FRACTION}, "
        f"local_search_iters={LOCAL_SEARCH_ITERS}"
    )

    all_results = []
    for condition in CONDITIONS:
        label = CONDITION_LABELS[condition]
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            t0 = time.time()
            record = run_single(seed, condition, training, test)
            all_results.append(record)
            print(
                f"[{label}] seed={seed} best={record['best_fitness']:.4f} "
                f"test={record['test_fitness']:.4f} conv_gen={record['convergence_gen']} "
                f"moves={record['moves_accepted']}/{record['moves_attempted']} "
                f"construct_failures={record['construct_failures']} "
                f"({time.time() - t0:.1f}s)"
            )

    with open(OUTPUT_DIR / "all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    analyze(all_results)


def analyze(all_results: list):
    labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    by_condition = {
        label: sorted([r for r in all_results if r["condition"] == label], key=lambda r: r["seed"])
        for label in labels
    }
    seeds_per_condition = {label: [r["seed"] for r in recs] for label, recs in by_condition.items()}
    first = next(iter(seeds_per_condition.values()))
    assert all(seeds == first for seeds in seeds_per_condition.values()), (
        "Seeds must match pairwise across conditions for a paired test"
    )

    print("\n" + "=" * 86)
    print("RESULTS TABLE (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 86)
    print(
        f"{'Method':<22}{'Mean':>9}{'Best':>9}{'Std':>9}{'ConvGen':>9}"
        f"{'TestMean':>11}{'Sec/run':>9}"
    )
    for label in labels:
        records = by_condition[label]
        best_vals = np.array([r["best_fitness"] for r in records], dtype=float)
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        test_vals = np.array([r["test_fitness"] for r in records], dtype=float)
        sec_vals = np.array([r["elapsed_sec"] for r in records], dtype=float)
        print(
            f"{label:<22}{best_vals.mean():>9.4f}{best_vals.min():>9.4f}{best_vals.std():>9.4f}"
            f"{conv_vals.mean():>9.2f}{test_vals.mean():>11.4f}{sec_vals.mean():>9.1f}"
        )

    print("\n" + "=" * 86)
    print("MOVE DIAGNOSTICS (summed across the whole run: all elites, all generations)")
    print("=" * 86)
    print(f"{'Method':<22}{'AcceptRate':>12}{'MeanAccepted':>14}{'ConstructFails':>16}")
    for label in labels:
        records = by_condition[label]
        attempted = np.array([r["moves_attempted"] for r in records], dtype=float)
        accepted = np.array([r["moves_accepted"] for r in records], dtype=float)
        fails = np.array([r["construct_failures"] for r in records], dtype=float)
        accept_rate = accepted.sum() / attempted.sum() if attempted.sum() > 0 else float("nan")
        print(
            f"{label:<22}{accept_rate:>12.3f}{accepted.mean():>14.1f}{fails.mean():>16.2f}"
        )

    print("\n" + "=" * 86)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test on best_fitness)")
    print("=" * 86)
    for label_a, label_b in combinations(labels, 2):
        a = np.array([r["best_fitness"] for r in by_condition[label_a]], dtype=float)
        b = np.array([r["best_fitness"] for r in by_condition[label_b]], dtype=float)
        diffs = a - b  # positive => label_b better (lower fitness)
        print(f"\n{label_a} vs {label_b}:")
        if np.all(diffs == 0):
            print("  All paired differences are zero; Wilcoxon test is not informative.")
            continue
        stat, p_value = wilcoxon(a, b)
        effect_size = rank_biserial_effect_size(diffs)
        direction = f"{label_b} better" if diffs.mean() > 0 else f"{label_a} better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(
            f"  W={stat:.4f}  p={p_value:.6f}  r={effect_size:.4f}  "
            f"mean({label_a}-{label_b})={diffs.mean():.4f} ({direction})  => {significance}"
        )


if __name__ == "__main__":
    main()
