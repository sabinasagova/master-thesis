"""
Experiment: exploratory strategy sweep vs baseline GPHH.

Runs the published GPHH baseline (gp_algorithms.standard_gp) plus all nine
restored exploratory strategies (yuantian/exploratory/) under identical GP
settings (population size, generations, crossover/mutation rates,
elitism), and one additional combo condition (lexicase + heuristic seeding,
matching how the original sweep tested the lexicase driver together with
seeding -- see yuantian/exploratory/heuristic_seeding.py).

Terminal-set note: baseline and the strategies that only change
selection/variation/evaluation (adaptive, surrogate, diverse, lexicase,
lexicase_seeded, multi_sgs, decision_trace) all run on the plain baseline
terminal set, so the EA mechanism is the only thing varying for those.
mod_integrated and trace_directed additionally get
yuantian.exploratory.diagnostic_graft.install_graft_terminals applied (their
graft is a no-op otherwise -- see exploratory/README.md). map_elites
additionally runs with --cp_propagation/--nr_terminals-equivalent flags on
(its CP/NR behaviour descriptor is degenerate otherwise). This matches how
each strategy needed its own flags in the original sweep, not a new
confound introduced by this script.

Settings below are intentionally reduced from the paper's defaults
(pop=1000, gen=50) to keep the full 10-seed x 11-condition comparison
runnable on a laptop, matching cp_propagation_experiment.py's scale.

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 -O \\
        yuantian/experiments/exploratory_sweep_experiment.py
"""
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from deap import creator, tools
from scipy.stats import wilcoxon

# gphh_solver.GPHH.init_model() does a bare `import multitreegp`, which
# requires yuantian/ itself (not this experiments/ subfolder) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.exploratory import EXPLORATORY_REGISTRY, GRAFT_DEPENDENT_STRATEGIES
from yuantian.exploratory.diagnostic_graft import install_graft_terminals
from yuantian.exploratory.heuristic_seeding import seed_then_run
from yuantian.exploratory.selection import lexicase_gp
from yuantian.gp_algorithms import standard_gp
from yuantian.gphh_solver import GPHH, ParametersGPHH, read_instances
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum

# ---------------------------------------------------------------------------
# Experiment configuration (reduced scale, fair across all conditions)
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_BASE = 5000
POP_SIZE = 60
N_GEN = 25
N_ELITE = 3
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE
SIMULATOR_TYPE = SimulatorTypeEnum.SERIAL_SGS

TRAIN_CLASSES = [1, 20, 40, 60, 80]
TRAIN_FILES = [f"J50{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"J50{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "exploratory_sweep_experiment"

# Conditions: "baseline" (standard_gp) + the 9 EXPLORATORY_REGISTRY strategies
# + "lexicase_seeded" (lexicase_gp combined with heuristic seeding).
CONDITIONS = ["baseline"] + list(EXPLORATORY_REGISTRY.keys()) + ["lexicase_seeded"]

# Strategies that need the richer terminal set (--cp_propagation +
# --nr_terminals equivalent) to be non-degenerate. See module docstring.
RICH_TERMINAL_STRATEGIES = GRAFT_DEPENDENT_STRATEGIES | {"map_elites"}


def build_params(condition: str) -> ParametersGPHH:
    rich = condition in RICH_TERMINAL_STRATEGIES
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        cp_propagation_feature=rich,
        nr_terminals_feature=rich,
    )
    params.pop_size = POP_SIZE
    params.n_gen = N_GEN
    params.n_elite = N_ELITE
    params.tournament_size = TOURNAMENT_SIZE
    params.crossover_rate = CROSSOVER_RATE
    params.mutation_rate = MUTATION_RATE
    params.deap_verbose = False
    return params


def run_single(seed: int, condition: str, training, test) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    needs_nr_instances = condition in RICH_TERMINAL_STRATEGIES
    training_set = read_instances(
        [str(f) for f in training], keep_non_renewable=needs_nr_instances
    )
    params = build_params(condition)
    training_provider = StaticDatasetProvider(training_set)
    solver = GPHH(training_set_provider=training_provider, params_gphh=params)
    solver.init_model()

    if condition in GRAFT_DEPENDENT_STRATEGIES:
        install_graft_terminals(solver.pset, solver.simulator)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    hof = tools.HallOfFame(1)
    pop_archive: list = []

    common_kwargs = dict(
        cxpb=CROSSOVER_RATE,
        mutpb=MUTATION_RATE,
        n_elite=N_ELITE,
        ngen=N_GEN,
        training_data_provider=training_provider,
        validation_data_provider=StaticDatasetProvider([]),
        stats=mstats,
        halloffame=hof,
        pop_archive=pop_archive,
        verbose=False,
    )

    t0 = time.time()
    if condition == "baseline":
        pop = solver.toolbox.population(n=POP_SIZE)
        _, logbook = standard_gp(pop, solver.toolbox, **common_kwargs)
    elif condition == "lexicase_seeded":
        _, logbook = seed_then_run(
            lexicase_gp,
            solver.toolbox,
            individual_class=creator.Individual,
            pop_size=POP_SIZE,
            decision_type=solver.decision_type,
            pset=solver.pset,
            mutate=solver.toolbox.mutate,
            **common_kwargs,
        )
    else:
        pop = solver.toolbox.population(n=POP_SIZE)
        driver = EXPLORATORY_REGISTRY[condition]
        _, logbook = driver(pop, solver.toolbox, **common_kwargs)
    elapsed = time.time() - t0

    best = hof[0]
    gen_best_log = logbook.chapters["generation_best"]
    fitness_curve = [g["fitness"] for g in gen_best_log]
    best_fitness = fitness_curve[-1]
    convergence_gen = next(i for i, f in enumerate(fitness_curve) if f <= best_fitness + 1e-9)
    final_pop_stats = logbook.chapters["fitness"][-1]
    test_fitness = solver.toolbox.evaluate(individual=best, domains=test)[0]

    return {
        "seed": seed,
        "condition": condition,
        "best_fitness": best_fitness,
        "mean_final_pop": float(final_pop_stats["avg"]),
        "std_final_pop": float(final_pop_stats["std"]),
        "convergence_gen": convergence_gen,
        "test_fitness": test_fitness,
        "elapsed_sec": elapsed,
        "best_tree": str(best),
        "feasible": getattr(best, "infeas_frac", 0.0) == 0.0,
    }


def rank_biserial_effect_size(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test."""
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

    training = [RCPSPDatabase.MMLIB_50_DIR + f for f in TRAIN_FILES]
    test_renewable_only = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES])
    print(f"Training instances: {TRAIN_FILES}")
    print(f"Test instances: {TEST_FILES}")
    print(
        f"pop_size={POP_SIZE}, n_gen={N_GEN}, n_elite={N_ELITE}, "
        f"tournament={TOURNAMENT_SIZE}, cx={CROSSOVER_RATE}, mut={MUTATION_RATE}"
    )
    print(f"Conditions: {CONDITIONS}")

    all_results = []
    for condition in CONDITIONS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            t0 = time.time()
            record = run_single(seed, condition, training, test_renewable_only)
            all_results.append(record)
            print(
                f"[{condition}] seed={seed} best={record['best_fitness']:.4f} "
                f"test={record['test_fitness']:.4f} conv_gen={record['convergence_gen']} "
                f"feasible={record['feasible']} ({time.time() - t0:.1f}s)"
            )

    with open(OUTPUT_DIR / "all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    analyze(all_results)


def analyze(all_results: list):
    by_condition = {
        c: sorted([r for r in all_results if r["condition"] == c], key=lambda r: r["seed"])
        for c in CONDITIONS
    }
    baseline = by_condition["baseline"]

    print("\n" + "=" * 100)
    print("RESULTS TABLE (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 100)
    print(
        f"{'Strategy':<18}{'Mean':>10}{'Best':>10}{'Std':>10}{'Feasible%':>11}"
        f"{'AvgConvGen':>12}{'p vs baseline':>15}"
    )
    best_baseline = np.array([r["best_fitness"] for r in baseline], dtype=float)
    for condition in CONDITIONS:
        records = by_condition[condition]
        best_vals = np.array([r["best_fitness"] for r in records], dtype=float)
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        feas_pct = 100.0 * sum(r["feasible"] for r in records) / len(records)
        if condition == "baseline" or np.all(best_vals == best_baseline):
            p_str = "--"
        else:
            try:
                _, p = wilcoxon(best_baseline, best_vals)
                p_str = f"{p:.4f}"
            except ValueError:
                p_str = "n/a"
        print(
            f"{condition:<18}{best_vals.mean():>10.4f}{best_vals.min():>10.4f}"
            f"{best_vals.std():>10.4f}{feas_pct:>10.0f}%{conv_vals.mean():>12.2f}{p_str:>15}"
        )

    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test on best_fitness, vs baseline)")
    print("=" * 100)
    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        records = by_condition[condition]
        vals = np.array([r["best_fitness"] for r in records], dtype=float)
        diffs = best_baseline - vals  # positive => condition better (lower fitness)
        if np.all(diffs == 0):
            print(f"{condition}: all paired differences are zero; Wilcoxon test is not informative.")
            continue
        stat, p_value = wilcoxon(best_baseline, vals)
        effect_size = rank_biserial_effect_size(diffs)
        direction = f"{condition} better" if diffs.mean() > 0 else "baseline better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(
            f"{condition}: W={stat:.4f} p={p_value:.6f} r={effect_size:.4f} "
            f"mean(baseline-{condition})={diffs.mean():.4f} ({direction}) => {significance}"
        )


if __name__ == "__main__":
    main()
