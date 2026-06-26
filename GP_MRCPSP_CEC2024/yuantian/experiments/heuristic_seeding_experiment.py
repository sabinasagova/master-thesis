"""
Experiment: heuristic-seeded GP initial population vs the baseline random
(ramped half-and-half) initial population.

Compares three configurations under identical GP settings (population size,
generations, crossover/mutation rates, tournament size, elitism, terminal
set):
  - baseline: gen-0 built purely at random (`seeding_strategy="random"`)
  - heuristic: 8 textbook priority-rule trees (4 activity rules x 2 mode
    rules: EST/EFT/LFT/MSLK x SPT_MODE/MIN_RES_MODE) inserted once each into
    gen-0, the rest filled at random (`seeding_strategy="heuristic"`)
  - heuristic_mutated: same 8 rules, each contributing itself plus 2
    mutated clones, the rest filled at random
    (`seeding_strategy="heuristic_mutated"`)

Settings mirror cp_propagation_experiment.py exactly (same reduced scale,
same instances, same seeds) so results sit alongside that experiment.

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 -O yuantian/experiments/heuristic_seeding_experiment.py
"""
import json
import random
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# gphh_solver.GPHH.init_model() does a bare `import multitreegp`, which
# requires yuantian/ itself (not this experiments/ subfolder) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.gphh_solver import GPHH, ParametersGPHH, read_instances
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum

# ---------------------------------------------------------------------------
# Experiment configuration (matches cp_propagation_experiment.py exactly)
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_BASE = 1000
POP_SIZE = 60
N_GEN = 25
N_ELITE = 3
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE
SIMULATOR_TYPE = SimulatorTypeEnum.SERIAL_SGS
N_MUTATED_CLONES = 2

CONDITIONS = ["baseline", "heuristic", "heuristic_mutated"]
# Display label -> actual ParametersGPHH.seeding_strategy value. "baseline"
# is a label for this experiment, not a seeding_strategy GPHH.solve()
# understands: ParametersGPHH/GPHH.solve() only special-case the literal
# string "random" for "no seeding" (anything else takes the heuristic-seed
# branch), so "baseline" must map to "random" here or it would silently run
# the same heuristic-seeded code path as the "heuristic" condition.
SEEDING_STRATEGY_BY_CONDITION = {
    "baseline": "random",
    "heuristic": "heuristic",
    "heuristic_mutated": "heuristic_mutated",
}

# Same MMLIB50 classes/files as cp_propagation_experiment.py, for direct
# comparability.
TRAIN_CLASSES = [1, 20, 40, 60, 80]
TRAIN_FILES = [f"J50{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"J50{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "heuristic_seeding_experiment"


def build_params(condition: str) -> ParametersGPHH:
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
    )
    params.pop_size = POP_SIZE
    params.n_gen = N_GEN
    params.n_elite = N_ELITE
    params.tournament_size = TOURNAMENT_SIZE
    params.crossover_rate = CROSSOVER_RATE
    params.mutation_rate = MUTATION_RATE
    params.deap_verbose = False
    params.seeding_strategy = SEEDING_STRATEGY_BY_CONDITION[condition]
    params.n_mutated_clones = N_MUTATED_CLONES
    return params


def run_single(seed: int, condition: str, training, test, output_path: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    params = build_params(condition)
    solver = GPHH(
        training_set_provider=StaticDatasetProvider(training),
        params_gphh=params,
    )
    solver.init_model()
    solver.solve(output_path=str(output_path))
    with open(output_path) as f:
        result = json.load(f)

    gen_best = result["generation_best"]
    fitness_curve = [g["fitness"] for g in gen_best]
    best_fitness = fitness_curve[-1]
    convergence_gen = next(
        i for i, f in enumerate(fitness_curve) if f <= best_fitness + 1e-9
    )
    final_pop_stats = result["fitness"][-1]
    # held-out generalization check: evaluate final best individual on unseen instances
    test_fitness = solver.toolbox.evaluate(individual=solver.best_heuristic, domains=test)[0]
    return {
        "seed": seed,
        "condition": condition,
        "best_fitness": best_fitness,
        "mean_final_pop": float(final_pop_stats["avg"]),
        "std_final_pop": float(final_pop_stats["std"]),
        "convergence_gen": convergence_gen,
        "test_fitness": test_fitness,
        "elapsed_sec": result["elapsed"],
        "best_tree": gen_best[-1]["tree"],
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

    training = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TRAIN_FILES])
    test = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES])
    print(f"Training instances: {TRAIN_FILES}")
    print(f"Test instances: {TEST_FILES}")
    print(f"pop_size={POP_SIZE}, n_gen={N_GEN}, n_elite={N_ELITE}, "
          f"tournament={TOURNAMENT_SIZE}, cx={CROSSOVER_RATE}, mut={MUTATION_RATE}, "
          f"n_mutated_clones={N_MUTATED_CLONES}")

    all_results = []
    for condition in CONDITIONS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            t0 = time.time()
            out_path = OUTPUT_DIR / f"{condition}_seed{seed}.json"
            record = run_single(seed, condition, training, test, out_path)
            all_results.append(record)
            print(
                f"[{condition}] seed={seed} best={record['best_fitness']:.4f} "
                f"test={record['test_fitness']:.4f} conv_gen={record['convergence_gen']} "
                f"({time.time() - t0:.1f}s)"
            )

    summary_path = OUTPUT_DIR / "all_runs.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    analyze(all_results)


def analyze(all_results: list):
    by_condition = {
        condition: sorted(
            [r for r in all_results if r["condition"] == condition], key=lambda r: r["seed"]
        )
        for condition in CONDITIONS
    }
    seeds_per_condition = {c: [r["seed"] for r in recs] for c, recs in by_condition.items()}
    first = next(iter(seeds_per_condition.values()))
    assert all(seeds == first for seeds in seeds_per_condition.values()), (
        "Seeds must match pairwise across conditions for a paired test"
    )

    def stats_block(records, key):
        vals = np.array([r[key] for r in records], dtype=float)
        return vals, vals.mean(), vals.min(), vals.std()

    print("\n" + "=" * 78)
    print("RESULTS TABLE (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 78)
    print(f"{'Method':<20}{'Mean':>9}{'Best':>9}{'Std':>9}{'AvgConvGen':>12}{'TestMean':>11}")
    for label in CONDITIONS:
        records = by_condition[label]
        best_vals, mean_best, min_best, std_best = stats_block(records, "best_fitness")
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        test_vals = np.array([r["test_fitness"] for r in records], dtype=float)
        print(
            f"{label:<20}{mean_best:>9.4f}{min_best:>9.4f}{std_best:>9.4f}"
            f"{conv_vals.mean():>12.2f}{test_vals.mean():>11.4f}"
        )

    print("\n" + "=" * 78)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test on best_fitness)")
    print("=" * 78)
    for cond_a, cond_b in combinations(CONDITIONS, 2):
        a = np.array([r["best_fitness"] for r in by_condition[cond_a]], dtype=float)
        b = np.array([r["best_fitness"] for r in by_condition[cond_b]], dtype=float)
        diffs = a - b  # positive => cond_b better (lower fitness)
        print(f"\n{cond_a} vs {cond_b}:")
        if np.all(diffs == 0):
            print("  All paired differences are zero; Wilcoxon test is not informative.")
            continue
        stat, p_value = wilcoxon(a, b)
        effect_size = rank_biserial_effect_size(diffs)
        direction = f"{cond_b} better" if diffs.mean() > 0 else f"{cond_a} better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(f"  W={stat:.4f}  p={p_value:.6f}  r={effect_size:.4f}  "
              f"mean({cond_a}-{cond_b})={diffs.mean():.4f} ({direction})  => {significance}")


if __name__ == "__main__":
    main()
