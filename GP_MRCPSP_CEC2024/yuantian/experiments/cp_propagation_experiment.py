"""
Experiment: critical path propagation terminals vs baseline GPHH terminal set.

Compares two configurations under identical GP settings (population size,
generations, crossover/mutation rates, tournament size, elitism):
  - baseline: original GPHH terminal set
  - cp_propagation: baseline terminal set + CP_FORWARD, CP_BACKWARD,
    CP_SLACK_SCORE, CP_PROB

Settings below are intentionally reduced from the paper's defaults
(pop=1000, gen=50) to keep the full 10-seed x 2-condition comparison
runnable on a laptop. Only the terminal set differs between conditions.

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python yuantian/experiments/cp_propagation_experiment.py
"""
import json
import random
import sys
import time
import warnings
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
# Experiment configuration (reduced scale, fair across both conditions)
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

# A handful of MMLIB50 classes for training (used to evaluate fitness during
# evolution) and a held-out case per class for an out-of-sample test score.
TRAIN_CLASSES = [1, 20, 40, 60, 80]
TRAIN_FILES = [f"J50{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"J50{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "cp_propagation_experiment"


def build_params(cp_propagation: bool) -> ParametersGPHH:
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        cp_propagation_feature=cp_propagation,
    )
    params.pop_size = POP_SIZE
    params.n_gen = N_GEN
    params.n_elite = N_ELITE
    params.tournament_size = TOURNAMENT_SIZE
    params.crossover_rate = CROSSOVER_RATE
    params.mutation_rate = MUTATION_RATE
    params.deap_verbose = False
    return params


def run_single(seed: int, cp_propagation: bool, training, test, output_path: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    params = build_params(cp_propagation)
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
        "condition": "cp_propagation" if cp_propagation else "baseline",
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
          f"tournament={TOURNAMENT_SIZE}, cx={CROSSOVER_RATE}, mut={MUTATION_RATE}")

    all_results = []
    for cp_propagation in (False, True):
        condition = "cp_propagation" if cp_propagation else "baseline"
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            t0 = time.time()
            out_path = OUTPUT_DIR / f"{condition}_seed{seed}.json"
            record = run_single(seed, cp_propagation, training, test, out_path)
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
    baseline = sorted(
        [r for r in all_results if r["condition"] == "baseline"], key=lambda r: r["seed"]
    )
    modified = sorted(
        [r for r in all_results if r["condition"] == "cp_propagation"], key=lambda r: r["seed"]
    )
    assert [r["seed"] for r in baseline] == [r["seed"] for r in modified], (
        "Seeds must match pairwise between conditions for a paired test"
    )

    def stats_block(records, key):
        vals = np.array([r[key] for r in records], dtype=float)
        return vals, vals.mean(), vals.min(), vals.std()

    print("\n" + "=" * 70)
    print("RESULTS TABLE (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 70)
    print(f"{'Method':<16}{'Mean':>10}{'Best':>10}{'Std':>10}{'AvgConvGen':>12}")
    for label, records in [("Baseline", baseline), ("CP Propagation", modified)]:
        best_vals, mean_best, min_best, std_best = stats_block(records, "best_fitness")
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        print(
            f"{label:<16}{mean_best:>10.4f}{min_best:>10.4f}{std_best:>10.4f}"
            f"{conv_vals.mean():>12.2f}"
        )

    test_baseline = np.array([r["test_fitness"] for r in baseline], dtype=float)
    test_modified = np.array([r["test_fitness"] for r in modified], dtype=float)
    print(f"\nHeld-out test fitness: baseline mean={test_baseline.mean():.4f}, "
          f"cp_propagation mean={test_modified.mean():.4f}")

    best_baseline = np.array([r["best_fitness"] for r in baseline], dtype=float)
    best_modified = np.array([r["best_fitness"] for r in modified], dtype=float)
    diffs = best_baseline - best_modified  # positive => cp_propagation better (lower fitness)

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test on best_fitness)")
    print("=" * 70)
    if np.all(diffs == 0):
        print("All paired differences are zero; Wilcoxon test is not informative.")
    else:
        stat, p_value = wilcoxon(best_baseline, best_modified)
        effect_size = rank_biserial_effect_size(diffs)
        print(f"W statistic = {stat:.4f}")
        print(f"p-value     = {p_value:.6f}")
        print(f"effect size (matched-pairs rank-biserial r) = {effect_size:.4f}")
        direction = "CP propagation better" if diffs.mean() > 0 else "Baseline better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(f"Mean(best_baseline - best_cp_propagation) = {diffs.mean():.4f} ({direction})")
        print(f"=> {significance}")


if __name__ == "__main__":
    main()
