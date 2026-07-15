"""
Mode-interaction terminals vs baseline, with RCCP as the comparison that
matters most here.

mode_interaction_terminals.py is asking a different question than the
other resource-aware extensions: not "how much is left" (nr_terminals.py)
or "who's contending for the bottleneck right now" (rccp_terminals.py),
but a consequence signal: how much a mode choice squeezes the other activities'
mode options later. It is also the least literature-established extension (see
that module's docstring), so this is the first real evidence either way.

Four conditions:
  - baseline: normal terminal set
  - mode_interaction: + MI_CONSTRAINT_TIGHTENING, MI_RECIPROCAL_SCARCITY,
    MI_ACTIVITY_PRESSURE
  - rccp: + RCCP_BOTTLENECK_UTIL, RCCP_CANDIDATE_CONTENTION, RCCP_SLACK,
    RCCP_PRESSURE_TREND (the most useful side-by-side, since both are
    resource-aware but looking at different things)
  - both: both terminal sets together

Scaled down from the paper defaults, same scale as
rccp_terminals_experiment.py so the tables line up. mode_interaction's own
module docstring measured roughly a 3x wall-clock slowdown for that
terminal set, so expect mode_interaction/both to run noticeably slower
than baseline/rccp here.

Run with (from the repo root):
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/experiments/mode_interaction_experiment.py
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
# Experiment configuration (reduced scale, fair across all conditions, same
# as cp_propagation_experiment.py / rccp_terminals_experiment.py)
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_BASE = 9000
POP_SIZE = 60
N_GEN = 25
N_ELITE = 3
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE
SIMULATOR_TYPE = SimulatorTypeEnum.SERIAL_SGS

CONDITIONS = ("baseline", "mode_interaction", "rccp", "both")

# Same train/test instances as cp_propagation_experiment.py /
# rccp_terminals_experiment.py, so this table lines up with both.
TRAIN_CLASSES = [1, 20, 40, 60, 80]
TRAIN_FILES = [f"J50{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"J50{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "mode_interaction_experiment"


def build_params(condition: str) -> ParametersGPHH:
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        rccp_terminals_feature=condition in ("rccp", "both"),
        mode_interaction_terminals_feature=condition in ("mode_interaction", "both"),
    )
    params.pop_size = POP_SIZE
    params.n_gen = N_GEN
    params.n_elite = N_ELITE
    params.tournament_size = TOURNAMENT_SIZE
    params.crossover_rate = CROSSOVER_RATE
    params.mutation_rate = MUTATION_RATE
    params.deap_verbose = False
    return params


def run_single(seed: int, condition: str, training, test, output_path: Path) -> dict:
    assert condition in CONDITIONS
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
          f"tournament={TOURNAMENT_SIZE}, cx={CROSSOVER_RATE}, mut={MUTATION_RATE}")

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
        c: sorted([r for r in all_results if r["condition"] == c], key=lambda r: r["seed"])
        for c in CONDITIONS
    }
    for c in CONDITIONS:
        assert [r["seed"] for r in by_condition[c]] == [r["seed"] for r in by_condition["baseline"]], (
            f"Seeds must match pairwise across conditions for a paired test ({c})"
        )

    print("\n" + "=" * 80)
    print("RESULTS TABLE (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 80)
    print(f"{'Method':<18}{'Mean':>10}{'Best':>10}{'Std':>10}{'AvgConvGen':>12}{'TestMean':>11}")
    for condition in CONDITIONS:
        records = by_condition[condition]
        best_vals = np.array([r["best_fitness"] for r in records], dtype=float)
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        test_vals = np.array([r["test_fitness"] for r in records], dtype=float)
        print(
            f"{condition:<18}{best_vals.mean():>10.4f}{best_vals.min():>10.4f}"
            f"{best_vals.std():>10.4f}{conv_vals.mean():>12.2f}{test_vals.mean():>11.4f}"
        )

    print("\n" + "=" * 80)
    print("WALL-CLOCK (seconds per run, mean -- mode_interaction.py documents the expected slowdown)")
    print("=" * 80)
    for condition in CONDITIONS:
        elapsed = np.array([r["elapsed_sec"] for r in by_condition[condition]], dtype=float)
        print(f"  {condition:<18} mean={elapsed.mean():.1f}s")

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test, baseline vs each condition)")
    print("=" * 80)

    def report_pair(label_a, label_b, metric):
        a = np.array([r[metric] for r in by_condition[label_a]], dtype=float)
        b = np.array([r[metric] for r in by_condition[label_b]], dtype=float)
        diffs = a - b  # positive => b better (lower fitness)
        if np.all(diffs == 0):
            print(f"  {label_a} vs {label_b} [{metric}]: all differences zero, not informative")
            return
        stat, p_value = wilcoxon(a, b)
        effect_size = rank_biserial_effect_size(diffs)
        direction = f"{label_b} better" if diffs.mean() > 0 else f"{label_a} better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(
            f"  {label_a} vs {label_b} [{metric}]: W={stat:.4f} p={p_value:.6f} r={effect_size:.4f} "
            f"mean({label_a}-{label_b})={diffs.mean():.4f} ({direction}) => {significance}"
        )

    for metric, metric_label in [("best_fitness", "training fitness"), ("test_fitness", "test fitness")]:
        print(f"\n-- {metric_label} --")
        for condition in ("mode_interaction", "rccp", "both"):
            report_pair("baseline", condition, metric)
        print("  -- direct comparison: mode-interaction (consequence) vs RCCP (contention) --")
        report_pair("mode_interaction", "rccp", metric)


if __name__ == "__main__":
    main()
