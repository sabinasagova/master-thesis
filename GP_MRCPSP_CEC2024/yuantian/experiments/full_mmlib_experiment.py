"""
Large-scale controlled comparison: Baseline GPHH vs. epsilon-lexicase GPHH vs.
epsilon-lexicase + critical-path local search ("full hybrid"), on Serial and
Parallel SGS, on a stratified subset of MMLIB50.

*** SCALE NOTE ***
A full-scale protocol (population=1000, generations=50, the complete
108-class MMLIB50 split, 30 seeds) was measured at ~2.25s per individual
evaluation on the real 324-instance training split; since the GP loop
re-evaluates the whole population every generation, that scale would cost
roughly 200-250 days of compute on a single machine. This script instead
runs a reduced "moderate tier" protocol (~5-6 hours total):
  - population=60 (vs. 1000 at full scale)
  - generations=20 (vs. 50)
  - 10 stratified classes out of 108, evenly spread (classes 1,12,23,...,100),
    using the same case-index split convention as the codebase's official
    split (cases 1-3 = train, case 4 = validation, case 5 = test) -> 30
    training / 10 validation / 10 test instances instead of 324/108/108.
  - 10 seeds (vs. 30)
Crossover=0.8, mutation=0.15, elitism=10, tournament k=7, the function set,
and decision type (activity-first) match the full-scale protocol exactly.

Model selection mirrors the codebase's own validation methodology
(GPHH.solve()): after evolution, the individual in the final population with
the best validation fitness is the one evaluated on the held-out test set,
not just the training hall-of-fame winner.

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python -O yuantian/experiments/full_mmlib_experiment.py
"""
import json
import random
import sys
import time
import warnings
from functools import partial
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
# Reduced ("moderate tier") protocol -- see module docstring for the deviation
# rationale and exact deltas from the full-scale protocol.
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_BASE = 4000
POP_SIZE = 60
N_GEN = 20
N_ELITE = 10
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE

ELITE_FRACTION = 0.08
LOCAL_SEARCH_ITERS = 10

STRATIFIED_CLASSES = list(range(1, 109, 11))[:10]  # [1, 12, 23, ..., 100]
TRAIN_FILES = [f"J50{c}_{case}.mm" for c in STRATIFIED_CLASSES for case in (1, 2, 3)]
VAL_FILES = [f"J50{c}_4.mm" for c in STRATIFIED_CLASSES]
TEST_FILES = [f"J50{c}_5.mm" for c in STRATIFIED_CLASSES]

METHODS = ["baseline", "lexicase", "ls_only", "lexicase_ls"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "lexicase": "Lexicase",
    "ls_only": "LS only",
    "lexicase_ls": "Lexicase + LS",
}
SGS_TYPES = ["serial", "parallel"]

OUTPUT_DIR = Path(__file__).parent / "results" / "full_mmlib_experiment"


def build_params(sgs_type: str) -> ParametersGPHH:
    simulator_type = (
        SimulatorTypeEnum.SERIAL_SGS if sgs_type == "serial" else SimulatorTypeEnum.PARALLEL_SGS
    )
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=simulator_type,
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


def run_single(seed: int, method: str, sgs_type: str, training, validation, test) -> dict:
    assert method in METHODS
    random.seed(seed)
    np.random.seed(seed)
    params = build_params(sgs_type)
    solver = GPHH(
        training_set_provider=StaticDatasetProvider(training),
        validation_set_provider=StaticDatasetProvider(validation),
        params_gphh=params,
    )
    solver.init_model()

    # 2x2 factorial: selection (tournament vs epsilon-lexicase) x local search (off/on)
    use_lexicase = method in ("lexicase", "lexicase_ls")
    use_local_search = method in ("ls_only", "lexicase_ls")

    if use_lexicase:
        solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)

    mstats = _make_mstats()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = RefreshHallOfFame(1)
    pop_archive = PopulationArchive()

    common_kwargs = dict(
        cxpb=params.crossover_rate,
        mutpb=params.mutation_rate,
        n_elite=params.n_elite,
        ngen=params.n_gen,
        training_data_provider=StaticDatasetProvider(training),
        validation_data_provider=StaticDatasetProvider(validation),
        stats=mstats,
        halloffame=hof,
        pop_archive=pop_archive,
        verbose=False,
    )

    if use_local_search:
        final_pop, log = lexicase_memetic_gp(
            pop,
            solver.toolbox,
            decision_type=params.decision_type,
            simulator=solver.simulator,
            pset=solver.pset,
            elite_fraction=ELITE_FRACTION,
            local_search_iters=LOCAL_SEARCH_ITERS,
            rng=random,
            **common_kwargs,
        )
    else:
        final_pop, log = standard_gp(pop, solver.toolbox, **common_kwargs)

    # Validation-based model selection (mirrors GPHH.solve()): the individual
    # in the final population with the best validation fitness is what gets
    # tested, not just the training hall-of-fame winner.
    validation_evaluate = partial(solver.toolbox.evaluate, domains=validation)
    val_fitnesses = solver.toolbox.map(validation_evaluate, final_pop)
    best_validated_ind = min(zip(final_pop, val_fitnesses), key=lambda x: x[1][0])[0]
    test_fitness = solver.toolbox.evaluate(individual=best_validated_ind, domains=test)[0]
    # this single evaluation has no selection pressure backing it up (unlike
    # best_fitness_train, where an infeasible individual would have to beat
    # out the entire rest of the population to end up reported), so a single
    # infeasible test instance can silently swap test_fitness for a sentinel
    # in the millions -- this PRIMARY metric needs feasibility tracked.
    test_feasible = all(best_validated_ind.case_feasible)

    gen_best = log.chapters["generation_best"]
    fitness_curve = [g["fitness"] for g in gen_best]
    best_fitness_train = fitness_curve[-1]
    convergence_gen = next(
        i for i, f in enumerate(fitness_curve) if f <= best_fitness_train + 1e-9
    )
    final_pop_stats = log.chapters["fitness"][-1]
    fitness_std_per_gen = [rec["std"] for rec in log.chapters["fitness"]]
    unique_trees_per_gen = [
        len(set(str(ind) for ind in gen_pop)) for gen_pop in pop_archive
    ]

    return {
        "seed": seed,
        "method": method,
        "sgs": sgs_type,
        "best_fitness_train": best_fitness_train,
        "test_fitness": test_fitness,
        "test_feasible": test_feasible,
        "mean_final_pop": float(final_pop_stats["avg"]),
        "std_final_pop": float(final_pop_stats["std"]),
        "convergence_gen": convergence_gen,
        "fitness_curve": fitness_curve,
        "avg_fitness_std_over_run": float(np.mean(fitness_std_per_gen)),
        "avg_unique_trees_over_run": float(np.mean(unique_trees_per_gen)),
        "best_tree": gen_best[-1]["tree"],
    }


def _feasible_test_vals(records: list) -> np.ndarray:
    # old result files (pre-feasibility-tracking) default to "assume feasible"
    # so they keep working, but every new run actually filters.
    return np.array([r["test_fitness"] for r in records if r.get("test_feasible", True)])


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

    train_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TRAIN_FILES]
    val_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in VAL_FILES]
    test_paths = [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in TEST_FILES]
    training = read_instances(train_paths)
    validation = read_instances(val_paths)
    test = read_instances(test_paths)
    print(f"Stratified classes: {STRATIFIED_CLASSES}")
    print(f"Train/Val/Test sizes: {len(training)}/{len(validation)}/{len(test)}")
    print(
        f"pop_size={POP_SIZE}, n_gen={N_GEN}, n_elite={N_ELITE}, tournament={TOURNAMENT_SIZE}, "
        f"cx={CROSSOVER_RATE}, mut={MUTATION_RATE}, elite_fraction={ELITE_FRACTION}, "
        f"local_search_iters={LOCAL_SEARCH_ITERS}, seeds={N_SEEDS}"
    )

    all_results = []
    for sgs_type in SGS_TYPES:
        for method in METHODS:
            for i in range(N_SEEDS):
                seed = SEED_BASE + i
                t0 = time.time()
                record = run_single(seed, method, sgs_type, training, validation, test)
                all_results.append(record)
                print(
                    f"[{sgs_type}/{method}] seed={seed} "
                    f"train_best={record['best_fitness_train']:.4f} "
                    f"test={record['test_fitness']:.4f} test_feasible={record['test_feasible']} "
                    f"conv_gen={record['convergence_gen']} "
                    f"uniq_trees={record['avg_unique_trees_over_run']:.1f} "
                    f"({time.time() - t0:.1f}s)"
                )
            with open(OUTPUT_DIR / "all_runs_partial.json", "w") as f:
                json.dump(all_results, f, indent=2)

    with open(OUTPUT_DIR / "all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    analyze(all_results)


def analyze(all_results: list):
    import sys

    report_lines = []

    def emit(line=""):
        print(line)
        report_lines.append(line)

    by_key = {}
    for r in all_results:
        by_key.setdefault((r["sgs"], r["method"]), []).append(r)
    for key in by_key:
        by_key[key] = sorted(by_key[key], key=lambda r: r["seed"])

    emit("=" * 90)
    emit("1. SUMMARY TABLES (per SGS): methods x mean +/- std test fitness")
    emit("=" * 90)
    emit("(TestMean/TestStd below are feasibility-filtered -- see n_feasible noted per row if any seed was dropped)")
    for sgs_type in SGS_TYPES:
        emit(f"\n--- SGS = {sgs_type} ---")
        emit(
            f"{'Method':<15}{'TrainMean':>11}{'TrainBest':>11}{'TrainStd':>10}"
            f"{'TrainMedian':>12}{'TestMean':>10}{'TestStd':>10}{'ConvGen':>9}"
        )
        for method in METHODS:
            records = by_key[(sgs_type, method)]
            train_vals = np.array([r["best_fitness_train"] for r in records])
            test_vals = _feasible_test_vals(records)
            conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
            n_dropped = len(records) - len(test_vals)
            dropped_note = f" (dropped {n_dropped} infeasible test)" if n_dropped else ""
            emit(
                f"{METHOD_LABELS[method]:<15}{train_vals.mean():>11.4f}{train_vals.min():>11.4f}"
                f"{train_vals.std():>10.4f}{np.median(train_vals):>12.4f}"
                f"{test_vals.mean():>10.4f}{test_vals.std():>10.4f}{conv_vals.mean():>9.2f}"
                f"{dropped_note}"
            )

    emit("\n" + "=" * 90)
    emit("2. FULL RANKING (by mean test fitness, lower=better)")
    emit("=" * 90)
    for sgs_type in SGS_TYPES:
        ranking = sorted(
            METHODS,
            key=lambda m: _feasible_test_vals(by_key[(sgs_type, m)]).mean(),
        )
        emit(f"{sgs_type}: " + " < ".join(METHOD_LABELS[m] for m in ranking))
    overall_ranking = sorted(
        [(sgs, m) for sgs in SGS_TYPES for m in METHODS],
        key=lambda sm: _feasible_test_vals(by_key[sm]).mean(),
    )
    emit(
        "Overall (across both SGS): "
        + " < ".join(f"{METHOD_LABELS[m]}/{sgs}" for sgs, m in overall_ranking)
    )

    # Full pairwise set for the 2x2 factorial (selection x local search):
    # isolates the effect of lexicase alone, LS alone, their combination,
    # and the incremental effect of adding one component on top of the other.
    comparisons = [
        ("baseline", "lexicase"),
        ("baseline", "ls_only"),
        ("baseline", "lexicase_ls"),
        ("lexicase", "lexicase_ls"),
        ("ls_only", "lexicase_ls"),
        ("lexicase", "ls_only"),
    ]

    def _wilcoxon_block(section_label, metric_key, metric_label, feasible_key=None):
        emit("\n" + "=" * 90)
        emit(f"{section_label}. STATISTICAL SIGNIFICANCE (paired Wilcoxon on {metric_label}, by seed)")
        if feasible_key:
            emit("Pairs where EITHER side was infeasible on this metric are dropped before testing.")
        emit("=" * 90)
        for sgs_type in SGS_TYPES:
            emit(f"\n--- SGS = {sgs_type} ---")
            for a, b in comparisons:
                recs_a = by_key[(sgs_type, a)]
                recs_b = by_key[(sgs_type, b)]
                vals_a = np.array([r[metric_key] for r in recs_a])
                vals_b = np.array([r[metric_key] for r in recs_b])
                if feasible_key:
                    feas_a = np.array([r.get(feasible_key, True) for r in recs_a])
                    feas_b = np.array([r.get(feasible_key, True) for r in recs_b])
                    mask = feas_a & feas_b
                    n_dropped = (~mask).sum()
                    vals_a, vals_b = vals_a[mask], vals_b[mask]
                else:
                    n_dropped = 0
                diffs = vals_a - vals_b  # positive => b better
                drop_note = f" [dropped {n_dropped} infeasible pair(s)]" if n_dropped else ""
                if len(diffs) < 1 or np.all(diffs == 0):
                    emit(
                        f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}: all differences zero or no "
                        f"pairs left, Wilcoxon not informative{drop_note}"
                    )
                    continue
                stat, p_value = wilcoxon(vals_a, vals_b)
                effect = rank_biserial_effect_size(diffs)
                label_a, label_b = METHOD_LABELS[a], METHOD_LABELS[b]
                direction = f"{label_b} better" if diffs.mean() > 0 else f"{label_a} better"
                sig = "significant (p<0.05)" if p_value < 0.05 else "not significant"
                emit(
                    f"{label_a} vs {label_b}: n={len(diffs)} W={stat:.3f}, p={p_value:.6f}, r={effect:.4f}, "
                    f"mean diff={diffs.mean():.4f} ({direction}) -> {sig}{drop_note}"
                )

    # Primary: validation-selected test fitness (what actually matters for
    # generalization). Secondary: training best_fitness (what the GP run
    # itself optimized), kept for transparency / overfitting checks.
    _wilcoxon_block("3a", "test_fitness", "held-out test_fitness [PRIMARY]", feasible_key="test_feasible")
    _wilcoxon_block("3b", "best_fitness_train", "training best_fitness [secondary]")

    emit("\n" + "=" * 90)
    emit("4. CONVERGENCE ANALYSIS (mean generation-to-best, lower=faster)")
    emit("=" * 90)
    for sgs_type in SGS_TYPES:
        emit(f"\n--- SGS = {sgs_type} ---")
        for method in METHODS:
            records = by_key[(sgs_type, method)]
            conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
            fit_std = np.array([r["avg_fitness_std_over_run"] for r in records])
            uniq = np.array([r["avg_unique_trees_over_run"] for r in records])
            emit(
                f"{METHOD_LABELS[method]:<15} avg_conv_gen={conv_vals.mean():.2f}, "
                f"avg_pop_fitness_std={fit_std.mean():.4f}, "
                f"avg_unique_trees/gen={uniq.mean():.1f}/{POP_SIZE}"
            )

    try:
        _plot_convergence(by_key)
        emit(f"\nConvergence plot saved to {OUTPUT_DIR / 'convergence.png'}")
    except Exception as e:
        emit(f"\n(plotting skipped: {e})")

    with open(OUTPUT_DIR / "report.txt", "w") as f:
        f.write("\n".join(report_lines))


def _plot_convergence(by_key):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SGS_TYPES), figsize=(7 * len(SGS_TYPES), 5), sharey=True)
    if len(SGS_TYPES) == 1:
        axes = [axes]
    for ax, sgs_type in zip(axes, SGS_TYPES):
        for method in METHODS:
            records = by_key[(sgs_type, method)]
            curves = np.array([r["fitness_curve"] for r in records])
            mean_curve = curves.mean(axis=0)
            ax.plot(
                range(len(mean_curve)),
                mean_curve,
                label=METHOD_LABELS[method],
                marker="o",
                markersize=3,
            )
        ax.set_title(f"SGS = {sgs_type}")
        ax.set_xlabel("generation")
        ax.set_ylabel("mean best fitness (train, % dev. from CPM lower bound)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
