"""
Epsilon-lexicase selection + critical-path local search vs the plain
baseline (tournament selection, no local search), plus a third arm
testing gap-aware early stopping/rollback. The gap-aware idea came from
noticing lexicase's train/val gap stays flat for a while then jumps to a
higher plateau, with none of the later training gains showing up on
held-out data.

Gap-aware stopping turned out to be a null result though (see readme.md
extension #2 and gap_aware_stopping.py) -- tested it both with and
without a real train-test gap present, no significant test-fitness
benefit either time, even though the detector itself works fine. Kept the
condition in this script rather than splitting it into its own experiment
since the fairest comparison is against "proposed" under the exact same
settings, which this script already gives for free. The actual gap-aware
driver got moved to exploratory/ once the result was in.

All three conditions share population size, generations, crossover/
mutation rates, elitism, and the train/val/test instances -- only the
selection operator, local search, and (for proposed_gap_aware) the
rollback differ.

On validation/test set size: with only 10 held-out instances, the
model-selection noise turned out to be bigger than the actual effect being
measured, so --n_classes controls how many stratified MMLIB50 classes
back the split (default 25, comfortably above the ~20-30 floor that
seemed to be needed). --known_gap_split swaps in
full_mmlib_experiment.py's split instead, the one known to actually
produce a train-test gap.

Run with (from the repo root), same pop=60/gen=20 scale used for the
validation run before spending real compute on the full paper spec:
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/experiments/lexicase_local_search_experiment.py

Time estimate without actually running it:
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/experiments/lexicase_local_search_experiment.py --dry_run

Full paper spec (pop=1000/gen=50), once the small run looks promising:
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/experiments/lexicase_local_search_experiment.py \\
        --pop_size 1000 --n_gen 50 --multiprocess
"""
import argparse
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

from yuantian.exploratory.gap_aware_stopping import lexicase_memetic_gp_gap_aware
from yuantian.gp_algorithms import standard_gp
from yuantian.gphh_solver import GPHH, ParametersGPHH, RefreshHallOfFame, read_instances
from yuantian.hybrid_gp import epsilon_lexicase_selection, lexicase_memetic_gp
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum
from yuantian.utils import PopulationArchive

DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE
SIMULATOR_TYPE = SimulatorTypeEnum.SERIAL_SGS

CONDITIONS = ("baseline", "proposed", "proposed_gap_aware")

OUTPUT_DIR = Path(__file__).parent / "results" / "lexicase_local_search_experiment"


def stratified_classes(n_classes: int, n_total: int = 108) -> list:
    """n_classes evenly spaced MMLIB50 class numbers in [1, n_total], same
    stratification idea as full_mmlib_experiment.py's STRATIFIED_CLASSES
    (there: a fixed 10; here: configurable, since validation/test size is
    exactly what this experiment needs to scale up)."""
    step = max(1, n_total // n_classes)
    return list(range(1, n_total + 1, step))[:n_classes]


def build_params(args) -> ParametersGPHH:
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        cp_propagation_feature=False,  # unchanged GP representation/terminal set
    )
    params.pop_size = args.pop_size
    params.n_gen = args.n_gen
    params.n_elite = args.n_elite
    params.tournament_size = args.tournament_size
    params.crossover_rate = args.crossover_rate
    params.mutation_rate = args.mutation_rate
    params.deap_verbose = False
    if args.multiprocess:
        params.cpu_cores = args.cpu_cores
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


def run_single(seed: int, condition: str, training, validation, test, args) -> dict:
    assert condition in CONDITIONS
    random.seed(seed)
    np.random.seed(seed)
    params = build_params(args)
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    mstats = _make_mstats()
    pop = solver.toolbox.population(n=params.pop_size)
    hof = RefreshHallOfFame(1)
    pop_archive = PopulationArchive()
    validation_provider = StaticDatasetProvider(validation)

    gap_aware_fields = {
        "onset_generation": None,
        "validation_fitness_at_onset": None,
        "validation_fitness_final": None,
        "returned_generation": None,
        "stopped_early": None,
    }

    if condition in ("proposed", "proposed_gap_aware"):
        solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)
        common_kwargs = dict(
            cxpb=params.crossover_rate,
            mutpb=params.mutation_rate,
            n_elite=params.n_elite,
            ngen=params.n_gen,
            training_data_provider=StaticDatasetProvider(training),
            validation_data_provider=validation_provider,
            decision_type=params.decision_type,
            simulator=solver.simulator,
            pset=solver.pset,
            elite_fraction=args.elite_fraction,
            local_search_iters=args.local_search_iters,
            stats=mstats,
            halloffame=hof,
            pop_archive=pop_archive,
            rng=random,
            verbose=False,
        )
        if condition == "proposed_gap_aware":
            _, log = lexicase_memetic_gp_gap_aware(
                pop,
                solver.toolbox,
                stopping_mode=args.stopping_mode,
                gap_onset_window=args.gap_onset_window,
                gap_onset_patience=args.gap_onset_patience,
                gap_onset_threshold_ratio=args.gap_onset_threshold_ratio,
                gap_onset_min_absolute_rise=args.gap_onset_min_absolute_rise,
                **common_kwargs,
            )
            gap_aware_fields.update(log.gap_aware_report)
        else:
            _, log = lexicase_memetic_gp(pop, solver.toolbox, **common_kwargs)
    else:
        _, log = standard_gp(
            pop,
            solver.toolbox,
            cxpb=params.crossover_rate,
            mutpb=params.mutation_rate,
            n_elite=params.n_elite,
            ngen=params.n_gen,
            training_data_provider=StaticDatasetProvider(training),
            validation_data_provider=validation_provider,
            stats=mstats,
            halloffame=hof,
            pop_archive=pop_archive,
            verbose=False,
        )

    test_fitness = solver.toolbox.evaluate(individual=hof[0], domains=test)[0]
    gen_best = log.chapters["generation_best"]
    fitness_curve = [g["fitness"] for g in gen_best]
    validation_curve = [g.get("validation_fitness") for g in gen_best]
    best_fitness = fitness_curve[-1]
    convergence_gen = next(
        i for i, f in enumerate(fitness_curve) if f <= best_fitness + 1e-9
    )
    final_pop_stats = log.chapters["fitness"][-1]
    fitness_std_per_gen = [rec["std"] for rec in log.chapters["fitness"]]
    unique_trees_per_gen = [
        len(set(str(ind) for ind in gen_pop)) for gen_pop in pop_archive
    ]

    record = {
        "seed": seed,
        "condition": condition,
        "best_fitness": best_fitness,
        "mean_final_pop": float(final_pop_stats["avg"]),
        "std_final_pop": float(final_pop_stats["std"]),
        "convergence_gen": convergence_gen,
        "test_fitness": test_fitness,
        "final_validation_fitness": validation_curve[-1],
        "avg_fitness_std_over_run": float(np.mean(fitness_std_per_gen)),
        "avg_unique_trees_over_run": float(np.mean(unique_trees_per_gen)),
        "best_tree": gen_best[-1]["tree"],
        "n_generations_run": len(gen_best) - 1,  # excludes gen 0
    }
    record.update(gap_aware_fields)
    return record


def rank_biserial_effect_size(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    return float((r_plus - r_minus) / ranks.sum())


def estimate_time(args, training, validation, test) -> None:
    """Run 1 seed at 1 generation per condition, then extrapolate wall-clock
    time for the full requested --n_seeds x --n_gen run. Printed, not
    returned -- this is purely an informational pre-flight check (see
    module docstring), so the requester can confirm a full-scale run's cost
    before committing compute to it."""
    print(
        f"\n--dry_run: timing 1 seed x 1 generation per condition "
        f"(pop_size={args.pop_size}) to estimate the full "
        f"--n_seeds={args.n_seeds} x --n_gen={args.n_gen} run..."
    )
    probe_args = argparse.Namespace(**vars(args))
    probe_args.n_gen = 1
    total_estimate = 0.0
    for condition in CONDITIONS:
        t0 = time.time()
        run_single(args.seed_base, condition, training, validation, test, probe_args)
        elapsed_one_gen = time.time() - t0
        # elapsed_one_gen includes gen-0 (always run) + 1 extra generation;
        # the marginal per-generation cost is what scales with --n_gen.
        per_gen = elapsed_one_gen / 2
        condition_estimate = per_gen * args.n_gen * args.n_seeds
        total_estimate += condition_estimate
        print(
            f"  {condition:<20} ~{per_gen:.2f}s/gen -> "
            f"~{condition_estimate:.0f}s for {args.n_seeds} seeds x {args.n_gen} gens"
        )
    print(
        f"  TOTAL estimate: ~{total_estimate:.0f}s (~{total_estimate / 60:.1f} min, "
        f"~{total_estimate / 3600:.2f} h) for all {len(CONDITIONS)} conditions x "
        f"{args.n_seeds} seeds x {args.n_gen} generations.\n"
        f"  (Rough: ignores any per-run fixed overhead beyond per-generation cost, "
        f"and assumes per-generation cost is ~constant across generations.)"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pop_size", type=int, default=60)
    p.add_argument("--n_gen", type=int, default=20)
    p.add_argument("--n_elite", type=int, default=3)
    p.add_argument("--tournament_size", type=int, default=7)
    p.add_argument("--crossover_rate", type=float, default=0.8)
    p.add_argument("--mutation_rate", type=float, default=0.15)
    p.add_argument("--elite_fraction", type=float, default=0.08)
    p.add_argument("--local_search_iters", type=int, default=10)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=2000)
    p.add_argument(
        "--n_classes", type=int, default=25,
        help="stratified MMLIB50 classes backing train/validation/test (one instance "
             "per class per split); >=20-30 recommended for validation/test (see module docstring)",
    )
    p.add_argument(
        "--known_gap_split", action="store_true", default=False,
        help="use full_mmlib_experiment.py's 10-class/3-train-case split instead of the "
             "1-instance-per-class default above. That split actually produces a real "
             "train-test gap (~18-19 test fitness); the default n_classes split doesn't "
             "reliably (in one run test fitness came out *better* than training, so "
             "gap-aware stopping had nothing to catch). Overrides --n_classes.",
    )
    p.add_argument("--stopping_mode", choices=["stop", "rollback"], default="rollback")
    p.add_argument("--gap_onset_window", type=int, default=3)
    p.add_argument("--gap_onset_patience", type=int, default=2)
    p.add_argument("--gap_onset_threshold_ratio", type=float, default=2.0)
    p.add_argument("--gap_onset_min_absolute_rise", type=float, default=0.1)
    p.add_argument("--multiprocess", action="store_true", default=False)
    p.add_argument("--cpu_cores", type=int, default=4)
    p.add_argument(
        "--dry_run", action="store_true", default=False,
        help="print a wall-clock time estimate for the requested configuration and exit",
    )
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.known_gap_split:
        # same classes/train/val/test as full_mmlib_experiment.py
        classes = list(range(1, 109, 11))[:10]
        train_files = [f"J50{c}_{case}.mm" for c in classes for case in (1, 2, 3)]
        val_files = [f"J50{c}_4.mm" for c in classes]
        test_files = [f"J50{c}_5.mm" for c in classes]
    else:
        classes = stratified_classes(args.n_classes)
        train_files = [f"J50{c}_1.mm" for c in classes]
        val_files = [f"J50{c}_2.mm" for c in classes]
        test_files = [f"J50{c}_3.mm" for c in classes]

    training = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in train_files])
    validation = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in val_files])
    test = read_instances([str(RCPSPDatabase.MMLIB_50_DIR + f) for f in test_files])
    print(f"Stratified classes ({len(classes)}, known_gap_split={args.known_gap_split}): {classes}")
    print(f"Train/Val/Test instance counts: {len(training)}/{len(validation)}/{len(test)}")
    print(
        f"pop_size={args.pop_size}, n_gen={args.n_gen}, n_elite={args.n_elite}, "
        f"cx={args.crossover_rate}, mut={args.mutation_rate}, "
        f"elite_fraction={args.elite_fraction}, local_search_iters={args.local_search_iters}"
    )
    print(
        f"gap-aware stopping: mode={args.stopping_mode}, "
        f"window={args.gap_onset_window}, patience={args.gap_onset_patience}, "
        f"threshold_ratio={args.gap_onset_threshold_ratio}, "
        f"min_absolute_rise={args.gap_onset_min_absolute_rise}"
    )

    if args.dry_run:
        estimate_time(args, training, validation, test)
        return

    all_results = []
    for condition in CONDITIONS:
        for i in range(args.n_seeds):
            seed = args.seed_base + i
            t0 = time.time()
            record = run_single(seed, condition, training, validation, test, args)
            all_results.append(record)
            extra = ""
            if condition == "proposed_gap_aware":
                extra = (
                    f" onset_gen={record['onset_generation']} "
                    f"returned_gen={record['returned_generation']} "
                    f"stopped_early={record['stopped_early']}"
                )
            print(
                f"[{condition}] seed={seed} best={record['best_fitness']:.4f} "
                f"test={record['test_fitness']:.4f} val={record['final_validation_fitness']:.4f} "
                f"conv_gen={record['convergence_gen']} "
                f"uniq_trees={record['avg_unique_trees_over_run']:.1f}"
                f"{extra} ({time.time() - t0:.1f}s)"
            )

    with open(OUTPUT_DIR / "all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    analyze(all_results, args)


def analyze(all_results: list, args):
    by_condition = {
        c: sorted([r for r in all_results if r["condition"] == c], key=lambda r: r["seed"])
        for c in CONDITIONS
    }
    baseline = by_condition["baseline"]
    for c in CONDITIONS:
        assert [r["seed"] for r in by_condition[c]] == [r["seed"] for r in baseline], (
            f"Seeds must match pairwise across conditions for a paired test ({c})"
        )

    print("\n" + "=" * 90)
    print("RESULTS TABLE (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 90)
    print(
        f"{'Method':<20}{'Mean':>9}{'Best':>9}{'Std':>9}{'ConvGen':>9}"
        f"{'TestMean':>11}{'ValMean':>10}"
    )
    for condition in CONDITIONS:
        records = by_condition[condition]
        best_vals = np.array([r["best_fitness"] for r in records], dtype=float)
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        test_vals = np.array([r["test_fitness"] for r in records], dtype=float)
        val_vals = np.array([r["final_validation_fitness"] for r in records], dtype=float)
        print(
            f"{condition:<20}{best_vals.mean():>9.4f}{best_vals.min():>9.4f}"
            f"{best_vals.std():>9.4f}{conv_vals.mean():>9.2f}"
            f"{test_vals.mean():>11.4f}{val_vals.mean():>10.4f}"
        )

    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test)")
    print("=" * 90)

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
        report_pair("baseline", "proposed", metric)
        report_pair("baseline", "proposed_gap_aware", metric)
        report_pair("proposed", "proposed_gap_aware", metric)

    print("\n" + "=" * 90)
    print("GENERALIZATION GAP (test_fitness - best_fitness; higher = more overfit)")
    print("=" * 90)
    for condition in CONDITIONS:
        records = by_condition[condition]
        gaps = np.array(
            [r["test_fitness"] - r["best_fitness"] for r in records], dtype=float
        )
        print(f"  {condition:<20} mean gap={gaps.mean():.4f}, std={gaps.std():.4f}")

    gap_aware = by_condition["proposed_gap_aware"]
    n_onset_detected = sum(1 for r in gap_aware if r["onset_generation"] is not None)
    print(
        f"\nGap-aware stopping diagnostics (n={len(gap_aware)} seeds, mode={args.stopping_mode}):"
    )
    print(f"  onset detected in {n_onset_detected}/{len(gap_aware)} runs")
    if n_onset_detected:
        onset_gens = [r["onset_generation"] for r in gap_aware if r["onset_generation"] is not None]
        returned_gens = [r["returned_generation"] for r in gap_aware]
        print(f"  onset generation: mean={np.mean(onset_gens):.1f}, values={onset_gens}")
        print(f"  returned generation: mean={np.mean(returned_gens):.1f}, values={returned_gens}")
    if args.stopping_mode == "stop":
        n_stopped = sum(1 for r in gap_aware if r["stopped_early"])
        gens_run = [r["n_generations_run"] for r in gap_aware]
        print(
            f"  runs that actually stopped early: {n_stopped}/{len(gap_aware)}; "
            f"generations actually run: mean={np.mean(gens_run):.1f} (budget was {args.n_gen})"
        )

    print("\n" + "=" * 90)
    print("DIAGNOSTICS: diversity and convergence character")
    print("=" * 90)
    for condition in CONDITIONS:
        records = by_condition[condition]
        fit_std = np.array([r["avg_fitness_std_over_run"] for r in records], dtype=float)
        uniq = np.array([r["avg_unique_trees_over_run"] for r in records], dtype=float)
        print(
            f"{condition:<20} avg population fitness std={fit_std.mean():.4f}, "
            f"avg unique trees/gen={uniq.mean():.1f} (out of {args.pop_size})"
        )


if __name__ == "__main__":
    main()
