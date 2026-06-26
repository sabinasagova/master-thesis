"""
Single-cell-single-seed entry point for the full paper-spec experiment
matrix (see matrix_config.py for axes/presets/cost model this composes
against).

    python -m yuantian.experiments.matrix_runner \\
        --condition nr --strategy S --sgs serial --dataset MMLIB50 \\
        --pop 1000 --gen 50 --seed 11000 --out results/matrix/

Writes results/matrix/{dataset}__{sgs}__{strategy}__{condition}__seed{seed}.json.
Idempotent: skips (and says so) if that file already exists, so a
resubmitted or relaunched PBS array element doesn't redo a finished cell.

IMPORTANT instance-set caveat for "nr": the paper's own baseline strips
non-renewable (NR) resources from every instance before the GP ever sees it
(to_renewable_only_rcpsp_model, the read_instances default). NR terminals
have nothing to read once that conversion has happened, so the "nr"
condition has to load instances WITHOUT that conversion
(keep_non_renewable=True) -- same thing gphh_solver.py's own CLI already
does, keyed off whether --nr_terminals is set (see read_instances call
there). That means "nr" runs on a different, slightly harder instance set
than baseline/lexicase/local_search/hybrid (all of which use the paper's
renewable-only conversion, exactly matching Tables V/VI), and the two
results are NOT a same-instance paired comparison. analyze_matrix.py
flags this explicitly rather than silently diffing mismatched instances.

Conditions:
  - baseline / nr: pure standard_gp (gp_algorithms.standard_gp), with
    nr_terminals_feature on for "nr". Goes through GPHH.solve() directly,
    so the result JSON's feasibility fields come from solve()'s own
    (already-fixed) write path natively -- this is also why this runner
    defaults to single-process: under --multiprocess, train_case_records
    would come back None and fall back to a recompute (see
    gphh_solver.py's docstring on that path), which is fine but
    unnecessary when the point of a result file is to feed an analysis
    later, not to race the clock.
  - lexicase / local_search / hybrid: solve() hardcodes standard_gp, so
    these can't go through it unmodified. Builds the population and drives
    the loop directly (standard_gp with epsilon-lexicase selection
    registered, or lexicase_memetic_gp), same pattern
    full_mmlib_experiment.py already uses, then calls the same
    evaluate_and_package_test_data + write_result gphh_solver.py uses
    internally so the result JSON has the identical shape regardless of
    which condition produced it.

Both paths build params via ParametersGPHH.fast(...) (a constructor
convenience that builds the identical pset/depth structure as .default()
-- verified directly against gphh_solver.py before writing this module),
then explicitly override pop_size/n_gen/n_elite/tournament_size/
crossover_rate/mutation_rate to the paper's Table IV values -- same
convention full_mmlib_experiment.py and friends already use. So "baseline"
needs zero additional overrides beyond decision_type/simulator_type to be
a valid paper reproduction.

--dry_run: times 1 seed at the REAL --pop for a couple of generations and
extrapolates to the requested --gen -- the trustworthy per-cell number,
same convention as nr_terminals_experiment.py's own --dry_run. (See
matrix_config.estimate_cell_seconds for the rough pre-submission number
generate_pbs_jobs.py uses before any real measurement exists.)
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

# gphh_solver.GPHH.init_model() does a bare `import multitreegp`, which
# requires yuantian/ itself (not this experiments/ subfolder) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.experiments.matrix_config import (
    CONDITIONS,
    DATASETS,
    SGS_TO_SIMULATOR_TYPE,
    SGS_TYPES,
    STRATEGIES,
    STRATEGY_TO_DECISION_TYPE,
    dataset_files,
    stratified_classes,
)
from yuantian.gp_algorithms import standard_gp
from yuantian.gphh_solver import (
    GPHH,
    ParametersGPHH,
    RefreshHallOfFame,
    evaluate_and_package_test_data,
    read_instances,
)
from yuantian.hybrid_gp import epsilon_lexicase_selection, lexicase_memetic_gp
from yuantian.rcpsp_dataset import StaticDatasetProvider
from yuantian.utils import PopulationArchive

# Paper Table IV values applied on top of ParametersGPHH.fast(...)'s pset/
# depth structure -- pop/gen are the only knobs the CLI actually varies.
N_ELITE = 10
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
ELITE_FRACTION = 0.08
LOCAL_SEARCH_ITERS = 10

DRY_RUN_PROBE_GENS = 2


def cell_filename(args) -> str:
    return f"{args.dataset}__{args.sgs}__{args.strategy}__{args.condition}__seed{args.seed}.json"


def build_params(args) -> ParametersGPHH:
    decision_type = STRATEGY_TO_DECISION_TYPE[args.strategy]
    simulator_type = SGS_TO_SIMULATOR_TYPE[args.sgs]
    params = ParametersGPHH.fast(
        decision_type=decision_type,
        simulator_type=simulator_type,
        nr_terminals_feature=(args.condition == "nr"),
    )
    params.pop_size = args.pop
    params.n_gen = args.gen
    params.n_elite = N_ELITE
    params.tournament_size = TOURNAMENT_SIZE
    params.crossover_rate = CROSSOVER_RATE
    params.mutation_rate = MUTATION_RATE
    params.deap_verbose = False
    if args.multiprocess:
        params.cpu_cores = args.cpu_cores
    return params


def load_split(args):
    classes = stratified_classes(args.dataset, args.n_classes)
    train_files, val_files, test_files = dataset_files(args.dataset, classes)
    # see module docstring: "nr" needs NR resources preserved, every other
    # condition uses the paper's own renewable-only conversion.
    keep_non_renewable = args.condition == "nr"
    training = read_instances(train_files, keep_non_renewable=keep_non_renewable)
    validation = read_instances(val_files, keep_non_renewable=keep_non_renewable)
    test = read_instances(test_files, keep_non_renewable=keep_non_renewable)
    return training, validation, test


def _make_mstats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats


def _patch_matrix_cell(args, output_path: Path) -> dict:
    """Stamp identity (condition/strategy/sgs/dataset/seed) directly into
    the result JSON, so analyze_matrix.py can group cells without parsing
    filenames back apart -- filenames are still informative for a human
    browsing the directory, but the JSON itself is the source of truth."""
    with open(output_path) as f:
        result = json.load(f)
    result["matrix_cell"] = {
        "condition": args.condition, "strategy": args.strategy,
        "sgs": args.sgs, "dataset": args.dataset, "seed": args.seed,
        "pop": args.pop, "gen": args.gen,
    }
    with open(output_path, "w") as f:
        json.dump(result, f)
    return result


def run_cell(args, training, validation, test, output_path: Path) -> dict:
    """Run exactly one (condition, strategy, sgs, dataset, seed) cell,
    write the result JSON to output_path, and return the same dict."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    params = build_params(args)

    if args.condition in ("baseline", "nr"):
        solver = GPHH(
            training_set_provider=StaticDatasetProvider(training),
            validation_set_provider=StaticDatasetProvider(validation),
            test_set_provider=StaticDatasetProvider(test),
            params_gphh=params,
        )
        solver.init_model()
        solver.solve(output_path=str(output_path))
        return _patch_matrix_cell(args, output_path)

    # lexicase / local_search / hybrid -- solve() can't drive these
    # (hardcodes standard_gp), so build the pop and run the loop directly,
    # same pattern full_mmlib_experiment.py already uses.
    solver = GPHH(
        training_set_provider=StaticDatasetProvider(training),
        validation_set_provider=StaticDatasetProvider(validation),
        params_gphh=params,
    )
    solver.init_model()

    use_lexicase = args.condition in ("lexicase", "hybrid")
    use_local_search = args.condition in ("local_search", "hybrid")
    if use_lexicase:
        solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)

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
        stats=_make_mstats(),
        halloffame=hof,
        pop_archive=pop_archive,
        verbose=False,
    )

    start = time.time()
    if use_local_search:
        final_pop, log = lexicase_memetic_gp(
            pop, solver.toolbox,
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
    elapsed = time.time() - start

    solver.best_heuristic = hof[0]
    # Explicit re-evaluation against training data, not a read of whatever
    # case_records happens to be sitting on best_heuristic -- both
    # standard_gp and lexicase_memetic_gp directly re-evaluate
    # halloffame[0] against the VALIDATION set at the end of every
    # generation (since validation_data_provider is always set above), so
    # by the time the loop returns, case_records reflects validation, not
    # training -- same bug, and same fix, as gphh_solver.GPHH.solve().
    #
    # This recompute is also deliberately repair-free (plain toolbox.evaluate,
    # no CP local search), to stay on the same footing as test evaluation
    # (evaluate_and_package_test_data never applies repair either) and as
    # baseline/nr/lexicase (which have no repair step to begin with) -- so
    # train_case_records means the same thing across all 5 conditions: how
    # the EVOLVED TREE performs on its own, not tree-plus-runtime-repair.
    #
    # For use_local_search, this means train_recheck will NOT equal
    # best_heuristic.fitness.values[0]: _local_search_elites overwrites the
    # elite fraction's fitness with the CP-repaired value every generation
    # (without changing the tree), and that refined fitness is what
    # halloffame ends up tracking (RefreshHallOfFame rebuilds from the
    # current, already-refined population each generation) -- so the
    # recorded fitness legitimately includes a repair benefit a repair-free
    # recompute cannot reproduce. A strict equality assert here would fire
    # on every single local_search/hybrid run, unconditionally; the
    # one-sided check below instead confirms repair only ever helped
    # (lower deviation == better), catching real data-mismatch bugs (e.g. a
    # stateful provider returning a different batch) without flagging this
    # expected, by-design gap.
    train_recheck = solver.toolbox.evaluate(individual=solver.best_heuristic, domains=training)[0]
    if use_local_search:
        assert train_recheck >= solver.best_heuristic.fitness.values[0] - 1e-6, (
            f"repair-free re-evaluation on training gave {train_recheck}, which is BETTER "
            f"than the recorded (repaired) training fitness {solver.best_heuristic.fitness.values[0]} "
            f"-- repair should only ever improve or match, never worsen, so this points at a real "
            f"data mismatch, not the expected repair gap"
        )
    else:
        assert abs(train_recheck - solver.best_heuristic.fitness.values[0]) < 1e-6, (
            f"re-evaluating best_heuristic on training gave {train_recheck}, expected to "
            f"match the recorded training fitness {solver.best_heuristic.fitness.values[0]}"
        )
    train_case_records = solver.best_heuristic.case_records

    test_data = evaluate_and_package_test_data(
        toolbox=solver.toolbox,
        best_heuristic=solver.best_heuristic,
        pop=final_pop,
        validation_data_provider=StaticDatasetProvider(validation),
        test_data_provider=StaticDatasetProvider(test),
        train_case_records=train_case_records,
    )
    solver.write_result(
        log, filepath=str(output_path), pop_archive=pop_archive, elapsed=elapsed, others=test_data
    )
    return _patch_matrix_cell(args, output_path)


def estimate_time(args) -> None:
    """Pre-flight check: time 1 seed at the REAL --pop for
    DRY_RUN_PROBE_GENS generations (cost scales with population, so this
    has to be measured at the real --pop, not extrapolated up from a
    smaller one) and extrapolate to the requested --gen. Also prints the
    output JSON's feasibility fields so a wrong/empty field shows up here,
    not after committing real grid time to it."""
    training, validation, test = load_split(args)
    probe_args = argparse.Namespace(**vars(args))
    probe_args.gen = DRY_RUN_PROBE_GENS
    tmp_dir = Path(args.out) / "_dry_run_probe"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / cell_filename(probe_args)

    print(
        f"\n--dry_run: {args.condition}/{args.strategy}/{args.sgs}/{args.dataset} -- "
        f"timing 1 seed x {DRY_RUN_PROBE_GENS} generations at the REAL pop={args.pop} "
        f"(measured here, not extrapolated up from a smaller pop) on "
        f"{len(training)} train / {len(validation)} val / {len(test)} test instances, "
        f"multiprocess={args.multiprocess}..."
    )
    t0 = time.time()
    result = run_cell(probe_args, training, validation, test, out_path)
    elapsed = time.time() - t0
    per_gen = elapsed / (DRY_RUN_PROBE_GENS + 1)  # gen-0 + probe gens, see nr_terminals_experiment.py's same convention
    estimate = per_gen * args.gen
    print(f"  ~{per_gen:.2f}s/gen -> ~{estimate:.0f}s (~{estimate / 3600:.2f}h) for --gen={args.gen}, this seed alone")

    train_case_records = result.get("train_case_records")
    if train_case_records is None:
        print("    train_case_records: None in JSON (expected only under --multiprocess)")
    else:
        n_feas = sum(r["feasible"] for r in train_case_records)
        print(f"    train_case_records: present, {n_feas}/{len(train_case_records)} feasible")
    best_heuristic = result.get("best_heuristic")
    if best_heuristic:
        test_case_records = best_heuristic.get("test_case_records")
        n_feas = sum(r["feasible"] for r in test_case_records) if test_case_records else 0
        n_tot = len(test_case_records) if test_case_records else 0
        print(f"    test_fitness={best_heuristic['test_fitness']:.4f}, test_case_records {n_feas}/{n_tot} feasible")
    out_path.unlink(missing_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--strategy", required=True, choices=STRATEGIES)
    p.add_argument("--sgs", required=True, choices=SGS_TYPES)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--pop", type=int, required=True)
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--n_classes", type=int, default=10,
        help="stratified classes backing the 60/20/20 train/val/test split (paper "
             "itself uses all classes; see matrix_config.PAPER_N_CLASSES for why a "
             "reduced count is the practical default)",
    )
    p.add_argument(
        "--multiprocess", action="store_true", default=False,
        help="off by default -- see module docstring on why single-process is the "
             "default for this runner specifically (native feasibility serialization)",
    )
    p.add_argument("--cpu_cores", type=int, default=4)
    p.add_argument(
        "--dry_run", action="store_true", default=False,
        help="pre-flight timing + feasibility-field check for this one cell, doesn't "
             "write into --out's real path",
    )
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        estimate_time(args)
        return

    output_path = Path(args.out) / cell_filename(args)
    if output_path.exists():
        print(f"SKIP (already exists): {output_path}")
        return

    training, validation, test = load_split(args)
    print(
        f"[{args.dataset}/{args.sgs}/{args.strategy}/{args.condition}] seed={args.seed} "
        f"pop={args.pop} gen={args.gen} train={len(training)} val={len(validation)} test={len(test)}"
    )
    t0 = time.time()
    result = run_cell(args, training, validation, test, output_path)
    elapsed = time.time() - t0
    best_heuristic = result.get("best_heuristic") or {}
    print(
        f"  done in {elapsed:.1f}s -- train_best={result['generation_best'][-1]['fitness']:.4f} "
        f"test_fitness={best_heuristic.get('test_fitness', float('nan')):.4f} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
