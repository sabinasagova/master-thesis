"""
Experiment: non-renewable (NR) resource terminals vs the baseline GPHH
terminal set.

Three conditions, all run on NR-preserving instances (keep_non_renewable=
True, see nr_terminals.py for why that means this experiment's "baseline"
isn't directly comparable to cp_propagation_experiment.py's):
  - baseline: normal terminal set, NR resources exist in the model but the
    GP can't see them (no NR_* terminals)
  - baseline+nr: baseline + NR_STOCK_RATIO, NR_MODE_DEMAND_RATIO,
    NR_BUDGET_PRESSURE
  - baseline+nr+cp: baseline+nr + CP_FORWARD, CP_BACKWARD, CP_SLACK_SCORE,
    CP_PROB (so --nr_terminals + --cp_propagation together)

Defaults below (pop=60, gen=25, 5 classes) are scaled down from the
paper's (pop=1000, gen=50, full 108-class split) so a run finishes on a
laptop -- but every one of those numbers is now a CLI flag, precisely so
that going from validation-scale to paper-scale is changing arguments,
not editing code.

Before launching a much larger run (e.g. close to the paper's actual
scale), run with --dry_run first. That measures 1 seed's per-generation
cost at the REAL --pop_size you're about to commit to (cost scales with
population size, so timing it at a smaller pop_size and extrapolating up
would be measuring the wrong thing), extrapolates to your full
--n_gen x --n_seeds x conditions budget, and sanity-checks the output
JSON's feasibility fields (including the --multiprocess fallback path)
before any real compute gets spent:
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 -O \\
        yuantian/experiments/nr_terminals_experiment.py --dry_run \\
        --pop_size 1000 --n_gen 50 --n_seeds 10 --multiprocess

Run with (from the repo root):
    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 -O \\
        yuantian/experiments/nr_terminals_experiment.py
"""
import argparse
import json
import random
import sys
import tempfile
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
# Experiment configuration. These are just the defaults -- main() overwrites
# POP_SIZE/N_GEN/CPU_CORES from CLI args before anything runs. Left as
# module globals (read directly by build_params/run_single below) rather
# than threaded through as a parameter, since nr_terminals_followup.py and
# nr_terminals_mmlib_plus_experiment.py import run_single directly and
# expect these defaults untouched -- only main()'s own argparse path
# overrides them, and only for this script's own direct invocation.
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_BASE = 7000
POP_SIZE = 60
N_GEN = 25
N_ELITE = 3
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
CPU_CORES = 1
DECISION_TYPE = DecisionTypeEnum.ACTIVITY_THEN_MODE
SIMULATOR_TYPE = SimulatorTypeEnum.SERIAL_SGS

N_CLASSES = 5
TRAIN_CLASSES = [1, 20, 40, 60, 80]
TRAIN_FILES = [f"J50{c}_1.mm" for c in TRAIN_CLASSES]
TEST_FILES = [f"J50{c}_2.mm" for c in TRAIN_CLASSES]

OUTPUT_DIR = Path(__file__).parent / "results" / "nr_terminals_experiment"

CONDITIONS = ["baseline", "baseline+nr", "baseline+nr+cp"]

# how many extra generations the --dry_run probe actually runs, on top of
# the always-run gen 0 -- small enough to be cheap, more than 1 so the
# per-generation cost isn't just a single noisy sample.
DRY_RUN_PROBE_GENS = 2


def stratified_classes(n_classes: int, n_total: int = 108) -> list:
    """n_classes evenly spaced MMLIB50 class numbers in [1, n_total]."""
    step = max(1, n_total // n_classes)
    return list(range(1, n_total + 1, step))[:n_classes]


def build_params(condition: str) -> ParametersGPHH:
    params = ParametersGPHH.fast(
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        nr_terminals_feature=condition in ("baseline+nr", "baseline+nr+cp"),
        cp_propagation_feature=condition == "baseline+nr+cp",
    )
    params.pop_size = POP_SIZE
    params.n_gen = N_GEN
    params.n_elite = N_ELITE
    params.tournament_size = TOURNAMENT_SIZE
    params.crossover_rate = CROSSOVER_RATE
    params.mutation_rate = MUTATION_RATE
    params.deap_verbose = False
    if CPU_CORES > 1:
        params.cpu_cores = CPU_CORES
    return params


def run_single(seed: int, condition: str, training, test, output_path: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    params = build_params(condition)
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    solver.solve(output_path=str(output_path))
    with open(output_path) as f:
        result = json.load(f)

    gen_best = result["generation_best"]
    fitness_curve = [g["fitness"] for g in gen_best]
    best_fitness = fitness_curve[-1]
    convergence_gen = next(i for i, f in enumerate(fitness_curve) if f <= best_fitness + 1e-9)
    final_pop_stats = result["fitness"][-1]

    # solve() now writes per-instance feasibility straight into the JSON
    # (train_case_records, see gphh_solver.py), so prefer that instead of
    # re-evaluating just to get case_feasible back. Falls back to the old
    # recompute for result files from before this field existed, or for the
    # rare case it's null (e.g. a --multiprocess run -- see solve()'s
    # comment on why that attribute doesn't survive a worker process).
    train_case_records = result.get("train_case_records")
    if train_case_records is not None:
        train_feasible = all(r["feasible"] for r in train_case_records)
    else:
        train_recheck = solver.toolbox.evaluate(individual=solver.best_heuristic, domains=training)[0]
        train_feasible = all(solver.best_heuristic.case_feasible)
        assert abs(train_recheck - best_fitness) < 1e-6, (
            f"re-evaluating best_heuristic on training gave {train_recheck}, "
            f"expected to match the JSON's best_fitness {best_fitness}"
        )

    # this script's test set never flows through solve() itself (only
    # training_set_provider is wired up above), so there's no JSON field to
    # read for it -- still has to be evaluated directly.
    test_fitness = solver.toolbox.evaluate(individual=solver.best_heuristic, domains=test)[0]
    test_feasible = all(solver.best_heuristic.case_feasible)

    return {
        "seed": seed,
        "condition": condition,
        "best_fitness": best_fitness,
        "train_feasible": train_feasible,
        "mean_final_pop": float(final_pop_stats["avg"]),
        "std_final_pop": float(final_pop_stats["std"]),
        "convergence_gen": convergence_gen,
        "test_fitness": test_fitness,
        "test_feasible": test_feasible,
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pop_size", type=int, default=POP_SIZE)
    p.add_argument("--n_gen", type=int, default=N_GEN)
    p.add_argument("--n_seeds", type=int, default=N_SEEDS)
    p.add_argument("--seed_base", type=int, default=SEED_BASE)
    p.add_argument(
        "--n_classes", type=int, default=N_CLASSES,
        help="stratified MMLIB50 classes backing train/test (1 instance per class per "
             "split, same convention as TRAIN_CLASSES' current default of 5)",
    )
    p.add_argument("--multiprocess", action="store_true", default=False)
    p.add_argument("--cpu_cores", type=int, default=4)
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="override the results directory (default: results/nr_terminals_experiment)",
    )
    p.add_argument(
        "--dry_run", action="store_true", default=False,
        help="pre-flight check only: measure 1 seed's per-generation cost at the real "
             "--pop_size and extrapolate to --n_gen x --n_seeds x conditions, and verify "
             "the output JSON's feasibility fields. Writes to a throwaway temp dir, never "
             "to --output_dir, and never runs the real experiment.",
    )
    return p.parse_args()


def estimate_time(args) -> None:
    """Pre-flight check before committing to a much larger run than the
    defaults above. Times 1 seed at the REAL --pop_size for
    DRY_RUN_PROBE_GENS generations -- fitness evaluation cost scales with
    population size, so this has to be measured at the population size
    you're actually about to run, not extrapolated up from a smaller one
    -- then scales that up to the full --n_gen x --n_seeds x conditions
    budget. Also prints what actually came back in the output JSON's
    feasibility fields, with --multiprocess on if that's what the real run
    will use, so a wrong/empty feasibility field shows up here instead of
    after committing hours of compute to it.
    """
    global POP_SIZE, N_GEN, CPU_CORES
    classes = stratified_classes(args.n_classes)
    train_files = [f"J50{c}_1.mm" for c in classes]
    test_files = [f"J50{c}_2.mm" for c in classes]
    training = read_instances(
        [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in train_files], keep_non_renewable=True
    )
    test = read_instances(
        [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in test_files], keep_non_renewable=True
    )

    POP_SIZE = args.pop_size
    N_GEN = DRY_RUN_PROBE_GENS
    CPU_CORES = args.cpu_cores if args.multiprocess else 1

    print(
        f"\n--dry_run: timing 1 seed x {DRY_RUN_PROBE_GENS} generations per condition "
        f"at the REAL pop_size={args.pop_size} (measured here, not extrapolated up from "
        f"a smaller pop_size, since eval cost scales with population) on "
        f"{len(training)} training / {len(test)} test instances "
        f"(--n_classes={args.n_classes}), multiprocess={args.multiprocess} "
        f"(cpu_cores={CPU_CORES}), to estimate the full --n_seeds={args.n_seeds} "
        f"x --n_gen={args.n_gen} run..."
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="nr_terminals_dry_run_"))
    total_estimate = 0.0
    for condition in CONDITIONS:
        out_path = tmp_dir / f"{condition.replace('+', '_')}_probe.json"
        t0 = time.time()
        record = run_single(args.seed_base, condition, training, test, out_path)
        elapsed = time.time() - t0
        # elapsed covers gen-0 (always run) + DRY_RUN_PROBE_GENS extra
        # generations -- DRY_RUN_PROBE_GENS + 1 generation-equivalents of
        # evaluation cost, same convention as lexicase_local_search_
        # experiment.py's estimate_time.
        per_gen = elapsed / (DRY_RUN_PROBE_GENS + 1)
        condition_estimate = per_gen * args.n_gen * args.n_seeds
        total_estimate += condition_estimate
        print(
            f"  {condition:<18} ~{per_gen:.2f}s/gen -> "
            f"~{condition_estimate:.0f}s for {args.n_seeds} seeds x {args.n_gen} gens"
        )
        with open(out_path) as f:
            result = json.load(f)
        train_case_records = result.get("train_case_records")
        if train_case_records is None:
            print(
                f"    train_case_records: None in JSON (expected under --multiprocess) "
                f"-> fallback recompute fired correctly, train_feasible={record['train_feasible']}"
            )
        else:
            n_feas = sum(r["feasible"] for r in train_case_records)
            print(
                f"    train_case_records: present, {n_feas}/{len(train_case_records)} feasible "
                f"-> train_feasible={record['train_feasible']} (consistent)"
            )
        print(
            f"    test_fitness={record['test_fitness']:.4f} "
            f"test_feasible={record['test_feasible']}"
        )

    print(
        f"\n  TOTAL estimate: ~{total_estimate:.0f}s (~{total_estimate / 60:.1f} min, "
        f"~{total_estimate / 3600:.2f} h) for all {len(CONDITIONS)} conditions x "
        f"{args.n_seeds} seeds x {args.n_gen} generations.\n"
        f"  (Rough: ignores per-run fixed overhead beyond per-generation cost, and "
        f"assumes per-generation cost is ~constant across generations.)\n"
        f"  Probe output written to {tmp_dir}, not --output_dir -- nothing real touched."
    )


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    global POP_SIZE, N_GEN, N_SEEDS, SEED_BASE, CPU_CORES, OUTPUT_DIR

    if args.dry_run:
        estimate_time(args)
        return

    POP_SIZE = args.pop_size
    N_GEN = args.n_gen
    N_SEEDS = args.n_seeds
    SEED_BASE = args.seed_base
    CPU_CORES = args.cpu_cores if args.multiprocess else 1
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = stratified_classes(args.n_classes)
    train_files = [f"J50{c}_1.mm" for c in classes]
    test_files = [f"J50{c}_2.mm" for c in classes]
    training = read_instances(
        [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in train_files], keep_non_renewable=True
    )
    test = read_instances(
        [str(RCPSPDatabase.MMLIB_50_DIR + f) for f in test_files], keep_non_renewable=True
    )
    print(f"Training instances: {train_files}")
    print(f"Test instances: {test_files}")
    print(
        f"pop_size={POP_SIZE}, n_gen={N_GEN}, n_elite={N_ELITE}, "
        f"tournament={TOURNAMENT_SIZE}, cx={CROSSOVER_RATE}, mut={MUTATION_RATE}"
    )

    all_results = []
    for condition in CONDITIONS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            t0 = time.time()
            out_path = OUTPUT_DIR / f"{condition.replace('+', '_')}_seed{seed}.json"
            record = run_single(seed, condition, training, test, out_path)
            all_results.append(record)
            print(
                f"[{condition}] seed={seed} best={record['best_fitness']:.4f} "
                f"test={record['test_fitness']:.4f} conv_gen={record['convergence_gen']} "
                f"({time.time() - t0:.1f}s)"
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
    seeds = [r["seed"] for r in baseline]
    for c in CONDITIONS:
        assert [r["seed"] for r in by_condition[c]] == seeds, (
            f"Seeds must match pairwise across conditions for a paired test ({c})"
        )

    print("\n" + "=" * 80)
    print("RESULTS TABLE -- RAW (training-set fitness = % deviation from CPM lower bound)")
    print("=" * 80)
    print("WARNING: includes any infeasible-seed sentinel values uncorrected; see FEASIBILITY-FILTERED below.")
    print(f"{'Condition':<18}{'Mean':>10}{'Best':>10}{'Std':>10}{'AvgConvGen':>12}")
    for condition in CONDITIONS:
        records = by_condition[condition]
        best_vals = np.array([r["best_fitness"] for r in records], dtype=float)
        conv_vals = np.array([r["convergence_gen"] for r in records], dtype=float)
        print(
            f"{condition:<18}{best_vals.mean():>10.4f}{best_vals.min():>10.4f}"
            f"{best_vals.std():>10.4f}{conv_vals.mean():>12.2f}"
        )

    print("\n" + "=" * 80)
    print("RESULTS TABLE -- FEASIBILITY-FILTERED (uses individual.case_feasible, not a magnitude guess)")
    print("=" * 80)
    for condition in CONDITIONS:
        records = by_condition[condition]
        feas_vals = np.array([r["best_fitness"] for r in records if r["train_feasible"]], dtype=float)
        n_infeasible = len(records) - len(feas_vals)
        if len(feas_vals):
            print(
                f"{condition:<18} n_feasible={len(feas_vals)}/{len(records)} "
                f"mean={feas_vals.mean():.4f} min={feas_vals.min():.4f} std={feas_vals.std():.4f}"
            )
        else:
            print(f"{condition:<18} n_feasible=0/{len(records)} -- every seed was infeasible on training")

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (paired Wilcoxon signed-rank test on best_fitness, vs baseline)")
    print("Pairs where EITHER condition's training run was infeasible are dropped (not just baseline's),")
    print("so the comparison is never contaminated by a sentinel value from either side.")
    print("=" * 80)
    best_baseline = np.array([r["best_fitness"] for r in baseline], dtype=float)
    baseline_feasible = np.array([r["train_feasible"] for r in baseline])
    for condition in CONDITIONS:
        if condition == "baseline":
            continue
        records = by_condition[condition]
        vals = np.array([r["best_fitness"] for r in records], dtype=float)
        cond_feasible = np.array([r["train_feasible"] for r in records])
        mask = baseline_feasible & cond_feasible
        n_dropped = (~mask).sum()
        if n_dropped:
            print(f"{condition}: dropping {n_dropped}/{len(records)} seed(s) where baseline and/or {condition} was infeasible on training")
        b_f, v_f = best_baseline[mask], vals[mask]
        diffs = b_f - v_f  # positive => condition better (lower fitness)
        if len(diffs) < 1 or np.all(diffs == 0):
            print(f"{condition}: not enough informative paired differences after filtering.")
            continue
        stat, p_value = wilcoxon(b_f, v_f)
        effect_size = rank_biserial_effect_size(diffs)
        direction = f"{condition} better" if diffs.mean() > 0 else "baseline better"
        significance = "significant (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        print(
            f"{condition}: n={len(diffs)} W={stat:.4f} p={p_value:.6f} r={effect_size:.4f} "
            f"mean(baseline-{condition})={diffs.mean():.4f} ({direction}) => {significance}"
        )


if __name__ == "__main__":
    main()
