"""
Read every result JSON under a matrix output directory and produce:

  (a) the paper-baseline reproduction report -- our reproduced "baseline"
      condition vs the paper's published Tables V (serial) / VI (parallel)
      numbers and its manual-rule baseline, per dataset x strategy x sgs.
  (b) extension-comparison tables -- each non-baseline condition vs
      baseline: feasibility-filtered mean fitness, std, feasibility rate,
      paired Wilcoxon, formatted to sit next to the paper's own table
      layout.

    python -m yuantian.experiments.analyze_matrix --results_dir results/matrix/

Feasibility handling (train and test): every result JSON carries per-
instance {instance, fitness (None if infeasible), feasible} records
(train_case_records at the top level, test_case_records under
best_heuristic / best_heuristic_validation -- see gphh_solver.py). For each
seed's run, this script computes a FEASIBLE-ONLY mean (averaging only the
instances that were actually feasible within that run) rather than the
coarser all-instances-must-be-feasible-or-drop-the-whole-seed convention
used elsewhere in this repo (e.g. nr_terminals_experiment.py): per-instance
feasible-mean per seed, not a whole-seed drop, so a single infeasible
instance doesn't asymmetrically discard an entire seed's data. A seed is
only excluded entirely if it has ZERO feasible instances (nothing to
average). Paired
Wilcoxon comparisons additionally drop a (baseline, condition) pair if
EITHER side has zero feasible instances for that seed -- same "drop pairs
contaminated on either side" principle the rest of this repo already uses,
just applied at the finer per-instance level now that case_records makes
that possible instead of only a whole-run feasible/infeasible flag.

IMPORTANT instance-set caveat: the "nr" condition runs on NR-preserving
instances, every other condition (including "baseline") runs on the
paper's renewable-only-converted instances (see matrix_runner.py's module
docstring). So "nr" is NOT a same-instance paired comparison against this
matrix's "baseline" row -- this script prints it as its own block with
that caveat stated, not silently folded into the main extension-comparison
table.

"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.experiments.matrix_config import (
    CELL_EXCLUSIONS,
    CONDITIONS,
    DATASETS,
    PAPER_TABLE_V_SERIAL,
    PAPER_TABLE_VI_PARALLEL,
    SGS_TYPES,
    STRATEGIES,
)


def rank_biserial_effect_size(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    return float((r_plus - r_minus) / ranks.sum())


def load_results(results_dir: Path) -> list:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        with open(path) as f:
            r = json.load(f)
        if "matrix_cell" not in r:
            continue  # not one of ours (e.g. a stray non-matrix result file)
        results.append(r)
    return results


def feasible_mean(case_records: list) -> float:
    """Mean fitness over only the feasible instances in case_records, or
    None if every instance in this run was infeasible (nothing to average).
    """
    if not case_records:
        return None
    feas = [r["fitness"] for r in case_records if r["feasible"]]
    return float(np.mean(feas)) if feas else None


def feasibility_rate(case_records: list) -> float:
    if not case_records:
        return float("nan")
    return float(np.mean([r["feasible"] for r in case_records]))


def per_seed_summary(result: dict) -> dict:
    """One row per result file: seed, train feasible-mean + rate, test
    feasible-mean + rate (test_case_records lives under best_heuristic,
    which is only present when the run had validation/test providers --
    true for every matrix_runner.py cell)."""
    cell = result["matrix_cell"]
    train_records = result.get("train_case_records")
    best = result.get("best_heuristic") or {}
    test_records = best.get("test_case_records")
    return {
        **cell,
        "train_mean": feasible_mean(train_records),
        "train_feas_rate": feasibility_rate(train_records),
        "test_mean": feasible_mean(test_records),
        "test_feas_rate": feasibility_rate(test_records),
    }


def group_by(rows: list, *keys) -> dict:
    grouped = {}
    for r in rows:
        grouped.setdefault(tuple(r[k] for k in keys), []).append(r)
    return grouped


def paired_wilcoxon(base_rows: list, cond_rows: list, metric: str):
    """Pair by seed, drop a pair if either side's metric is None (zero
    feasible instances that seed), return (n, p, r, mean_diff) or None if
    fewer than 1 informative pair remains."""
    base_by_seed = {r["seed"]: r[metric] for r in base_rows}
    cond_by_seed = {r["seed"]: r[metric] for r in cond_rows}
    common_seeds = sorted(set(base_by_seed) & set(cond_by_seed))
    pairs = [
        (base_by_seed[s], cond_by_seed[s]) for s in common_seeds
        if base_by_seed[s] is not None and cond_by_seed[s] is not None
    ]
    if len(pairs) < 1:
        return None
    b = np.array([p[0] for p in pairs])
    c = np.array([p[1] for p in pairs])
    diffs = b - c  # positive => condition better (lower fitness)
    if np.all(diffs == 0):
        return dict(n=len(pairs), p=None, r=0.0, mean_diff=0.0, n_dropped=len(common_seeds) - len(pairs))
    stat, p_value = wilcoxon(b, c)
    return dict(
        n=len(pairs), p=p_value, r=rank_biserial_effect_size(diffs),
        mean_diff=float(diffs.mean()), n_dropped=len(common_seeds) - len(pairs),
    )


def reproduction_report(rows: list, emit):
    emit("=" * 100)
    emit("PAPER-BASELINE REPRODUCTION REPORT (our 'baseline' condition vs paper Tables V/VI)")
    emit("=" * 100)
    emit("Manual-rule reference is the paper's strongest hand-crafted heuristic for that dataset")
    emit("(name varies by dataset/SGS -- shown per row). Paper GP column is mean+-std over 30 runs.\n")

    baseline_rows = [r for r in rows if r["condition"] == "baseline"]
    by_sgs_ds_strat = group_by(baseline_rows, "sgs", "dataset", "strategy")

    col = "{:<16} {:<9} {:>22} {:>14} {:>9} {:>8} {:>8} {:>3}"
    for sgs, paper_table in (("serial", PAPER_TABLE_V_SERIAL), ("parallel", PAPER_TABLE_VI_PARALLEL)):
        emit(f"--- SGS = {sgs} ({'Table V' if sgs == 'serial' else 'Table VI'}) ---")
        emit(col.format("Dataset", "Strategy", "Manual rule", "Paper GP", "Our mean", "Our std", "Diff", "n"))
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                manual_name, manual_val, paper_val, paper_std = paper_table[dataset][strategy]
                cell_rows = by_sgs_ds_strat.get((sgs, dataset, strategy), [])
                train_means = [r["train_mean"] for r in cell_rows if r["train_mean"] is not None]
                manual_str = f"{manual_val:.2f} ({manual_name})"
                paper_str = f"{paper_val:.2f}+-{paper_std:.2f}"
                if train_means:
                    our_mean = float(np.mean(train_means))
                    our_std = float(np.std(train_means))
                    diff_str = f"{our_mean - paper_val:+.2f}"
                    emit(col.format(
                        dataset, strategy, manual_str, paper_str,
                        f"{our_mean:.2f}", f"{our_std:.2f}", diff_str, str(len(train_means)),
                    ))
                else:
                    emit(col.format(dataset, strategy, manual_str, paper_str, "no data yet", "", "", ""))
        emit("")


def extension_comparison_report(rows: list, emit):
    emit("=" * 100)
    emit("EXTENSION COMPARISON (each condition vs baseline, feasibility-filtered)")
    emit("=" * 100)
    emit("Per-instance feasible-mean per seed (see module docstring) feeds both the table and the")
    emit("paired Wilcoxon test below -- a seed only drops out entirely if it has zero feasible instances.\n")

    for split, metric in (("train", "train_mean"), ("test", "test_mean")):
        emit(f"--- {split} fitness ---")
        by_cell = group_by(rows, "dataset", "sgs", "strategy")
        for (dataset, sgs, strategy), cell_rows in sorted(by_cell.items()):
            baseline_rows = [r for r in cell_rows if r["condition"] == "baseline"]
            if not baseline_rows:
                continue
            emit(f"\n  {dataset} / {sgs} / {strategy}")
            emit(f"  {'Condition':<14}{'n_feas/n':>10}{'Mean':>10}{'Std':>10}{'FeasRate':>10}  vs baseline")
            for condition in CONDITIONS:
                if (condition, strategy, sgs, dataset) in CELL_EXCLUSIONS:
                    emit(f"  {condition:<14}  [excluded from this matrix -- see matrix_config.CELL_EXCLUSIONS, not a missing/pending run]")
                    continue
                cond_rows = [r for r in cell_rows if r["condition"] == condition]
                if not cond_rows:
                    continue
                vals = [r[metric] for r in cond_rows if r[metric] is not None]
                feas_rate = float(np.mean([r[f"{split}_feas_rate"] for r in cond_rows]))
                if condition == "nr":
                    note = "[different instance set, see caveat below -- not in vs-baseline column]"
                elif condition == "baseline":
                    note = "--"
                else:
                    cmp = paired_wilcoxon(baseline_rows, cond_rows, metric)
                    if cmp is None:
                        note = "not enough paired data"
                    elif cmp["p"] is None:
                        note = f"n={cmp['n']} all differences zero"
                    else:
                        sig = "significant" if cmp["p"] < 0.05 else "not significant"
                        note = f"n={cmp['n']} p={cmp['p']:.4f} r={cmp['r']:.3f} ({sig})"
                mean_str = f"{np.mean(vals):.2f}" if vals else "n/a"
                std_str = f"{np.std(vals):.2f}" if vals else "n/a"
                emit(
                    f"  {condition:<14}{len(vals):>5}/{len(cond_rows):<4}{mean_str:>10}{std_str:>10}"
                    f"{feas_rate:>10.2f}  {note}"
                )
        emit("")

    emit("--- 'nr' condition caveat ---")
    emit(
        "The 'nr' condition runs on NR-preserving instances (keep_non_renewable=True); every "
        "other condition above (including this matrix's 'baseline' row) runs on the paper's "
        "renewable-only-converted instances. They are NOT the same problem instances, so 'nr' "
        "vs this matrix's 'baseline' is not a valid same-instance paired comparison -- see "
        "nr_terminals_experiment.py for the dedicated, instance-matched nr-vs-baseline comparison."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=str, default="yuantian/experiments/results/matrix")
    p.add_argument("--out", type=str, default=None, help="also write the report text to this path")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    raw_results = load_results(results_dir)
    if not raw_results:
        print(f"No matrix result files found under {results_dir}")
        return
    rows = [per_seed_summary(r) for r in raw_results]
    print(f"Loaded {len(rows)} matrix result files from {results_dir}\n")

    report_lines = []

    def emit(line=""):
        print(line)
        report_lines.append(line)

    reproduction_report(rows, emit)
    extension_comparison_report(rows, emit)

    if args.out:
        Path(args.out).write_text("\n".join(report_lines))
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
