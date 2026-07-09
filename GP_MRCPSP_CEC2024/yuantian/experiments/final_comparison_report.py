"""
Cross-condition ranking for the "final comparison" table/CD-diagram used in
the thesis's experimental chapter: baseline vs lexicase vs local_search vs
hybrid (the "nr" condition is deliberately excluded here -- see
analyze_matrix.py's docstring on why it runs on a different, NR-preserving
instance set and is never a same-instance comparison against these four).

Per (sgs, strategy) cell, each condition's 30-seed mean test/train fitness
is one Demsar-style "problem" observation; conditions are ranked 1 (best,
lowest ARD%) to 4 (worst) within each cell, ranks are averaged across the 6
cells, and a Friedman test (blocks = the 6 cells, k = 4 conditions) checks
whether the average-rank differences are unlikely under the null of no
systematic difference between conditions.

    python -m yuantian.experiments.final_comparison_report --results_dir results/matrix/ --glob "MMLIB50__*.json"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yuantian.experiments.analyze_matrix import group_by, load_summaries

CONDITIONS = ["baseline", "lexicase", "local_search", "hybrid"]


def cell_means(rows: list, metric: str) -> dict:
    """(sgs, strategy) -> {condition: mean(metric) over that cell's 30 seeds}."""
    by_cell = group_by(rows, "sgs", "strategy")
    out = {}
    for cell, cell_rows in sorted(by_cell.items()):
        means = {}
        for c in CONDITIONS:
            vals = [r[metric] for r in cell_rows if r["condition"] == c and r[metric] is not None]
            if vals:
                means[c] = float(np.mean(vals))
        if len(means) == len(CONDITIONS):
            out[cell] = means
    return out


def average_ranks(means_by_cell: dict) -> dict:
    """condition -> mean rank across cells (1 = best/lowest fitness)."""
    ranks_per_cell = []
    for means in means_by_cell.values():
        order = sorted(CONDITIONS, key=lambda c: means[c])
        ranks_per_cell.append({c: order.index(c) + 1 for c in CONDITIONS})
    return {c: float(np.mean([r[c] for r in ranks_per_cell])) for c in CONDITIONS}


def report(rows: list, emit):
    for split, metric in (("test", "test_mean"), ("train", "train_mean")):
        emit("=" * 100)
        emit(f"FINAL COMPARISON -- {split} fitness, baseline vs lexicase vs local_search vs hybrid")
        emit("=" * 100)
        means = cell_means(rows, metric)
        if not means:
            emit("(no cell has all four conditions with data -- skipping)\n")
            continue
        for cell, m in means.items():
            emit(f"  {cell}: " + ", ".join(f"{c}={m[c]:.2f}" for c in CONDITIONS))
        ranks = average_ranks(means)
        emit("\nAverage rank across cells (1=best/lowest ARD%, 4=worst):")
        for c in sorted(CONDITIONS, key=lambda c: ranks[c]):
            emit(f"  {c:<14} {ranks[c]:.2f}")
        data = [[means[cell][c] for cell in means] for c in CONDITIONS]
        if len(means) >= 3:
            stat, p = friedmanchisquare(*data)
            emit(f"\nFriedman chi2={stat:.4f} p={p:.4f} (blocks={len(means)} sgs x strategy cells, k={len(CONDITIONS)} conditions)")
        else:
            emit("\n(fewer than 3 cells with complete data -- Friedman test skipped)")
        emit("")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=str, default="yuantian/experiments/results/matrix")
    p.add_argument("--glob", type=str, default="*.json")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    rows = load_summaries(Path(args.results_dir), args.glob)
    if not rows:
        print(f"No matrix result files found under {args.results_dir} matching {args.glob!r}")
        return
    print(f"Loaded {len(rows)} matrix result files\n")

    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    report(rows, emit)

    if args.out:
        Path(args.out).write_text("\n".join(lines))
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
