"""
Merges the per-(dataset, config, seed) result.csv files produced by CI matrix
jobs (ci_single_run.py, one row each, downloaded into --in-dir's subfolders
by actions/download-artifact) into the usual raw.csv + summary.csv shape that
experiment_runner.py / opportunity_terminals_experiment.py write locally.

Usage
-----
    python -m yuantian.ci_merge_results --in-dir artifacts --out results/ci_matrix
"""
import csv
import glob
import os
from collections import defaultdict
from optparse import OptionParser

import numpy as np
from scipy import stats


def wilcoxon(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b, alternative="less", zero_method="zsplit")[1])
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--in-dir", dest="in_dir", default="artifacts")
    parse.add_option("--out", dest="out", default="results/ci_matrix")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(opt.in_dir, "**", "result.csv"), recursive=True)):
        with open(path) as f:
            rows.append(next(csv.DictReader(f)))

    with open(os.path.join(opt.out, "raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "seed", "dev_all", "feasible", "dev_feas", "time_s"])
        for r in rows:
            w.writerow([r["dataset"], r["config"], r["seed"], r["dev_all"], r["feasible"],
                        r["dev_feas"], r["time_s"]])

    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["dataset"], r["config"])].append(r)

    summary_rows = []
    datasets = sorted({d for d, _ in by_key})
    for dataset in datasets:
        baseline = sorted(by_key.get((dataset, "baseline_gphh"), []), key=lambda r: r["seed"])
        baseline_dev_all = [float(r["dev_all"]) for r in baseline]
        configs = sorted({c for d, c in by_key if d == dataset})
        for config in configs:
            entries = sorted(by_key[(dataset, config)], key=lambda r: r["seed"])
            dev_all = [float(r["dev_all"]) for r in entries]
            feasible = [float(r["feasible"]) for r in entries]
            dev_feas = [float(r["dev_feas"]) for r in entries]
            p = wilcoxon(dev_all, baseline_dev_all) if config != "baseline_gphh" and len(dev_all) == len(baseline_dev_all) else float("nan")
            summary_rows.append([dataset, config, np.mean(dev_all), np.std(dev_all),
                                  np.mean(feasible), np.mean(dev_feas), p])
            print(f"{dataset:15s} {config:22s} n={len(entries):>2d}  dev_feas={np.mean(dev_feas):>8.2f}  "
                  f"feasible={np.mean(feasible)*100:>5.1f}%  p_vs_baseline={p}")

    with open(os.path.join(opt.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "dev_all_mean", "dev_all_std",
                    "feasible_mean", "dev_feas_mean", "p_vs_baseline"])
        w.writerows(summary_rows)
    print(f"\nMerged {len(rows)} results -> {opt.out}/raw.csv, {opt.out}/summary.csv")
