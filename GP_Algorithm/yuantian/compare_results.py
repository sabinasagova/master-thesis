"""
Compare baseline vs. modified GPHH results.

Reads the JSON files produced by run_comparison.py (or gphh_solver.py) and
prints a summary table of training fitness, validation fitness, and test
fitness for each run and each configuration.

Usage
-----
    python yuantian/compare_results.py \
        --baseline results/baseline \
        --modified  results/modified

Output columns
--------------
  run          – run index (JSON file name without extension)
  train        – best training fitness (mean ARD% over training instances)
  val          – validation fitness of the best individual
  test_hof     – test fitness of the hall-of-fame individual
  test_val     – test fitness of the best-validated individual (if available)
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np


def load_run(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def best_fitness(data: dict) -> float:
    """Best (minimum) training fitness recorded across all generations."""
    gen_bests = [g["fitness"] for g in data.get("generation_best", [])]
    return min(gen_bests) if gen_bests else float("nan")


def val_fitness(data: dict) -> float:
    gen_bests = data.get("generation_best", [])
    vals = [g["validation_fitness"] for g in gen_bests
            if g.get("validation_fitness") is not None]
    return min(vals) if vals else float("nan")


def summarise_dir(directory: str) -> list[dict]:
    rows = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        data = load_run(os.path.join(directory, fname))
        run_id = fname.replace(".json", "")

        hof_test = (data.get("best_heuristic", {}).get("test_fitness")
                    or float("nan"))
        val_test = (data.get("best_heuristic_validation", {}).get("test_fitness")
                    or float("nan"))

        rows.append({
            "run":       run_id,
            "train":     best_fitness(data),
            "val":       val_fitness(data),
            "test_hof":  hof_test,
            "test_val":  val_test,
        })
    return rows


def print_table(label: str, rows: list[dict]):
    header = f"{'run':>5}  {'train':>10}  {'val':>10}  {'test_hof':>10}  {'test_val':>10}"
    print(f"\n── {label} {'─' * (len(header) - len(label) - 4)}")
    print(header)
    print("─" * len(header))
    for r in rows:
        print(f"{r['run']:>5}  {r['train']:>10.4f}  {r['val']:>10.4f}"
              f"  {r['test_hof']:>10.4f}  {r['test_val']:>10.4f}")

    if rows:
        trains    = [r["train"]    for r in rows if r["train"]    == r["train"]]
        test_hofs = [r["test_hof"] for r in rows if r["test_hof"] == r["test_hof"]]
        test_vals = [r["test_val"] for r in rows if r["test_val"] == r["test_val"]]
        avg = lambda lst: sum(lst) / len(lst) if lst else float("nan")
        print("─" * len(header))
        print(f"{'avg':>5}  {avg(trains):>10.4f}  {'':>10}  "
              f"{avg(test_hofs):>10.4f}  {avg(test_vals):>10.4f}")


def compare(baseline_rows: list[dict], modified_rows: list[dict]):
    """Print side-by-side delta for matched run indices."""
    b = {r["run"]: r for r in baseline_rows}
    m = {r["run"]: r for r in modified_rows}
    shared = sorted(set(b) & set(m))
    if not shared:
        return

    print(f"\n── Delta (modified − baseline, negative = improvement) "
          f"{'─' * 10}")
    header = f"{'run':>5}  {'Δtrain':>10}  {'Δtest_hof':>10}  {'Δtest_val':>10}"
    print(header)
    print("─" * len(header))
    deltas_test = []
    for run in shared:
        dt   = m[run]["train"]    - b[run]["train"]
        dh   = m[run]["test_hof"] - b[run]["test_hof"]
        dv   = m[run]["test_val"] - b[run]["test_val"]
        deltas_test.append(dh)
        print(f"{run:>5}  {dt:>+10.4f}  {dh:>+10.4f}  {dv:>+10.4f}")

    avg_delta = sum(deltas_test) / len(deltas_test) if deltas_test else float("nan")
    direction = "improvement" if avg_delta < 0 else "regression"
    print(f"\n  Average Δtest_hof: {avg_delta:+.4f}  ({direction})")


def save_plot(baseline_rows: list[dict], modified_rows: list[dict], out_path: str):
    b = {r["run"]: r for r in baseline_rows}
    m = {r["run"]: r for r in modified_rows}
    shared = sorted(set(b) & set(m))
    if not shared:
        print("No shared runs to plot.")
        return

    runs = [f"Run {r}" for r in shared]
    b_hof  = [b[r]["test_hof"]  for r in shared]
    m_hof  = [m[r]["test_hof"]  for r in shared]
    b_avg  = sum(b_hof) / len(b_hof)
    m_avg  = sum(m_hof) / len(m_hof)

    x = np.arange(len(runs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPHH: Baseline vs. Modified — test makespan deviation (%)",
                 fontsize=13, fontweight="bold")

    # ── Left: grouped bar per run ─────────────────────────────────────────
    ax = axes[0]
    bars_b = ax.bar(x - width / 2, b_hof, width, label="Baseline", color="#4C72B0")
    bars_m = ax.bar(x + width / 2, m_hof, width, label="Modified", color="#DD8452")
    ax.axhline(b_avg, color="#4C72B0", linestyle="--", linewidth=1,
               label=f"Baseline avg ({b_avg:.1f}%)")
    ax.axhline(m_avg, color="#DD8452", linestyle="--", linewidth=1,
               label=f"Modified avg ({m_avg:.1f}%)")
    ax.set_xlabel("Run")
    ax.set_ylabel("test_hof  (mean ARD%, lower = better)")
    ax.set_title("Per-run test fitness")
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.legend(fontsize=8)
    ax.bar_label(bars_b, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(bars_m, fmt="%.1f", padding=2, fontsize=8)
    ax.set_ylim(0, max(max(b_hof), max(m_hof)) * 1.25)

    # ── Right: delta per run ──────────────────────────────────────────────
    ax2 = axes[1]
    deltas = [m[r]["test_hof"] - b[r]["test_hof"] for r in shared]
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in deltas]
    bars_d = ax2.bar(runs, deltas, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    avg_delta = sum(deltas) / len(deltas)
    ax2.axhline(avg_delta, color="grey", linestyle="--", linewidth=1,
                label=f"avg Δ = {avg_delta:+.1f}%")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Δ test_hof  (modified − baseline)")
    ax2.set_title("Delta per run  (green = improvement)")
    ax2.legend(fontsize=8)

    # Manual labels so zero-delta bars stay readable inside the chart
    ax2.figure.canvas.draw()          # force layout so get_ylim() is final
    y_min, y_max = ax2.get_ylim()
    offset = (y_max - y_min) * 0.04
    for bar, d in zip(bars_d, deltas):
        x_pos = bar.get_x() + bar.get_width() / 2
        if d == 0:
            # No visible bar — place label below the zero line
            ax2.text(x_pos, -offset, "+0.0", ha="center", va="top", fontsize=8)
        elif d < 0:
            # Negative bar — place label inside near the bottom tip
            ax2.text(x_pos, d + offset, f"{d:+.1f}", ha="center", va="bottom", fontsize=8)
        else:
            # Positive bar — place label inside near the top tip
            ax2.text(x_pos, d - offset, f"{d:+.1f}", ha="center", va="top", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--baseline", dest="baseline_dir", default="results/baseline",
                     help="Directory with baseline JSON files")
    parse.add_option("--modified", dest="modified_dir", default="results/modified",
                     help="Directory with modified JSON files")
    parse.add_option("--plot", dest="plot_path", default="",
                     help="Save comparison plot to this path (e.g. results/comparison.png)")
    (options, _) = parse.parse_args()

    baseline_rows = summarise_dir(options.baseline_dir)
    modified_rows = summarise_dir(options.modified_dir)

    print_table("Baseline (Yuan Tian et al., CEC 2024)", baseline_rows)
    print_table("Modified (proposed approach)", modified_rows)
    compare(baseline_rows, modified_rows)

    if options.plot_path:
        save_plot(baseline_rows, modified_rows, options.plot_path)
