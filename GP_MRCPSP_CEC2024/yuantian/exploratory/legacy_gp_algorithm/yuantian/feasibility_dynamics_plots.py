"""
Category C (Feasibility Dynamics) plots.

C1-C3 consume results/<out>/<dataset>__<config>__feasibility_traces.json from
feasibility_dynamics.py (population-level TRAINING feasibility per
generation). C4 consumes raw.csv from experiment_runner.py directly (final
TEST-set dev_feas/feasible per seed) -- a different signal from C1-C3, and
one that needs no new instrumentation at all.

Usage
-----
    python -m yuantian.feasibility_dynamics_plots --in results/feas_dynamics_j20 \
        --raw results/two_way_j20_v2/raw.csv --out ../thesis-en-master/img --dataset PSPLIB_J20
"""
import json
import os
import glob
from optparse import OptionParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 10, "figure.dpi": 200,
    "savefig.dpi": 200, "axes.linewidth": 0.8,
})
sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.linewidth": 0.4, "grid.color": "0.85"})
PALETTE = sns.color_palette("colorblind")
LABELS = {"baseline_gphh": "Baseline GPHH", "custom_ea": "Custom EA (DMGE)"}


def load_traces(in_dir, dataset):
    out = {}
    for fp in sorted(glob.glob(os.path.join(in_dir, f"{dataset}__*__feasibility_traces.json"))):
        config = os.path.basename(fp).split("__")[1]
        with open(fp) as f:
            out[config] = json.load(f)  # list of per-seed lists
    return out


def c1_population_feasibility(traces, out_path, title):
    """C1: mean +/- IQR population feasibility fraction over generations."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (cfg, seed_traces) in enumerate(traces.items()):
        max_len = max(len(t) for t in seed_traces)
        padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in seed_traces])
        median = np.median(padded, axis=0) * 100
        q25, q75 = np.percentile(padded, 25, axis=0) * 100, np.percentile(padded, 75, axis=0) * 100
        gens = np.arange(max_len)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(gens, median, color=color, linewidth=2.2, marker="o", markersize=4,
                label=LABELS.get(cfg, cfg))
        ax.fill_between(gens, q25, q75, color=color, alpha=0.2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population feasible (%, training set)")
    ax.set_ylim(0, 102)
    ax.set_title(title, fontsize=12)
    ax.legend()
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def c2_time_to_first_feasible(traces, out_path, title):
    """C2: ECDF of generation-of-first-nonzero-population-feasibility, per config."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (cfg, seed_traces) in enumerate(traces.items()):
        first_gen = []
        for t in seed_traces:
            nonzero = [g for g, frac in enumerate(t) if frac > 0]
            first_gen.append(nonzero[0] if nonzero else len(t))  # never -> censored at run length
        vals = np.sort(first_gen)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=PALETTE[i % len(PALETTE)], linewidth=2,
                label=LABELS.get(cfg, cfg))
    ax.set_xlabel("Generation of first feasible individual")
    ax.set_ylabel("Cumulative probability across seeds")
    ax.set_title(title, fontsize=12)
    ax.legend()
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def c3_seed_generation_heatmap(traces, out_path, title):
    """C3: one heatmap per config, rows=seed, cols=generation, color=feasible fraction."""
    fig, axes = plt.subplots(1, len(traces), figsize=(6 * len(traces), 4))
    axes = np.atleast_1d(axes)
    for ax, (cfg, seed_traces) in zip(axes, traces.items()):
        max_len = max(len(t) for t in seed_traces)
        padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in seed_traces]) * 100
        sns.heatmap(padded, ax=ax, cmap="viridis", vmin=0, vmax=100, cbar=True,
                    cbar_kws={"label": "feasible %"})
        ax.set_title(LABELS.get(cfg, cfg))
        ax.set_xlabel("Generation")
        ax.set_ylabel("Seed index")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def c4_pareto_frontier(raw_csv, out_path, title):
    """C4: final test-set dev_feas vs feasible%, per seed, with a Pareto
    frontier traced over the pooled (all-config) points. Needs no new
    instrumentation -- reads experiment_runner.py's existing raw.csv."""
    df = pd.read_csv(raw_csv)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, cfg in enumerate(df["config"].unique()):
        sub = df[df.config == cfg]
        ax.scatter(sub["dev_feas"], sub["feasible"] * 100, color=PALETTE[i % len(PALETTE)],
                   edgecolor="0.2", linewidth=0.5, s=50, alpha=0.8, label=LABELS.get(cfg, cfg))

    pts = df[["dev_feas", "feasible"]].dropna().values
    pts[:, 1] *= 100
    # Pareto frontier: lower dev_feas AND higher feasible% is better:
    order = pts[pts[:, 0].argsort()]
    frontier = [order[0]]
    for p in order[1:]:
        if p[1] >= frontier[-1][1]:
            frontier.append(p)
    frontier = np.array(frontier)
    ax.plot(frontier[:, 0], frontier[:, 1], color="black", linestyle="--",
            linewidth=1.2, alpha=0.6, label="Pareto frontier (pooled)")

    ax.set_xlabel("Test-set dev_feas (%)")
    ax.set_ylabel("Test-set feasible (%)")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--in", dest="in_dir", default="results/feas_dynamics_j20")
    parse.add_option("--raw", dest="raw_csv", default="results/two_way_j20_v2/raw.csv")
    parse.add_option("--dataset", dest="dataset", default="PSPLIB_J20")
    parse.add_option("--out", dest="out_dir", default="results/plots")
    parse.add_option("--format", dest="fmt", default="pdf")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    if os.path.isdir(opt.in_dir):
        traces = load_traces(opt.in_dir, opt.dataset)
        if traces:
            c1_population_feasibility(traces, os.path.join(opt.out_dir, f"feasdyn-c1-population.{opt.fmt}"), opt.dataset)
            c2_time_to_first_feasible(traces, os.path.join(opt.out_dir, f"feasdyn-c2-time-to-feasible.{opt.fmt}"), opt.dataset)
            c3_seed_generation_heatmap(traces, os.path.join(opt.out_dir, f"feasdyn-c3-heatmap.{opt.fmt}"), opt.dataset)
        else:
            print(f"No traces found in {opt.in_dir} for dataset {opt.dataset}; skipping C1-C3.")
    else:
        print(f"{opt.in_dir} does not exist yet; skipping C1-C3.")

    if os.path.isfile(opt.raw_csv):
        c4_pareto_frontier(opt.raw_csv, os.path.join(opt.out_dir, f"feasdyn-c4-pareto.{opt.fmt}"), opt.dataset)
