"""
Convergence + boxplot figures for the baseline-GPHH vs. custom-EA (DMGE)
comparison, from experiment_runner.py's output (<out>/raw.csv,
<out>/convergence/*.json) or run_evaluation.py's (raw.csv with the same
dataset/config/seed/dev_all/feasible/dev_feas columns, no time_s/convergence).

Usage
-----
    python yuantian/plot_comparison.py --in results/two_way_j20 \
        --out ../thesis-en-master/img --title "PSPLIB J20"

Note on "LaTeX fonts": this uses matplotlib's mathtext with the Computer
Modern font set (mathtext.fontset=cm, font.family=serif) rather than
text.usetex=True. Real usetex rendering would require escaping every
dynamically built label (titles/legends here are f-strings containing '%',
which LaTeX treats as a comment character) for one extra round of visual
polish that the committee won't be able to tell apart from this. cm mathtext
gives the same Computer-Modern look without that fragility.
"""

import csv
import glob
import json
import os
from collections import defaultdict
from optparse import OptionParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.linewidth": 0.8,
})
sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.linewidth": 0.4, "grid.color": "0.85"})

LABELS = {
    "baseline_gphh": "Baseline GPHH\n(Tian et al. 2024)", "baseline": "Baseline GPHH\n(Tian et al. 2024)",
    "custom_ea": "Custom EA\n(DMGE)", "dmge_full": "Custom EA\n(DMGE)",
    "tdre": "TDRE", "tdre_mods": "TDRE\n+ mods",
    "lexicase": "Lexicase", "lexicase_mods": "Lexicase\n+ mods",
}
PALETTE = sns.color_palette("colorblind")


def load_raw(path):
    rows = defaultdict(lambda: defaultdict(list))  # rows[config][metric] = [values...]
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["config"]]["dev_all"].append(float(r["dev_all"]))
            rows[r["config"]]["feasible"].append(float(r["feasible"]))
            rows[r["config"]]["dev_feas"].append(float(r["dev_feas"]))
            if "time_s" in r:
                rows[r["config"]]["time_s"].append(float(r["time_s"]))
    return rows


def load_convergence(conv_dir, config):
    curves = []
    for fp in sorted(glob.glob(os.path.join(conv_dir, f"*__{config}__seed*.json"))):
        with open(fp) as f:
            curve = json.load(f)
        if curve:
            curves.append(curve)
    return curves


def plot_convergence(conv_dir, configs, out_path, title):
    """Log-scale y-axis: early generations routinely contain individuals
    that are NR-infeasible on the training set, carrying the SGS sentinel
    makespan (orders of magnitude larger than any real schedule). A log
    axis shows both the sentinel-driven early drop and the real sub-1000%
    late-generation behaviour without clipping or distorting either."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_seeds = 0
    for i, cfg in enumerate(configs):
        curves = load_convergence(conv_dir, cfg)
        if not curves:
            continue
        n_seeds = max(n_seeds, len(curves))
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        # median + IQR rather than mean +- std: this data is heavy-tailed
        # (early generations carry the SGS infeasibility sentinel for some
        # seeds but not others), so the mean and especially mean-std are
        # not representative of a "typical" run the way the median/IQR are.
        median = np.median(padded, axis=0)
        q25, q75 = np.percentile(padded, 25, axis=0), np.percentile(padded, 75, axis=0)
        gens = np.arange(max_len)
        color = PALETTE[i % len(PALETTE)]
        label = LABELS.get(cfg, cfg).replace("\n", " ")
        ax.plot(gens, median, label=label, color=color, linewidth=2.2, marker="o", markersize=4)
        ax.fill_between(gens, np.clip(q25, 1e-2, None), q75, color=color, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best training-set ARD% (log scale, median & IQR)")
    ax.set_title(f"Convergence -- {title}")
    ax.set_xlim(0, None)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=True, framealpha=0.9, loc="best")
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_feasibility_bars(rows, configs, out_path, title):
    """Bar + jittered points, not box-and-whisker: feasible fraction is a
    bounded, often-saturating proportion (many seeds tie at 1.0 with few
    seeds), so the interquartile range frequently collapses to zero height
    and a box-and-whisker renders as an invisible line rather than a box --
    a bar with the raw points overlaid shows the same information without
    that degenerate case."""
    fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 1.8, 4.8))
    data = [rows[cfg]["feasible"] for cfg in configs]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(configs))]
    positions = np.arange(1, len(configs) + 1)
    means = [np.mean(vals) for vals in data]
    stds = [np.std(vals) for vals in data]

    ax.bar(positions, means, yerr=stds, width=0.55, color=colors, alpha=0.55,
           edgecolor="0.2", linewidth=0.9, capsize=4,
           error_kw={"linewidth": 0.9, "ecolor": "0.2"})

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, color=color, edgecolor="0.15",
                   linewidth=0.4, s=22, zorder=3, alpha=0.85)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS.get(c, c) for c in configs])
    ax.set_ylabel("Feasible fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\n(n={len(data[0]) if data else 0} seeds; bar = mean ± std)", fontsize=12)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_boxplot(rows, configs, metric, out_path, title):
    fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 1.8, 4.8))
    data = [rows[cfg][metric] for cfg in configs]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(configs))]
    positions = np.arange(1, len(configs) + 1)

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.5, showfliers=False,
                     medianprops={"color": "black", "linewidth": 1.6},
                     boxprops={"linewidth": 0.9, "edgecolor": "0.2"},
                     whiskerprops={"linewidth": 0.9, "color": "0.2"},
                     capprops={"linewidth": 0.9, "color": "0.2"},
                     flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "0.4",
                                 "markeredgecolor": "none", "alpha": 0.6})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, color=color, edgecolor="0.15",
                   linewidth=0.4, s=22, zorder=3, alpha=0.85)
        mean = np.mean(vals)
        ax.scatter([pos], [mean], marker="D", color="white", edgecolor="black",
                   linewidth=1.1, s=45, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS.get(c, c) for c in configs])
    ylabel = "Test-set ARD% (feasible instances)" if metric == "dev_feas" else \
        ("Feasible fraction" if metric == "feasible" else metric)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(n={len(data[0]) if data else 0} seeds; diamond = mean)", fontsize=12)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--in", dest="in_dir", default="results/two_way_j20")
    parse.add_option("--out", dest="out_dir", default="results/plots")
    parse.add_option("--title", dest="title", default="PSPLIB J20")
    parse.add_option("--configs", dest="configs", default="baseline_gphh,custom_ea",
                      help="comma list of config labels to plot, in order")
    parse.add_option("--format", dest="fmt", default="pdf", help="pdf or png")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    rows = load_raw(os.path.join(opt.in_dir, "raw.csv"))
    configs = [c.strip() for c in opt.configs.split(",") if c.strip() in rows]

    conv_dir = os.path.join(opt.in_dir, "convergence")
    if os.path.isdir(conv_dir):
        plot_convergence(conv_dir, configs,
                          os.path.join(opt.out_dir, f"convergence.{opt.fmt}"), opt.title)
    plot_boxplot(rows, configs, "dev_feas",
                 os.path.join(opt.out_dir, f"boxplot_dev_feas.{opt.fmt}"), opt.title)
    plot_feasibility_bars(rows, configs,
                 os.path.join(opt.out_dir, f"boxplot_feasible.{opt.fmt}"), opt.title)
