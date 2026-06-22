"""
Category B (Ablation & Mechanics) plots, consuming results/nr50_powered/raw.csv
(produced by run_evaluation.py's CONFIGS: baseline, mods_standard, dmge_full,
dmge_no_nr, dmge_no_cp, dmge_no_renew, dmge_nograft). Fully runnable today --
no new instrumentation needed, unlike hardness_plots.py.

Usage
-----
    python -m yuantian.ablation_plots --in results/nr50_powered --out ../thesis-en-master/img
"""
import os
from optparse import OptionParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 10, "figure.dpi": 200,
    "savefig.dpi": 200, "axes.linewidth": 0.8,
})
sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.linewidth": 0.4, "grid.color": "0.85"})
PALETTE = sns.color_palette("colorblind")

# Ordered nograft -> full so the waterfall reads as "grafts switched on one at a time"
ORDER = ["baseline", "mods_standard", "dmge_nograft", "dmge_no_renew",
         "dmge_no_cp", "dmge_no_nr", "dmge_full"]
LABELS = {
    "baseline": "Baseline GPHH", "mods_standard": "+ terminals,\nstandard driver",
    "dmge_nograft": "DMGE driver,\nno grafts", "dmge_no_renew": "+ NR, CP\n(no RENEWABLE)",
    "dmge_no_cp": "+ NR, RENEWABLE\n(no CP)", "dmge_no_nr": "+ CP, RENEWABLE\n(no NR)",
    "dmge_full": "DMGE\n(all 3 grafts)",
}


def load(in_dir):
    df = pd.read_csv(os.path.join(in_dir, "raw.csv"))
    return df[df["config"].isin(ORDER)]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def b1_waterfall(df, out_path):
    """B1: horizontal forest plot of mean dev_feas +/- 95% CI per ablation step."""
    order = [c for c in ORDER if c in df["config"].unique()]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(order) + 1.5))
    for i, cfg in enumerate(order):
        vals = df[df.config == cfg]["dev_feas"].dropna()
        mean = vals.mean()
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.errorbar(mean, i, xerr=ci, fmt="o", color=PALETTE[i % len(PALETTE)],
                   markersize=8, capsize=5, linewidth=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS.get(c, c) for c in order])
    ax.set_xlabel("dev_feas (%), mean $\\pm$ 95% CI")
    ax.invert_yaxis()
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def b2_slopegraph(df, out_path):
    """B2: seed-level connected lines across the ablation sequence."""
    order = [c for c in ORDER if c in df["config"].unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = df.pivot_table(values="dev_feas", index="seed", columns="config")[order]
    for seed, row in piv.iterrows():
        ax.plot(range(len(order)), row.values, color="0.6", alpha=0.5, linewidth=1, marker="o", markersize=3)
    mean_row = piv.mean()
    ax.plot(range(len(order)), mean_row.values, color=PALETTE[3], linewidth=3,
            marker="o", markersize=8, label="mean across seeds", zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(c, c) for c in order], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("dev_feas (%)")
    ax.legend()
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def b3_feasibility_wilson(df, out_path, n_test_instances):
    """B3: feasibility rate with Wilson 95% CI, per ablation step. n_test_instances
    is needed because raw.csv's 'feasible' column is already a per-seed fraction,
    not a count -- pass the test-set size used for that run to recover k."""
    order = [c for c in ORDER if c in df["config"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, cfg in enumerate(order):
        fracs = df[df.config == cfg]["feasible"].dropna()
        mean_frac = fracs.mean()
        k = round(mean_frac * n_test_instances * len(fracs))
        n = n_test_instances * len(fracs)
        lo, hi = wilson_ci(k, n)
        ax.bar(i, mean_frac * 100, color=PALETTE[i % len(PALETTE)], alpha=0.7,
               edgecolor="0.2", linewidth=0.9)
        ax.errorbar(i, mean_frac * 100, yerr=[[mean_frac * 100 - lo * 100],
                    [hi * 100 - mean_frac * 100]], fmt="none", color="black", capsize=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(c, c) for c in order], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Feasible (%), Wilson 95% CI")
    ax.set_ylim(0, 105)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def b4_attribution_heatmap(df, out_path):
    """B4: standardized effect of removing each graft, relative to dmge_full,
    on dev_feas and feasible."""
    grafts = {"dmge_no_nr": "NR removed", "dmge_no_cp": "CP removed",
              "dmge_no_renew": "RENEWABLE removed"}
    full = df[df.config == "dmge_full"]
    rows = []
    for cfg, label in grafts.items():
        sub = df[df.config == cfg]
        for metric in ["dev_feas", "feasible"]:
            a, b = full[metric].dropna(), sub[metric].dropna()
            pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            cohens_d = (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0
            rows.append({"graft": label, "metric": metric, "cohens_d": cohens_d})
    piv = pd.DataFrame(rows).pivot(index="graft", columns="metric", values="cohens_d")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Cohen's d vs. dmge_full (ablation - full)"})
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def b5_operator_vs_terminal(df, out_path):
    """B5: baseline vs mods_standard vs dmge_full -- isolates terminal-only
    gains (mods_standard) from the graft operator's additional contribution
    (dmge_full vs mods_standard)."""
    order = ["baseline", "mods_standard", "dmge_full"]
    order = [c for c in order if c in df["config"].unique()]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    data = [df[df.config == c]["dev_feas"].dropna() for c in order]
    means = [d.mean() for d in data]
    sems = [d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0 for d in data]
    ax.bar(range(len(order)), means, yerr=sems, capsize=5,
           color=[PALETTE[i] for i in range(len(order))], alpha=0.75, edgecolor="0.2")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Baseline\n(operator + terminals)", "+ terminals only\n(standard driver)",
                        "+ graft operator\n(DMGE)"][:len(order)])
    ax.set_ylabel("dev_feas (%), mean $\\pm$ SEM")
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--in", dest="in_dir", default="results/nr50_powered")
    parse.add_option("--out", dest="out_dir", default="results/plots")
    parse.add_option("--n-test", dest="n_test", type="int", default=16,
                      help="test-set size used for that run, for B3's Wilson CI")
    parse.add_option("--format", dest="fmt", default="pdf")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    df = load(opt.in_dir)
    b1_waterfall(df, os.path.join(opt.out_dir, f"ablation-b1-waterfall.{opt.fmt}"))
    b2_slopegraph(df, os.path.join(opt.out_dir, f"ablation-b2-slopegraph.{opt.fmt}"))
    b3_feasibility_wilson(df, os.path.join(opt.out_dir, f"ablation-b3-wilson.{opt.fmt}"), opt.n_test)
    b4_attribution_heatmap(df, os.path.join(opt.out_dir, f"ablation-b4-attribution.{opt.fmt}"))
    b5_operator_vs_terminal(df, os.path.join(opt.out_dir, f"ablation-b5-operator-vs-terminal.{opt.fmt}"))
