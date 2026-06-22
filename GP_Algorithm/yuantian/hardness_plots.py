"""
Category A (Instance Hardness) plots, consuming results/<out>/per_instance.csv
from instance_hardness.py.

Usage
-----
    python -m yuantian.hardness_plots --in results/hardness_j20 --out ../thesis-en-master/img
"""
import os
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


def load(in_dir):
    df = pd.read_csv(os.path.join(in_dir, "per_instance.csv"))
    df["dev_feas"] = df["dev"].where(df["is_feasible"])
    return df


def a1_ard_vs_rs_nr(df, out_path, n_bins=4):
    """A1: binned line + ribbon, dev_feas vs resource_strength_nr bin, by config."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df = df.copy()
    df["rs_bin"] = pd.qcut(df["resource_strength_nr"], n_bins, duplicates="drop")
    for i, cfg in enumerate(df["config"].unique()):
        sub = df[df["config"] == cfg]
        grp = sub.groupby("rs_bin")["dev_feas"].agg(["mean", "std", "count"])
        x = np.arange(len(grp))
        color = PALETTE[i % len(PALETTE)]
        ax.errorbar(x, grp["mean"], yerr=grp["std"], marker="o", color=color,
                    capsize=4, linewidth=2, label=cfg)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([str(iv) for iv in grp.index], rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("Resource Strength (NR), binned -- low = tighter")
    ax.set_ylabel("dev_feas (%)")
    ax.legend()
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def a2_feasibility_gap_heatmap(df, out_path, config_a, config_b, n_bins=4):
    """A2: heatmap of feasible%(config_a) - feasible%(config_b) over OS x RF_NR
    grid. Degrades to a 1-row heatmap (OS only) if RF_NR is constant across
    the instance pool -- pd.qcut on a constant column with duplicates="drop"
    returns zero bins rather than one, which would otherwise crash here."""
    df = df.copy()
    df["os_bin"] = pd.qcut(df["order_strength"], n_bins, duplicates="drop")
    if df["resource_factor_nr"].nunique() > 1:
        df["rf_bin"] = pd.qcut(df["resource_factor_nr"], n_bins, duplicates="drop")
    else:
        print(f"  note: resource_factor_nr is constant "
              f"({df['resource_factor_nr'].iloc[0]:.3f}) across this instance "
              f"pool -- A2 degrades to Order Strength only.")
        df["rf_bin"] = "(constant)"
    piv_a = df[df.config == config_a].pivot_table(
        values="is_feasible", index="rf_bin", columns="os_bin", aggfunc="mean")
    piv_b = df[df.config == config_b].pivot_table(
        values="is_feasible", index="rf_bin", columns="os_bin", aggfunc="mean")
    gap = (piv_a - piv_b) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(gap, annot=True, fmt=".0f", cmap="RdBu", center=0, ax=ax,
                cbar_kws={"label": f"feasible% ({config_a}) - feasible% ({config_b})"})
    ax.set_xlabel("Order Strength bin")
    ax.set_ylabel("Resource Factor (NR) bin")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def a3_ecdf_by_tercile(df, out_path, n_terciles=3):
    """A3: ECDF of dev_feas, faceted by RS_NR tercile, lines per config."""
    df = df.copy()
    cut = pd.qcut(df["resource_strength_nr"], n_terciles, duplicates="drop")
    # duplicates="drop" can return fewer bins than n_terciles when the
    # underlying instance set has few distinct RS(NR) values (e.g. a small
    # smoke-test sample) -- label whatever bins actually resulted, in order,
    # rather than assuming exactly 3.
    names = ["tight", "medium", "loose", "looser", "loosest"]
    code_to_name = {code: names[i] for i, code in enumerate(cut.cat.categories)}
    df["rs_tercile"] = cut.cat.rename_categories(code_to_name)
    terciles = list(df["rs_tercile"].cat.categories)
    fig, axes = plt.subplots(1, len(terciles), figsize=(4.5 * len(terciles), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tercile in zip(axes, terciles):
        sub = df[df["rs_tercile"] == tercile]
        for i, cfg in enumerate(sub["config"].unique()):
            vals = sub[sub.config == cfg]["dev_feas"].dropna().sort_values()
            if vals.empty:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", color=PALETTE[i % len(PALETTE)], label=cfg, linewidth=2)
        ax.set_title(f"RS(NR) = {tercile}")
        ax.set_xlabel("dev_feas (%)")
        sns.despine(fig=fig, ax=ax)
    axes[0].set_ylabel("Cumulative probability")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def a5_advantage_scatter(df, out_path, config_a, config_b):
    """A5: per-instance advantage (config_b dev_feas - config_a dev_feas) vs
    resource_factor_nr, restricted to pairs where BOTH configs produced a
    feasible schedule (dev_feas is NaN otherwise) -- using raw dev here would
    let the SGS infeasibility sentinel (orders of magnitude larger than any
    real ARD%) swamp the real signal."""
    piv = df.pivot_table(values="dev_feas", index=["dataset", "instance_id", "seed"],
                          columns="config", aggfunc="first").dropna(subset=[config_a, config_b])
    hardness = df.drop_duplicates(["dataset", "instance_id"]).set_index(["dataset", "instance_id"])
    piv = piv.reset_index()
    piv["advantage"] = piv[config_b] - piv[config_a]  # negative = config_a better
    piv = piv.merge(hardness[["order_strength", "resource_factor_nr", "resource_strength_nr"]],
                     on="instance_id", how="left")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(piv["resource_factor_nr"], piv["advantage"], color=PALETTE[0],
               edgecolor="0.2", linewidth=0.4, alpha=0.7, s=30)
    order = piv["resource_factor_nr"].argsort()
    if len(piv) > 5:
        z = np.polyfit(piv["resource_factor_nr"], piv["advantage"], 1)
        xs = np.linspace(piv["resource_factor_nr"].min(), piv["resource_factor_nr"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="black", linestyle="--", linewidth=1.5,
                label="linear trend")
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("Resource Factor (NR)")
    ax.set_ylabel(f"dev[{config_b}] - dev[{config_a}]  (negative = {config_a} better)")
    ax.legend()
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--in", dest="in_dir", default="results/hardness_j20")
    parse.add_option("--out", dest="out_dir", default="results/plots")
    parse.add_option("--config-a", dest="config_a", default="custom_ea")
    parse.add_option("--config-b", dest="config_b", default="baseline_gphh")
    parse.add_option("--format", dest="fmt", default="pdf")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    df = load(opt.in_dir)
    a1_ard_vs_rs_nr(df, os.path.join(opt.out_dir, f"hardness-a1-rs-nr.{opt.fmt}"))
    a2_feasibility_gap_heatmap(df, os.path.join(opt.out_dir, f"hardness-a2-gap-heatmap.{opt.fmt}"),
                                opt.config_a, opt.config_b)
    a3_ecdf_by_tercile(df, os.path.join(opt.out_dir, f"hardness-a3-ecdf.{opt.fmt}"))
    a5_advantage_scatter(df, os.path.join(opt.out_dir, f"hardness-a5-advantage.{opt.fmt}"),
                         opt.config_a, opt.config_b)
