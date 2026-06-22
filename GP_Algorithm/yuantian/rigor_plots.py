"""
Category E (Statistical Rigor) plots, consuming raw.csv from
experiment_runner.py (results/two_way_j20_v2, results/two_way_mmlib50_v2).
Fully runnable today against existing per-seed aggregates -- no new
instrumentation needed.

Usage
-----
    python -m yuantian.rigor_plots --j20 results/two_way_j20_v2 \
        --mmlib results/two_way_mmlib50_v2 --out ../thesis-en-master/img
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


def load_paired(in_dir, flagship="custom_ea", baseline="baseline_gphh"):
    df = pd.read_csv(os.path.join(in_dir, "raw.csv"))
    piv = df.pivot_table(values="dev_all", index="seed", columns="config")
    return piv[flagship].values, piv[baseline].values


def e1_bootstrap(samples, out_path, title, n_boot=10000, seed=0):
    """E1: bootstrap distribution of the paired mean difference (flagship -
    baseline), with the observed statistic and the one-sided Wilcoxon p
    marked. samples = (flagship_vals, baseline_vals), paired by seed."""
    flagship, baseline = samples
    diffs = flagship - baseline
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                            for _ in range(n_boot)])
    observed = diffs.mean()
    _, p = stats.wilcoxon(flagship, baseline, alternative="less", zero_method="zsplit")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(boot_means, bins=50, color=PALETTE[0], alpha=0.7, edgecolor="0.3", linewidth=0.3)
    ax.axvline(observed, color="black", linewidth=2, linestyle="--",
               label=f"observed mean diff = {observed:,.0f}")
    ax.axvline(0, color="0.3", linewidth=1)
    frac_favorable = (boot_means < 0).mean()
    ax.set_title(f"{title}\n(one-sided Wilcoxon p={p:.3f}; "
                f"{frac_favorable*100:.0f}% of bootstrap mass favors DMGE)", fontsize=11)
    ax.set_xlabel("Bootstrapped mean(DMGE dev_all - baseline dev_all)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return observed, p


def e2_forest(results, out_path):
    """E2: cross-dataset standardized effect-size forest plot.
    results: dict[label] -> (flagship_vals, baseline_vals)."""
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(results) + 1.5))
    for i, (label, (flagship, baseline)) in enumerate(results.items()):
        diffs = flagship - baseline
        d = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0.0
        se = 1 / np.sqrt(len(diffs))
        ci = 1.96 * se
        ax.errorbar(d, i, xerr=ci, fmt="o", color=PALETTE[i % len(PALETTE)],
                   markersize=10, capsize=5, linewidth=2)
    ax.axvline(0, color="0.3", linewidth=1)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(list(results.keys()))
    ax.set_xlabel("Standardized effect size (Cohen's d, DMGE - baseline on dev_all)\n"
                 "negative = DMGE favorable")
    ax.invert_yaxis()
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def e3_power_curve(samples, out_path, title, max_n=100, alpha=0.05):
    """E3: post-hoc power curve. Estimates power for a one-sided paired t-test
    at the OBSERVED effect size/variance, as a function of hypothetical seed
    count, with the current n marked. (A paired t-test approximation is used
    here for closed-form power; the thesis's actual test is the nonparametric
    Wilcoxon, which has *somewhat* lower power than its parametric analogue at
    small n -- treat this curve as an optimistic upper bound, not an exact
    figure, and say so wherever it's cited.)"""
    flagship, baseline = samples
    diffs = flagship - baseline
    d = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0.0
    ns = np.arange(5, max_n + 1)
    t_crit = stats.t.ppf(1 - alpha, df=ns - 1)
    ncp = d * np.sqrt(ns)
    power = 1 - stats.nct.cdf(t_crit, df=ns - 1, nc=ncp)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ns, power, color=PALETTE[0], linewidth=2)
    ax.axhline(0.8, color="0.4", linestyle=":", linewidth=1, label="80% power")
    ax.axvline(len(diffs), color="black", linestyle="--", linewidth=1.5,
              label=f"current n = {len(diffs)}")
    ax.set_xlabel("Hypothetical seed count")
    ax.set_ylabel("Estimated power (one-sided paired test, $\\alpha$=0.05)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    # the seed count at which the curve first crosses 80% power, if any
    above = ns[power >= 0.8]
    return int(above.min()) if len(above) else None


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--j20", dest="j20_dir", default="results/two_way_j20_v2")
    parse.add_option("--mmlib", dest="mmlib_dir", default="results/two_way_mmlib50_v2")
    parse.add_option("--out", dest="out_dir", default="results/plots")
    parse.add_option("--format", dest="fmt", default="pdf")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    j20 = load_paired(opt.j20_dir)
    mmlib = load_paired(opt.mmlib_dir)

    e1_bootstrap(j20, os.path.join(opt.out_dir, f"rigor-e1-bootstrap-j20.{opt.fmt}"), "PSPLIB J20")
    e1_bootstrap(mmlib, os.path.join(opt.out_dir, f"rigor-e1-bootstrap-mmlib50.{opt.fmt}"), "MMLIB+ NR50")
    e2_forest({"PSPLIB J20": j20, "MMLIB+ NR50": mmlib},
              os.path.join(opt.out_dir, f"rigor-e2-forest.{opt.fmt}"))
    n80_j20 = e3_power_curve(j20, os.path.join(opt.out_dir, f"rigor-e3-power-j20.{opt.fmt}"), "PSPLIB J20")
    n80_mmlib = e3_power_curve(mmlib, os.path.join(opt.out_dir, f"rigor-e3-power-mmlib50.{opt.fmt}"), "MMLIB+ NR50")
    print(f"Seeds needed for ~80% power: J20={n80_j20}, MMLIB+NR50={n80_mmlib}")
