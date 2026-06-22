"""Two-panel line plot (dev_feas vs. population size, dev_feas vs. generation
count) for the parameter sensitivity sweep (param_sensitivity.py)."""
import csv
import os
from collections import defaultdict
from optparse import OptionParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titlesize": 13,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.linewidth": 0.8,
})
sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.linewidth": 0.4, "grid.color": "0.85"})
COLOR = sns.color_palette("colorblind")[0]

if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--in", dest="in_dir", default="results/param_sensitivity")
    parse.add_option("--out", dest="out_path", default="results/plots/sensitivity.pdf")
    (opt, _) = parse.parse_args()
    os.makedirs(os.path.dirname(opt.out_path), exist_ok=True)

    rows = defaultdict(list)
    with open(os.path.join(opt.in_dir, "summary.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["axis"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, axis, xlabel in zip(axes, ["pop", "gen"], ["Population size", "Generations"]):
        data = sorted(rows[axis], key=lambda r: float(r["value"]))
        x = [float(r["value"]) for r in data]
        y = [float(r["dev_feas_mean"]) for r in data]
        yerr = [float(r["dev_feas_std"]) for r in data]
        ax.errorbar(x, y, yerr=yerr, marker="o", color=COLOR, capsize=4, linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test-set ARD% (feasible instances)" if axis == "pop" else "")
        sns.despine(fig=fig, ax=ax)
    fig.suptitle("DMGE parameter sensitivity -- PSPLIB J20 (5 seeds)")
    fig.tight_layout()
    fig.savefig(opt.out_path, bbox_inches="tight")
    print(f"Saved: {opt.out_path}")
