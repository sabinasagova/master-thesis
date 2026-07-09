"""
Generate the boxplot and critical-difference-diagram figures referenced by
chap05.tex's [BOXPLOT ...] / [CRITICAL DIFFERENCE DIAGRAM ...] placeholders,
directly from the raw per-seed result JSONs (no numbers are hand-entered).

    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 \\
        yuantian/experiments/generate_thesis_plots.py \\
        --results_dir yuantian/experiments/results/matrix \\
        --out_dir ../thesis-en-master/img/generated
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CELLS = [("parallel", "AF"), ("parallel", "MF"), ("parallel", "S"),
         ("serial", "AF"), ("serial", "MF"), ("serial", "S")]
CELL_LABELS = [f"{sgs[:4]}/{strat}" for sgs, strat in CELLS]

# Nemenyi critical-value table (Demsar 2006, alpha=0.05), q_alpha by k (# conditions)
Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def rank_biserial(diffs):
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    return float((r_plus - r_minus) / ranks.sum())


def load_raw(results_dir: Path, glob: str, conditions: set):
    """dict: (sgs, strategy, condition) -> {"test": [30 seed means], "train": [...]}"""
    out = {}
    for path in sorted(results_dir.glob(glob)):
        with open(path) as f:
            r = json.load(f)
        if "matrix_cell" not in r:
            continue
        cell = r["matrix_cell"]
        cond = cell["condition"]
        if cond not in conditions:
            del r
            continue
        key = (cell["sgs"], cell["strategy"], cond)
        train_records = r.get("train_case_records") or []
        best = r.get("best_heuristic") or {}
        test_records = best.get("test_case_records") or []
        train_feas = [x["fitness"] for x in train_records if x["feasible"]]
        test_feas = [x["fitness"] for x in test_records if x["feasible"]]
        train_mean = float(np.mean(train_feas)) if train_feas else None
        test_mean = float(np.mean(test_feas)) if test_feas else None
        out.setdefault(key, {"test": [], "train": []})
        if test_mean is not None:
            out[key]["test"].append(test_mean)
        if train_mean is not None:
            out[key]["train"].append(train_mean)
        del r
    return out


def boxplot_grid(data, conditions, colors, title, ylabel, out_path, metric="test", ncols=3):
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 7.5), sharey=False)
    axes = axes.flatten()
    for i, (sgs, strategy) in enumerate(CELLS):
        ax = axes[i]
        box_data = [data.get((sgs, strategy, c), {}).get(metric, []) for c in conditions]
        bp = ax.boxplot(box_data, labels=conditions, patch_artist=True, widths=0.6,
                         showmeans=False, flierprops=dict(marker="o", markersize=3, alpha=0.6))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_title(f"{sgs}/{strategy}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.tick_params(axis="y", labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    for j in range(len(CELLS), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def effect_size_plot(data, baseline_cond, other_cond, title, out_path):
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ys = np.arange(len(CELLS))
    rs, sigs = [], []
    for sgs, strategy in CELLS:
        base = np.array(data.get((sgs, strategy, baseline_cond), {}).get("test", []))
        cond = np.array(data.get((sgs, strategy, other_cond), {}).get("test", []))
        n = min(len(base), len(cond))
        base, cond = base[:n], cond[:n]
        diffs = base - cond  # positive => cond better (lower ARD%)
        if n == 0 or np.all(diffs == 0):
            rs.append(0.0); sigs.append(False); continue
        try:
            _, p = wilcoxon(base, cond)
        except ValueError:
            p = 1.0
        rs.append(rank_biserial(diffs))
        sigs.append(p < 0.05)
    colors = ["#1f77b4" if s else "#bbbbbb" for s in sigs]
    ax.hlines(ys, 0, rs, color=colors, linewidth=2, zorder=1)
    ax.scatter(rs, ys, c=colors, s=70, zorder=2,
               edgecolors="black", linewidths=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(CELL_LABELS, fontsize=9)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(f"rank-biserial $r$  (favours {other_cond} $\\rightarrow$)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                          markeredgecolor="black", label="p < 0.05 (significant)"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb",
                          markeredgecolor="black", label="not significant")]
    ax.legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def cd_diagram(avg_ranks: dict, n_blocks: int, out_path, title):
    conditions = list(avg_ranks.keys())
    k = len(conditions)
    q_alpha = Q_ALPHA_005[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    order = sorted(conditions, key=lambda c: avg_ranks[c])
    ranks = [avg_ranks[c] for c in order]

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([lo, hi], [0.75, 0.75], color="black", linewidth=1)
    for tick in range(lo, hi + 1):
        ax.plot([tick, tick], [0.72, 0.78], color="black", linewidth=1)
        ax.text(tick, 0.82, str(tick), ha="center", fontsize=8)
    ax.text((lo + hi) / 2, 0.86, "average rank (1 = best, "
            f"{hi} = worst)", ha="center", fontsize=8, style="italic")

    # stagger label depth when two ranks are close enough that their text
    # would otherwise overlap, instead of overlapping "baselinelexicase"-style
    label_y = 0.48
    prev_r = None
    depth = 0
    for cond, r in zip(order, ranks):
        if prev_r is not None and abs(r - prev_r) < 0.35:
            depth += 1
        else:
            depth = 0
        prev_r = r
        this_y = label_y - depth * 0.16
        ax.plot([r, r], [0.75, this_y + 0.05], color="black", linewidth=0.8)
        ax.plot(r, 0.75, marker="o", color="#1f77b4", markersize=5, zorder=3)
        ax.text(r, this_y, f"{cond} ({r:.2f})", ha="center", va="top", fontsize=8)

    # CD reference bar: a ruler showing how long a "critical difference" is,
    # anchored at the best (leftmost, rank 1) end of the axis -- it is a
    # scale reference, not a claim that ranks 1..(1+CD) are themselves a group.
    cd_y = 0.94
    ax.annotate("", xy=(lo + cd, cd_y), xytext=(lo, cd_y),
                arrowprops=dict(arrowstyle="<->", color="red", linewidth=1.6))
    ax.text(lo + cd / 2, cd_y + 0.05, f"critical difference, CD = {cd:.2f}",
            ha="center", fontsize=8, color="red")

    # Thick grey bars connect every GROUP of conditions whose average ranks
    # are mutually within one CD of each other. Conditions tied (or nearly
    # tied) on rank are merged into a single point first, so two conditions
    # sharing the same rank don't each spawn their own near-duplicate bar.
    unique_ranks = []
    for r in ranks:
        if not unique_ranks or r - unique_ranks[-1] > 1e-9:
            unique_ranks.append(r)
    i = 0
    bar_y = 0.65
    while i < len(unique_ranks):
        j = i
        while j + 1 < len(unique_ranks) and unique_ranks[j + 1] - unique_ranks[i] <= cd:
            j += 1
        if j > i:
            ax.plot([unique_ranks[i], unique_ranks[j]], [bar_y, bar_y],
                     color="black", linewidth=4, solid_capstyle="butt", alpha=0.6)
            bar_y -= 0.07
        i += 1

    ax.set_title(title, fontsize=10, pad=18)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="yuantian/experiments/results/matrix")
    p.add_argument("--out_dir", default="../thesis-en-master/img/generated")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_conditions = {"baseline", "lexicase", "local_search", "hybrid", "baseline_nr", "nr"}

    for dataset in ["MMLIB50", "MMLIB100"]:
        print(f"=== {dataset} ===")
        data = load_raw(results_dir, f"{dataset}__*.json", all_conditions)

        # 1. Lexicase / local search / NR vs baseline boxplot + effect-size plot,
        # generated for both MMLIB50 (primary evidence) and MMLIB100 (scalability check)
        boxplot_grid(data, ["baseline", "lexicase"], ["#7f7f7f", "#2ca02c"],
                     f"Lexicase vs. baseline, test-fitness ARD% ({dataset})",
                     "ARD%", out_dir / f"lexicase_boxplot_{dataset.lower()}.pdf")
        effect_size_plot(data, "baseline", "lexicase",
                          f"Lexicase vs. baseline: effect size per cell ({dataset})",
                          out_dir / f"lexicase_effectsize_{dataset.lower()}.pdf")

        boxplot_grid(data, ["baseline", "local_search"], ["#7f7f7f", "#ff7f0e"],
                     f"Local search vs. baseline, test-fitness ARD% ({dataset})",
                     "ARD%", out_dir / f"localsearch_boxplot_{dataset.lower()}.pdf")
        effect_size_plot(data, "baseline", "local_search",
                          f"Local search vs. baseline: effect size per cell ({dataset})",
                          out_dir / f"localsearch_effectsize_{dataset.lower()}.pdf")

        boxplot_grid(data, ["baseline_nr", "nr"], ["#7f7f7f", "#d62728"],
                     f"NR terminals vs. baseline_nr, test-fitness ARD% ({dataset})",
                     "ARD%", out_dir / f"nr_boxplot_{dataset.lower()}.pdf")
        effect_size_plot(data, "baseline_nr", "nr",
                          f"NR terminals vs. baseline_nr: effect size per cell ({dataset})",
                          out_dir / f"nr_effectsize_{dataset.lower()}.pdf")

        # 2. Final comparison boxplot (baseline/lexicase/local_search/hybrid), both datasets
        boxplot_grid(data, ["baseline", "lexicase", "local_search", "hybrid"],
                     ["#7f7f7f", "#2ca02c", "#ff7f0e", "#9467bd"],
                     f"Final comparison, test-fitness ARD% ({dataset})",
                     "ARD%", out_dir / f"final_comparison_boxplot_{dataset.lower()}.pdf")

        # 3. CD diagram only for MMLIB50 (Friedman significant there, not on MMLIB100)
        if dataset == "MMLIB50":
            conds = ["baseline", "lexicase", "local_search", "hybrid"]
            cell_means = {}
            for sgs, strategy in CELLS:
                m = {}
                for c in conds:
                    vals = data.get((sgs, strategy, c), {}).get("test", [])
                    if vals:
                        m[c] = float(np.mean(vals))
                if len(m) == len(conds):
                    cell_means[(sgs, strategy)] = m
            ranks_per_cell = []
            for m in cell_means.values():
                order = sorted(conds, key=lambda c: m[c])
                ranks_per_cell.append({c: order.index(c) + 1 for c in conds})
            avg_rank = {c: float(np.mean([r[c] for r in ranks_per_cell])) for c in conds}
            cd_diagram(avg_rank, len(cell_means), out_dir / "final_comparison_cd_mmlib50.pdf",
                       "Critical-difference diagram: final comparison, MMLIB50 (test ARD%)")

    print("\nAll figures written to", out_dir.resolve())


if __name__ == "__main__":
    main()
