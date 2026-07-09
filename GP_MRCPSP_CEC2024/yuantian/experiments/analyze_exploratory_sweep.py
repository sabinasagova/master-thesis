"""
Analyse the Phase 0 exploratory sweep re-run (exploratory_sweep_experiment.py's
all_runs.json) and generate the pilot tables/figures referenced by the thesis
(chap04's pilot-selection and pilot-DMGE subsections).

Comparison metric: HELD-OUT TEST FITNESS, not training fitness. Two of the
sweep's own conditions make training fitness incomparable across conditions:
(a) the graft-dependent strategies (mod_integrated, trace_directed,
map_elites) train on NR-preserving instances, whose CPM-relative ARD% sits on
a much looser scale (~110+) than the renewable-only instances every other
condition trains on (~12); (b) the sweep's lexicase driver evaluates fitness
on rotating mini-batches, so its recorded best-of-run training fitness (0.0)
is a batch-level number, not a full-training-set one. The test set, by
contrast, is the SAME renewable-only instance list for every condition
(NR terminals degrade to neutral values there), so test fitness is the one
scale on which all eleven conditions are directly comparable.

    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 \\
        yuantian/experiments/analyze_exploratory_sweep.py

Reads  yuantian/experiments/results/exploratory_sweep_experiment/all_runs.json
Writes yuantian/experiments/results/exploratory_sweep_report.txt
       ../thesis-en-master/img/generated/pilot_selection_boxplot.pdf
       ../thesis-en-master/img/generated/pilot_selection_cd.pdf
       ../thesis-en-master/img/generated/pilot_dmge_boxplot.pdf
       ../thesis-en-master/img/generated/pilot_dmge_cd.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path(__file__).parent / "results"
OUT_IMG = Path(__file__).parent / "../../../thesis-en-master/img/generated"

# chap04's grouping: the selection/search sweep vs the DMGE/graft family
SELECTION_GROUP = ["baseline", "lexicase", "lexicase_seeded", "diverse",
                   "adaptive", "surrogate", "map_elites", "multi_sgs"]
DMGE_GROUP = ["baseline", "mod_integrated", "trace_directed", "decision_trace"]
LABELS = {
    "baseline": "baseline", "lexicase": "lexicase", "lexicase_seeded": "lexicase+seed",
    "diverse": "diverse", "adaptive": "adaptive", "surrogate": "surrogate",
    "map_elites": "MAP-Elites", "multi_sgs": "multi-SGS",
    "mod_integrated": "DMGE", "trace_directed": "TDRE", "decision_trace": "QD-illum.",
}

# Demsar (2006) two-tailed Nemenyi q_alpha at alpha=0.05, by k
Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219}


def rank_biserial(diffs):
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    return float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())


def load():
    runs = json.load(open(RESULTS / "exploratory_sweep_experiment" / "all_runs.json"))
    by = {}
    for r in runs:
        by.setdefault(r["condition"], []).append(r)
    for c in by:
        by[c].sort(key=lambda x: x["seed"])
    return by


def stats_table(by, conds, emit):
    base = np.array([r["test_fitness"] for r in by["baseline"]])
    emit(f"{'condition':<16}{'test mean':>10}{'std':>8}{'best':>8}{'convgen':>9}{'p':>10}{'r':>8}")
    rows = {}
    for c in conds:
        t = np.array([r["test_fitness"] for r in by[c]])
        conv = float(np.mean([r["convergence_gen"] for r in by[c]]))
        if c == "baseline":
            p, rb = None, None
        else:
            _, p = wilcoxon(base, t)
            rb = rank_biserial(base - t)  # positive => condition better (lower test ARD%)
        rows[c] = dict(mean=t.mean(), std=t.std(), best=t.min(), conv=conv, p=p, r=rb)
        emit(f"{c:<16}{t.mean():>10.2f}{t.std():>8.2f}{t.min():>8.2f}{conv:>9.1f}"
             f"{('--' if p is None else f'{p:.4f}'):>10}{('--' if rb is None else f'{rb:+.3f}'):>8}")
    return rows


def avg_ranks(by, conds):
    """Rank conditions within each seed (block) by test fitness; average."""
    seeds = [r["seed"] for r in by["baseline"]]
    mat = np.array([[by[c][i]["test_fitness"] for c in conds] for i in range(len(seeds))])
    ranks = np.argsort(np.argsort(mat, axis=1), axis=1) + 1
    return {c: float(ranks[:, j].mean()) for j, c in enumerate(conds)}, mat


def boxplot(by, conds, title, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    data = [[r["test_fitness"] for r in by[c]] for c in conds]
    bp = ax.boxplot(data, tick_labels=[LABELS[c] for c in conds], patch_artist=True,
                    widths=0.6, flierprops=dict(marker="o", markersize=3, alpha=0.6))
    for patch, c in zip(bp["boxes"], conds):
        patch.set_facecolor("#7f7f7f" if c == "baseline" else "#1f77b4")
        patch.set_alpha(0.65)
    ax.set_ylabel("test-fitness ARD%", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def cd_diagram(ranks, n_blocks, out_path, title):
    """Classic Demsar-style layout: rank axis on top, condition labels
    stacked at the left (better half) and right (worse half) margins with
    horizontal connector lines, so labels can never collide regardless of
    how close the average ranks are."""
    conds = sorted(ranks, key=lambda c: ranks[c])
    k = len(conds)
    cd = Q_ALPHA_005[k] * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    n_left = (k + 1) // 2
    row_h = 0.09
    fig, ax = plt.subplots(figsize=(7.2, 1.8 + 0.32 * n_left))
    lo, hi = 1, k
    ax.set_xlim(lo - 2.2, hi + 2.2)
    ax.set_ylim(-row_h * (n_left + 1), 1.18)
    ax.axis("off")
    axis_y = 0.9
    ax.plot([lo, hi], [axis_y, axis_y], color="black", linewidth=1)
    for tick in range(lo, hi + 1):
        ax.plot([tick, tick], [axis_y - 0.03, axis_y + 0.03], color="black", linewidth=1)
        ax.text(tick, axis_y + 0.06, str(tick), ha="center", fontsize=8)
    cd_y = 1.08
    ax.annotate("", xy=(lo + cd, cd_y), xytext=(lo, cd_y),
                arrowprops=dict(arrowstyle="<->", color="red", linewidth=1.6))
    ax.text(lo + cd / 2, cd_y + 0.045, f"CD = {cd:.2f}",
            ha="center", fontsize=8, color="red")
    # labels: better half stacked on the left margin, worse half on the right
    for i, c in enumerate(conds):
        r = ranks[c]
        if i < n_left:
            row = i
            x_lab, ha = lo - 2.1, "left"
        else:
            row = k - 1 - i
            x_lab, ha = hi + 2.1, "right"
        y_lab = -row_h * (row + 1)
        ax.plot([r, r], [axis_y, y_lab], color="black", linewidth=0.8)
        ax.plot([min(r, x_lab), max(r, x_lab)], [y_lab, y_lab],
                color="black", linewidth=0.8)
        ax.plot(r, axis_y, marker="o", color="#1f77b4", markersize=5, zorder=3)
        ax.text(x_lab, y_lab + 0.02, f"{LABELS[c]} ({r:.2f})",
                ha=ha, va="bottom", fontsize=8)
    # thick bars connecting groups whose ranks are within one CD
    vals = [ranks[c] for c in conds]
    i, bar_y = 0, axis_y - 0.10
    drawn = []
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in drawn):
            ax.plot([vals[i] - 0.05, vals[j] + 0.05], [bar_y, bar_y],
                    color="black", linewidth=4, solid_capstyle="butt", alpha=0.6)
            drawn.append((i, j))
            bar_y -= 0.07
        i += 1
    ax.set_title(title, fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    by = load()
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 90)
    emit("PILOT SWEEP -- selection/search strategies (held-out test fitness, common scale)")
    emit("=" * 90)
    stats_table(by, SELECTION_GROUP, emit)
    ranks, mat = avg_ranks(by, SELECTION_GROUP)
    stat, p = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    emit(f"\nAverage ranks: " + ", ".join(f"{LABELS[c]}={ranks[c]:.2f}"
                                          for c in sorted(ranks, key=lambda c: ranks[c])))
    emit(f"Friedman chi2={stat:.4f} p={p:.4f} (blocks=10 seeds, k={len(SELECTION_GROUP)})")
    boxplot(by, SELECTION_GROUP,
            "Pilot sweep: selection/search strategies, test-fitness ARD% (MMLIB50)",
            OUT_IMG / "pilot_selection_boxplot.pdf")
    if p < 0.05:
        cd_diagram(ranks, 10, OUT_IMG / "pilot_selection_cd.pdf",
                   "Critical-difference diagram: pilot selection/search sweep (test ARD%)")

    emit()
    emit("=" * 90)
    emit("PILOT SWEEP -- DMGE/graft family (held-out test fitness, common scale)")
    emit("=" * 90)
    stats_table(by, DMGE_GROUP, emit)
    ranks2, mat2 = avg_ranks(by, DMGE_GROUP)
    stat2, p2 = friedmanchisquare(*[mat2[:, j] for j in range(mat2.shape[1])])
    emit(f"\nAverage ranks: " + ", ".join(f"{LABELS[c]}={ranks2[c]:.2f}"
                                          for c in sorted(ranks2, key=lambda c: ranks2[c])))
    emit(f"Friedman chi2={stat2:.4f} p={p2:.4f} (blocks=10 seeds, k={len(DMGE_GROUP)})")
    boxplot(by, DMGE_GROUP,
            "Pilot sweep: DMGE/graft family, test-fitness ARD% (MMLIB50)",
            OUT_IMG / "pilot_dmge_boxplot.pdf")
    if p2 < 0.05:
        cd_diagram(ranks2, 10, OUT_IMG / "pilot_dmge_cd.pdf",
                   "Critical-difference diagram: pilot DMGE-family sweep (test ARD%)")

    out = RESULTS / "exploratory_sweep_report.txt"
    out.write_text("\n".join(lines))
    print("\nReport written to", out)


if __name__ == "__main__":
    main()
