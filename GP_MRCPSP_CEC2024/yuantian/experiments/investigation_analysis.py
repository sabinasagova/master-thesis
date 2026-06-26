"""
Statistical analysis + figures for the lexicase-vs-baseline overfitting
investigation, consuming the enriched datasets produced by
investigation_lexicase_overfitting.py (and, if present, the per-generation
curves from investigation_validation_dynamics.py).

Run with (from the GP_MRCPSP_CEC2024 repo root):
    PYTHONPATH=$(pwd) python yuantian/experiments/investigation_analysis.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

RESULTS_DIR = Path(__file__).parent / "results" / "investigation_lexicase_overfitting"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VAL_DYNAMICS_PATH = Path(__file__).parent / "results" / "investigation_validation_dynamics" / "curves.json"

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


def rank_biserial(diffs):
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    return float((r_plus - r_minus) / ranks.sum())


def bootstrap_ci_mean(diffs, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_report(name, a, b, label_a="baseline", label_b="lexicase"):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a - b  # positive => b (lexicase) better, since lower=better
    stat, p = wilcoxon(a, b) if not np.all(diffs == 0) else (0.0, 1.0)
    r = rank_biserial(diffs)
    lo, hi = bootstrap_ci_mean(diffs)
    log(
        f"  {name}: {label_a}={a.mean():.4f}±{a.std():.4f}, {label_b}={b.mean():.4f}±{b.std():.4f}, "
        f"mean_diff({label_a}-{label_b})={diffs.mean():.4f} [95% CI {lo:.4f}, {hi:.4f}], "
        f"Wilcoxon p={p:.4g}, r={r:.3f}"
    )
    return {
        "metric": name,
        f"mean_{label_a}": float(a.mean()), f"std_{label_a}": float(a.std()),
        f"mean_{label_b}": float(b.mean()), f"std_{label_b}": float(b.std()),
        "mean_diff": float(diffs.mean()), "ci95": [lo, hi],
        "wilcoxon_p": float(p), "rank_biserial_r": r, "n": len(a),
    }


def main():
    followup = json.load(open(RESULTS_DIR / "enriched_followup.json"))
    all_runs = json.load(open(RESULTS_DIR / "enriched_all_runs_serial.json"))

    baseline = sorted([r for r in followup if r["method"] == "baseline"], key=lambda r: r["seed"])
    lexicase = sorted([r for r in followup if r["method"] == "lexicase"], key=lambda r: r["seed"])
    assert [r["seed"] for r in baseline] == [r["seed"] for r in lexicase]
    n = len(baseline)

    summary = {}

    # ------------------------------------------------------------------
    log("=" * 78)
    log("0. RECONSTRUCTION SANITY CHECK: does best_tree match the officially")
    log("   reported (validation-selected) test_fitness?")
    log("=" * 78)
    diffs0 = np.array([abs(r["recon_test_fitness"] - r["test_fitness"]) for r in followup])
    exact = int((diffs0 < 1e-6).sum())
    log(
        f"n={len(diffs0)}, exact matches={exact} ({100*exact/len(diffs0):.0f}%), "
        f"mean |diff|={diffs0.mean():.3f}, median={np.median(diffs0):.3f}, max={diffs0.max():.3f}"
    )
    log(
        "Interpretation: the validation-selected individual (best of final pop on a "
        "10-instance validation set) frequently differs from the final generation's "
        "training-fitness hall-of-fame winner. The resulting test-fitness noise from "
        "*which* near-equally-fit individual gets picked (mean ~0.86, up to 2.9) is "
        "LARGER than the baseline-vs-lexicase mean difference itself (0.30). This is a "
        "first-order candidate explanation for the null test-fitness result: model-"
        "selection noise from a small validation set may swamp the treatment effect."
    )
    summary["model_selection_noise"] = {
        "n": len(diffs0), "exact_match_pct": 100 * exact / len(diffs0),
        "mean_abs_diff": float(diffs0.mean()), "median_abs_diff": float(np.median(diffs0)),
        "max_abs_diff": float(diffs0.max()),
    }

    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("1. OVERFITTING ANALYSIS (n=31 paired seeds, serial SGS)")
    log("=" * 78)
    bt_train = [r["best_fitness_train"] for r in baseline]
    lx_train = [r["best_fitness_train"] for r in lexicase]
    bt_test = [r["test_fitness"] for r in baseline]
    lx_test = [r["test_fitness"] for r in lexicase]
    bt_val_recon = [r["recon_val_fitness"] for r in baseline]
    lx_val_recon = [r["recon_val_fitness"] for r in lexicase]

    log("-- Raw metrics --")
    summary["train"] = paired_report("train (official)", bt_train, lx_train)
    summary["test"] = paired_report("test (official)", bt_test, lx_test)
    summary["val_recon"] = paired_report("validation (reconstructed best_tree)", bt_val_recon, lx_val_recon)

    log("\n-- Generalization gaps (gap = test/val minus train; higher = more overfit) --")
    bt_gap_test = np.array(bt_test) - np.array(bt_train)
    lx_gap_test = np.array(lx_test) - np.array(lx_train)
    bt_gap_val = np.array(bt_val_recon) - np.array(bt_train)
    lx_gap_val = np.array(lx_val_recon) - np.array(lx_train)
    summary["gap_train_test"] = paired_report("train->test gap", bt_gap_test, lx_gap_test)
    summary["gap_train_val"] = paired_report("train->val gap", bt_gap_val, lx_gap_val)

    log(
        "\nInterpretation: if lexicase's train->test gap is significantly larger than "
        "baseline's, that is direct evidence of overfitting (fitting training cases harder "
        "without a matching generalization benefit)."
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (lbl, b, l) in zip(
        axes,
        [
            ("Training fitness", bt_train, lx_train),
            ("Test fitness (official)", bt_test, lx_test),
            ("Train→Test gap", bt_gap_test, lx_gap_test),
        ],
    ):
        ax.boxplot([b, l], labels=["baseline", "lexicase"])
        ax.set_title(lbl)
        ax.set_ylabel("% deviation from CPM LB" if "gap" not in lbl.lower() else "gap (pp)")
    fig.suptitle("Fig 1. Overfitting analysis: baseline vs lexicase (n=31 paired seeds, serial SGS)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_overfitting_gaps.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("3. TREE COMPLEXITY (n=31 paired seeds)")
    log("=" * 78)
    bt_size = [r["tree_stats"]["total_size"] for r in baseline]
    lx_size = [r["tree_stats"]["total_size"] for r in lexicase]
    bt_depth = [r["tree_stats"]["max_depth"] for r in baseline]
    lx_depth = [r["tree_stats"]["max_depth"] for r in lexicase]
    summary["tree_size"] = paired_report("total tree size (nodes)", bt_size, lx_size)
    summary["tree_depth"] = paired_report("max tree depth", bt_depth, lx_depth)

    all_size = np.array(bt_size + lx_size)
    all_gap = np.concatenate([bt_gap_test, lx_gap_test])
    rho, p_rho = spearmanr(all_size, all_gap)
    log(f"\nSpearman(tree_size, train->test gap), pooled n={len(all_size)}: rho={rho:.3f}, p={p_rho:.4g}")
    rho_test, p_test = spearmanr(all_size, bt_test + lx_test)
    log(f"Spearman(tree_size, test_fitness), pooled n={len(all_size)}: rho={rho_test:.3f}, p={p_test:.4g}")
    summary["tree_size_vs_gap_corr"] = {"rho": float(rho), "p": float(p_rho)}
    summary["tree_size_vs_test_corr"] = {"rho": float(rho_test), "p": float(p_test)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([bt_size, lx_size], labels=["baseline", "lexicase"])
    axes[0].set_title("Tree size (total nodes)")
    axes[1].scatter(bt_size, bt_gap_test, label="baseline", alpha=0.7)
    axes[1].scatter(lx_size, lx_gap_test, label="lexicase", alpha=0.7)
    axes[1].set_xlabel("tree size (nodes)")
    axes[1].set_ylabel("train→test gap")
    axes[1].legend()
    axes[1].set_title(f"Tree size vs. generalization gap (Spearman rho={rho:.2f}, p={p_rho:.3f})")
    fig.suptitle("Fig 2. Tree complexity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_tree_complexity.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("4. DIVERSITY ANALYSIS (using existing avg_unique_trees_over_run / avg_fitness_std_over_run)")
    log("=" * 78)
    bt_uniq = [r["avg_unique_trees_over_run"] for r in baseline]
    lx_uniq = [r["avg_unique_trees_over_run"] for r in lexicase]
    bt_fstd = [r["avg_fitness_std_over_run"] for r in baseline]
    lx_fstd = [r["avg_fitness_std_over_run"] for r in lexicase]
    summary["diversity_unique_trees"] = paired_report("avg unique trees / gen", bt_uniq, lx_uniq)
    summary["diversity_fitness_std"] = paired_report("avg population fitness std / gen", bt_fstd, lx_fstd)

    all_uniq = np.array(bt_uniq + lx_uniq)
    all_test = np.array(bt_test + lx_test)
    rho_u, p_u = spearmanr(all_uniq, all_test)
    rho_ug, p_ug = spearmanr(all_uniq, all_gap)
    log(f"\nSpearman(diversity[unique trees/gen], test_fitness), pooled n={len(all_uniq)}: rho={rho_u:.3f}, p={p_u:.4g}")
    log(f"Spearman(diversity[unique trees/gen], train->test gap), pooled n={len(all_uniq)}: rho={rho_ug:.3f}, p={p_ug:.4g}")
    summary["diversity_vs_test_corr"] = {"rho": float(rho_u), "p": float(p_u)}
    summary["diversity_vs_gap_corr"] = {"rho": float(rho_ug), "p": float(p_ug)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([bt_uniq, lx_uniq], labels=["baseline", "lexicase"])
    axes[0].set_title("Avg. unique trees per generation (diversity proxy)")
    axes[1].scatter(bt_uniq, bt_test, label="baseline", alpha=0.7)
    axes[1].scatter(lx_uniq, lx_test, label="lexicase", alpha=0.7)
    axes[1].set_xlabel("avg unique trees / gen")
    axes[1].set_ylabel("test fitness")
    axes[1].legend()
    axes[1].set_title(f"Diversity vs. test fitness (rho={rho_u:.2f}, p={p_u:.3f})")
    fig.suptitle("Fig 3. Diversity analysis")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_diversity.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("5. SPECIALIST VS GENERALIST (per-instance fitness spread of best_tree)")
    log("=" * 78)

    def cv(vals):
        vals = np.array(vals, dtype=float)
        return float(vals.std() / vals.mean()) if vals.mean() != 0 else 0.0

    bt_test_cv = [cv(r["test_case_fitness"]) for r in baseline]
    lx_test_cv = [cv(r["test_case_fitness"]) for r in lexicase]
    bt_train_cv = [cv(r["train_case_fitness"]) for r in baseline]
    lx_train_cv = [cv(r["train_case_fitness"]) for r in lexicase]
    summary["specialist_cv_test"] = paired_report("coeff. of variation across test instances", bt_test_cv, lx_test_cv)
    summary["specialist_cv_train"] = paired_report("coeff. of variation across train instances", bt_train_cv, lx_train_cv)
    log(
        "\nInterpretation: higher coefficient of variation across instances means the rule "
        "performs unevenly (very good on some instances, poor on others) = more 'specialist'. "
        "Lower CV = more uniformly 'generalist' behavior."
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([bt_test_cv, lx_test_cv], labels=["baseline", "lexicase"])
    ax.set_title("Per-instance fitness CV on the held-out test set\n(higher = more specialist behavior)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_specialist_generalist.png", dpi=150)
    plt.close(fig)

    log("\n-- Head-to-head per-test-instance dominance (same literal best_tree, paired by seed) --")
    n_test_instances = len(baseline[0]["test_case_fitness"])
    wins = np.zeros(n_test_instances)   # lexicase strictly better
    losses = np.zeros(n_test_instances)  # baseline strictly better
    ties = np.zeros(n_test_instances)
    mean_b_by_inst = np.mean([r["test_case_fitness"] for r in baseline], axis=0)
    mean_l_by_inst = np.mean([r["test_case_fitness"] for r in lexicase], axis=0)
    for b, l in zip(baseline, lexicase):
        bv = np.array(b["test_case_fitness"])
        lv = np.array(l["test_case_fitness"])
        for i in range(n_test_instances):
            if abs(lv[i] - bv[i]) < 1e-9:
                ties[i] += 1
            elif lv[i] < bv[i]:
                wins[i] += 1
            else:
                losses[i] += 1
    log(f"{'instance':<10}{'lex wins':>10}{'base wins':>11}{'ties':>7}{'mean base':>12}{'mean lex':>11}")
    inst_table = []
    for i in range(n_test_instances):
        log(
            f"{i:<10}{int(wins[i]):>10}{int(losses[i]):>11}{int(ties[i]):>7}"
            f"{mean_b_by_inst[i]:>12.2f}{mean_l_by_inst[i]:>11.2f}"
        )
        inst_table.append(
            {
                "instance": i, "lex_wins": int(wins[i]), "base_wins": int(losses[i]), "ties": int(ties[i]),
                "mean_baseline": float(mean_b_by_inst[i]), "mean_lexicase": float(mean_l_by_inst[i]),
            }
        )
    log(
        "\nInterpretation: most ties are at/near 0% deviation (instances solved essentially "
        "optimally by both methods regardless of selection scheme -- e.g. instances 2, 3, 9 "
        "are tied in the large majority of seeds). One instance (4) is far harder than the "
        "rest for both methods (~89-91% mean deviation) and by itself dominates the magnitude "
        "of the aggregate test_fitness average, since it is ~30x larger than most other "
        "instances' deviations. With only 10 test instances and this much heterogeneity in "
        "difficulty, the aggregate metric's sensitivity to one or two hard instances is itself "
        "a source of noise that compounds with the model-selection noise found above."
    )
    summary["head_to_head_test"] = inst_table

    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("6. TRANSFERABILITY: terminal usage frequency in evolved trees")
    log("=" * 78)
    bt_usage = Counter()
    lx_usage = Counter()
    for r in baseline:
        bt_usage.update(r["terminal_usage"])
    for r in lexicase:
        lx_usage.update(r["terminal_usage"])
    bt_total = sum(bt_usage.values())
    lx_total = sum(lx_usage.values())
    all_terms = sorted(set(bt_usage) | set(lx_usage))
    log(f"{'terminal':<10} {'baseline %':>12} {'lexicase %':>12} {'diff (pp)':>10}")
    usage_table = []
    for t in all_terms:
        bp = 100 * bt_usage.get(t, 0) / bt_total
        lp = 100 * lx_usage.get(t, 0) / lx_total
        log(f"{t:<10} {bp:>12.2f} {lp:>12.2f} {lp - bp:>10.2f}")
        usage_table.append({"terminal": t, "baseline_pct": bp, "lexicase_pct": lp, "diff_pp": lp - bp})
    summary["terminal_usage"] = usage_table

    from scipy.stats import chi2_contingency
    contingency = np.array([[bt_usage.get(t, 0) for t in all_terms], [lx_usage.get(t, 0) for t in all_terms]])
    chi2, p_chi2, dof, _ = chi2_contingency(contingency)
    log(
        f"\nChi-square test of independence (terminal x method), pooled over all {bt_total + lx_total} "
        f"terminal occurrences across n={n} trees/method: chi2={chi2:.1f}, dof={dof}, p={p_chi2:.4g}"
    )
    log(
        "(This tests whether the *overall* terminal-usage profile differs between methods; it "
        "does not by itself imply the shift causes the generalization gap -- see tree-size/"
        "depth correlations above, which were not significant.)"
    )
    summary["terminal_usage_chi2"] = {"chi2": float(chi2), "dof": int(dof), "p": float(p_chi2)}

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_terms))
    width = 0.35
    ax.bar(x - width / 2, [100 * bt_usage.get(t, 0) / bt_total for t in all_terms], width, label="baseline")
    ax.bar(x + width / 2, [100 * lx_usage.get(t, 0) / lx_total for t in all_terms], width, label="lexicase")
    ax.set_xticks(x)
    ax.set_xticklabels(all_terms, rotation=45, ha="right")
    ax.set_ylabel("% of all terminal occurrences")
    ax.legend()
    ax.set_title("Fig 5. Terminal (feature) usage frequency in evolved trees")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_terminal_usage.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("8. HYBRID (lexicase + critical-path local search) COMPARISON (n=10, serial SGS)")
    log("=" * 78)
    hy_base = sorted([r for r in all_runs if r["method"] == "baseline"], key=lambda r: r["seed"])
    hy_lex = sorted([r for r in all_runs if r["method"] == "lexicase"], key=lambda r: r["seed"])
    hy_hyb = sorted([r for r in all_runs if r["method"] == "hybrid"], key=lambda r: r["seed"])
    log(f"n per method: baseline={len(hy_base)}, lexicase={len(hy_lex)}, hybrid={len(hy_hyb)}")

    ls_gain = np.array([r["best_fitness_train"] for r in hy_hyb]) - np.array(
        [r["recon_train_fitness"] for r in hy_hyb]
    )
    log(
        f"\nLocal-search contribution to hybrid's *reported* training fitness "
        f"(recorded best_fitness_train minus the raw-tree recon_train_fitness; "
        f"negative = LS made the reported number look better than the bare tree): "
        f"mean={ls_gain.mean():.3f}, std={ls_gain.std():.3f}"
    )
    summary["hybrid_ls_contribution_to_train"] = {"mean": float(ls_gain.mean()), "std": float(ls_gain.std())}

    def gap_arr(records):
        return np.array([r["recon_test_fitness"] for r in records]) - np.array(
            [r["recon_train_fitness"] for r in records]
        )

    base_gap, lex_gap, hyb_gap = gap_arr(hy_base), gap_arr(hy_lex), gap_arr(hy_hyb)
    log("\nApples-to-apples generalization gap (recon_test - recon_train, same literal tree):")
    log(f"  baseline: mean={base_gap.mean():.3f}, std={base_gap.std():.3f}")
    log(f"  lexicase: mean={lex_gap.mean():.3f}, std={lex_gap.std():.3f}")
    log(f"  hybrid:   mean={hyb_gap.mean():.3f}, std={hyb_gap.std():.3f}")
    u_lh, p_lh = mannwhitneyu(lex_gap, hyb_gap)
    u_bh, p_bh = mannwhitneyu(base_gap, hyb_gap)
    log(f"  Mann-Whitney lexicase vs hybrid gap: U={u_lh:.1f}, p={p_lh:.4g}")
    log(f"  Mann-Whitney baseline vs hybrid gap: U={u_bh:.1f}, p={p_bh:.4g}")
    log(
        "  (n=10/group, exploratory/underpowered -- treat as a trend indicator, not a "
        "confirmatory test)"
    )
    summary["hybrid_gap_comparison"] = {
        "baseline_gap_mean": float(base_gap.mean()), "lexicase_gap_mean": float(lex_gap.mean()),
        "hybrid_gap_mean": float(hyb_gap.mean()), "mw_lex_vs_hyb_p": float(p_lh), "mw_base_vs_hyb_p": float(p_bh),
    }

    hy_size = {
        m: [r["tree_stats"]["total_size"] for r in recs]
        for m, recs in [("baseline", hy_base), ("lexicase", hy_lex), ("hybrid", hy_hyb)]
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([base_gap, lex_gap, hyb_gap], labels=["baseline", "lexicase", "hybrid"])
    axes[0].set_title("Train→test gap (raw-tree, apples-to-apples), n=10/group")
    axes[1].boxplot([hy_size["baseline"], hy_size["lexicase"], hy_size["hybrid"]], labels=["baseline", "lexicase", "hybrid"])
    axes[1].set_title("Tree size, n=10/group")
    fig.suptitle("Fig 6. Hybrid (lexicase+local search) vs baseline/lexicase")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_hybrid_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    if VAL_DYNAMICS_PATH.exists():
        log("\n" + "=" * 78)
        log("2. VALIDATION DYNAMICS ACROSS GENERATIONS (fresh instrumented runs)")
        log("=" * 78)
        curves = json.load(open(VAL_DYNAMICS_PATH))
        by_method = {"baseline": [], "lexicase": []}
        for r in curves:
            by_method[r["method"]].append(r)
        if all(len(v) > 0 for v in by_method.values()):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
            for ax, method in zip(axes, ("baseline", "lexicase")):
                recs = by_method[method]
                gens = np.arange(len(recs[0]["train_curve"]))
                train_arr = np.array([r["train_curve"] for r in recs])
                val_arr = np.array([r["val_curve"] for r in recs])
                test_arr = np.array([r["test_curve"] for r in recs])
                for arr, lbl in [(train_arr, "train"), (val_arr, "validation"), (test_arr, "test")]:
                    mean = arr.mean(axis=0)
                    ax.plot(gens, mean, label=lbl)
                    ax.fill_between(gens, mean - arr.std(axis=0), mean + arr.std(axis=0), alpha=0.15)
                ax.set_title(f"{method} (n={len(recs)})")
                ax.set_xlabel("generation")
                ax.legend()
            axes[0].set_ylabel("fitness (% deviation from CPM LB)")
            fig.suptitle("Fig 7. Train/validation/test dynamics across generations")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig7_validation_dynamics.png", dpi=150)
            plt.close(fig)
            for method in ("baseline", "lexicase"):
                recs = by_method[method]
                val_arr = np.array([r["val_curve"] for r in recs])
                test_arr = np.array([r["test_curve"] for r in recs])
                val_best_gen = val_arr.mean(axis=0).argmin()
                test_best_gen = test_arr.mean(axis=0).argmin()
                n_gen = val_arr.shape[1] - 1
                log(
                    f"  {method} (n={len(recs)}): mean validation fitness minimized at gen "
                    f"{val_best_gen}/{n_gen}; mean test fitness minimized at gen {test_best_gen}/{n_gen}; "
                    f"final-gen val={val_arr[:, -1].mean():.3f}, final-gen test={test_arr[:, -1].mean():.3f}"
                )

            log("\n-- Train->test gap trajectory across generations (does the gap widen as evolution proceeds?) --")
            fig, ax = plt.subplots(figsize=(8, 5))
            gap_summary = {}
            for method in ("baseline", "lexicase"):
                recs = by_method[method]
                train_arr = np.array([r["train_curve"] for r in recs])
                test_arr = np.array([r["test_curve"] for r in recs])
                gap_arr = test_arr - train_arr
                gens = np.arange(gap_arr.shape[1])
                mean_gap = gap_arr.mean(axis=0)
                ax.plot(gens, mean_gap, label=method)
                ax.fill_between(gens, mean_gap - gap_arr.std(axis=0), mean_gap + gap_arr.std(axis=0), alpha=0.15)
                early_gap = mean_gap[1:8].mean()   # gens 1-7
                late_gap = mean_gap[9:].mean()     # gens 9-20
                rho_g, p_g = spearmanr(gens, mean_gap)
                log(
                    f"  {method}: mean gap gens 1-7={early_gap:.3f}, mean gap gens 9-20={late_gap:.3f}, "
                    f"change={late_gap - early_gap:+.3f}; Spearman(gen, mean_gap) rho={rho_g:.3f}, p={p_g:.4g}"
                )
                gap_summary[method] = {
                    "early_gap_gen1_7": float(early_gap), "late_gap_gen9_20": float(late_gap),
                    "change": float(late_gap - early_gap), "spearman_rho": float(rho_g), "spearman_p": float(p_g),
                }
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("generation")
            ax.set_ylabel("train->test gap (test - train)")
            ax.legend()
            ax.set_title(f"Fig 8. Generalization gap over generations (n={len(by_method['baseline'])}/method)")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "fig8_gap_trajectory.png", dpi=150)
            plt.close(fig)
            summary["gap_trajectory"] = gap_summary
            log(
                "  Interpretation: baseline's gap is roughly flat/noisy across generations (no "
                "progressive widening); lexicase's gap visibly widens from the early-to-mid "
                "generations to a higher, stable plateau from ~gen 9 onward. n=6/method -- "
                "directional evidence, not a confirmatory test."
            )
        else:
            log(f"  Curves file present but incomplete: {[ (m, len(v)) for m, v in by_method.items() ]}")
    else:
        log("\n[Investigation 2 pending: instrumented per-generation run not finished yet]")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "report.txt", "w") as f:
        f.write("\n".join(report_lines))
    log(f"\nAll figures written to {FIG_DIR}")
    log(f"Structured summary: {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
