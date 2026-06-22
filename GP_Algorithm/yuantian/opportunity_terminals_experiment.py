"""
Exploratory check: does adding URGENCY_SCORE (activity tree) and
MODE_DURATION_REGRET (mode tree) on top of the full DMGE configuration
change its held-out test-set ARD%/feasibility, relative to DMGE without
them and the unmodified baseline?

Background: a proposal argued the baseline is "information-poor" because it
lacks dynamic urgency and opportunity-cost signals. Checking the actual
codebase found that a dynamic-urgency feature (DYNAMIC_SLACK) already
exists but was never enabled in any experiment this thesis reports
(ParametersGPHH.medium()'s dynamic_CPM_feature defaults to False and
experiment_runner.py never overrides it); a true CPM-recompute opportunity
cost was not implemented and would require an extra dynamic-CPM pass per
candidate mode, so MODE_DURATION_REGRET here is a cheaper static-duration
proxy instead. Both are real terminals added to rcpsp_simulation.py /
modifications.py (use_opportunity_terminals=True), not placeholders -- this
script runs them for real and reports whatever the result is.

Reuses experiment_runner.py's run_gphh / _cached_dataset / make_params
directly so the methodology (validation-based final-individual selection,
held-out test set, paired Wilcoxon) is identical to the rest of the thesis.

Usage
-----
    python -m yuantian.opportunity_terminals_experiment --datasets PSPLIB_J20 \
        -n 10 --pop 60 --gen 20 --train 16 --val 8 --test 16 --out results/opportunity_j20
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from optparse import OptionParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from yuantian.experiment_runner import (
    _cached_dataset, make_params, run_gphh, wilcoxon, star,
)

CONFIGS = [
    ("baseline_gphh", "standard", False, {}, False),
    ("custom_ea", "mod_integrated", True, {"enabled_grafts": ("NR", "CP", "RENEWABLE")}, False),
    ("custom_ea_opportunity", "mod_integrated", True, {"enabled_grafts": ("NR", "CP", "RENEWABLE")}, True),
]


def _task(args):
    (dataset, label, algo, all_mods, kwargs, use_opportunity, seed, pop, gen,
     n_train, n_val, n_test, out_dir) = args
    conv_path = os.path.join(out_dir, "convergence", f"{dataset}__{label}__seed{seed}.json")
    try:
        train, val, test = _cached_dataset(dataset, n_train, n_val, n_test, False)
        params = make_params(all_mods, pop, gen, use_opportunity=use_opportunity)
        res = run_gphh(algo, params, kwargs, train, val, test, seed, conv_path)
    except Exception as exc:
        print(f"!! FAILED {dataset}/{label}/seed{seed}: {exc}", flush=True)
        res = (float("nan"), float("nan"), float("nan"), float("nan"))
    return dataset, label, seed, res


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--datasets", dest="datasets", default="PSPLIB_J20")
    parse.add_option("--pop", dest="pop", type="int", default=60)
    parse.add_option("--gen", dest="gen", type="int", default=20)
    parse.add_option("-n", dest="n_seeds", type="int", default=10)
    parse.add_option("--seed", dest="seed", type="int", default=42)
    parse.add_option("--train", dest="n_train", type="int", default=16)
    parse.add_option("--val", dest="n_val", type="int", default=8)
    parse.add_option("--test", dest="n_test", type="int", default=16)
    parse.add_option("--out", dest="out", default="results/opportunity")
    parse.add_option("--workers", dest="workers", type="int", default=0)
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    datasets = [d.strip() for d in opt.datasets.split(",") if d.strip()]
    seeds = [opt.seed + 100 * s for s in range(opt.n_seeds)]
    workers = opt.workers or os.cpu_count()

    tasks = [(d, label, algo, all_mods, kwargs, use_opp, sd, opt.pop, opt.gen,
              opt.n_train, opt.n_val, opt.n_test, opt.out)
             for d in datasets for sd in seeds
             for (label, algo, all_mods, kwargs, use_opp) in CONFIGS]
    print(f"Datasets={datasets}  configs={[c[0] for c in CONFIGS]}  "
          f"seeds={opt.n_seeds}  pop/gen={opt.pop}/{opt.gen}  "
          f"{len(tasks)} runs on {workers} workers")

    raw_rows = []
    raw = {d: {label: {} for label, *_ in CONFIGS} for d in datasets}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            dataset, label, seed, res = fut.result()
            raw[dataset][label][seed] = res
            raw_rows.append([dataset, label, seed, *res])
            elapsed = time.time() - t0
            eta = elapsed / k * (len(tasks) - k)
            print(f"[{k:>3}/{len(tasks)}] {dataset:15s} {label:22s} seed {seed}  "
                  f"dev_all={res[0]:>9.2f} feas={res[1] * 100:>3.0f}% dev_feas={res[2]:>8.2f} "
                  f"time={res[3]:>7.1f}s  elapsed={elapsed:>5.0f}s ETA={eta:>5.0f}s", flush=True)

    with open(os.path.join(opt.out, "raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "seed", "dev_all", "feasible", "dev_feas", "time_s"])
        w.writerows(raw_rows)

    print("\n=== Summary ===")
    summary_rows = []
    for dataset in datasets:
        dev_all = {label: [raw[dataset][label][s][0] for s in seeds] for label, *_ in CONFIGS}
        feas = {label: [raw[dataset][label][s][1] for s in seeds] for label, *_ in CONFIGS}
        dev_feas = {label: [raw[dataset][label][s][2] for s in seeds] for label, *_ in CONFIGS}
        baseline = dev_all["baseline_gphh"]
        for label, *_ in CONFIGS:
            _, p = wilcoxon(dev_all[label], baseline) if label != "baseline_gphh" else (float("nan"), float("nan"))
            row = [dataset, label,
                   float(np.mean(dev_all[label])), float(np.std(dev_all[label])),
                   float(np.mean(feas[label])), float(np.mean(dev_feas[label])), p]
            summary_rows.append(row)
            p_str = f"{p:.3f} ({star(p)})" if not np.isnan(p) else "--"
            print(f"{dataset:15s} {label:22s} dev_feas={np.mean(dev_feas[label]):>8.2f}  "
                  f"feasible={np.mean(feas[label]) * 100:>5.1f}%  p_vs_baseline={p_str}")

    with open(os.path.join(opt.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "dev_all_mean", "dev_all_std",
                     "feasible_mean", "dev_feas_mean", "p_vs_baseline"])
        w.writerows(summary_rows)
    print(f"\nWrote {opt.out}/raw.csv and {opt.out}/summary.csv ({time.time()-t0:.0f}s total)")
