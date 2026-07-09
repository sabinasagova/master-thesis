"""
Parameter sensitivity sweep for DMGE (custom_ea / mod_integrated) on PSPLIB
J20: one-factor-at-a-time variation of population size and generation count
around the defaults used for the head-to-head comparison in
experiment_runner.py (pop=60, gen=20), holding the other fixed. Reuses
experiment_runner.py's dataset split, training loop, validation-based
selection, and test-set evaluation directly so this sweep is methodologically
identical to the benchmark comparison, just varying one knob at a time.

Usage
-----
    python -m yuantian.param_sensitivity --datasets PSPLIB_J20 -n 5 \
        --out results/param_sensitivity
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from optparse import OptionParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from yuantian.experiment_runner import (_cached_dataset, make_params, run_gphh)

DEFAULT_POP, DEFAULT_GEN = 60, 20
POP_GRID = [20, 40, 60, 100]
GEN_GRID = [5, 10, 20, 40]
ALGO, ALL_MODS, KWARGS = "mod_integrated", True, {"enabled_grafts": ("NR", "CP", "RENEWABLE")}


def _task(args):
    dataset, axis, value, seed, n_train, n_val, n_test, out_dir = args
    pop = value if axis == "pop" else DEFAULT_POP
    gen = value if axis == "gen" else DEFAULT_GEN
    conv_path = os.path.join(out_dir, "convergence", f"{dataset}__{axis}{value}__seed{seed}.json")
    try:
        train, val, test = _cached_dataset(dataset, n_train, n_val, n_test, False)
        dev_all, feasible, dev_feas, elapsed = run_gphh(
            ALGO, make_params(ALL_MODS, pop, gen), KWARGS, train, val, test, seed, conv_path)
    except Exception as exc:
        print(f"!! FAILED {dataset}/{axis}={value}/seed{seed}: {exc}", flush=True)
        dev_all, feasible, dev_feas, elapsed = (float("nan"),) * 4
    return dataset, axis, value, seed, dev_all, feasible, dev_feas, elapsed


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--datasets", dest="datasets", default="PSPLIB_J20")
    parse.add_option("-n", dest="n_seeds", type="int", default=5)
    parse.add_option("--seed", dest="seed", type="int", default=42)
    parse.add_option("--train", dest="n_train", type="int", default=16)
    parse.add_option("--val", dest="n_val", type="int", default=8)
    parse.add_option("--test", dest="n_test", type="int", default=16)
    parse.add_option("--out", dest="out", default="results/param_sensitivity")
    parse.add_option("--workers", dest="workers", type="int", default=0)
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    datasets = [d.strip() for d in opt.datasets.split(",") if d.strip()]
    seeds = [opt.seed + 100 * s for s in range(opt.n_seeds)]
    workers = opt.workers or os.cpu_count()

    tasks = []
    for d in datasets:
        for value in POP_GRID:
            for sd in seeds:
                tasks.append((d, "pop", value, sd, opt.n_train, opt.n_val, opt.n_test, opt.out))
        for value in GEN_GRID:
            if value == DEFAULT_GEN:
                continue  # already covered by pop=DEFAULT_POP sweep point
            for sd in seeds:
                tasks.append((d, "gen", value, sd, opt.n_train, opt.n_val, opt.n_test, opt.out))
    print(f"Datasets={datasets}  pop_grid={POP_GRID}  gen_grid={GEN_GRID}  "
          f"seeds={opt.n_seeds}  {len(tasks)} runs on {workers} workers")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            dataset, axis, value, seed, dev_all, feasible, dev_feas, elapsed = fut.result()
            rows.append([dataset, axis, value, seed, dev_all, feasible, dev_feas, elapsed])
            el = time.time() - t0
            eta = el / k * (len(tasks) - k)
            print(f"[{k:>3}/{len(tasks)}] {dataset:12s} {axis}={value:<4} seed{seed}  "
                  f"dev_feas={dev_feas:>7.2f} feas={feasible*100:>3.0f}%  "
                  f"elapsed={el:>5.0f}s ETA={eta:>5.0f}s", flush=True)

    with open(os.path.join(opt.out, "raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "axis", "value", "seed", "dev_all", "feasible", "dev_feas", "time_s"])
        w.writerows(rows)

    summary_rows = []
    for dataset in datasets:
        print(f"\n{'=' * 70}\n{dataset}\n{'=' * 70}")
        for axis, grid in [("pop", POP_GRID), ("gen", GEN_GRID)]:
            print(f"-- varying {axis} (other fixed at default) --")
            for value in grid:
                sel = [r for r in rows if r[0] == dataset and r[1] == axis and r[2] == value]
                if not sel and value == (DEFAULT_POP if axis == "pop" else DEFAULT_GEN):
                    sel = [r for r in rows if r[0] == dataset and r[1] == "pop" and r[2] == DEFAULT_POP]
                if not sel:
                    continue
                df = np.array([r[6] for r in sel])
                df = df[~np.isnan(df)]
                fe = np.array([r[5] for r in sel])
                ts = np.array([r[7] for r in sel])
                df_str = f"{df.mean():.2f} (±{df.std():.2f})" if df.size else "n/a"
                print(f"  {axis}={value:<4} dev_feas={df_str:>18s}  feas={fe.mean()*100:5.1f}%  "
                      f"time_s={ts.mean():7.1f}")
                summary_rows.append([dataset, axis, value, df.mean() if df.size else float("nan"),
                                      df.std() if df.size else float("nan"), fe.mean(), ts.mean()])

    with open(os.path.join(opt.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "axis", "value", "dev_feas_mean", "dev_feas_std", "feasible", "time_s_mean"])
        w.writerows(summary_rows)
    print(f"\nWrote {opt.out}/raw.csv, {opt.out}/summary.csv  ({time.time() - t0:.0f}s total)")
