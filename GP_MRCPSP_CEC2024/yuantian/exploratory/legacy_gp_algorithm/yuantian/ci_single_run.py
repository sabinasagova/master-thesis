"""
Single-(dataset, config, seed) runner for CI matrix jobs. Each GitHub Actions
matrix entry invokes this once and gets one row of output; a separate merge
step concatenates all rows into the usual raw.csv/summary.csv format that
experiment_runner.py / opportunity_terminals_experiment.py produce locally.

Exists because those two scripts parallelize across seeds within ONE process
via ProcessPoolExecutor, which only scales to local core count. A CI matrix
job fans the same (dataset, config, seed) combinations out across many
separate runners instead, so each runner only ever does one combination.

Uses ParametersGPHH.default() by default, which matches Tian et al. (2024)
exactly (population 1000, 50 generations, tournament size 7, 10 elites,
Table IV of the paper) rather than the reduced .medium() budget the local
scripts use for fast iteration -- pass --pop/--gen explicitly to override.

Configs (label -> algo, all_mods, kwargs, use_opportunity):
    baseline_gphh          standard,        all_mods=False
    custom_ea              mod_integrated,  all_mods=True   (DMGE)
    custom_ea_opportunity  mod_integrated,  all_mods=True, use_opportunity=True

Usage
-----
    python -m yuantian.ci_single_run --dataset PSPLIB_J20 --config custom_ea \
        --seed 42 --pop 1000 --gen 50 --train 16 --val 8 --test 16 --out result.csv
"""
import csv
import os
import sys
from optparse import OptionParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yuantian.experiment_runner import _cached_dataset, run_gphh, DT, SIM
from yuantian.gphh_solver import ParametersGPHH

CONFIGS = {
    "baseline_gphh": ("standard", False, {}, False),
    "custom_ea": ("mod_integrated", True, {"enabled_grafts": ("NR", "CP", "RENEWABLE")}, False),
    "custom_ea_opportunity": ("mod_integrated", True, {"enabled_grafts": ("NR", "CP", "RENEWABLE")}, True),
}

if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--dataset", dest="dataset", default="PSPLIB_J20")
    parse.add_option("--config", dest="config", default="custom_ea",
                      help="one of: " + ", ".join(CONFIGS))
    parse.add_option("--seed", dest="seed", type="int", default=42)
    parse.add_option("--pop", dest="pop", type="int", default=1000)
    parse.add_option("--gen", dest="gen", type="int", default=50)
    parse.add_option("--train", dest="n_train", type="int", default=16)
    parse.add_option("--val", dest="n_val", type="int", default=8)
    parse.add_option("--test", dest="n_test", type="int", default=16)
    parse.add_option("--out", dest="out", default="result.csv")
    (opt, _) = parse.parse_args()

    algo, all_mods, kwargs, use_opportunity = CONFIGS[opt.config]
    conv_path = os.path.join(os.path.dirname(opt.out) or ".", "convergence.json")
    train, val, test = _cached_dataset(opt.dataset, opt.n_train, opt.n_val, opt.n_test, False)
    params = ParametersGPHH.default(
        decision_type=DT, simulator_type=SIM, cpu=1,
        use_modifications=all_mods, use_nr_terminals=all_mods,
        use_scheduling_state_terminals=all_mods, use_cp_mutation=all_mods,
        use_opportunity_terminals=use_opportunity,
    )
    params.pop_size, params.n_gen = opt.pop, opt.gen
    dev_all, feasible, dev_feas, elapsed = run_gphh(
        algo, params, kwargs, train, val, test, opt.seed, conv_path
    )

    os.makedirs(os.path.dirname(opt.out) or ".", exist_ok=True)
    with open(opt.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "seed", "dev_all", "feasible", "dev_feas", "time_s"])
        w.writerow([opt.dataset, opt.config, opt.seed, dev_all, feasible, dev_feas, elapsed])
    print(f"{opt.dataset} {opt.config} seed={opt.seed}: "
          f"dev_all={dev_all:.2f} feasible={feasible*100:.0f}% dev_feas={dev_feas:.2f} time={elapsed:.1f}s")
