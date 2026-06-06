"""
Comparison runner: baseline GPHH vs. proposed modifications.

Runs both configurations with identical seeds and saves results to:
  results/baseline/   – Yuan Tian's original GPHH
  results/modified/   – GPHH with modifications from modifications.py

Usage
-----
Quick smoke-test (fast parameters, small dataset):
    python yuantian/run_comparison.py

Full experiment (paper parameters, MMLIB50, 5 runs):
    python yuantian/run_comparison.py --default --dataset MMLIB50 -n 5

Then inspect results with:
    python yuantian/compare_results.py --baseline results/baseline \
                                       --modified  results/modified
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from optparse import OptionParser

from yuantian.gphh_solver import GPHH, ParametersGPHH, read_instances
from yuantian.rcpsp_dataset import (
    RCPSPDatabase, StaticDatasetProvider, EvenlyDividedDatasetProvider
)
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum


def build_params(use_modifications: bool, default: bool, medium: bool,
                 decision_type, simulator_type, cpu_cores,
                 dynamic_terminals, split_training,
                 fixed_activity_rule="", fixed_mode_rule="") -> ParametersGPHH:
    if default:
        return ParametersGPHH.default(
            decision_type=decision_type,
            simulator_type=simulator_type,
            cpu=cpu_cores,
            dynamic_CPM_feature=dynamic_terminals,
            fixed_activity_rule=fixed_activity_rule,
            fixed_mode_rule=fixed_mode_rule,
            use_modifications=use_modifications,
        )
    if medium:
        return ParametersGPHH.medium(
            decision_type=decision_type,
            simulator_type=simulator_type,
            cpus=cpu_cores,
            dynamic_CPM_feature=dynamic_terminals,
            use_modifications=use_modifications,
        )
    return ParametersGPHH.fast(
        decision_type=decision_type,
        simulator_type=simulator_type,
        cpus=cpu_cores,
        dynamic_CPM_feature=dynamic_terminals,
        use_modifications=use_modifications,
    )


def run_config(label: str, use_modifications: bool, params: ParametersGPHH,
               training_provider, validation_provider, test_provider,
               n_runs: int, start_index: int, seed: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Modifications: {use_modifications}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}\n")

    solver = GPHH(
        training_set_provider=training_provider,
        validation_set_provider=validation_provider,
        test_set_provider=test_provider,
        params_gphh=params,
    )
    solver.init_model()

    for n in range(start_index, start_index + n_runs):
        current_seed = seed + n * 100
        random.seed(current_seed)
        np.random.seed(current_seed)
        for provider in [training_provider, validation_provider, test_provider]:
            provider.reset()

        print(f"[{label}] Run {n} — seed {current_seed}")
        solver.solve(output_path=os.path.join(output_dir, f"{n}.json"))
        print(f"[{label}] Run {n} done.\n")


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("-s", dest="sgs", default="serial",
                     help="SGS type: serial / parallel")
    parse.add_option("-d", dest="decision_type", default="activity_first",
                     help="Decision type: activity_first / mode_first / simultaneous")
    parse.add_option("--default", action="store_true", dest="default", default=False,
                     help="Use paper parameters (pop=1000, gen=50).")
    parse.add_option("--medium", action="store_true", dest="medium", default=False,
                     help="Use medium parameters (pop=50, gen=10). "
                          "Faster than --default, shows ETA per generation.")
    parse.add_option("--dataset", dest="dataset", default="",
                     help="MMLIB50 / MMLIB100 / MMLIBPLUS_50 / MMLIBPLUS_100")
    parse.add_option("-n", dest="n_runs", type="int", default=1,
                     help="Number of runs per configuration")
    parse.add_option("--start_index", dest="start_index", type="int", default=0)
    parse.add_option("--seed", dest="seed", type="int", default=1)
    parse.add_option("--dynamic", action="store_true", dest="dynamic_terminals",
                     default=False)
    parse.add_option("--split", action="store_true", dest="split_training_set",
                     default=False)
    parse.add_option("--multiprocess", action="store_true", dest="multi_process",
                     default=False)
    parse.add_option("--log", dest="output_dir", default="./results",
                     help="Root directory for results. "
                          "Subfolders baseline/ and modified/ are created automatically.")
    (options, _) = parse.parse_args()

    SIMULATOR_TYPE = SimulatorTypeEnum(options.sgs)
    DECISION_TYPE = DecisionTypeEnum(options.decision_type)
    CPU_CORES = 8 if options.multi_process else 1

    # ── Dataset ───────────────────────────────────────────────────────────
    match options.dataset:
        case "MMLIB50":
            train_files = RCPSPDatabase.get_some_MMLIB_50_each_class_files(1, 4)
            val_files   = RCPSPDatabase.get_some_MMLIB_50_each_class_files(4, 5)
            test_files  = RCPSPDatabase.get_some_MMLIB_50_each_class_files(5, 6)
        case "MMLIB100":
            train_files = RCPSPDatabase.get_some_MMLIB_100_each_class_files(1, 4)
            val_files   = RCPSPDatabase.get_some_MMLIB_100_each_class_files(4, 5)
            test_files  = RCPSPDatabase.get_some_MMLIB_100_each_class_files(5, 6)
        case "MMLIBPLUS_50":
            train_files = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(1, 4)
            val_files   = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(4, 5)
            test_files  = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(5, 6)
        case "MMLIBPLUS_100":
            train_files = RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files(1, 4)
            val_files   = RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files(4, 5)
            test_files  = RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files(5, 6)
        case _:
            train_files = ["discrete_optimization_data/mm/MMLIB/MMLIB50/J501_4.mm"]
            val_files   = ["discrete_optimization_data/mm/MMLIB/MMLIB50/J501_3.mm",
                           "discrete_optimization_data/mm/MMLIB/MMLIB50/J501_4.mm"]
            test_files  = ["discrete_optimization_data/mm/MMLIB/MMLIB50/J501_5.mm"]

    training_set = read_instances(train_files)
    val_set      = read_instances(val_files)
    test_set     = read_instances(test_files)

    def make_providers():
        train_p = (EvenlyDividedDatasetProvider(training_set, 51)
                   if options.split_training_set and options.dataset
                   else StaticDatasetProvider(training_set))
        val_p   = StaticDatasetProvider(val_set)
        test_p  = StaticDatasetProvider(test_set)
        return train_p, val_p, test_p

    # ── Run baseline ──────────────────────────────────────────────────────
    train_p, val_p, test_p = make_providers()
    baseline_params = build_params(
        use_modifications=False,
        default=options.default,
        medium=options.medium,
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        cpu_cores=CPU_CORES,
        dynamic_terminals=options.dynamic_terminals,
        split_training=options.split_training_set,
    )
    run_config(
        label="baseline",
        use_modifications=False,
        params=baseline_params,
        training_provider=train_p,
        validation_provider=val_p,
        test_provider=test_p,
        n_runs=options.n_runs,
        start_index=options.start_index,
        seed=options.seed,
        output_dir=os.path.join(options.output_dir, "baseline"),
    )

    # ── Run with modifications ─────────────────────────────────────────────
    train_p, val_p, test_p = make_providers()
    modified_params = build_params(
        use_modifications=True,
        default=options.default,
        medium=options.medium,
        decision_type=DECISION_TYPE,
        simulator_type=SIMULATOR_TYPE,
        cpu_cores=CPU_CORES,
        dynamic_terminals=options.dynamic_terminals,
        split_training=options.split_training_set,
    )
    run_config(
        label="modified",
        use_modifications=True,
        params=modified_params,
        training_provider=train_p,
        validation_provider=val_p,
        test_provider=test_p,
        n_runs=options.n_runs,
        start_index=options.start_index,
        seed=options.seed,
        output_dir=os.path.join(options.output_dir, "modified"),
    )

    print("\nBoth configurations finished.")
    print(f"Baseline results : {os.path.join(options.output_dir, 'baseline')}/")
    print(f"Modified results : {os.path.join(options.output_dir, 'modified')}/")
    print("Run compare_results.py to see the summary table.")
