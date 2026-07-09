"""
Manual (non-learned) priority-rule baselines, evaluated on the SAME held-out
test sets as experiment_runner.py's GPHH/DMGE head-to-head, to round out the
comparison with a second class of "existing approach" (Tian et al. 2024,
Table II, also compare their GPHH against these). Unlike the GP arms, these
rules are deterministic and untrained: no training set, no seeds, just one
decode-and-measure pass per rule per test instance.

Rules evaluated (serial SGS, activity-first decision, matching the GPHH
arms' DT/SIM in experiment_runner.py):
  EST-SFM     earliest-start-time priority, shortest-first mode selection
  GRPW-EFFT   greatest-rank-positional-weight priority, earliest-feasible-
              finish-time mode selection
  LSTLFT-EFFT latest-start-and-finish-time priority, EFFT mode selection --
              Tian et al.'s own reported best manual rule for MMLIB50/serial

Usage
-----
    python -m yuantian.manual_heuristic_baseline --datasets PSPLIB_J20 \
        --test 16 --out results/two_way_j20_v2
"""
import csv
import os
import sys
from functools import partial
from optparse import OptionParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from yuantian.experiment_runner import DATASET_SPLITS, evenly_sampled, DATASET_FETCHERS
from yuantian.gphh_solver import read_instances_with_nr
from yuantian.rcpsp_simulation import FeatureEnum, SerialSimulator

RULES = [
    ("manual_EST_SFM", FeatureEnum.EARLIEST_START_DATE, "min", "duration"),
    ("manual_GRPW_EFFT", FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT, "max", "efft"),
    ("manual_LSTLFT_EFFT", FeatureEnum.DYNAMIC_LATEST_START_AND_FINISH_TIME, "min", "efft"),
]


def load_test_set(dataset, n_test):
    fetch = DATASET_FETCHERS[dataset]
    _train_range, _val_range, test_range = DATASET_SPLITS[dataset]
    files = evenly_sampled(fetch(*test_range), n_test)
    return read_instances_with_nr(files)


def evaluate_rule(simulator, priority_feature, priority_extre, mode_kind, test_domains):
    priority_rule = simulator.feature_function_map[priority_feature]
    mode_rule = (simulator.feature_duration if mode_kind == "duration"
                 else simulator.heuristic_earliest_feasible_finish_time)
    heuristic = partial(simulator.activity_first_choose, priority_func=priority_rule,
                         mode_func=mode_rule, priority_extre=priority_extre, mode_extre="min")
    devs, feas_devs, n_feasible = [], [], 0
    for domain in test_domains:
        sol = simulator.buildSolution(domain=domain, choose=heuristic)
        mk = sol.get_end_time(domain.sink_task)
        dev = (mk - domain.cpm_esd) * 100 / domain.cpm_esd
        devs.append(dev)
        if getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7:
            n_feasible += 1
            feas_devs.append(dev)
    return (float(np.mean(devs)), n_feasible / len(test_domains),
            float(np.mean(feas_devs)) if feas_devs else float("nan"))


if __name__ == "__main__":
    parse = OptionParser()
    parse.add_option("--datasets", dest="datasets", default="PSPLIB_J20")
    parse.add_option("--test", dest="n_test", type="int", default=16)
    parse.add_option("--out", dest="out", default="results/two_way")
    (opt, _) = parse.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    datasets = [d.strip() for d in opt.datasets.split(",") if d.strip()]

    rows = []
    for dataset in datasets:
        test = load_test_set(dataset, opt.n_test)
        simulator = SerialSimulator()
        print(f"\n{dataset}  ({len(test)} test instances)")
        for label, feature, extre, mode_kind in RULES:
            dev_all, feasible, dev_feas = evaluate_rule(simulator, feature, extre, mode_kind, test)
            dev_feas_str = f"{dev_feas:.2f}" if not np.isnan(dev_feas) else "n/a"
            print(f"  {label:20s} dev_all={dev_all:8.2f}  feas={feasible*100:5.1f}%  dev_feas={dev_feas_str}")
            rows.append([dataset, label, dev_all, feasible, dev_feas])

    out_path = os.path.join(opt.out, "manual_baselines.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "dev_all", "feasible", "dev_feas"])
        w.writerows(rows)
    print(f"\nWrote {out_path}")
