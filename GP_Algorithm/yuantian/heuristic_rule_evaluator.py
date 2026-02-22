"""
This script evaluates various manually designed heuristic priority rules on a set of MRCPSP instances
from the MMLIB_PLUS dataset using both serial and parallel scheduling generation schemes.
It computes the average deviation from the CPM lower bound and the accumulated makespan
for each heuristic and records the results in a JSON file.
"""
from yuantian.gphh_solver import read_instances
from yuantian.rcpsp_dataset import StaticDatasetProvider, RCPSPDatabase
from yuantian.rcpsp_simulation import SerialSimulator, ParallelSimulator, FeatureEnum
from functools import partial
from typing import Union
import numpy as np
import json

# Some constant strings
NAME_str = "name" # name of the priority rule
EXTRE_str = "extre" # "min" or "max"
FUNC_str = "func" # function code of the priority rule
MODE_FUNC_str = "mode_rule" # function code of the mode selection rule
SGS_str = "sgs" # "serial" or "parallel"
DECISION_str = "decision_type" # "activity_first" or "mode_first"
FITNESS_str = "fitness" # fitness value


def run_test(test_set, simulator, heuristic_func):
    """
    2 indicators are recorded:
    (1) average deviation against CPM lower bound `vals`
    (2) accumated_makespan for the whole problem set `accumulated_makespan`
    """
    vals: list[Union[int, float]] = []
    accumulated_makespan = 0
    for domain in test_set:
        solution = simulator.buildSolution(domain, heuristic_func)
        if domain.satisfy(solution) == False:
            print(
                "infeasible solution! {simulator=}, {domain.file_path=}, {heuristic_func}"
            )
        # assert domain.satisfy(solution)==False, f"infeasible solution! {simulator=}, {domain.file_path=}, {heuristic_func}"
        do_makespan = solution.get_end_time(domain.sink_task)
        accumulated_makespan += do_makespan
        vals.append((do_makespan - domain.cpm_esd) * 100 / domain.cpm_esd)

    return [np.mean(vals), accumulated_makespan]


if __name__ == "__main__":
    # Uncomment to use all MMLIB_PLUS instances
    # test_set_files: list[str] = RCPSPDatabase.get_all_MMLIB_PLUS_files()

    # test_set_files: list[str] = RCPSPDatabase.get_instance_list_from_txt(r"./discrete_optimization_data/mm/MMLIB/selected_100.txt")
    # A small set for quick testing
    test_set_files: list[str] = [
        "discrete_optimization_data/mm/MMLIB/MMLIB50/J501_3.mm",
        "discrete_optimization_data/mm/MMLIB/MMLIB50/J501_5.mm",
        "discrete_optimization_data/mm//MMLIB//MMLIB100/J100100_2.mm",
    ]
    test_set = read_instances(test_set_files)

    priority_rules: list = [
        {NAME_str: "EarliestStart", EXTRE_str: "min", FUNC_str: "ES"},
        {NAME_str: "EarliestFinish", EXTRE_str: "min", FUNC_str: "EF"},
        {NAME_str: "LatestStart", EXTRE_str: "min", FUNC_str: "LS"},
        {NAME_str: "LatestFinish", EXTRE_str: "min", FUNC_str: "LF"},
        {NAME_str: "DynamicEarliestStart", EXTRE_str: "min", FUNC_str: "ES_d"},
        {NAME_str: "DynamicEarliestFinish", EXTRE_str: "min", FUNC_str: "EF_d"},
        {NAME_str: "DynamicLatestStart", EXTRE_str: "min", FUNC_str: "LS_d"},
        {NAME_str: "DynamicLatestFinish", EXTRE_str: "min", FUNC_str: "LF_d"},
        {NAME_str: "LatestStartandFinishTime", EXTRE_str: "min", FUNC_str: "LSTLFT"},
        # {
        #     NAME_str: "GreatestResourceDemand",
        #     EXTRE_str: "max",
        #     FUNC_str: "GRD"
        # },
        {NAME_str: "GreatRankPositionalWeight", EXTRE_str: "max", FUNC_str: "GRPW"},
        {
            NAME_str: "GreatRankPositionalWeightAll",
            EXTRE_str: "max",
            FUNC_str: "GRPW_all",
        },
        {NAME_str: "MostTotalSuccessor", EXTRE_str: "max", FUNC_str: "TSC"},
    ]
    mode_selection_rules: list = [
        {NAME_str: "SFM", EXTRE_str: "min", FUNC_str: "task_duration"}
    ]

    results: list = []
    # serial
    simulator = SerialSimulator()
    # serial + activity_first
    for rule in priority_rules:
        priority_rule = simulator.feature_function_map[FeatureEnum(
            rule[FUNC_str])]
        mode_rule = simulator.heuristic_earliest_feasible_finish_time
        heuristic_func = partial(simulator.activity_first_choose,
                                 priority_func=priority_rule,
                                 mode_func=mode_rule,
                                 priority_extre=rule[EXTRE_str],
                                 mode_extre="min")
        fitness = run_test(test_set, simulator, heuristic_func)
        record = {
            NAME_str: rule[FUNC_str],
            SGS_str: "serial",
            MODE_FUNC_str: "EFFT",
            DECISION_str: "activity_first",
            FITNESS_str: fitness[0]
        }
        print(record)
        results.append(record)

    # serial + mode_first
    for rule in priority_rules:
        priority_rule = simulator.feature_function_map[FeatureEnum(rule[FUNC_str])]
        mode_rule = simulator.heuristic_earliest_feasible_finish_time
        heuristic_func = partial(
            simulator.mode_first_choose,
            priority_func=priority_rule,
            mode_func=mode_rule,
            priority_extre=rule[EXTRE_str],
            mode_extre="min",
        )
        fitness = run_test(test_set, simulator, heuristic_func)
        record = {
            NAME_str: rule[FUNC_str],
            SGS_str: "serial",
            MODE_FUNC_str: "EFFT",
            DECISION_str: "mode_first",
            FITNESS_str: fitness[0],
        }
        print(record)
        results.append(record)

    # parallel + activity first
    simulator = ParallelSimulator()
    for rule in priority_rules:
        priority_rule = simulator.feature_function_map[FeatureEnum(rule[FUNC_str])]
        mode_rule = simulator.feature_duration
        heuristic_func = partial(
            simulator.activity_first_choose,
            priority_func=priority_rule,
            mode_func=mode_rule,
            priority_extre=rule[EXTRE_str],
            mode_extre="min",
        )
        fitness = run_test(test_set, simulator, heuristic_func)
        record = {
            NAME_str: rule[FUNC_str],
            SGS_str: "parallel",
            MODE_FUNC_str: "SFM",
            DECISION_str: "activity_first",
            FITNESS_str: fitness[0],
        }
        print(record)
        results.append(record)

    # parallel + mode first
    for rule in priority_rules:
        priority_rule = simulator.feature_function_map[FeatureEnum(rule[FUNC_str])]
        mode_rule = simulator.feature_duration
        heuristic_func = partial(
            simulator.activity_first_choose,
            priority_func=priority_rule,
            mode_func=mode_rule,
            priority_extre=rule[EXTRE_str],
            mode_extre="min",
        )
        fitness = run_test(test_set, simulator, heuristic_func)
        record = {
            NAME_str: rule[FUNC_str],
            SGS_str: "parallel",
            MODE_FUNC_str: "SFM",
            DECISION_str: "mode_first",
            FITNESS_str: fitness[0],
        }
        print(record)
        results.append(record)

    json.dump(
        results, open("20231106heuristics_result_MMLIB_PLUS_dynamic.json", mode="w")
    )
