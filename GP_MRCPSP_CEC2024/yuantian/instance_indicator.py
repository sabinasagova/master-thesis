"""
Compute instance indicators for RCPSP problems.
Indicators include:
- Order Strength
- Resource Strength (renewable and non-renewable)
- Number of activities
- Number of modes
- Resource Factor (renewable and non-renewable)
You can find more details in the respective functions or the reference literature below.

Van Peteghem, V. and Vanhoucke, M., 2014,
"An experimental investigation of metaheuristics for the multi-mode resource-constrained project scheduling problem on new dataset instances",
European Journal of Operational Research, 235(1), 62–72 (doi:10.1016/j.ejor.2013.10.012).

The result is saved as 'instance_report.json' and 'instance_report.xlsx'.
"""
import logging
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
logging.basicConfig(level=logging.INFO)


def OrderStrength(problem: RCPSPModel):
    """
    the number of precedence relations (including the transitive ones) divided by the theoretical maximum number of precedence relations.
    Theoretical maximum precedence relations = n(n-1)/2
    OS can be 0.25/0.5/0.75
    """
    num_activities = problem.n_jobs_non_dummy
    num_precedence_relations = sum([
        len(relations)
        for relations in problem.graph.full_successors.values()
    ])
    # exclude relations with dummy start and finish
    num_precedence_relations -= num_activities
    num_precedence_relations -= (num_activities+1)

    max_precedence_relations = num_activities*(num_activities-1)/2

    return num_precedence_relations/max_precedence_relations


def ResourceStrength(problem: RCPSPModel):
    """
    RS=0 -> highly resource constrained problem
    RS=1 -> resource-unstrained CPM-case
    """
    pass


def ResourceStrength_NR(problem: RCPSPModel) -> float:
    """
    RS_r = (a_r - K_r^min)/(K_r^max - K_r^min)
    K_r^min and K_r^max are the minimum and maximum nonrenewable resource consumption 
    which can be obtained by cumulating the consumptions obtained
    when performing each activity in the mode having minimum and maximum consumptions.
    RS can be 0.25/0.5/0.75
    """
    NR = problem.non_renewable_resources_list
    capacity: dict = {
        res: problem.resources[res]
        for res in problem.non_renewable_resources_list
    }
    max_: dict = {
        res: sum(
            [
                max(
                    [
                        mode_detail[res]
                        for mode_id, mode_detail in modes.items()
                    ]
                )
                for act, modes in problem.mode_details.items()
            ]
        )
        for res in NR
    }
    min_: dict = {
        res: sum(
            [
                min(
                    [
                        mode_detail[res]
                        for mode_id, mode_detail in modes.items()
                    ]
                )
                for act, modes in problem.mode_details.items()
            ]
        )
        for res in NR
    }
    RS = {
        res: (capacity[res]-min_[res])/(max_[res]-min_[res])
        for res in NR
    }
    return RS


def ResourceStrength_R(problem: RCPSPModel):
    """    
    RS= (K_r - K_r^min)/(K_r^max - K_r^min)
    K_r^min： minimal availability of resource type in order to assure feasibility of RCPSP
    K_r^max: peak demand of resource type r in a CPM schedule. 
    RS can be 0.25/0.5/0.75
    RS=0 -> highly resource constrained problem
    RS=1 -> resource-unstrained CPM-case
    """
    R = problem.renewable_resources_list
    capacity: dict = {
        res: problem.resources[res]
        for res in R
    }
    min_: dict = {
        res: max(
            [
                min(
                    [
                        mode_detail[res]
                        for mode_id, mode_detail in modes.items()
                    ]
                )
                for act, modes in problem.mode_details.items()
            ]
        )
        for res in R
    }
    """
    Maximum value is determined via the resource dependant earliest start schedule 
    obtained when performing the activities in the lowest indexed modes.
    一般情况下，每个mode的duration是按照升序排列的
    """
    # Tailor to single-mode problem
    from discrete_optimization.rcpsp.transform_model import to_single_mode_rcpsp_model
    single_mode_problem = to_single_mode_rcpsp_model(problem)
    # single_mode_problem=problem
    # Generate CPM schedule
    single_mode_problem.cpm, single_mode_problem.cpm_esd = compute_cpm(
        single_mode_problem)
    resource_usage: dict = {
        res: [0]*single_mode_problem.horizon
        for res in R
    }
    for act_id, act_node in single_mode_problem.cpm.items():
        for t in range(act_node._ESD, act_node._EFD):
            for res in R:
                resource_usage[res][t] += single_mode_problem.mode_details[act_id][1][res]

    max_: dict = {
        res: max(
            resource_usage[res]
        )
        for res in R
    }

    RS = {
        res: (capacity[res]-min_[res])/(max_[res]-min_[res])
        for res in R
    }
    # stat: dict = {
    #     res: [max_[res], capacity[res]]
    #     for res in R
    # }
    # unconstrained = False
    # for res in R:
    #     if capacity[res] > max_[res]:
    #         unconstrained = True
    # if unconstrained:
    #     stat: dict = {
    #         res: f"{max_[res]} / {capacity[res]}"
    #         for res in R
    #     }
    #     logging.info("".join([problem.file_path, "\n", str(stat), "\n\n"]))
    return RS


def ResourcePeak_R(problem: RCPSPModel) -> dict:
    resource_usage = ResourceUsage_R(problem)
    max_: dict = {
        res: max(
            resource_usage[res]
        )
        for res in resource_usage
    }
    return max_


def ResourceUsage_R(problem: RCPSPModel) -> dict:
    """
    Maximum value is determined via the resource dependant earliest start schedule 
    obtained when performing the activities in the lowest indexed modes.
    一般情况下，每个mode的duration是按照升序排列的
    """
    # Tailor to single-mode problem
    from discrete_optimization.rcpsp.transform_model import to_single_mode_rcpsp_model
    from yuantian.gphh_solver import compute_cpm
    single_mode_problem = to_single_mode_rcpsp_model(problem)
    R = single_mode_problem.renewable_resources_list
    # single_mode_problem=problem
    single_mode_problem.cpm, single_mode_problem.cpm_esd = compute_cpm(
        single_mode_problem)
    resource_usage: dict = {
        res: [0]*single_mode_problem.horizon
        for res in R
    }
    for act_id, act_node in single_mode_problem.cpm.items():
        for t in range(act_node._ESD, act_node._EFD):
            for res in R:
                resource_usage[res][t] += single_mode_problem.mode_details[act_id][1][res]
    return resource_usage


def ResourceFactor_R(problem: RCPSPModel) -> float:
    """
    Number of requirements for resource types
    RF can be 0.5/1
    RF = 1 -> each non-dummy activity requests the full complement of all resource types
    RF = 0 -> None of activities requests any of the resource types.
    """
    sum_modes: int = 0
    sum_res_requests: int = 0
    n_res: int = len(problem.renewable_resources_list)
    for act in problem.tasks_list_non_dummy:
        for m in problem.mode_details[act].values():
            sum_modes += 1
            for res in problem.renewable_resources_list:
                if m[res] > 0:
                    sum_res_requests += 1
    return sum_res_requests/(sum_modes*n_res)


def ResourceFactor_NR(problem: RCPSPModel) -> float:
    """
    Number of requirements for resource types
    RF can be 0.5/1
    RF = 1 -> each non-dummy activity requests the full complement of all resource types
    RF = 0 -> None of activities requests any of the resource types.
    """
    sum_modes: int = 0
    sum_res_requests: int = 0
    n_res: int = len(problem.non_renewable_resources_list)
    for act in problem.tasks_list_non_dummy:
        for m in problem.mode_details[act].values():
            sum_modes += 1
            for res in problem.non_renewable_resources_list:
                if m[res] > 0:
                    sum_res_requests += 1
    return sum_res_requests/(sum_modes*n_res)


def Number_of_modes(problem: RCPSPModel):
    """
    Number of modes for each activity: 3/6/9
    """
    return problem.max_number_of_mode


def Number_of_activities(problem: RCPSPModel) -> int:
    """
    Number of activities in the project: 50/100
    Dummy nodes are excluded.
    """
    return problem.n_jobs_non_dummy


def Number_of_resources_R(problem: RCPSPModel):
    return len(problem.renewable_resources_list)


def Number_of_resources_NR(problem: RCPSPModel):
    return len(problem.non_renewable_resources_list)


def ResourceCapacity_R(problem: RCPSPModel):

    return {
        res: problem.resources[res]
        for res in problem.renewable_resources_list
    }


def plot_resource_usage(usage: list[int]):
    from matplotlib import pyplot as plt
    plt.figure()
    trimmed_values = list(usage)
    while trimmed_values[-1] == 0:
        trimmed_values.pop()
    x = list(range(len(trimmed_values)))
    plt.plot(x, trimmed_values)
    plt.xlim(0, len(x))
    plt.show()


if __name__ == "__main__":
    from yuantian.rcpsp_dataset import RCPSPDatabase
    files: list[str] = ["./discrete_optimization_data/mm//MMLIB//MMLIB50/J5097_1.mm"]
    files = RCPSPDatabase.get_all_MMLIB_PLUS_files()
    print(f"{len(files)} files total.")
    print("Reading instances...")
    from discrete_optimization.rcpsp.rcpsp_parser import parse_file
    problems = [
        parse_file(f)
        for f in files
    ]
    print("Reading completed!")
    report = []
    from gphh_solver import compute_cpm
    for problem in problems:
        problem.cpm, problem.cpm_esd = compute_cpm(problem)
        problem.graph.full_predecessors = problem.graph.ancestors_map()
        problem.graph.full_successors = problem.graph.descendants_map()
        instance_name = problem.file_path
        max_modes = Number_of_modes(problem)
        activities = Number_of_activities(problem)
        num_R = Number_of_resources_R(problem)
        num_NR = Number_of_resources_NR(problem)
        OS = OrderStrength(problem)
        RF_R = ResourceFactor_R(problem)
        RF_NR = ResourceFactor_NR(problem)
        RS_R = ResourceStrength_R(problem)
        RS_NR = ResourceStrength_NR(problem)
        resource_peak_R = ResourcePeak_R(problem)
        capacity_R = ResourceCapacity_R(problem)
        peak_capacity_R = [
            f"{res}:{resource_peak_R[res]}/{capacity_R[res]}"
            for res in problem.renewable_resources_list
        ]
        conditions: list[bool] = [
            True if resource_peak_R[res] <= capacity_R[res] else False
            for res in problem.renewable_resources_list
        ]
        unconstrained: bool = True if any(conditions) else False
        strict_unconstrained: bool = True if all(conditions) else False

        record: dict = {
            "instance_name": instance_name,
            "max_modes": max_modes,
            "number_activities": activities,
            "num_R": num_R,
            "num_NR": num_NR,
            "OS": OS,
            "RF_R": RF_R,
            "RF_NR": RF_NR,
            "RS_R": RS_R,
            "RS_NR": RS_NR,
            "Peak/Capacity_R": str(peak_capacity_R),
            "unconstrained": unconstrained,
            "strict_unconstrained": strict_unconstrained
        }
        report.append(record)

    import pandas as pd
    import json
    json.dump(report, open("./instance_report.json", mode="w"))
    pd.DataFrame(report).to_excel("./instance_report.xlsx", index=False)
    print("Finished!")
