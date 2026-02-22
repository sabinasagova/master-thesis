#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.rcpsp import RCPSPModelSpecialConstraintsPreemptive
from discrete_optimization.rcpsp.rcpsp_model import (
    RCPSPModel,
    SpecialConstraintsDescription,
)


def from_rcpsp_model(
    rcpsp_model: RCPSPModel,
    constraints: SpecialConstraintsDescription,
    preemptive=False,
):
    """Transform a model without special constraints into one with those.
    Also permits to pass from a classic RCPSP to a preemptive version

    Args:
        rcpsp_model (RCPSPModel): _description_
        constraints (SpecialConstraintsDescription): _description_
        preemptive (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if preemptive:
        return RCPSPModelSpecialConstraintsPreemptive(
            resources=rcpsp_model.resources,
            non_renewable_resources=rcpsp_model.non_renewable_resources_list,
            mode_details=rcpsp_model.mode_details,
            successors=rcpsp_model.successors,
            horizon=rcpsp_model.horizon,
            special_constraints=constraints,
            tasks_list=rcpsp_model.tasks_list,
            source_task=rcpsp_model.source_task,
            sink_task=rcpsp_model.sink_task,
            name_task=rcpsp_model.name_task,
        )
    return RCPSPModel(
        resources=rcpsp_model.resources,
        non_renewable_resources_list=rcpsp_model.non_renewable_resources_list,
        mode_details=rcpsp_model.mode_details,
        successors=rcpsp_model.successors,
        horizon=rcpsp_model.horizon,
        special_constraints=constraints,
        tasks_list=rcpsp_model.tasks_list,
        source_task=rcpsp_model.source_task,
        sink_task=rcpsp_model.sink_task,
        name_task=rcpsp_model.name_task,
    )


def to_renewable_only_rcpsp_model(
        rcpsp_model: RCPSPModel
) -> RCPSPModel:
    """Transform a model with non-renewable resources into one without those.

    Args:
        rcpsp_model (RCPSPModel): _description_

    Returns:
        RCPSPModel: _description_
    """
    renewable_only_model = rcpsp_model.copy()
    if renewable_only_model.non_renewable_resources_list:
        # Remove non-renewable resources in resources
        for res in renewable_only_model.non_renewable_resources_list:
            renewable_only_model.resources.pop(res)
            renewable_only_model.resources_list.remove(res)
        # Remove non-renewable resources in mode_details
        for act in renewable_only_model.mode_details:
            for m in renewable_only_model.mode_details[act]:
                for res in renewable_only_model.non_renewable_resources_list:
                    renewable_only_model.mode_details[act][m].pop(res)
        # Set non_renewable_resources as empty list
        renewable_only_model.non_renewable_resources_list = []

    return renewable_only_model


def to_single_mode_rcpsp_model(
        rcpsp_model: RCPSPModel
) -> RCPSPModel:
    single_mode_model = rcpsp_model.copy()
    for act, modes in single_mode_model.mode_details.items():
        for mode_id in list(modes.keys()):
            if mode_id != 1:
                modes.pop(mode_id)
    single_mode_model.max_number_of_mode = 1

    return single_mode_model
