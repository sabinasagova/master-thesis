#  Copyright (c) 2022-2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  This file is modified by Yuan Tian to implement GPHH for MRCPSP.


import logging
import multiprocessing
import operator
import os
import random
from functools import partial
from typing import Dict, List, Set, Union

import numpy as np
import psutil
from deap import creator, gp, tools
from deap.base import Fitness, Toolbox
from deap.gp import PrimitiveSet, PrimitiveTree, genGrow, genHalfAndHalf

from discrete_optimization.generic_rcpsp_tools.generic_rcpsp_solver import \
    SolverGenericRCPSP
from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction, Problem)
from discrete_optimization.rcpsp.solver.cpm import CPM
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import parse_file

from yuantian.gp_algorithms import mutBiased, standard_gp
from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_dataset import (DatasetProvider, EvenlyDividedDatasetProvider, RCPSPDatabase,
                                    StaticDatasetProvider)
from yuantian.rcpsp_simulation import (BackwardSerialSimulator, DecisionTypeEnum,
                                       FeatureEnum, ParallelSimulator, SerialSimulator,
                                       Simulator, SimulatorTypeEnum)
from yuantian.modifications import (ACTIVE_MODIFICATIONS, MUTATION_MODIFICATIONS,
                                    NR_TERMINALS, SCHEDULING_STATE_TERMINALS,
                                    OPPORTUNITY_TERMINALS)

logger = logging.getLogger(__name__)


# ── Baseline operator (Yuan Tian, CEC 2024) ───────────────────────────────
def if_then_else_operator(input1, output1, output2):
    if input1:
        return output1
    else:
        return output2


def protected_div(left, right):
    if right != 0.0:
        return left / right
    else:
        return 1.0


def max_operator(left, right):
    def max_():
        return max(left(), right())

    return max_


def min_operator(left, right):
    def min_():
        return min(left(), right())

    return min_


def negative_operator(terminal):
    def negative():
        return -terminal()

    return negative


def add_operator(left, right):
    def add():
        return left() + right()

    return add


def sub_operator(left, right):
    def sub():
        return left() - right()

    return sub


def mul_operator(left, right):
    def mul():
        return left() * right()

    return mul


def protected_div_operator(left, right):
    def protected_div():
        try:
            return left() / right()
        except ZeroDivisionError:
            return 1

    return protected_div


def compute_cpm(problem: ANY_RCPSP):
    cpm_solver = CPM(problem)
    path = cpm_solver.run_classic_cpm()
    cpm = cpm_solver.map_node
    cpm_esd = cpm[path[-1]]._ESD  # to normalize...
    return cpm, cpm_esd


class RefreshHallOfFame(tools.HallOfFame):
    def update(self, population):
        super().clear()  # del all fitnesses in the HOF
        super().update(population)


class ParametersGPHH:
    set_feature: Set[FeatureEnum] = None
    set_primitves: PrimitiveSet = None
    pop_size: int = None
    n_gen: int = None
    min_tree_depth: int = None
    max_tree_depth: int = None
    crossover_rate: float = None
    mutation_rate: float = None
    deap_verbose: bool = None
    simulator: SimulatorTypeEnum = None
    decision_type: DecisionTypeEnum = None

    def __init__(
            self,
            set_feature,
            set_primitves,
            n_tournament,
            pop_size,
            n_gen,
            n_elite,
            max_program_depth,
            init_min_tree_depth,
            init_max_tree_depth,
            crossover_rate,
            mutation_rate,
            mut_min_depth,
            mut_max_depth,
            deap_verbose,
            decision_type,
            simulator,
            cpu_cores=1,
    ):
        self.set_feature = set_feature
        self.set_primitves = set_primitves
        self.tournament_size = n_tournament
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_elite = n_elite
        self.max_program_depth = max_program_depth
        self.init_min_tree_depth = init_min_tree_depth
        self.init_max_tree_depth = init_max_tree_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mut_min_depth = mut_min_depth
        self.mut_max_depth = mut_max_depth
        self.deap_verbose = deap_verbose
        self.simulator = simulator
        self.decision_type = decision_type
        self.cpu_cores = cpu_cores
        self.use_modifications = False
        self.use_nr_terminals = False
        self.use_scheduling_state_terminals = False
        self.use_cp_mutation = False
        self.use_opportunity_terminals = False

    @staticmethod
    def init_simulator_pset(
            simulator_type: SimulatorTypeEnum = SimulatorTypeEnum.SERIAL_SGS,
    ):
        if simulator_type == SimulatorTypeEnum.SERIAL_SGS:
            simulator = SerialSimulator()
        elif simulator_type == SimulatorTypeEnum.PARALLEL_SGS:
            simulator = ParallelSimulator()
        elif simulator_type == SimulatorTypeEnum.BACKWARD_SERIAL_SGS:
            simulator = BackwardSerialSimulator()
        else:
            simulator = SerialSimulator()
        return simulator

    static_CPM_features = [
        FeatureEnum.EARLIEST_START_DATE,
        FeatureEnum.EARLIEST_FINISH_DATE,
        FeatureEnum.LATEST_START_DATE,
        FeatureEnum.LATEST_FINISH_DATE,
        FeatureEnum.SLACK,                
        FeatureEnum.IS_ON_CRITICAL_PATH,
    ]
    dynamic_CPM_features = [
        FeatureEnum.DYNAMIC_EARLIEST_START_DATE,
        FeatureEnum.DYNAMIC_EARLIEST_FINISH_DATE,
        FeatureEnum.DYNAMIC_LATEST_START_DATE,
        FeatureEnum.DYNAMIC_LATEST_FINISH_DATE,
        FeatureEnum.DYNAMIC_SLACK
    ]

    @staticmethod
    def init_feature_set(
            decision_type: DecisionTypeEnum = DecisionTypeEnum.ACTIVITY_THEN_MODE,
            simulator_type: SimulatorTypeEnum = SimulatorTypeEnum.SERIAL_SGS,
            dynamic_CPM: bool = False,
    ):
        if decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE:
            set_feature = {
                TerminalTypeEnum.ACTIVITY.value: [
                    # Prcedence relations
                    FeatureEnum.IMMEDIATE_PREDECESSOR_COUNT,
                    FeatureEnum.TOTAL_PREDECESSOR_COUNT,
                    FeatureEnum.IMMEDIATE_SUCCESSOR_COUNT,
                    FeatureEnum.TOTAL_SUCCESSOR_COUNT,
                    FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT,
                    FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT_ALL,
                    # Exclusive terminals for activity first
                    FeatureEnum.AVG_TASK_DURATION,
                    FeatureEnum.MAX_TASK_DURATION,
                    FeatureEnum.MIN_TASK_DURATION,
                    FeatureEnum.MIN_RESOURCE_REQUIREMENT_ACROSS_MODES,
                    FeatureEnum.MAX_RESOURCE_REQUIREMENT_ACROSS_MODES,
                    FeatureEnum.AVG_RESOURCE_REQUIREMENT_ACROSS_MODES,
                ],
                TerminalTypeEnum.MODE.value: [
                    FeatureEnum.TASK_DURATION,
                    FeatureEnum.DYNAMIC_EARLIEST_FEASIBLE_FINISH_TIME,
                    FeatureEnum.RESOURCE_REQUIRED,
                    FeatureEnum.GREATEST_RESOURCE_DEMAND,
                    FeatureEnum.MAX_RESOURCE_REQUIREMENT,
                    FeatureEnum.MIN_RESOURCE_REQUIREMENT,
                    FeatureEnum.AVG_RESOURCE_REQUIREMENT,
                    FeatureEnum.AVG_RESOURCE_CAPACITY,
                    FeatureEnum.MAX_RESOURCE_CAPACITY,
                    FeatureEnum.MIN_RESOURCE_CAPACITY,
                ],
            }
            set_feature[TerminalTypeEnum.ACTIVITY.value] += (
                ParametersGPHH.dynamic_CPM_features
                if dynamic_CPM
                else ParametersGPHH.static_CPM_features
            )
        elif decision_type == DecisionTypeEnum.MODE_THEN_ACTIVITY:
            set_feature = {
                TerminalTypeEnum.ACTIVITY.value: [
                    # Prcedence relations
                    FeatureEnum.IMMEDIATE_PREDECESSOR_COUNT,
                    FeatureEnum.TOTAL_PREDECESSOR_COUNT,
                    FeatureEnum.IMMEDIATE_SUCCESSOR_COUNT,
                    FeatureEnum.TOTAL_SUCCESSOR_COUNT,
                    FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT,
                    FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT_ALL,
                ],
                TerminalTypeEnum.MODE.value: [
                    FeatureEnum.TASK_DURATION,
                    FeatureEnum.DYNAMIC_EARLIEST_FEASIBLE_FINISH_TIME,
                    FeatureEnum.RESOURCE_REQUIRED,
                    FeatureEnum.GREATEST_RESOURCE_DEMAND,
                    FeatureEnum.MAX_RESOURCE_REQUIREMENT,
                    FeatureEnum.MIN_RESOURCE_REQUIREMENT,
                    FeatureEnum.AVG_RESOURCE_REQUIREMENT,
                    FeatureEnum.AVG_RESOURCE_CAPACITY,
                    FeatureEnum.MAX_RESOURCE_CAPACITY,
                    FeatureEnum.MIN_RESOURCE_CAPACITY,
                ],
            }
            set_feature[TerminalTypeEnum.ACTIVITY.value] += (
                ParametersGPHH.dynamic_CPM_features
                if dynamic_CPM
                else ParametersGPHH.static_CPM_features
            )
        elif decision_type == DecisionTypeEnum.SIMULTANEOUS:
            set_feature = {
                TerminalTypeEnum.INTEGRATED.value: [
                    FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT,
                    FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT_ALL,
                    FeatureEnum.IMMEDIATE_PREDECESSOR_COUNT,
                    FeatureEnum.TOTAL_PREDECESSOR_COUNT,
                    FeatureEnum.IMMEDIATE_SUCCESSOR_COUNT,
                    FeatureEnum.TOTAL_SUCCESSOR_COUNT,
                    FeatureEnum.TASK_DURATION,
                    FeatureEnum.DYNAMIC_EARLIEST_FEASIBLE_FINISH_TIME,
                    FeatureEnum.RESOURCE_REQUIRED,
                    FeatureEnum.GREATEST_RESOURCE_DEMAND,
                    FeatureEnum.MAX_RESOURCE_REQUIREMENT,
                    FeatureEnum.MIN_RESOURCE_REQUIREMENT,
                    FeatureEnum.AVG_RESOURCE_REQUIREMENT,
                    FeatureEnum.AVG_RESOURCE_CAPACITY,
                    FeatureEnum.MAX_RESOURCE_CAPACITY,
                    FeatureEnum.MIN_RESOURCE_CAPACITY,
                ]
            }
            set_feature[TerminalTypeEnum.INTEGRATED.value] += (
                ParametersGPHH.dynamic_CPM_features
                if dynamic_CPM
                else ParametersGPHH.static_CPM_features
            )
        # if simulator_type == SimulatorTypeEnum.SERIAL_SGS:
        #     if decision_type == DecisionTypeEnum.SIMULTANEOUS:
        #         set_feature[TerminalTypeEnum.INTEGRATED.value].append(
        #             FeatureEnum.DYNAMIC_EARLIEST_FEASIBLE_FINISH_TIME
        #         )
        #     else:
        #         set_feature[TerminalTypeEnum.MODE.value].append(
        #             FeatureEnum.DYNAMIC_EARLIEST_FEASIBLE_FINISH_TIME
        #         )

        return set_feature

    @staticmethod
    def default(
            simulator_type: SimulatorTypeEnum = SimulatorTypeEnum.SERIAL_SGS,
            decision_type: DecisionTypeEnum = DecisionTypeEnum.ACTIVITY_THEN_MODE,
            dynamic_CPM_feature: bool = False,
            fixed_activity_rule="",
            fixed_mode_rule="",
            cpu=1,
            use_modifications: bool = False,
            use_nr_terminals: bool = False,
            use_scheduling_state_terminals: bool = False,
            use_cp_mutation: bool = False,
            use_opportunity_terminals: bool = False,
    ):
        simulator = ParametersGPHH.init_simulator_pset(simulator_type)
        set_feature = ParametersGPHH.init_feature_set(
            decision_type, simulator_type, dynamic_CPM_feature
        )
        pset: Dict[
            TerminalTypeEnum, PrimitiveSet
        ] = {}  # {DecisionTypeEnum: PrimitiveSet}
        if fixed_activity_rule or fixed_mode_rule:
            if fixed_activity_rule:
                set_feature[TerminalTypeEnum.ACTIVITY.value] = [
                    FeatureEnum(fixed_activity_rule)
                ]
            if fixed_mode_rule:
                set_feature[TerminalTypeEnum.MODE.value] = [
                    FeatureEnum(fixed_mode_rule)
                ]
        for terminal_type in set_feature:
            pset[terminal_type] = PrimitiveSet(decision_type, 0)
            # add terminal set
            for feature in set_feature[terminal_type]:
                pset[terminal_type].addTerminal(
                    simulator.feature_function_map[feature], feature.value
                )
            # inject NR-awareness terminals (Modification 2)
            if use_nr_terminals:
                for feature in NR_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
            # inject scheduling-state terminals (Modification 3)
            if use_scheduling_state_terminals:
                for feature in SCHEDULING_STATE_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
            # inject dynamic urgency / mode-regret terminals (Modification 7, exploratory)
            if use_opportunity_terminals:
                for feature in OPPORTUNITY_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
            # add function set
            pset[terminal_type].addPrimitive(add_operator, 2, name="add")
            if fixed_activity_rule and terminal_type == TerminalTypeEnum.ACTIVITY.value:
                continue
            if fixed_mode_rule and terminal_type == TerminalTypeEnum.MODE.value:
                continue
            pset[terminal_type].addPrimitive(sub_operator, 2, name="sub")
            pset[terminal_type].addPrimitive(mul_operator, 2, name="mul")
            pset[terminal_type].addPrimitive(protected_div_operator, 2, name="div")
            pset[terminal_type].addPrimitive(min_operator, 2, name="min")
            pset[terminal_type].addPrimitive(max_operator, 2, name="max")
            if_else_op = ACTIVE_MODIFICATIONS.get("if_else", if_then_else_operator) if use_modifications else if_then_else_operator
            pset[terminal_type].addPrimitive(if_else_op, 3, name="if_else")
            pset[terminal_type].addPrimitive(negative_operator, 1, name="neg")

        if decision_type == DecisionTypeEnum.SIMULTANEOUS:
            init_min_tree_depth = {TerminalTypeEnum.INTEGRATED.value: 2}
            init_max_tree_depth = {TerminalTypeEnum.INTEGRATED.value: 6}
            mut_min_depth = {TerminalTypeEnum.INTEGRATED.value: 4}
            mut_max_depth = {TerminalTypeEnum.INTEGRATED.value: 4}
            max_program_depth = {TerminalTypeEnum.INTEGRATED.value: 8}
        elif (
                decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE
                or decision_type == DecisionTypeEnum.MODE_THEN_ACTIVITY
        ):
            init_min_tree_depth = {
                TerminalTypeEnum.ACTIVITY.value: 2,
                TerminalTypeEnum.MODE.value: 2,
            }
            init_max_tree_depth = {
                TerminalTypeEnum.ACTIVITY.value: 6,
                TerminalTypeEnum.MODE.value: 6,
            }
            mut_min_depth = {
                TerminalTypeEnum.ACTIVITY.value: 4,
                TerminalTypeEnum.MODE.value: 4,
            }
            mut_max_depth = {
                TerminalTypeEnum.ACTIVITY.value: 4,
                TerminalTypeEnum.MODE.value: 4,
            }
            max_program_depth = {
                TerminalTypeEnum.ACTIVITY.value: 8,
                TerminalTypeEnum.MODE.value: 8,
            }
            if fixed_activity_rule:
                init_min_tree_depth[TerminalTypeEnum.ACTIVITY.value] = 1
                init_max_tree_depth[TerminalTypeEnum.ACTIVITY.value] = 1
                mut_min_depth[TerminalTypeEnum.ACTIVITY.value] = 1
                mut_max_depth[TerminalTypeEnum.ACTIVITY.value] = 1
                max_program_depth[TerminalTypeEnum.ACTIVITY.value] = 1
            if fixed_mode_rule:
                init_min_tree_depth[TerminalTypeEnum.MODE.value] = 1
                init_max_tree_depth[TerminalTypeEnum.MODE.value] = 1
                mut_min_depth[TerminalTypeEnum.MODE.value] = 1
                mut_max_depth[TerminalTypeEnum.MODE.value] = 1
                max_program_depth[TerminalTypeEnum.MODE.value] = 1

        params = ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            n_tournament=7,
            pop_size=1000,
            n_elite=10,
            n_gen=50,
            max_program_depth=max_program_depth,
            init_min_tree_depth=init_min_tree_depth,
            init_max_tree_depth=init_max_tree_depth,
            crossover_rate=0.8,
            mutation_rate=0.15,
            mut_min_depth=mut_min_depth,
            mut_max_depth=mut_max_depth,
            deap_verbose=True,
            simulator=simulator,
            decision_type=decision_type,
            cpu_cores=cpu,
        )
        params.use_modifications = use_modifications
        params.use_nr_terminals = use_nr_terminals
        params.use_scheduling_state_terminals = use_scheduling_state_terminals
        params.use_cp_mutation = use_cp_mutation
        params.use_opportunity_terminals = use_opportunity_terminals
        return params

    @staticmethod
    def medium(
            simulator_type: SimulatorTypeEnum = SimulatorTypeEnum.SERIAL_SGS,
            decision_type: DecisionTypeEnum = DecisionTypeEnum.ACTIVITY_THEN_MODE,
            dynamic_CPM_feature: bool = False,
            cpus=1,
            use_modifications: bool = False,
            use_nr_terminals: bool = False,
            use_scheduling_state_terminals: bool = False,
            use_cp_mutation: bool = False,
            use_opportunity_terminals: bool = False,
    ):
        """Intermediate config: pop=50, gen=10. Quick experiment runs with ETA logging."""
        simulator = ParametersGPHH.init_simulator_pset(simulator_type)
        set_feature = ParametersGPHH.init_feature_set(
            decision_type, simulator_type, dynamic_CPM_feature
        )
        pset: Dict[TerminalTypeEnum, PrimitiveSet] = {}
        for terminal_type in set_feature:
            pset[terminal_type] = PrimitiveSet(decision_type, 0)
            pset[terminal_type].addPrimitive(add_operator, 2, name="add")
            pset[terminal_type].addPrimitive(sub_operator, 2, name="sub")
            pset[terminal_type].addPrimitive(mul_operator, 2, name="mul")
            pset[terminal_type].addPrimitive(protected_div_operator, 2, name="div")
            pset[terminal_type].addPrimitive(min_operator, 2, name="min")
            pset[terminal_type].addPrimitive(max_operator, 2, name="max")
            if_else_op = ACTIVE_MODIFICATIONS.get("if_else", if_then_else_operator) if use_modifications else if_then_else_operator
            pset[terminal_type].addPrimitive(if_else_op, 3, name="if_else")
            for feature in set_feature[terminal_type]:
                pset[terminal_type].addTerminal(
                    simulator.feature_function_map[feature], feature.value
                )
            if use_nr_terminals:
                for feature in NR_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
            if use_scheduling_state_terminals:
                for feature in SCHEDULING_STATE_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
            if use_opportunity_terminals:
                for feature in OPPORTUNITY_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
        if decision_type == DecisionTypeEnum.SIMULTANEOUS:
            init_min_tree_depth = {TerminalTypeEnum.INTEGRATED.value: 2}
            init_max_tree_depth = {TerminalTypeEnum.INTEGRATED.value: 6}
            mut_min_depth       = {TerminalTypeEnum.INTEGRATED.value: 4}
            mut_max_depth       = {TerminalTypeEnum.INTEGRATED.value: 4}
            max_program_depth   = {TerminalTypeEnum.INTEGRATED.value: 8}
        else:
            init_min_tree_depth = {TerminalTypeEnum.ACTIVITY.value: 2, TerminalTypeEnum.MODE.value: 2}
            init_max_tree_depth = {TerminalTypeEnum.ACTIVITY.value: 6, TerminalTypeEnum.MODE.value: 6}
            mut_min_depth       = {TerminalTypeEnum.ACTIVITY.value: 4, TerminalTypeEnum.MODE.value: 4}
            mut_max_depth       = {TerminalTypeEnum.ACTIVITY.value: 4, TerminalTypeEnum.MODE.value: 4}
            max_program_depth   = {TerminalTypeEnum.ACTIVITY.value: 8, TerminalTypeEnum.MODE.value: 8}

        params = ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            n_tournament=5,
            pop_size=50,
            n_elite=5,
            n_gen=10,
            max_program_depth=max_program_depth,
            init_min_tree_depth=init_min_tree_depth,
            init_max_tree_depth=init_max_tree_depth,
            crossover_rate=0.8,
            mutation_rate=0.15,
            mut_min_depth=mut_min_depth,
            mut_max_depth=mut_max_depth,
            deap_verbose=True,
            simulator=simulator,
            decision_type=decision_type,
            cpu_cores=cpus,
        )
        params.use_modifications = use_modifications
        params.use_nr_terminals = use_nr_terminals
        params.use_scheduling_state_terminals = use_scheduling_state_terminals
        params.use_opportunity_terminals = use_opportunity_terminals
        params.use_cp_mutation = use_cp_mutation
        return params

    @staticmethod
    def fast(
            simulator_type: SimulatorTypeEnum = SimulatorTypeEnum.SERIAL_SGS,
            decision_type: DecisionTypeEnum = DecisionTypeEnum.ACTIVITY_THEN_MODE,
            dynamic_CPM_feature: bool = False,
            cpus=1,
            use_modifications: bool = False,
            use_nr_terminals: bool = False,
            use_scheduling_state_terminals: bool = False,
            use_cp_mutation: bool = False,
    ):
        simulator = ParametersGPHH.init_simulator_pset(simulator_type)
        set_feature = ParametersGPHH.init_feature_set(
            decision_type, simulator_type, dynamic_CPM_feature
        )
        # {DecisionTypeEnum: PrimitiveSet}
        pset: Dict[TerminalTypeEnum, PrimitiveSet] = {}
        for terminal_type in set_feature:
            pset[terminal_type] = PrimitiveSet(decision_type, 0)
            # add function set
            pset[terminal_type].addPrimitive(add_operator, 2, name="add")
            pset[terminal_type].addPrimitive(sub_operator, 2, name="sub")
            pset[terminal_type].addPrimitive(mul_operator, 2, name="mul")
            pset[terminal_type].addPrimitive(protected_div_operator, 2, name="div")
            pset[terminal_type].addPrimitive(min_operator, 2, name="min")
            pset[terminal_type].addPrimitive(max_operator, 2, name="max")
            if_else_op = ACTIVE_MODIFICATIONS.get("if_else", if_then_else_operator) if use_modifications else if_then_else_operator
            pset[terminal_type].addPrimitive(if_else_op, 3, name="if_else")
            # add terminal set
            for feature in set_feature[terminal_type]:
                pset[terminal_type].addTerminal(
                    simulator.feature_function_map[feature], feature.value
                )
            if use_nr_terminals:
                for feature in NR_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
            if use_scheduling_state_terminals:
                for feature in SCHEDULING_STATE_TERMINALS.get(terminal_type, []):
                    pset[terminal_type].addTerminal(
                        simulator.feature_function_map[feature], feature.value
                    )
        if decision_type == DecisionTypeEnum.SIMULTANEOUS:
            init_min_tree_depth = {TerminalTypeEnum.INTEGRATED.value: 2}
            init_max_tree_depth = {TerminalTypeEnum.INTEGRATED.value: 6}
            mut_min_depth = {TerminalTypeEnum.INTEGRATED.value: 4}
            mut_max_depth = {TerminalTypeEnum.INTEGRATED.value: 4}
            max_program_depth = {TerminalTypeEnum.INTEGRATED.value: 8}
        elif (
                decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE
                or decision_type == DecisionTypeEnum.MODE_THEN_ACTIVITY
        ):
            init_min_tree_depth = {
                TerminalTypeEnum.ACTIVITY.value: 2,
                TerminalTypeEnum.MODE.value: 2,
            }
            init_max_tree_depth = {
                TerminalTypeEnum.ACTIVITY.value: 6,
                TerminalTypeEnum.MODE.value: 6,
            }
            mut_min_depth = {
                TerminalTypeEnum.ACTIVITY.value: 4,
                TerminalTypeEnum.MODE.value: 4,
            }
            mut_max_depth = {
                TerminalTypeEnum.ACTIVITY.value: 4,
                TerminalTypeEnum.MODE.value: 4,
            }
            max_program_depth = {
                TerminalTypeEnum.ACTIVITY.value: 8,
                TerminalTypeEnum.MODE.value: 8,
            }

        params = ParametersGPHH(
            set_feature=set_feature,
            set_primitves=pset,
            n_tournament=2,
            pop_size=25,
            n_elite=1,
            n_gen=5,
            max_program_depth=max_program_depth,
            init_min_tree_depth=init_min_tree_depth,
            init_max_tree_depth=init_max_tree_depth,
            crossover_rate=0.8,
            mutation_rate=0.15,
            mut_min_depth=mut_min_depth,
            mut_max_depth=mut_max_depth,
            deap_verbose=True,
            simulator=simulator,
            decision_type=decision_type,
            cpu_cores=cpus,
        )
        params.use_modifications = use_modifications
        params.use_nr_terminals = use_nr_terminals
        params.use_scheduling_state_terminals = use_scheduling_state_terminals
        params.use_cp_mutation = use_cp_mutation
        return params

    @staticmethod
    def get_complete_primitive_set(simulator: Simulator) -> PrimitiveSet:
        pset: PrimitiveSet = PrimitiveSet("MAIN", 0)
        pset.addPrimitive(add_operator, 2, name="add")
        pset.addPrimitive(sub_operator, 2, name="sub")
        pset.addPrimitive(mul_operator, 2, name="mul")
        pset.addPrimitive(protected_div_operator, 2, name="div")
        pset.addPrimitive(min_operator, 2, name="min")
        pset.addPrimitive(max_operator, 2, name="max")

        for feature in FeatureEnum:
            pset.addTerminal(simulator.feature_function_map[feature], feature.value)

        return pset


class GPHH(SolverGenericRCPSP):
    training_data_provider: List[Problem]
    weight: int
    pset: PrimitiveSet
    toolbox: Toolbox
    params_gphh: ParametersGPHH
    evaluation_method: SimulatorTypeEnum
    reference_permutations: Dict

    def __init__(
            self,
            training_set_provider: DatasetProvider,
            validation_set_provider: DatasetProvider = None,
            test_set_provider: DatasetProvider = None,
            rcpsp_model: Problem = None,
            weight: int = -1,
            params_gphh: ParametersGPHH = None,
            params_objective_function: ParamsObjectiveFunction = None,
    ):
        # This framework was originally used to solve a single RCPSP problem
        # GPHH inherits from SolverGenricRCPSP, self.rcpsp_model is the problem needs to be solved
        SolverGenericRCPSP.__init__(self, rcpsp_model=rcpsp_model)
        self.training_data_provider = training_set_provider
        self.validation_data_provider = validation_set_provider
        self.test_data_provider = test_set_provider
        self.params_gphh = params_gphh
        if self.params_gphh is None:
            self.params_gphh = ParametersGPHH.fast()
        self.set_feature = self.params_gphh.set_feature
        self.pset: Dict[
            TerminalTypeEnum, list[FeatureEnum]
        ] = self.params_gphh.set_primitves
        self.weight = weight
        self.simulator: Simulator = self.params_gphh.simulator
        self.decision_type = self.params_gphh.decision_type
        self.toolbox = None

    def init_model(self):
        tournament_size = self.params_gphh.tournament_size
        init_min_tree_depth = self.params_gphh.init_min_tree_depth
        init_max_tree_depth = self.params_gphh.init_max_tree_depth
        mut_min_depth = self.params_gphh.mut_min_depth
        mut_max_depth = self.params_gphh.mut_max_depth
        max_program_depth = self.params_gphh.max_program_depth

        creator.create("FitnessMin", Fitness, weights=(self.weight,))

        self.toolbox = Toolbox()
        # multi-process map
        if self.params_gphh.cpu_cores > 1:
            pool = multiprocessing.Pool(processes=self.params_gphh.cpu_cores)
            self.toolbox.register("map", pool.map)

        if self.decision_type == DecisionTypeEnum.SIMULTANEOUS:
            # single tree
            creator.create("Individual", PrimitiveTree, fitness=creator.FitnessMin)
            self.toolbox.register(
                "expr",
                genHalfAndHalf,
                pset=self.pset[TerminalTypeEnum.INTEGRATED.value],
                min_=init_min_tree_depth[TerminalTypeEnum.INTEGRATED.value],
                max_=init_max_tree_depth[TerminalTypeEnum.INTEGRATED.value],
            )
            self.toolbox.register(
                "individual", tools.initIterate, creator.Individual, self.toolbox.expr
            )
            self.toolbox.register(
                "population", tools.initRepeat, list, self.toolbox.individual
            )
            self.toolbox.register("compile", gp.compile)

            self.toolbox.register(
                "evaluate",
                evaluate_heuristic,
                compile_func=self.toolbox.compile,
                pset=self.pset,
                decision_type=self.decision_type,
                simulator=self.simulator,
            )
            self.toolbox.register(
                "select", tools.selTournament, tournsize=tournament_size
            )
            self.toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
            self.toolbox.register(
                "expr_mut",
                gp.genGrow,
                min_=mut_min_depth[TerminalTypeEnum.INTEGRATED.value],
                max_=mut_max_depth[TerminalTypeEnum.INTEGRATED.value],
            )
            _mut_fn = (MUTATION_MODIFICATIONS.get("mutate", mutBiased)
                       if self.params_gphh.use_cp_mutation else mutBiased)
            self.toolbox.register(
                "mutate",
                _mut_fn,
                expr=self.toolbox.expr_mut,
                pset=self.pset[TerminalTypeEnum.INTEGRATED.value],
                nonterminal_prob=0.9,
                terminal_prob=0.1,
                root_prob=0,
            )
            self.toolbox.decorate(
                "mate",
                gp.staticLimit(
                    key=operator.attrgetter("height"),
                    max_value=max_program_depth[TerminalTypeEnum.INTEGRATED.value],
                ),
            )
            self.toolbox.decorate(
                "mutate",
                gp.staticLimit(
                    key=operator.attrgetter("height"),
                    max_value=max_program_depth[TerminalTypeEnum.INTEGRATED.value],
                ),
            )

        else:
            # multi-tree
            import multitreegp

            creator.create(
                "Individual", multitreegp.MultiPrimitiveTree, fitness=creator.FitnessMin
            )
            self.toolbox.register(
                "expr",
                lambda: {
                    TerminalTypeEnum.ACTIVITY.value: genHalfAndHalf(
                        pset=self.pset[TerminalTypeEnum.ACTIVITY.value],
                        min_=init_min_tree_depth[TerminalTypeEnum.ACTIVITY.value],
                        max_=init_max_tree_depth[TerminalTypeEnum.ACTIVITY.value],
                    ),
                    TerminalTypeEnum.MODE.value: genHalfAndHalf(
                        pset=self.pset[TerminalTypeEnum.MODE.value],
                        min_=init_min_tree_depth[TerminalTypeEnum.MODE.value],
                        max_=init_max_tree_depth[TerminalTypeEnum.MODE.value],
                    ),
                },
            )
            self.toolbox.register(
                "individual", tools.initIterate, creator.Individual, self.toolbox.expr
            )
            self.toolbox.register(
                "population", tools.initRepeat, list, self.toolbox.individual
            )
            self.toolbox.register("compile", gp.compile)
            self.toolbox.register(
                "evaluate",
                evaluate_heuristic,
                compile_func=self.toolbox.compile,
                pset=self.pset,
                decision_type=self.decision_type,
                simulator=self.simulator,
            )
            self.toolbox.register(
                "select", tools.selTournament, tournsize=tournament_size
            )
            self.toolbox.register("mate", multitreegp.cxOnePoint_type_wise_leaf_biased, termpb=0.1)
            expr_mut = {
                TerminalTypeEnum.ACTIVITY.value: partial(
                    genGrow,
                    min_=mut_min_depth[TerminalTypeEnum.ACTIVITY.value],
                    max_=mut_max_depth[TerminalTypeEnum.ACTIVITY.value],
                ),
                TerminalTypeEnum.MODE.value: partial(
                    genGrow,
                    min_=mut_min_depth[TerminalTypeEnum.MODE.value],
                    max_=mut_max_depth[TerminalTypeEnum.MODE.value],
                ),
            }
            _mut_fn = (MUTATION_MODIFICATIONS.get("mutate", mutBiased)
                       if self.params_gphh.use_cp_mutation else mutBiased)
            self.toolbox.register(
                "mutate",
                multitreegp.multi_tree_mutate,
                expr=expr_mut,
                pset=self.pset,
                mutate_func=partial(
                    _mut_fn,
                    nonterminal_prob=0.9,
                    terminal_prob=0.1,
                    root_prob=0,
                ),
            )

            self.toolbox.decorate(
                "mate",
                multitreegp.staticLimit(
                    key=operator.attrgetter("height"),
                    max_value=max_program_depth,
                ),
            )
            self.toolbox.decorate(
                "mutate",
                multitreegp.staticLimit(
                    key=operator.attrgetter("height"),
                    max_value=max_program_depth,
                ),
            )

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        # stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

    def solve(self, **kwargs):
        if self.toolbox is None:
            self.init_model()
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        pop = self.toolbox.population(n=self.params_gphh.pop_size)
        hof = RefreshHallOfFame(1)
        self.hof = hof
        from utils import PopulationArchive

        pop_archive = PopulationArchive()
        import time

        start = time.time()
        pop, log = standard_gp(
            pop,
            self.toolbox,
            cxpb=self.params_gphh.crossover_rate,
            mutpb=self.params_gphh.mutation_rate,
            n_elite=self.params_gphh.n_elite,
            ngen=self.params_gphh.n_gen,
            training_data_provider=self.training_data_provider,
            validation_data_provider=self.validation_data_provider,
            stats=mstats,
            halloffame=hof,
            pop_archive=pop_archive,
            verbose=self.params_gphh.deap_verbose,
        )
        elapsed = time.time() - start

        print(f"Running time: {elapsed}")
        self.best_heuristic = hof[0]
        logger.debug(f"best_heuristic: {self.best_heuristic}")
        output_path = kwargs.get("output_path", "result.json")

        test_data: dict = {}
        if all([self.validation_data_provider, self.test_data_provider]):
            # Put current best_heuristic into test set
            test_set = self.test_data_provider.next()
            if test_set:
                total_dev_percent = self.toolbox.evaluate(
                    individual=self.best_heuristic, domains=test_set
                )
                test_data["best_heuristic"] = {
                    "tree": str(self.best_heuristic),
                    "fitness": self.best_heuristic.fitness.values[0],
                    "test_fitness": total_dev_percent[0],
                }
                # Apply validation test to choose the best individual
                min_deviation = 100000
                best_validated_individual = None
                validation_set = self.validation_data_provider.next()
                if validation_set:
                    validation_evaluate = partial(
                        self.toolbox.evaluate, domains=validation_set
                    )
                    validation_fitnesses = self.toolbox.map(validation_evaluate, pop)
                    for ind, total_dev_percent in zip(pop, validation_fitnesses):
                        if total_dev_percent[0] < min_deviation:
                            min_deviation = total_dev_percent[0]
                            best_validated_individual = ind
                    # Put this ind test set
                    test_set = self.test_data_provider.next()
                    total_dev_percent = self.toolbox.evaluate(
                        individual=best_validated_individual, domains=test_set
                    )
                    test_data["best_heuristic_validation"] = {
                        "tree": str(best_validated_individual),
                        "fitness": best_validated_individual.fitness.values[0],
                        "validation_fitness": min_deviation,
                        "test_fitness": total_dev_percent[0],
                    }

        self.write_result(
            log, filepath=output_path, pop_archive=pop_archive, elapsed=elapsed, others=test_data
        )

    def write_result(self, log: dict, filepath: str, pop_archive, elapsed=0, others: dict = None):
        """Write configuration, fitness, etc to a json file for further analysis

        Args:
            log (dict): log generated by algorithms
            filepath (str): output filepath
            pop_archive (list): An archive which stores individuals in all generation
        """
        import datetime
        import json

        # set up data
        pset: dict = {
            type: [k for k in self.pset[type].mapping.keys()] for type in self.pset
        }
        config: dict = {
            "simulator": str(self.simulator.__class__),
            "decision_type": self.params_gphh.decision_type.value,
            "pop_size": self.params_gphh.pop_size,
            "gen": self.params_gphh.n_gen,
            "selection": ".".join(
                [self.toolbox.select.func.__module__, self.toolbox.select.func.__name__]
            ),
            "cx_operator": ".".join(
                [self.toolbox.mate.func.__module__, self.toolbox.mate.func.__name__]
            ),
            "mut_operator": ".".join(
                [
                    self.toolbox.mutate.func.__module__,
                    self.toolbox.mutate.func.__name__,
                ]
            ),
            "cx_rate": self.params_gphh.crossover_rate,
            "mut_rate": self.params_gphh.mutation_rate,
            "tournament_size": self.params_gphh.tournament_size,
            "tree_type": creator.Individual.reduce_args[1].__name__,
            "pset": pset,
        }
        """
        pop_archive: list[list[Individual]]
        E.g. [
                [ind, ind],  <--- gen 0
                [ind, ind],  <--- gen 1
                ...
            ]
        """
        population: list = [
            {"gen": gen, "tree": str(ind), "fitness": ind.fitness.values[0]}
            for gen, pop in enumerate(pop_archive)
            for ind in pop
        ]
        fitness_log: list = log.chapters["fitness"]
        generation_best_log: list = log.chapters["generation_best"]

        result: dict = {
            "configuration": config,
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": elapsed,
            "population": population,
            "fitness": fitness_log,
            "generation_best": generation_best_log,
        }
        if others:
            # add other data into result
            for key, value in others.items():
                result[key] = value
        # write to file
        json.dump(result, fp=open(filepath, mode="w"))

    def init_primitives(self, pset) -> PrimitiveSet:
        for i in range(len(self.list_feature)):
            pset.renameArguments(**{"ARG" + str(i): self.list_feature[i].value})
        return pset


def evaluate_heuristic(
        individual,
        domains: ANY_RCPSP,
        compile_func: callable,
        pset,
        decision_type: DecisionTypeEnum,
        simulator: Simulator,
        heuristic_func: callable = None,
) -> Union[int, float]:
    vals: list[Union[int, float]] = []
    if not heuristic_func:
        if decision_type == DecisionTypeEnum.SIMULTANEOUS:
            # for one-step decision-making
            heuristic_func = partial(
                simulator.together,
                priority_func=compile_func(
                    expr=individual, pset=pset[TerminalTypeEnum.INTEGRATED.value]
                ),
                mode_func=None,
                priority_extre="min",
                mode_extre="min",
            )
        elif decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE:
            # for two-step decision-making
            heuristic_func = partial(
                simulator.activity_first_choose,
                priority_func=compile_func(
                    expr=individual[TerminalTypeEnum.ACTIVITY.value],
                    pset=pset[TerminalTypeEnum.ACTIVITY.value],
                ),
                mode_func=compile_func(
                    expr=individual[TerminalTypeEnum.MODE.value],
                    pset=pset[TerminalTypeEnum.MODE.value],
                ),
                priority_extre="min",
                mode_extre="min",
            )
        elif decision_type == DecisionTypeEnum.MODE_THEN_ACTIVITY:
            # for two-step decision-making
            heuristic_func = partial(
                simulator.mode_first_choose,
                priority_func=compile_func(
                    expr=individual[TerminalTypeEnum.ACTIVITY.value],
                    pset=pset[TerminalTypeEnum.ACTIVITY.value],
                ),
                mode_func=compile_func(
                    expr=individual[TerminalTypeEnum.MODE.value],
                    pset=pset[TerminalTypeEnum.MODE.value],
                ),
                priority_extre="min",
                mode_extre="min",
            )
    # build solutions & get objective values
    for domain in domains:
        solution = simulator.buildSolution(domain=domain, choose=heuristic_func)
        do_makespan = solution.get_end_time(domain.sink_task)
        vals.append(
            (do_makespan - domain.cpm_esd)
            * 100
            / domain.cpm_esd
        )
    fitness = [np.mean(vals)]

    return fitness


def read_instances(filepaths: list[str]) -> list[RCPSPModel]:
    from discrete_optimization.rcpsp.transform_model import \
        to_renewable_only_rcpsp_model

    instances: List[RCPSPModel] = [
        to_renewable_only_rcpsp_model(parse_file(f)) for f in filepaths
    ]
    for problem in instances:
        problem.cpm, problem.cpm_esd = compute_cpm(problem)
        problem.graph.full_predecessors = problem.graph.ancestors_map()
        problem.graph.full_successors = problem.graph.descendants_map()
    return instances


def read_instances_with_nr(filepaths: list[str]) -> list[RCPSPModel]:
    """Load MRCPSP instances preserving nonrenewable resources.

    Unlike read_instances(), this does NOT call to_renewable_only_rcpsp_model,
    so NR resources (N1, N2, …) remain in the model.  The Serial/Parallel SGS
    already handles NR feasibility tracking in resource_avail_in_time; the NR
    terminals NR_STOCK_RATIO and NR_MODE_DEMAND_RATIO then have real signal to
    work with.

    Use with MMLIB+ files (Jall*.mm), which carry both renewable and
    nonrenewable resources.  Standard MMLIB50/100 files have no NR resources
    and behave identically to read_instances().
    """
    instances: List[RCPSPModel] = [parse_file(f) for f in filepaths]
    for problem in instances:
        problem.cpm, problem.cpm_esd = compute_cpm(problem)
        problem.graph.full_predecessors = problem.graph.ancestors_map()
        problem.graph.full_successors = problem.graph.descendants_map()
    return instances


if __name__ == "__main__":
    from optparse import OptionParser

    parse = OptionParser()
    parse.add_option(
        "-s",
        dest="sgs",
        help="Schedule Generation Scheme: serial/parallel",
        type="string",
        default="serial",
    )
    parse.add_option(
        "-d",
        dest="decision_type",
        help="Decision types: activity_first/mode_first/simultaneous",
        type="string",
        default="activity_first",
    )
    parse.add_option(
        "--fixed_activity_rule",
        dest="fixed_activity_rule",
        help="Specify the fixed activity rule",
        type="string",
        default="",
    )
    parse.add_option(
        "--fixed_mode_rule",
        dest="fixed_mode_rule",
        help="Specify the fixed mode rule",
        type="string",
        default="",
    )
    parse.add_option(
        "--default",
        action="store_true",
        dest="default",
        help="Use default GP parameters.",
        default=False,
    )
    parse.add_option(
        "--start_index",
        action="store",
        dest="start_index",
        help="Start index of this run. The result json starts from this index. ",
        type="int",
        default=0,
    )
    parse.add_option(
        "-n",
        action="store",
        dest="n_runs",
        help="Number of runs",
        type="int",
        default=1,
    )
    parse.add_option(
        "--log",
        action="store",
        dest="output_dir",
        help="Directory to store result",
        type="string",
        default="./results/",
    )
    parse.add_option(
        "--dataset",
        action="store",
        dest="dataset",
        help="Specify dataset to use: MMLIB50/MMLIB100/MMLIBPLUS_50/MMLIBPLUS_100. If not specified, use a small dataset for quick test",
        type="string",
        default="",
    )
    parse.add_option(
        "--split",
        action="store_true",
        help="Split training set into several groups",
        dest="split_training_set",
        default=False,
    )
    parse.add_option(
        "--dynamic",
        action="store_true",
        help="Use dynamic terminals (e.g., dynamic CPM)",
        dest="dynamic_terminals",
        default=False,
    )
    parse.add_option(
        "--multiprocess",
        action="store_true",
        dest="multi_process",
        help="Enable multi process (Disable by default)",
        default=False,
    )
    parse.add_option(
        "--seed",
        action="store",
        dest="seed",
        help="Seed number for this run",
        type="int",
        default=1,
    )
    parse.add_option(
        "--medium",
        action="store_true",
        dest="medium",
        help="Use medium parameters (pop=50, gen=10). Faster than --default, "
             "more reliable than the smoke-test default.",
        default=False,
    )
    parse.add_option(
        "--modifications",
        action="store_true",
        dest="use_modifications",
        help="Enable modifications from modifications.py (proposed approach). "
             "Without this flag the original baseline is used.",
        default=False,
    )
    (options, args) = parse.parse_args()
    print(options)
    SIMULATOR_TYPE = SimulatorTypeEnum(options.sgs)
    DECISION_TYPE = DecisionTypeEnum(options.decision_type)
    FIXED_ACTIVITY_RULE = options.fixed_activity_rule
    FIXED_MODE_RULE = options.fixed_mode_rule
    DEFAULT_PARAMETER = options.default
    MEDIUM_PARAMETER = options.medium
    START_INDEX = options.start_index
    N_RUNS = options.n_runs
    DATASET = options.dataset
    DYNAMIC_TERMINAL = options.dynamic_terminals
    SPLIT_TRAINING_SET = options.split_training_set
    MULTI_PROCESS = options.multi_process
    OUTPUT_DIR = options.output_dir
    SEED: int = options.seed
    USE_MODIFICATIONS: bool = options.use_modifications

    match DATASET:
        case "MMLIB50":
            training_files = RCPSPDatabase.get_some_MMLIB_50_each_class_files(1, 4)
            validation_set_files = RCPSPDatabase.get_some_MMLIB_50_each_class_files(4, 5)
            test_set_files = RCPSPDatabase.get_some_MMLIB_50_each_class_files(5, 6)
        case "MMLIB100":
            training_files = RCPSPDatabase.get_some_MMLIB_100_each_class_files(1, 4)
            validation_set_files = RCPSPDatabase.get_some_MMLIB_100_each_class_files(4, 5)
            test_set_files = RCPSPDatabase.get_some_MMLIB_100_each_class_files(5, 6)
        case "MMLIBPLUS_50":
            training_files = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(1, 4)
            validation_set_files = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(4, 5)
            test_set_files = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(5, 6)
        case "MMLIBPLUS_100":
            training_files = RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files(1, 4)
            validation_set_files = RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files(4, 5)
            test_set_files = RCPSPDatabase.get_some_MMLIB_PLUS_100_each_class_files(5, 6)
        case _:
            # quick setup training set
            training_files: List[str] = [
                # "./discrete_optimization_data/mm//MMLIB//MMLIB50/J5097_1.mm",
                # "./discrete_optimization_data/mm//MMLIB//MMLIB50/J5025_2.mm",
                "discrete_optimization_data/mm//MMLIB//MMLIB50/J501_4.mm"
            ]
            # quick setup validation set
            validation_set_files: list[str] = [
                "discrete_optimization_data/mm/MMLIB/MMLIB50/J501_3.mm",
                "discrete_optimization_data/mm/MMLIB/MMLIB50/J501_4.mm",
            ]

            # quick setup test set
            test_set_files: list[str] = [
                # "discrete_optimization_data/mm/MMLIB/MMLIB50/J501_3.mm",
                "discrete_optimization_data/mm/MMLIB/MMLIB50/J501_5.mm",
            ]

    training_set: list = read_instances(training_files)
    training_data_provider = (
        EvenlyDividedDatasetProvider(training_set, 51)
        if SPLIT_TRAINING_SET and DATASET
        else StaticDatasetProvider(training_set)
    )

    validation_set: list = read_instances(validation_set_files)
    validation_data_provider = StaticDatasetProvider(validation_set)
    # validation_data_provider = EmptyDataSetProvider()

    test_set: list = read_instances(test_set_files)
    test_data_provider: list = StaticDatasetProvider(test_set)

    CPU_CORES = 8 if MULTI_PROCESS else 1

    # set up parameters

    if DEFAULT_PARAMETER:
        params: ParametersGPHH = ParametersGPHH.default(
            decision_type=DECISION_TYPE,
            simulator_type=SIMULATOR_TYPE,
            cpu=CPU_CORES,
            dynamic_CPM_feature=DYNAMIC_TERMINAL,
            fixed_activity_rule=FIXED_ACTIVITY_RULE,
            fixed_mode_rule=FIXED_MODE_RULE,
            use_modifications=USE_MODIFICATIONS,
        )
    elif MEDIUM_PARAMETER:
        params: ParametersGPHH = ParametersGPHH.medium(
            decision_type=DECISION_TYPE,
            simulator_type=SIMULATOR_TYPE,
            cpus=CPU_CORES,
            dynamic_CPM_feature=DYNAMIC_TERMINAL,
            use_modifications=USE_MODIFICATIONS,
        )
    else:
        params: ParametersGPHH = ParametersGPHH.fast(
            decision_type=DECISION_TYPE,
            simulator_type=SIMULATOR_TYPE,
            cpus=CPU_CORES,
            dynamic_CPM_feature=DYNAMIC_TERMINAL,
            use_modifications=USE_MODIFICATIONS,
        )

    solver = GPHH(
        training_set_provider=training_data_provider,
        validation_set_provider=validation_data_provider,
        test_set_provider=test_data_provider,
        params_gphh=params,
    )
    solver.init_model()
    # create folders to store result

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import datetime

    print(
        f"""
        Current configuration:
        Decision type: {DECISION_TYPE}
        Simulator type: {SIMULATOR_TYPE}
        Start index: {START_INDEX}
        Number of runs: {N_RUNS}
        Program starts at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        CPU cores: {CPU_CORES}
        Seed number starts from {SEED}
        Dynamic terminals? {DYNAMIC_TERMINAL}
        Training set: {len(training_set)} cases {training_data_provider.__class__}
        Test Set: {len(test_set_files) if "test_set_files" in locals() else 0} case(s)
        """
    )
    for n in range(START_INDEX, START_INDEX + N_RUNS):
        print(f"Round {n} starts!!")
        SEED += 100
        random.seed(SEED)
        np.random.seed(SEED)
        for provider in [
            training_data_provider,
            validation_data_provider,
            test_data_provider,
        ]:
            provider.reset()
        solver.solve(output_path=os.path.join(OUTPUT_DIR, f"{n}.json"))
        print(f"Round {n} finished!!!")
