"""
RCPSP Simulator module
"""

import logging
from enum import Enum
from typing import Dict, List, Set, Hashable, Tuple, Union
from abc import abstractmethod

import numpy as np

from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP


from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel


class SimulatorTypeEnum(Enum):
    SERIAL_SGS = "serial"
    PARALLEL_SGS = "parallel"


class DecisionTypeEnum(Enum):
    ACTIVITY_THEN_MODE = "activity_first"
    MODE_THEN_ACTIVITY = "mode_first"
    SIMULTANEOUS = "simultaneous"


class FeatureEnum(Enum):
    # static CPM feature
    EARLIEST_START_DATE = "ES"
    EARLIEST_FINISH_DATE = "EF"
    LATEST_START_DATE = "LS"
    LATEST_FINISH_DATE = "LF"
    TASK_DURATION = "task_duration"
    # dynamic CPM feature
    DYNAMIC_EARLIEST_START_DATE = "ES_d"
    DYNAMIC_EARLIEST_FINISH_DATE = "EF_d"
    DYNAMIC_LATEST_START_DATE = "LS_d"
    DYNAMIC_LATEST_FINISH_DATE = "LF_d"
    DYNAMIC_EARLIEST_FEASIBLE_FINISH_TIME = "EFFT"  # Lova 2007, Multi-mode RCPSP ...
    DYNAMIC_LATEST_START_AND_FINISH_TIME = "LSTLFT"  # Lova 2007, multi-mode RCPSP ...
    GREATEST_RANK_POSITIONAL_WEIGHT = "GRPW"
    GREATEST_RANK_POSITIONAL_WEIGHT_ALL = "GRPW_all"
    GREATEST_RESOURCE_DEMAND = "GRD"
    TOTAL_PREDECESSOR_COUNT = "TPC"
    IMMEDIATE_PREDECESSOR_COUNT = "PC"
    TOTAL_SUCCESSOR_COUNT = "TSC"
    IMMEDIATE_SUCCESSOR_COUNT = "SC"
    RESOURCE_REQUIRED = "RR"
    AVG_TASK_DURATION = "avg_task_duration"
    MAX_TASK_DURATION = "max_task_duration"
    MIN_TASK_DURATION = "min_task_duration"
    AVG_RESOURCE_REQUIREMENT = "avg_RReq"
    MAX_RESOURCE_REQUIREMENT = "max_RReq"
    MIN_RESOURCE_REQUIREMENT = "min_RReq"
    AVG_RESOURCE_REQUIREMENT_ACROSS_MODES = "avg_RReq_m"
    MAX_RESOURCE_REQUIREMENT_ACROSS_MODES = "max_RReq_m"
    MIN_RESOURCE_REQUIREMENT_ACROSS_MODES = "min_RReq_m"
    AVG_RESOURCE_CAPACITY = "avg_ResCap"
    MAX_RESOURCE_CAPACITY = "max_ResCap"
    MIN_RESOURCE_CAPACITY = "min_ResCap"
    SLACK = "Slack"  # Latest Start - Earliest Start
    IS_ON_CRITICAL_PATH = "Is_On_Critical_Path" 
    # Dynamic version that calculates if delaying it *now* based on current state prolongs makespan
    DYNAMIC_SLACK = "Dynamic_Slack"


class Simulator(object):
    """
    Base class for RCPSP simulators
    1. Serial SGS
    2. Parallel SGS
    These simulators can build a schedule from a priority rule and a mode selection rule
    `buildSolution` function implements the schedule building procedure
    --------------
    The base simulator contains common functions for both types of simulators, including:
    1. Decision functions:
        a. activity_first_choose (activity selection first)
        b. mode_first_choose (mode selection first)
        c. together (simultaneous selection in the paper)
    2. Feature functions: EST,EFT,....
    """
    def __init__(self) -> None:
        self.rcpsp_problem: ANY_RCPSP = None
        self.type = None
        self.eligibles = None
        self.cur_act = None
        self.cur_mode = None
        self.dynamic_cpm: Dict[Any, CPMObject] = None
        self.feature_function_map = {
            FeatureEnum.EARLIEST_START_DATE: self.feature_early_start,
            FeatureEnum.EARLIEST_FINISH_DATE: self.feature_early_finish,
            FeatureEnum.LATEST_START_DATE: self.feature_late_start,
            FeatureEnum.LATEST_FINISH_DATE: self.feature_late_finish,
            FeatureEnum.DYNAMIC_EARLIEST_START_DATE: self.feature_dynamic_earliest_start,
            FeatureEnum.DYNAMIC_EARLIEST_FINISH_DATE: self.feature_dynamic_earliest_finish,
            FeatureEnum.DYNAMIC_LATEST_START_DATE: self.feature_dynamic_late_start,
            FeatureEnum.DYNAMIC_LATEST_FINISH_DATE: self.feature_dynamic_late_finish,
            FeatureEnum.DYNAMIC_EARLIEST_FEASIBLE_FINISH_TIME: self.heuristic_earliest_feasible_finish_time,
            FeatureEnum.TASK_DURATION: self.feature_duration,
            FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT: self.feature_greatest_rank_positional_weight,
            FeatureEnum.GREATEST_RANK_POSITIONAL_WEIGHT_ALL: self.feature_greatest_rank_positional_weight_all,
            FeatureEnum.GREATEST_RESOURCE_DEMAND: self.feature_greatest_resource_demand,
            FeatureEnum.TOTAL_PREDECESSOR_COUNT: self.feature_total_predecessor_count,
            FeatureEnum.IMMEDIATE_PREDECESSOR_COUNT: self.feature_immediate_predecessor_count,
            FeatureEnum.TOTAL_SUCCESSOR_COUNT: self.feature_total_successor_count,
            FeatureEnum.IMMEDIATE_SUCCESSOR_COUNT: self.feature_immediate_successor_count,
            FeatureEnum.RESOURCE_REQUIRED: self.feature_resources_required,
            FeatureEnum.AVG_RESOURCE_REQUIREMENT: self.feature_avg_resource_requirement,
            FeatureEnum.MAX_RESOURCE_REQUIREMENT: self.feature_max_resource_requirement,
            FeatureEnum.MIN_RESOURCE_REQUIREMENT: self.feature_min_resource_requirement,
            FeatureEnum.AVG_RESOURCE_CAPACITY: self.feature_avg_resource_capacity,
            FeatureEnum.MAX_RESOURCE_CAPACITY: self.feature_max_resource_capacity,
            FeatureEnum.MIN_RESOURCE_CAPACITY: self.feature_min_resource_capacity,
            FeatureEnum.AVG_TASK_DURATION: self.feature_average_duration,
            FeatureEnum.MAX_TASK_DURATION: self.feature_max_duration,
            FeatureEnum.MIN_TASK_DURATION: self.feature_min_duration,
            FeatureEnum.AVG_RESOURCE_REQUIREMENT_ACROSS_MODES: self.feature_avg_resource_requirement_across_modes,
            FeatureEnum.MAX_RESOURCE_REQUIREMENT_ACROSS_MODES: self.feature_max_resource_requirement_across_modes,
            FeatureEnum.MIN_RESOURCE_REQUIREMENT_ACROSS_MODES: self.feature_min_resource_requirement_across_modes,
            FeatureEnum.DYNAMIC_LATEST_START_AND_FINISH_TIME: self.heuristic_latest_start_and_finish_time,
            
            FeatureEnum.SLACK: self.feature_slack,
            FeatureEnum.IS_ON_CRITICAL_PATH: self.feature_is_on_critical_path,
            FeatureEnum.DYNAMIC_SLACK: self.feature_dynamic_slack,
        }

    @classmethod
    def from_type_name(cls, type_name: str):
        """Return a simulator with given `type_name`

        Args:
            type_name (str): Options can be `serial`, `parallel`

        Returns:
            _type_: A specific-type simulator
        """
        type = None
        try:
            type = SimulatorTypeEnum(type_name)
        except ValueError as e:
            print(e.with_traceback())
        if type == SimulatorTypeEnum.SERIAL_SGS:
            return SerialSimulator()
        elif type == SimulatorTypeEnum.PARALLEL_SGS:
            return ParallelSimulator()

    # Decision Functions
    def activity_first_choose(
        self,
        eligibles: Dict[Hashable, Tuple[int]],
        priority_func: callable,
        mode_func: callable,
        priority_extre: str = "min",
        mode_extre: str = "min",
    ) -> Tuple[int, int]:
        eligibles = dict(sorted(eligibles.items()))
        self._compute_dynamic_cpm(eligibles)  # compute dynamic CPM
        # print("----decisions-----")
        # activity selection first
        activity_priority_values: Dict[Hashable, float] = {}  # {act_id: values}
        for act in eligibles:
            self.cur_act = act
            self.cur_mode = None
            activity_priority_values[act] = priority_func()
        if priority_extre == "min":
            act_id = min(activity_priority_values, key=activity_priority_values.get)
        elif priority_extre == "max":
            act_id = max(activity_priority_values, key=activity_priority_values.get)
        else:
            raise ValueError(
                f"priority_extre could be either 'min' or 'max', but not {priority_extre}"
            )
        # print(activity_priority_values)
        # mode selection
        mode_priority_values = {}  # {mode_id: values}
        for m in eligibles[act_id]:
            self.cur_act = act_id
            self.cur_mode = m
            mode_priority_values[m] = mode_func()
        if mode_extre == "min":
            mode_id = min(mode_priority_values, key=mode_priority_values.get)
        elif mode_extre == "max":
            mode_id = max(mode_priority_values, key=mode_priority_values.get)
        else:
            raise ValueError(
                f"mode_extre could be either 'min' or 'max', but not {mode_extre}"
            )
        # print(mode_priority_values)
        # print(act_id, mode_id)
        return act_id, mode_id

    def mode_first_choose(
        self,
        eligibles: Dict[Hashable, Tuple[int]],
        priority_func: callable,
        mode_func: callable,
        priority_extre: str = "min",
        mode_extre: str = "min",
    ) -> Tuple[int, int]:
        eligibles = dict(sorted(eligibles.items()))
        self._compute_dynamic_cpm(eligibles)  # compute dynamic CPM
        mode_dict = (
            {}
        )  # selected mode for activities in eligibles {act id: selected mode id}
        # mode selection first
        for act in eligibles:
            self.cur_act = act
            mode_priority_values = {}  # {mode_id: values}
            for m in eligibles[act]:
                self.cur_mode = m
                mode_priority_values[m] = mode_func()
            if mode_extre == "min":
                mode_dict[act] = min(mode_priority_values, key=mode_priority_values.get)
            elif mode_extre == "max":
                mode_dict[act] = max(mode_priority_values, key=mode_priority_values.get)
            else:
                raise ValueError(
                    f"mode_extre could be either 'min' or 'max', but not {mode_extre}"
                )
        # activity selection
        activity_priority_values = {}
        for act in eligibles:
            self.cur_act = act
            self.cur_mode = mode_dict[act]
            activity_priority_values[act] = priority_func()
        if priority_extre == "min":
            act_id = min(activity_priority_values, key=activity_priority_values.get)
        elif priority_extre == "max":
            act_id = max(activity_priority_values, key=activity_priority_values.get)
        else:
            raise ValueError(
                f"priority_extre could be either 'min' or 'max', but not {priority_extre}"
            )
        return act_id, mode_dict[act_id]

    def together(
        self,
        eligibles: Dict[Hashable, Tuple[int]],
        priority_func: callable,
        mode_func: callable,
        priority_extre: str = "min",
        mode_extre: str = "min",
    ):
        eligibles = dict(sorted(eligibles.items()))
        self._compute_dynamic_cpm(eligibles)  # compute dynamic CPM
        # {(act_id, mode_id): values}
        priority_values: Dict[Hashable, Tuple[int]] = {}
        for act in eligibles:
            for m in eligibles[act]:
                self.cur_act, self.cur_mode = act, m
                priority_values[(act, m)] = priority_func()
        if priority_extre == "min":
            act_id, mode_id = min(priority_values, key=priority_values.get)
        elif priority_extre == "max":
            act_id, mode_id = max(priority_values, key=priority_values.get)
        else:
            raise ValueError(
                f"priority_extre could be either 'min' or 'max', but not {priority_extre}"
            )
        return act_id, mode_id

    #######################################################################
    def feature_early_start(self) -> Union[int, float]:
        return self.rcpsp_problem.cpm[self.cur_act]._ESD
        # return self.rcpsp_problem.cpm[self.cur_act]._ESD / self.rcpsp_problem.cpm_esd

    def feature_early_finish(self) -> Union[int, float]:
        return self.rcpsp_problem.cpm[self.cur_act]._EFD
        # return self.rcpsp_problem.cpm[self.cur_act]._EFD / self.rcpsp_problem.cpm_esd

    def feature_late_start(self) -> Union[int, float]:
        return self.rcpsp_problem.cpm[self.cur_act]._LSD
        # return self.rcpsp_problem.cpm[self.cur_act]._LSD / self.rcpsp_problem.cpm_esd

    def feature_late_finish(self) -> Union[int, float]:
        return self.rcpsp_problem.cpm[self.cur_act]._LFD
        # return self.rcpsp_problem.cpm[self.cur_act]._LFD / self.rcpsp_problem.cpm_esd

    def feature_dynamic_earliest_start(self) -> Union[int, float]:
        return self.dynamic_cpm[self.cur_act]._ESD

    def feature_dynamic_earliest_finish(self) -> Union[int, float]:
        return self.dynamic_cpm[self.cur_act]._EFD

    def feature_dynamic_late_start(self) -> Union[int, float]:
        # print(f"{self.dynamic_cpm[self.cur_act]._LSD}|{self.feature_late_start()}")
        return self.dynamic_cpm[self.cur_act]._LSD

    def feature_dynamic_late_finish(self) -> Union[int, float]:
        # print(f"{self.dynamic_cpm[self.cur_act]._LFD}|{self.feature_late_start()}")
        return self.dynamic_cpm[self.cur_act]._LFD

    def feature_duration(self) -> Union[int, float]:
        return self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]["duration"]

    def feature_average_duration(self) -> Union[int, float]:
        return sum(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m]["duration"]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
            ]
        ) / len(self.rcpsp_problem.mode_details[self.cur_act])

    def feature_max_duration(self) -> Union[int, float]:
        return max(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m]["duration"]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
            ]
        )

    def feature_min_duration(self) -> Union[int, float]:
        return min(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m]["duration"]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
            ]
        )

    def feature_max_resource_requirement_across_modes(self) -> Union[int, float]:
        return max(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m][res]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
                for res in self.rcpsp_problem.resources
            ]
        )

    def feature_min_resource_requirement_across_modes(self) -> Union[int, float]:
        return min(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m][res]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
                for res in self.rcpsp_problem.resources
            ]
        )

    def feature_avg_resource_requirement_across_modes(self) -> Union[int, float]:
        return sum(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m][res]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
                for res in self.rcpsp_problem.resources
            ]
        ) / (len(self.rcpsp_problem.resources) * self.rcpsp_problem.max_number_of_mode)

    def feature_greatest_rank_positional_weight(self):
        v = min(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m]["duration"]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
            ]
        )
        for succ in self.rcpsp_problem.successors[self.cur_act]:
            v += min(
                [
                    self.rcpsp_problem.mode_details[succ][m]["duration"]
                    for m in self.rcpsp_problem.mode_details[succ]
                ]
            )
        return v

    def feature_greatest_rank_positional_weight_all(self) -> Union[int, float]:
        v = min(
            [
                self.rcpsp_problem.mode_details[self.cur_act][m]["duration"]
                for m in self.rcpsp_problem.mode_details[self.cur_act]
            ]
        )
        for succ in self.rcpsp_problem.graph.full_successors[self.cur_act]:
            v += min(
                [
                    self.rcpsp_problem.mode_details[succ][m]["duration"]
                    for m in self.rcpsp_problem.mode_details[succ]
                ]
            )
        return v

    def feature_total_predecessor_count(self) -> Union[int, float]:
        return len(self.rcpsp_problem.graph.full_predecessors[self.cur_act])

    def feature_immediate_predecessor_count(self) -> Union[int, float]:
        return len(self.rcpsp_problem.graph.predecessors_dict[self.cur_act])

    def feature_total_successor_count(self) -> Union[int, float]:
        return len(self.rcpsp_problem.graph.full_successors[self.cur_act])

    def feature_immediate_successor_count(self) -> Union[int, float]:
        return len(self.rcpsp_problem.successors[self.cur_act])

    def feature_resources_required(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        return sum(
            [
                1
                if self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][res]
                else 0
                for res in self.rcpsp_problem.resources
            ]
        )

    def feature_greatest_resource_demand(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        return self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][
            "duration"
        ] * sum(
            [
                1
                if self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][res]
                else 0
                for res in self.rcpsp_problem.resources
            ]
        )

    def feature_avg_resource_requirement(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        return sum(
            [
                self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][res]
                for res in self.rcpsp_problem.resources
            ]
        ) / len(self.rcpsp_problem.resources)

    def feature_max_resource_requirement(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        return max(
            [
                self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][res]
                for res in self.rcpsp_problem.resources
            ]
        )

    def feature_min_resource_requirement(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        return min(
            [
                self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][res]
                for res in self.rcpsp_problem.resources
            ]
        )

    def feature_min_resource_capacity(self) -> Union[int, float]:
        return min(self.rcpsp_problem.resources.values())

    def feature_max_resource_capacity(self) -> Union[int, float]:
        return max(self.rcpsp_problem.resources.values())

    def feature_avg_resource_capacity(self) -> Union[int, float]:
        return sum(self.rcpsp_problem.resources.values()) / len(
            self.rcpsp_problem.resources_list
        )

    def feature_slack(self) -> Union[int, float]:
        """Total slack based on static CPM: Latest Start - Earliest Start"""
        return self.rcpsp_problem.cpm[self.cur_act]._LSD - self.rcpsp_problem.cpm[self.cur_act]._ESD

    def feature_is_on_critical_path(self) -> Union[int, float]:
        """Returns 1.0 if the activity is on the static critical path (slack == 0), else 0.0"""
        slack = self.rcpsp_problem.cpm[self.cur_act]._LSD - self.rcpsp_problem.cpm[self.cur_act]._ESD
        return 1.0 if slack == 0 else 0.0

    def feature_dynamic_slack(self) -> Union[int, float]:
        """
        Total slack based on dynamic CPM (updating as the schedule is built):
        Dynamic Latest Start - Dynamic Earliest Start
        """
        return self.dynamic_cpm[self.cur_act]._LSD - self.dynamic_cpm[self.cur_act]._ESD
    
    @abstractmethod
    def heuristic_earliest_feasible_finish_time(self) -> Union[int, float]:
        """
        Lova, Antonio; Tormos, Pilar; Barber, Federico (2006): Multi-mode resource constrained project scheduling: scheduling schemes, priority rules and mode selection rules. In Inteligencia Artificial. Revista Iberoamericana de Inteligencia Artificial 10 (30).
        Mode selection rule.
        This mode selection rule selects for each activity the execution mode such that it is scheduled with the feasible finish time as early as possible.
        Therefore, in the S-SGS the following activities to be processed could be executed earlier.
        If ties occur (an activity has several feasible modes with the same minimum value of feasible finish time) the mode with the highest duration is selected.
        """
        pass

    def heuristic_latest_start_and_finish_time(self) -> Union[int, float]:
        """
        The best priority rule in paper
        Lova, Antonio; Tormos, Pilar; Barber, Federico (2006): Multi-mode resource constrained project scheduling: scheduling schemes, priority rules and mode selection rules. In Inteligencia Artificial. Revista Iberoamericana de Inteligencia Artificial 10 (30).
        Mode selection rule.
        """
        return self.dynamic_cpm[self.cur_act]._LSD + self.dynamic_cpm[self.cur_act]._LFD

    ########################################

    ########################################

    @abstractmethod
    def buildSolution(self, domain: ANY_RCPSP, choose: callable) -> RCPSPSolution:
        pass

    @abstractmethod
    def _compute_dynamic_cpm():
        pass


from discrete_optimization.rcpsp.solver.cpm import CPMObject
from typing import Any


class SerialSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()
        self.type = SimulatorTypeEnum.SERIAL_SGS

    def _compute_dynamic_cpm(self, eligible: list):
        import networkx as nx

        _eligible: list = list(eligible)
        _scheduled = list(self.scheduled)
        _subproblem = set(self.rcpsp_problem.tasks_list) - set(_scheduled)
        cpm_nodes: Dict[Any, CPMObject] = {
            n: CPMObject(None, None, None, None) for n in _subproblem
        }
        _subgraph: nx.DiGraph = self.rcpsp_problem.graph.graph_nx.subgraph(_subproblem)
        subgraph_size = _subgraph.number_of_nodes()
        # earliest start and finish
        ## for those activities can scheduled immediately (predecessors already finished)
        for act in _eligible:
            cpm_nodes[act]._ESD = self.minimum_starting_time[act]
            cpm_nodes[act]._EFD = (
                self.minimum_starting_time[act]
                + self.rcpsp_problem.mode_details[act][1]["duration"]
            )
        _scheduled = list(_eligible)

        while len(_scheduled) < subgraph_size:
            _unscheduled = list(set(_subproblem) - set(_scheduled))
            # find the first one satisfying precedence constraints
            for i in range(len(_unscheduled)):
                cur_act = _unscheduled[i]
                pred_acts = list(_subgraph.predecessors(cur_act))
                if set(pred_acts) <= set(_scheduled):
                    max_pred_finish_time = 0
                    for j in pred_acts:
                        max_pred_finish_time = max(
                            max_pred_finish_time, cpm_nodes[j]._EFD
                        )
                    cpm_nodes[cur_act]._ESD = max(
                        max_pred_finish_time, self.minimum_starting_time[cur_act]
                    )
                    cpm_nodes[cur_act]._EFD = (
                        max_pred_finish_time
                        + self.rcpsp_problem.mode_details[cur_act][1]["duration"]
                    )
                    _scheduled.append(cur_act)
        # backward
        ## initialization
        cpm_nodes[self.rcpsp_problem.sink_task]._LSD = cpm_nodes[
            self.rcpsp_problem.sink_task
        ]._ESD
        cpm_nodes[self.rcpsp_problem.sink_task]._LFD = cpm_nodes[
            self.rcpsp_problem.sink_task
        ]._ESD
        _scheduled = [self.rcpsp_problem.sink_task]
        while len(_scheduled) < subgraph_size:
            _unscheduled = list(set(_subproblem) - set(_scheduled))
            _unscheduled.reverse()
            for i in range(len(_unscheduled)):
                cur_act = _unscheduled[i]
                succ_acts = list(_subgraph.successors(cur_act))
                if set(succ_acts) <= set(_scheduled):
                    min_succ_start_time = 9223372036854775807
                    for j in succ_acts:
                        min_succ_start_time = min(
                            min_succ_start_time, cpm_nodes[j]._LSD
                        )
                    cpm_nodes[cur_act]._LFD = min_succ_start_time
                    cpm_nodes[cur_act]._LSD = (
                        min_succ_start_time
                        - self.rcpsp_problem.mode_details[cur_act][1]["duration"]
                    )
                    _scheduled.append(cur_act)
        self.dynamic_cpm = cpm_nodes

    def buildSolution(self, domain: RCPSPModel, choose: callable) -> RCPSPSolution:
        """
        Build a RCPSP solution from serial SGS
        Args:
            domain: problem insatance
            choose: A callable function to choose activity and mode from eligibles. It should be constructed using `evaluate_heuristic` in gphh_solver.py

        Returns:
            A RCPSP solution
        """
        self.rcpsp_problem: RCPSPModel = domain
        predecessors = self.rcpsp_problem.graph.graph_nx.predecessors
        # predecessors = self.rcpsp_problem.graph.get_predecessors

        # initialization
        all_jobs = self.rcpsp_problem.tasks_list
        activity_end_times = {
            1: 0,
        }  # {activity_id: end_time}
        mode_dict: Dict[Hashable, int] = {
            act: 1 for act in self.rcpsp_problem.tasks_list
        }  # {activity_id: mode_id}
        unfeasible_non_renewable_resources = False
        self.new_horizon = self.rcpsp_problem.horizon

        self.resource_avail_in_time = {}
        for res in self.rcpsp_problem.resources_list:
            if self.rcpsp_problem.is_varying_resource():
                resource_avail_in_time[res] = rcpsp_problem.resources[res][  # type: ignore
                    : self.new_horizon + 1
                ]
            else:
                self.resource_avail_in_time[res] = np.full(
                    self.new_horizon, self.rcpsp_problem.resources[res], dtype=np.int_
                ).tolist()
        # Earliest start time for activity under precedence relations and predefined time window
        self.minimum_starting_time = {act: 0 for act in self.rcpsp_problem.tasks_list}
        # put the dummy start in scheduled set
        self.scheduled = [
            1,
        ]
        # build a schedule from partial to completed
        while (
            len(self.scheduled) < self.rcpsp_problem.n_jobs
            and not unfeasible_non_renewable_resources
        ):
            # Step 1: find the eligible set
            eligibles = {}  # {act_id:modes}
            unscheduled = set(all_jobs) - set(self.scheduled)
            for act in unscheduled:
                # if the predecessors of i are scheduled, then add i to eligibles
                if set(predecessors(act)) <= set(self.scheduled):
                    eligibles[act] = []
            # Step 2: choose the activity and available modes to schedule
            # For serial SGS, a mode which satisfies non-renewable resource constaints is eligible
            for act in eligibles:
                for m in self.rcpsp_problem.mode_details[act]:
                    valid = True
                    for res in self.rcpsp_problem.resources:
                        if self.rcpsp_problem.mode_details[act][m].get(res, 0) == 0:
                            continue
                        if (
                            self.resource_avail_in_time[res][-1]
                            < self.rcpsp_problem.mode_details[act][m][res]
                        ):
                            valid = False
                    if valid:
                        eligibles[act].append(m)
            act_id, mode_id = choose(eligibles)
            # Step 3: find the earliest finish time of all predecessors
            current_min_time = self.minimum_starting_time[act_id]
            # Step 4: find the earliest available time when resource availablity constraints are considered
            # True if find the feasible time slot for act_id (resource constraints)
            valid = False
            while not valid:
                valid = True
                for t in range(
                    current_min_time,
                    current_min_time
                    + self.rcpsp_problem.mode_details[act_id][mode_id]["duration"],
                ):
                    # Check feasibility of each resource from current_min_time to current_min_time + duration
                    for res in self.rcpsp_problem.resources_list:
                        # skip check if act_id does not require current resource
                        if (
                            self.rcpsp_problem.mode_details[act_id][mode_id].get(res, 0)
                            == 0
                        ):
                            continue
                        if t < self.new_horizon:
                            if (
                                self.resource_avail_in_time[res][t]
                                < self.rcpsp_problem.mode_details[act_id][mode_id][res]
                            ):
                                valid = False
                        else:
                            unfeasible_non_renewable_resources = True
                if not valid:
                    current_min_time += 1
            # Step 5: Update resource availability
            if not unfeasible_non_renewable_resources:
                end_t = (
                    current_min_time
                    + self.rcpsp_problem.mode_details[act_id][mode_id]["duration"]
                )
                for t in range(current_min_time, end_t):
                    # Skip if act_id does not use current res
                    for res in self.resource_avail_in_time:
                        if (
                            self.rcpsp_problem.mode_details[act_id][mode_id].get(res, 0)
                            == 0
                        ):
                            continue
                        # update renewable resources
                        self.resource_avail_in_time[res][
                            t
                        ] -= self.rcpsp_problem.mode_details[act_id][mode_id][res]
                        # update non-renewable resources
                        if (
                            res in self.rcpsp_problem.non_renewable_resources_list
                            and t == end_t - 1
                        ):
                            for tt in range(end_t, self.new_horizon):
                                self.resource_avail_in_time[res][
                                    tt
                                ] -= self.rcpsp_problem.mode_details[act_id][mode_id][
                                    res
                                ]
                                if self.resource_avail_in_time[res][tt] < 0:
                                    unfeasible_non_renewable_resources = True

            # Step 6: Record work
            activity_end_times[act_id] = end_t  # set finish time for act_id
            self.scheduled.append(act_id)  # add act_id into scheduled list
            for s in self.rcpsp_problem.successors[act_id]:
                self.minimum_starting_time[s] = max(
                    self.minimum_starting_time[s], activity_end_times[act_id]
                )
            mode_dict[act_id] = mode_id
            # print(f"{list(eligibles.keys())},{act_id},{activity_end_times[act_id]}")

        # Convert to readable schedule
        rcpsp_schedule: Dict[Hashable, Dict[str, int]] = {}
        for act_id in activity_end_times:
            rcpsp_schedule[act_id] = {}
            rcpsp_schedule[act_id]["start_time"] = (
                activity_end_times[act_id]
                - self.rcpsp_problem.mode_details[act_id][mode_dict[act_id]]["duration"]
            )
            rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
        if unfeasible_non_renewable_resources:
            rcpsp_schedule_feasible = False
            last_act_id = self.rcpsp_problem.sink_task
            if last_act_id not in rcpsp_schedule:
                rcpsp_schedule[last_act_id] = {}
                rcpsp_schedule[last_act_id]["start_time"] = 99999999
                rcpsp_schedule[last_act_id]["end_time"] = 9999999
        else:
            rcpsp_schedule_feasible = True
        # remove the source and sink in mode_dict (compatibility)
        del mode_dict[self.rcpsp_problem.source_task]
        del mode_dict[self.rcpsp_problem.sink_task]
        return RCPSPSolution(
            problem=self.rcpsp_problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=list(mode_dict.values()),
            rcpsp_schedule_feasible=rcpsp_schedule_feasible,
        )

    def heuristic_earliest_feasible_finish_time(self) -> Union[int, float]:
        """
        Lova, Antonio; Tormos, Pilar; Barber, Federico (2006): Multi-mode resource constrained project scheduling: scheduling schemes, priority rules and mode selection rules. In Inteligencia Artificial. Revista Iberoamericana de Inteligencia Artificial 10 (30).
        Mode selection rule.
        This mode selection rule selects for each activity the execution mode such that it is scheduled with the feasible finish time as early as possible.
        Therefore, in the S-SGS the following activities to be processed could be executed earlier.
        If ties occur (an activity has several feasible modes with the same minimum value of feasible finish time) the mode with the highest duration is selected.

        """
        unfeasible_non_renewable_resources = False
        current_min_time = self.minimum_starting_time[self.cur_act]
        # Step 4: find the earliest available time when resource availablity constraints are considered
        # True if find the feasible time slot for act_id (resource constraints)
        valid = False
        while not valid:
            valid = True
            for t in range(
                current_min_time,
                current_min_time
                + self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][
                    "duration"
                ],
            ):
                # Check feasibility of each resource from current_min_time to current_min_time + duration
                for res in self.rcpsp_problem.resources_list:
                    # skip check if act_id does not require current resource
                    if (
                        self.rcpsp_problem.mode_details[self.cur_act][
                            self.cur_mode
                        ].get(res, 0)
                        == 0
                    ):
                        continue
                    if t < self.new_horizon:
                        if (
                            self.resource_avail_in_time[res][t]
                            < self.rcpsp_problem.mode_details[self.cur_act][
                                self.cur_mode
                            ][res]
                        ):
                            valid = False
                    else:
                        unfeasible_non_renewable_resources = True
            if not valid:
                current_min_time += 1
        end_t = (
            current_min_time
            + self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]["duration"]
            if not unfeasible_non_renewable_resources
            else 99999
        )
        return end_t


class ParallelSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()
        self.type = SimulatorTypeEnum.PARALLEL_SGS

    def buildSolution(self, domain: RCPSPModel, choose: callable) -> RCPSPSolution:
        """
        Build a RCPSP solution from parallel SGS.
        Args:
            domain: problem instance
            choose:  A callable function to choose activity and mode from eligibles. It should be constructed using `evaluate_heuristic` in gphh_solver.py

        Returns:
            A RCPSP solution.
        """
        self.rcpsp_problem: RCPSPModel = domain
        # initialization
        activity_end_times = {
            1: 0,
        }  # {activity_id: end_time}
        mode_dict = {
            act: 1 for act in self.rcpsp_problem.tasks_list
        }  # {activity_id: mode_id}
        unfeasible_non_renewable_resources = False
        new_horizon = self.rcpsp_problem.horizon

        resource_avail_in_time = {}
        for res in self.rcpsp_problem.resources_list:
            if self.rcpsp_problem.is_varying_resource():
                resource_avail_in_time[res] = rcpsp_problem.resources[res][  # type: ignore
                    : new_horizon + 1
                ]
            else:
                resource_avail_in_time[res] = np.full(
                    new_horizon, self.rcpsp_problem.resources[res], dtype=np.int_
                ).tolist()
        resource_avail = self.rcpsp_problem.resources.copy()
        self.minimum_starting_time = {act: 0 for act in self.rcpsp_problem.tasks_list}
        completed: list[int] = []  # completed activities
        active_jobs: list[int] = [1]  # processing activities
        # feasible activities and their modes
        eligibles: Dict[Hashable, list[int]] = {}
        # already scheduled activities (completed + active)
        self.scheduled: list[int] = [
            1,
        ]
        self.current_time: int = 0
        while len(set(active_jobs) | set(completed)) < self.rcpsp_problem.n_jobs:
            # NOTE: variables active_jobs, ... can defined as set in release version
            # Step 1:
            # Step 1.1: find time t of a stage equals the earliest completion time of active activities
            self.current_time = min([activity_end_times[act] for act in active_jobs])
            # print("============================")
            # print(f"Current Time:{current_time}")
            # active_end_times = {act: activity_end_times[act] for act in active_jobs}
            # print(f"Active end times:{active_end_times}")
            # print(f"Before Removal resources: {resource_avail}")
            # Step 1.2: activities with a finish time equal to the (new) schedule time are removed from the active set and put into the complete set
            removal: list[int] = []
            for act in active_jobs:
                if activity_end_times[act] == self.current_time:
                    completed.append(act)
                    removal.append(act)
            active_jobs = list(set(active_jobs) - set(removal))
            # Step 1.3: update resource: release resources when active activities are completed.
            for act in removal:
                for res in self.rcpsp_problem.resources:
                    if res in self.rcpsp_problem.non_renewable_resources_list:
                        # don't update non-renewable resources
                        continue
                    resource_avail[res] += self.rcpsp_problem.mode_details[act][
                        mode_dict[act]
                    ][res]
            # Step 1.4: update eligible set
            eligibles = self.get_eligibles(resource_avail, completed, self.scheduled)
            # print(f"removal:{removal}")
            # print(f"After removal: {resource_avail}")

            # Step 2: Schedule activities in eligible set until this set is empty
            if len(eligibles) == 0 and len(active_jobs) == 0:
                unfeasible_non_renewable_resources = True
                print("Infeasible solution!")
                break
            else:
                while len(eligibles):
                    # Step 2.1: Choose a activity and its mode from eligbles
                    act_id, mode_id = choose(eligibles)
                    mode_dict[act_id] = mode_id
                    # print(f"eligibles: {eligibles}")
                    # print(f"decision: act {act_id}, \
                    #       mode: {mode_id}, \
                    #         {self.rcpsp_problem.mode_details[act_id][mode_id]}")
                    # print(f"Before update: {resource_avail}")
                    # Set finish time for selected act_id
                    activity_end_times[act_id] = (
                        self.current_time
                        + self.rcpsp_problem.mode_details[act_id][mode_id]["duration"]
                    )
                    eligibles.pop(act_id)
                    # add selected act_id to active job set
                    active_jobs.append(act_id)
                    self.scheduled.append(act_id)
                    # Update the successors' minimum start time (used for recording and dynamic CPM, not used in parallel SGS)
                    for s in self.rcpsp_problem.successors[act_id]:
                        self.minimum_starting_time[s] = max(
                            self.minimum_starting_time[s], activity_end_times[act_id]
                        )
                    # update resource availability
                    for res in resource_avail:
                        if (
                            self.rcpsp_problem.mode_details[act_id][mode_id].get(res, 0)
                            == 0
                        ):
                            continue
                        # update renewable & non-renewables resources
                        resource_avail[res] -= self.rcpsp_problem.mode_details[act_id][
                            mode_id
                        ][res]
                    # print(f"After update: {resource_avail}")
                    # re-update eligible set
                    eligibles = self.get_eligibles(
                        resource_avail, completed, self.scheduled
                    )
                    # print(
                    #     f"Active:{active_jobs}, Completed:{completed}, Scheduled:{scheduled}")
        # Convert to readable schedule
        rcpsp_schedule: Dict[Hashable, Dict[str, int]] = {}
        for act_id in activity_end_times:
            rcpsp_schedule[act_id] = {}
            rcpsp_schedule[act_id]["start_time"] = (
                activity_end_times[act_id]
                - self.rcpsp_problem.mode_details[act_id][mode_dict[act_id]]["duration"]
            )
            rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
        if unfeasible_non_renewable_resources:
            rcpsp_schedule_feasible = False
            last_act_id = self.rcpsp_problem.sink_task
            if last_act_id not in rcpsp_schedule:
                rcpsp_schedule[last_act_id] = {}
                rcpsp_schedule[last_act_id]["start_time"] = 99999999
                rcpsp_schedule[last_act_id]["end_time"] = 9999999
        else:
            rcpsp_schedule_feasible = True
        # remove the source and sink in mode_dict (compatibility)
        del mode_dict[self.rcpsp_problem.source_task]
        del mode_dict[self.rcpsp_problem.sink_task]
        return RCPSPSolution(
            problem=self.rcpsp_problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=list(mode_dict.values()),
            rcpsp_schedule_feasible=rcpsp_schedule_feasible,
        )

    def get_eligibles(self, resource_avail, completed, scheduled):
        eligibles: dict[Hashable, list[int]] = {}
        unscheduled = set(self.rcpsp_problem.tasks_list) - set(scheduled)
        if __debug__:
            print("------start get eligibles-----")
            print(f"resource avail:{resource_avail}")
            print(f"{eligibles=}")
        for act in unscheduled:
            # check precedence constraints
            if set(self.rcpsp_problem.graph.graph_nx.predecessors(act)) <= set(
                completed
            ):
                # check resource constraints for each mode
                for m in self.rcpsp_problem.mode_details[act]:
                    valid = True
                    if __debug__:
                        print(
                            f"Act:{act}, mode:{m}, {self.rcpsp_problem.mode_details[act][m]}"
                        )
                    for res in self.rcpsp_problem.resources:
                        if self.rcpsp_problem.mode_details[act][m].get(res, 0) == 0:
                            continue
                        if (
                            resource_avail[res]
                            < self.rcpsp_problem.mode_details[act][m][res]
                        ):
                            valid = False
                    if valid:
                        if eligibles.get(act, None) == None:
                            eligibles[act] = [m]
                        else:
                            eligibles[act].append(m)
        if __debug__:
            print(f"Eligibles:{eligibles}")
            print("------end get eligibles-----")
        return eligibles

    def _compute_dynamic_cpm(self, eligible: list):
        import networkx as nx

        _eligible: list = list(eligible)
        _scheduled = list(self.scheduled)
        _subproblem = set(self.rcpsp_problem.tasks_list) - set(_scheduled)
        cpm_nodes: Dict[Any, CPMObject] = {
            n: CPMObject(None, None, None, None) for n in _subproblem
        }
        _subgraph: nx.DiGraph = self.rcpsp_problem.graph.graph_nx.subgraph(_subproblem)
        subgraph_size = _subgraph.number_of_nodes()
        # earliest start and finish
        ## for those activities can scheduled immediately (predecessors already finished)
        for act in _eligible:
            cpm_nodes[act]._ESD = self.current_time
            cpm_nodes[act]._EFD = (
                self.current_time + self.rcpsp_problem.mode_details[act][1]["duration"]
            )
        _scheduled = list(_eligible)

        while len(_scheduled) < subgraph_size:
            _unscheduled = list(set(_subproblem) - set(_scheduled))
            # find the first one satisfying precedence constraints
            for i in range(len(_unscheduled)):
                cur_act = _unscheduled[i]
                pred_acts = list(_subgraph.predecessors(cur_act))
                if set(pred_acts) <= set(_scheduled):
                    # if no precedence tasks, earliest start time would be current time
                    max_pred_finish_time = max(
                        self.current_time, self.minimum_starting_time[cur_act]
                    )
                    for j in pred_acts:
                        max_pred_finish_time = max(
                            max_pred_finish_time, cpm_nodes[j]._EFD
                        )
                    cpm_nodes[cur_act]._ESD = max_pred_finish_time
                    cpm_nodes[cur_act]._EFD = (
                        max_pred_finish_time
                        + self.rcpsp_problem.mode_details[cur_act][1]["duration"]
                    )
                    _scheduled.append(cur_act)
        # backward
        ## initialization
        cpm_nodes[self.rcpsp_problem.sink_task]._LSD = cpm_nodes[
            self.rcpsp_problem.sink_task
        ]._ESD
        cpm_nodes[self.rcpsp_problem.sink_task]._LFD = cpm_nodes[
            self.rcpsp_problem.sink_task
        ]._EFD
        _scheduled = [self.rcpsp_problem.sink_task]
        while len(_scheduled) < subgraph_size:
            _unscheduled = list(set(_subproblem) - set(_scheduled))
            _unscheduled.reverse()
            for i in range(len(_unscheduled)):
                cur_act = _unscheduled[i]
                succ_acts = list(_subgraph.successors(cur_act))
                if set(succ_acts) <= set(_scheduled):
                    min_succ_start_time = 9223372036854775807
                    for j in succ_acts:
                        min_succ_start_time = min(
                            min_succ_start_time, cpm_nodes[j]._LSD
                        )
                    cpm_nodes[cur_act]._LFD = min_succ_start_time
                    cpm_nodes[cur_act]._LSD = (
                        min_succ_start_time
                        - self.rcpsp_problem.mode_details[cur_act][1]["duration"]
                    )
                    _scheduled.append(cur_act)
        self.dynamic_cpm = cpm_nodes

    @abstractmethod
    def heuristic_earliest_feasible_finish_time(self) -> Union[int, float]:
        """
        Lova, Antonio; Tormos, Pilar; Barber, Federico (2006): Multi-mode resource constrained project scheduling: scheduling schemes, priority rules and mode selection rules. In Inteligencia Artificial. Revista Iberoamericana de Inteligencia Artificial 10 (30).
        Mode selection rule.
        This mode selection rule selects for each activity the execution mode such that it is scheduled with the feasible finish time as early as possible.
        For parallel SGS, it returns
        If ties occur (an activity has several feasible modes with the same minimum value of feasible finish time) the mode with the highest duration is selected.

        """
        return (
            self.current_time
            + self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]["duration"]
        )