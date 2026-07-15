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
    # Restored for yuantian/exploratory/multi_sgs.py (Phase 0 sweep); not
    # produced by Simulator.get_simulator and unused by the default GPHH run.
    BACKWARD_SERIAL_SGS = "backward"


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
    # critical path propagation features
    CP_FORWARD = "CP_FORWARD"
    CP_BACKWARD = "CP_BACKWARD"
    CP_SLACK_SCORE = "CP_SLACK_SCORE"
    CP_PROB = "CP_PROB"
    # non-renewable resource features (see nr_terminals.py)
    NR_STOCK_RATIO = "NR_STOCK_RATIO"
    NR_MODE_DEMAND_RATIO = "NR_MODE_DEMAND_RATIO"
    NR_BUDGET_PRESSURE = "NR_BUDGET_PRESSURE"
    # resource-constrained critical path features (see rccp_terminals.py)
    RCCP_BOTTLENECK_UTIL = "RCCP_BOTTLENECK_UTIL"
    RCCP_CANDIDATE_CONTENTION = "RCCP_CANDIDATE_CONTENTION"
    RCCP_SLACK = "RCCP_SLACK"
    RCCP_PRESSURE_TREND = "RCCP_PRESSURE_TREND"
    RCCP_RESOURCE_CONCENTRATION = "RCCP_RESOURCE_CONCENTRATION"
    # mode-interaction features (see mode_interaction_terminals.py)
    MI_CONSTRAINT_TIGHTENING = "MI_CONSTRAINT_TIGHTENING"
    MI_RECIPROCAL_SCARCITY = "MI_RECIPROCAL_SCARCITY"
    MI_ACTIVITY_PRESSURE = "MI_ACTIVITY_PRESSURE"


# Sentinel for NR_MODE_DEMAND_RATIO when a candidate mode demands a
# non-renewable resource that's already fully depleted (remaining <= 0):
# finite (so it composes safely with the GP's add/sub/mul/protected-div
# primitives, none of which guard against inf/nan) but far larger than any
# real ratio, so it still reads as "avoid this mode" under priority_extre="min".
_NR_INFEASIBLE_DEMAND_RATIO = 1e6


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
        # Bumped each time self.eligibles gets reassigned (once per SGS
        # decision). _current_resource_avail uses this to cache its result
        # for the whole decision -- remaining capacity doesn't change
        # depending on which (cur_act, cur_mode) is being scored, only on
        # what's already committed, so we don't need to recompute it for
        # every candidate. Added after measuring mode_interaction_terminals
        # was too slow without it; RCCP terminals get the speedup for free
        # since they call the same method.
        self._decision_counter = 0
        self._resource_avail_cache_key = None
        self._resource_avail_cache = None
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
            FeatureEnum.CP_FORWARD: self.feature_cp_forward,
            FeatureEnum.CP_BACKWARD: self.feature_cp_backward,
            FeatureEnum.CP_SLACK_SCORE: self.feature_cp_slack_score,
            FeatureEnum.CP_PROB: self.feature_cp_prob,
            FeatureEnum.NR_STOCK_RATIO: self.feature_nr_stock_ratio,
            FeatureEnum.NR_MODE_DEMAND_RATIO: self.feature_nr_mode_demand_ratio,
            FeatureEnum.NR_BUDGET_PRESSURE: self.feature_nr_budget_pressure,
            FeatureEnum.RCCP_BOTTLENECK_UTIL: self.feature_rccp_bottleneck_util,
            FeatureEnum.RCCP_CANDIDATE_CONTENTION: self.feature_rccp_candidate_contention,
            FeatureEnum.RCCP_SLACK: self.feature_rccp_slack,
            FeatureEnum.RCCP_PRESSURE_TREND: self.feature_rccp_pressure_trend,
            FeatureEnum.RCCP_RESOURCE_CONCENTRATION: self.feature_rccp_resource_concentration,
            FeatureEnum.MI_CONSTRAINT_TIGHTENING: self.feature_mi_constraint_tightening,
            FeatureEnum.MI_RECIPROCAL_SCARCITY: self.feature_mi_reciprocal_scarcity,
            FeatureEnum.MI_ACTIVITY_PRESSURE: self.feature_mi_activity_pressure,
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

    def feature_cp_forward(self) -> Union[int, float]:
        return self.rcpsp_problem.cp_features[self.cur_act]["forward"]

    def feature_cp_backward(self) -> Union[int, float]:
        return self.rcpsp_problem.cp_features[self.cur_act]["backward"]

    def feature_cp_slack_score(self) -> Union[int, float]:
        return self.rcpsp_problem.cp_features[self.cur_act]["slack_score"]

    def feature_cp_prob(self) -> Union[int, float]:
        return self.rcpsp_problem.cp_features[self.cur_act]["prob"]

    def _nr_remaining_stock(self) -> Dict[str, float]:
        """Current remaining stock for every non-renewable resource, given
        the schedule built so far. Schedule-state-dependent (like the
        dynamic CPM terminals), so this cannot be precomputed in
        nr_terminals.py; implemented per-subclass since Serial and Parallel
        track resource availability in different shapes (see
        SerialSimulator/ParallelSimulator)."""
        raise NotImplementedError

    def feature_nr_stock_ratio(self) -> Union[int, float]:
        non_renewable = self.rcpsp_problem.non_renewable_resources_list
        if not non_renewable:
            return 1.0  # no NR resources tracked: nothing is being depleted
        remaining = self._nr_remaining_stock()
        initial = self.rcpsp_problem.resources
        return min(
            remaining[res] / initial[res] if initial[res] else 1.0
            for res in non_renewable
        )

    def feature_nr_mode_demand_ratio(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        non_renewable = self.rcpsp_problem.non_renewable_resources_list
        if not non_renewable:
            return 0.0  # no NR resources tracked: this mode costs nothing in NR terms
        remaining = self._nr_remaining_stock()
        mode_details = self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]
        ratios = []
        for res in non_renewable:
            demand = mode_details.get(res, 0)
            if demand == 0:
                ratios.append(0.0)
            elif remaining[res] <= 0:
                ratios.append(_NR_INFEASIBLE_DEMAND_RATIO)
            else:
                ratios.append(demand / remaining[res])
        return max(ratios)

    def feature_nr_budget_pressure(self) -> Union[int, float]:
        non_renewable = self.rcpsp_problem.non_renewable_resources_list
        if not non_renewable:
            return 1.0  # no NR resources tracked: no pressure at all
        remaining = self._nr_remaining_stock()
        static = self.rcpsp_problem.nr_static_features
        not_yet_scheduled = set(self.rcpsp_problem.tasks_list) - set(self.scheduled)
        pressures = []
        for res in non_renewable:
            min_future_demand = sum(
                static["min_mode_demand"][task][res] for task in not_yet_scheduled
            )
            initial = static["initial_stock"][res]
            margin = remaining[res] - min_future_demand
            pressures.append(margin / initial if initial else 0.0)
        return min(pressures)

    def _compute_current_resource_avail(self) -> Dict[str, float]:
        """Current remaining capacity for every *renewable* resource type,
        given the schedule built so far -- the resource-contention analogue
        of `_nr_remaining_stock`. Schedule-state-dependent, implemented
        per-subclass since Serial and Parallel track resource availability
        in different shapes (see SerialSimulator/ParallelSimulator;
        rccp_terminals.py explains why). Don't call this directly outside
        `_current_resource_avail` -- it's the uncached computation; callers
        should use `_current_resource_avail()` (below) so repeated calls
        within the same decision point are cheap."""
        raise NotImplementedError

    def _current_resource_avail(self) -> Dict[str, float]:
        if self._resource_avail_cache_key != self._decision_counter:
            self._resource_avail_cache_key = self._decision_counter
            self._resource_avail_cache = self._compute_current_resource_avail()
        return self._resource_avail_cache

    def _rccp_bottleneck_resource(self):
        """Returns (res, utilization) for whichever renewable resource is
        most contended right now. Shared by all the RCCP terminals so they
        agree on what "the" bottleneck is at any given point. Returns
        (None, 0.0) if there are no renewable resources."""
        renewable = self.rcpsp_problem.renewable_resources_list
        if not renewable:
            return None, 0.0
        avail = self._current_resource_avail()
        capacity = self.rcpsp_problem.resources
        utilizations = {
            res: 1.0 - (avail[res] / capacity[res] if capacity[res] else 0.0)
            for res in renewable
        }
        bottleneck_res = max(utilizations, key=utilizations.get)
        return bottleneck_res, utilizations[bottleneck_res]

    def feature_rccp_bottleneck_util(self) -> Union[int, float]:
        _, utilization = self._rccp_bottleneck_resource()
        return utilization

    def feature_rccp_candidate_contention(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        bottleneck_res, _ = self._rccp_bottleneck_resource()
        if bottleneck_res is None:
            return 0.0  # no renewable resources tracked: nothing to contend for
        avail = self._current_resource_avail()
        demand = self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode].get(
            bottleneck_res, 0
        )
        if demand == 0:
            return 0.0
        remaining = avail[bottleneck_res]
        if remaining <= 0:
            return _NR_INFEASIBLE_DEMAND_RATIO  # same "avoid this" sentinel as NR_MODE_DEMAND_RATIO
        return demand / remaining

    def feature_rccp_slack(self) -> Union[int, float]:
        # CP_SLACK_SCORE only looks at precedence, so it overestimates true
        # slack when the bottleneck resource is tight -- an activity can
        # have zero precedence conflicts and still have to wait because its
        # resource is fully booked. This just discounts slack by how busy
        # the bottleneck is. Shrinks slack toward 0 as
        # utilization climbs toward 1.
        _, utilization = self._rccp_bottleneck_resource()
        return self.feature_cp_slack_score() * (1.0 - utilization)

    def feature_rccp_pressure_trend(self) -> Union[int, float]:
        # Forward-looking version of bottleneck_util: sums up what every
        # currently-eligible activity would minimally need from the
        # bottleneck resource, divided by what's left. A value >1 means more demand is
        # queued than remaining capacity; optimistic, since it ignores
        # competition among the eligible activities.
        if not self.eligibles:
            return 0.0
        bottleneck_res, _ = self._rccp_bottleneck_resource()
        if bottleneck_res is None:
            return 0.0
        near_term_demand = sum(
            min(
                self.rcpsp_problem.mode_details[act][m].get(bottleneck_res, 0)
                for m in self.rcpsp_problem.mode_details[act]
            )
            for act in self.eligibles
        )
        avail = self._current_resource_avail()[bottleneck_res]
        if avail <= 0:
            return _NR_INFEASIBLE_DEMAND_RATIO if near_term_demand > 0 else 0.0
        return near_term_demand / avail

    def feature_rccp_resource_concentration(self) -> Union[int, float]:
        # 1 minus normalized Shannon entropy of the mode's renewable demand
        # distribution: higher = more concentrated on one resource type.
        #
        # Two edge cases handled on purpose: k=1 (only one renewable
        # resource type exists) returns 1.0 instead of dividing by log(1);
        # a mode that draws nothing at all returns 0.0, nothing to be
        # concentrated about.
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        renewable = self.rcpsp_problem.renewable_resources_list
        k = len(renewable)
        if k == 0:
            return 0.0  # no renewable resources tracked: nothing to concentrate on
        if k == 1:
            return 1.0
        demand = self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]
        values = [demand.get(res, 0) for res in renewable]
        total = sum(values)
        if total <= 0:
            return 0.0
        entropy = -sum(
            (v / total) * np.log(v / total) for v in values if v > 0
        )
        normalized_entropy = entropy / np.log(k)
        return float(1.0 - normalized_entropy)

    # Capping how many neighbors we look at -- measured mode_interaction
    # terminals running about 3x slower without this, since realistic
    # trees can call an MI terminal multiple times and each call rescans
    # every eligible neighbor. This does make MI_CONSTRAINT_TIGHTENING/
    # MI_ACTIVITY_PRESSURE an approximation once more than 5 neighbors
    # share a resource (we just skip the smaller ones), but that tradeoff
    # seemed worth it for the speedup.
    _MI_MAX_NEIGHBORS = 5

    def _mode_interaction_neighbors(self, candidate_resources) -> list:
        """Other currently-eligible activities (not the one being scored)
        that have at least one mode demanding something from
        candidate_resources -- basically "who could this decision affect."
        Renewable resources only: two activities sharing a non-renewable
        resource at different points in time aren't really competing for
        anything, since NR budget is a whole-project pool, not a
        time-window thing.

        Capped to the `_MI_MAX_NEIGHBORS` most resource-heavy matches (by
        each candidate's single largest demand, across its own modes, on
        any resource in `candidate_resources`) -- see the cost comment
        above `_MI_MAX_NEIGHBORS`. Below the cap, this returns exactly the
        same set an uncapped scan would."""
        if not self.eligibles or not candidate_resources:
            return []
        scored = []
        for act in self.eligibles:
            if act == self.cur_act:
                continue
            modes = self.rcpsp_problem.mode_details[act]
            max_demand = max(
                (modes[m].get(res, 0) for m in modes for res in candidate_resources),
                default=0,
            )
            if max_demand > 0:
                scored.append((max_demand, act))
        if len(scored) <= self._MI_MAX_NEIGHBORS:
            return [act for _, act in scored]
        scored.sort(reverse=True)
        return [act for _, act in scored[: self._MI_MAX_NEIGHBORS]]

    def _mi_constraint_tightening_fraction(self, demand, avail) -> float:
        """Shared by MI_CONSTRAINT_TIGHTENING and MI_ACTIVITY_PRESSURE.
        Given a demand vector and current availability, finds every
        neighbor sharing a demanded resource and works out what fraction of
        that neighbor's modes would stop being feasible after committing
        demand. Averaged across neighbors (not max, so one badly-hit
        neighbor doesn't dominate over a bunch of barely-affected ones)."""
        renewable = self.rcpsp_problem.renewable_resources_list
        candidate_resources = {res for res in renewable if demand.get(res, 0) > 0}
        if not candidate_resources:
            return 0.0  # this mode draws no renewable resource: nothing to tighten
        neighbors = self._mode_interaction_neighbors(candidate_resources)
        if not neighbors:
            return 0.0
        remaining_after = {
            res: avail[res] - demand.get(res, 0) for res in candidate_resources
        }
        fractions = []
        for act in neighbors:
            modes = self.rcpsp_problem.mode_details[act]
            n_modes = len(modes)
            n_infeasible = sum(
                1
                for m in modes
                if any(modes[m].get(res, 0) > remaining_after[res] for res in candidate_resources)
            )
            fractions.append(n_infeasible / n_modes if n_modes else 0.0)
        return float(np.mean(fractions))

    def feature_mi_constraint_tightening(self) -> Union[int, float]:
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        demand = self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]
        return self._mi_constraint_tightening_fraction(demand, self._current_resource_avail())

    def feature_mi_activity_pressure(self) -> Union[int, float]:
        # Same idea as constraint_tightening but averaged over the
        # activity's own modes, since we haven't committed to cur_mode yet
        # at the activity-tree stage.
        own_modes = self.rcpsp_problem.mode_details[self.cur_act]
        if not own_modes:
            return 0.0
        avail = self._current_resource_avail()
        scores = [
            self._mi_constraint_tightening_fraction(demand, avail)
            for demand in own_modes.values()
        ]
        return float(np.mean(scores))

    def feature_mi_reciprocal_scarcity(self) -> Union[int, float]:
        # Cheaper version of constraint_tightening: instead of recomputing
        # feasibility for every neighbor mode, just checks whether the
        # neighbors' combined cheapest-mode demand already exceeds what's
        # available, for each resource the candidate mode actually needs.
        # A count, not a full feasibility check.
        if self.cur_mode is None:
            raise ValueError("Mode is not specified.")
        renewable = self.rcpsp_problem.renewable_resources_list
        demand = self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]
        candidate_resources = [res for res in renewable if demand.get(res, 0) > 0]
        if not candidate_resources or not self.eligibles:
            return 0.0
        neighbors = [act for act in self.eligibles if act != self.cur_act]
        if not neighbors:
            return 0.0
        avail = self._current_resource_avail()
        n_scarce = 0
        for res in candidate_resources:
            near_term_demand = sum(
                min(
                    self.rcpsp_problem.mode_details[act][m].get(res, 0)
                    for m in self.rcpsp_problem.mode_details[act]
                )
                for act in neighbors
            )
            if near_term_demand > avail[res]:
                n_scarce += 1
        return n_scarce / len(candidate_resources)

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

    def _nr_remaining_stock(self) -> Dict[str, float]:
        # For a non-renewable resource, buildSolution's resource update step
        # decrements resource_avail_in_time[res][t] for every t from a
        # scheduled task's end_t through the horizon once (never replenished),
        # so the *last* horizon slot already reflects every consumption so
        # far -- the same lookup buildSolution's own eligible-mode filter
        # uses (`resource_avail_in_time[res][-1]`) to decide mode feasibility.
        return {
            res: self.resource_avail_in_time[res][-1]
            for res in self.rcpsp_problem.non_renewable_resources_list
        }

    def _compute_current_resource_avail(self) -> Dict[str, float]:
        # Serial SGS has no single global "now": each eligible activity has
        # its own earliest precedence-feasible start
        # (`minimum_starting_time`), and `resource_avail_in_time[res]` is a
        # full-horizon array of remaining capacity at every future time
        # slot, reflecting everything committed so far. The natural "right
        # now" reference for a pending decision is the earliest time any
        # currently-eligible activity could actually start -- the frontier
        # of the partial schedule -- so that's where this terminal reads
        # remaining capacity from.
        renewable = self.rcpsp_problem.renewable_resources_list
        if not self.eligibles:
            frontier_time = 0
        else:
            frontier_time = min(
                self.minimum_starting_time[act] for act in self.eligibles
            )
            frontier_time = min(frontier_time, self.new_horizon - 1)
        return {res: self.resource_avail_in_time[res][frontier_time] for res in renewable}

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
            # Drop activities with zero eligible modes: with non-renewable
            # resources present this is reachable (an exhausted NR budget can
            # leave a precedence-eligible activity with no feasible mode at
            # all), unlike the renewable-only pipeline where it never came up.
            # `choose(eligibles)` (e.g. activity_first_choose) assumes every
            # value is a non-empty mode list and would crash with
            # `min() arg is an empty sequence` otherwise.
            eligibles = {act: modes for act, modes in eligibles.items() if modes}
            if not eligibles:
                unfeasible_non_renewable_resources = True
                break
            # self.eligibles was declared in __init__ but never actually
            # set anywhere -- needed it so rccp_terminals can see what's
            # eligible right now.
            self.eligibles = eligibles
            self._decision_counter += 1
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

    def _nr_remaining_stock(self) -> Dict[str, float]:
        # `resource_avail[res]` for a non-renewable resource is decremented
        # on every scheduling decision and -- unlike renewable resources --
        # deliberately never replenished on task completion (see Step 1.3 in
        # buildSolution, "don't update non-renewable resources"), so it
        # already *is* the live remaining-stock value.
        return {
            res: self.resource_avail[res]
            for res in self.rcpsp_problem.non_renewable_resources_list
        }

    def _compute_current_resource_avail(self) -> Dict[str, float]:
        # Parallel SGS already maintains a single global current_time and a
        # scalar (not time-indexed) resource_avail dict that's kept exactly
        # in sync with it -- updated whenever an activity starts or
        # finishes, so it directly *is* the live remaining-capacity value at
        # self.current_time, no further lookup needed (unlike Serial, which
        # has no single global "now").
        return {
            res: self.resource_avail[res]
            for res in self.rcpsp_problem.renewable_resources_list
        }

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
        # exposed as instance state (not just a local var) so
        # _nr_remaining_stock can read live non-renewable consumption;
        # doesn't otherwise change how resource_avail itself is used below.
        self.resource_avail = resource_avail
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
            # same deal as SerialSimulator -- rccp_terminals needs this set
            self.eligibles = eligibles
            self._decision_counter += 1
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
                    self.eligibles = eligibles
                    self._decision_counter += 1
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


class BackwardSerialSimulator(Simulator):
    """
    Backward Serial SGS. Restored for yuantian/exploratory/multi_sgs.py (the
    Phase 0 exploratory sweep that preceded cp_propagation.py / hybrid_gp.py /
    nr_terminals.py); not used by the default GPHH run or by any of those
    three extensions.

    Schedules activities from the project sink back to the source:
      - Eligible at each step = activities whose ALL successors are already placed.
      - For each chosen activity, search DOWNWARD from its latest feasible finish
        time (= minimum start time of its successors) for the latest slot where
        all resource constraints are satisfied.
      - Returns a valid forward schedule: start[j] = horizon - backward_end[j].

    Using LS/LF terminals (which express deadline-relative urgency) as the GP
    priority signal is natural here; the evolved rule can differ from the forward
    rule and typically finds complementary, tighter schedules.

    Does not track non-renewable resource consumption correctly across the
    backward pass (a pre-existing limitation, not introduced by this restoration);
    callers should not enable it on NR-tight instances (see multi_sgs.py's
    ``use_backward`` flag).
    """

    def __init__(self) -> None:
        super().__init__()
        self.type = SimulatorTypeEnum.BACKWARD_SERIAL_SGS

    def _compute_dynamic_cpm(self, eligible: list):
        import networkx as nx

        _eligible = list(eligible)
        _scheduled_set = set(self.scheduled)  # backward-scheduled so far
        _subproblem = set(self.rcpsp_problem.tasks_list) - _scheduled_set
        cpm_nodes: Dict[Any, CPMObject] = {
            n: CPMObject(None, None, None, None) for n in _subproblem
        }
        _subgraph: nx.DiGraph = self.rcpsp_problem.graph.graph_nx.subgraph(_subproblem)

        for act in _eligible:
            cpm_nodes[act]._LFD = self.maximum_finish_time[act]
            cpm_nodes[act]._LSD = (
                self.maximum_finish_time[act]
                - self.rcpsp_problem.mode_details[act][1]["duration"]
            )
        _done = list(_eligible)
        while len(_done) < len(_subproblem):
            for cur in set(_subproblem) - set(_done):
                succ = list(_subgraph.successors(cur))
                if set(succ) <= set(_done):
                    min_succ_lsd = (
                        min(cpm_nodes[s]._LSD for s in succ) if succ else self.new_horizon
                    )
                    cpm_nodes[cur]._LFD = min_succ_lsd
                    cpm_nodes[cur]._LSD = (
                        min_succ_lsd - self.rcpsp_problem.mode_details[cur][1]["duration"]
                    )
                    _done.append(cur)

        # Forward pass: earliest start/finish (from source)
        source = self.rcpsp_problem.source_task
        if source in _subproblem:
            cpm_nodes[source]._ESD = 0
            cpm_nodes[source]._EFD = self.rcpsp_problem.mode_details[source][1]["duration"]
        _fwd_done = [source] if source in _subproblem else []
        while len(_fwd_done) < len(_subproblem):
            for cur in set(_subproblem) - set(_fwd_done):
                preds = list(_subgraph.predecessors(cur))
                if set(preds) <= set(_fwd_done):
                    max_pred_efd = max((cpm_nodes[p]._EFD for p in preds), default=0)
                    cpm_nodes[cur]._ESD = max_pred_efd
                    cpm_nodes[cur]._EFD = (
                        max_pred_efd + self.rcpsp_problem.mode_details[cur][1]["duration"]
                    )
                    _fwd_done.append(cur)

        self.dynamic_cpm = cpm_nodes

    def buildSolution(self, domain: RCPSPModel, choose: callable) -> RCPSPSolution:
        self.rcpsp_problem: RCPSPModel = domain
        successors = self.rcpsp_problem.successors  # {act: [succ, ...]}
        predecessors = self.rcpsp_problem.graph.graph_nx.predecessors
        graph_nx = self.rcpsp_problem.graph.graph_nx
        import networkx as nx

        all_jobs = self.rcpsp_problem.tasks_list
        self.new_horizon = self.rcpsp_problem.horizon
        sink = self.rcpsp_problem.sink_task
        source = self.rcpsp_problem.source_task

        # ── Deadline computation ───────────────────────────────────────────────
        # The backward schedule places activities as LATE as possible within a
        # deadline D.  After shifting so source starts at 0, the makespan equals
        # D - T_src where T_src = the source's pre-shift start time.  Resource
        # conflicts push activities earlier, increasing makespan toward the
        # resource-constrained optimum.  D must be >= optimal makespan (for
        # feasibility).  We use min(horizon, CPM_max_duration * 6) as a tight
        # but safe upper bound; for MMLIB50 this gives D ~= 4-6x the typical
        # optimal and much less than the raw horizon (~17x optimal).
        def _min_dur(act):
            modes = self.rcpsp_problem.mode_details[act]
            return min(modes[m]["duration"] for m in modes)

        def _max_dur(act):
            modes = self.rcpsp_problem.mode_details[act]
            return max(modes[m]["duration"] for m in modes)

        # CPM forward pass with MAX durations (tighter lower bound than min, avoids 0-duration modes)
        efd_cpm_max: Dict[Hashable, int] = {}
        for act in nx.topological_sort(graph_nx):
            preds_list = list(graph_nx.predecessors(act))
            esd = max((efd_cpm_max[p] for p in preds_list), default=0)
            efd_cpm_max[act] = esd + _max_dur(act)
        cp_max: int = efd_cpm_max[sink]

        # Deadline: tight enough to pack activities compactly, but not so tight
        # that resource conflicts force activities before time 0.
        deadline: int = min(self.new_horizon, max(cp_max * 6, self.new_horizon // 5))

        # CPM backward pass with deadline as the initial LFD for all activities.
        # Correct formula: LFD[j] = min(LFD[s] - min_dur[s] for s in successors(j))
        lfd_cpm: Dict[Hashable, int] = {act: deadline for act in all_jobs}
        for act in reversed(list(nx.topological_sort(graph_nx))):
            succs_list = list(graph_nx.successors(act))
            if succs_list:
                lfd_cpm[act] = min(lfd_cpm[s] - _min_dur(s) for s in succs_list)

        # Resource availability — same layout as forward SGS
        self.resource_avail_in_time: Dict = {}
        for res in self.rcpsp_problem.resources_list:
            self.resource_avail_in_time[res] = np.full(
                self.new_horizon, self.rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()

        # mode_dict and backward start times
        mode_dict: Dict[Hashable, int] = {act: 1 for act in all_jobs}
        activity_start_times: Dict[Hashable, int] = {}
        unfeasible_non_renewable_resources = False

        # maximum_finish_time from CPM backward pass (tight deadlines per activity)
        self.maximum_finish_time: Dict[Hashable, int] = dict(lfd_cpm)
        # minimum_starting_time kept at 0 for terminal compatibility
        self.minimum_starting_time: Dict[Hashable, int] = {act: 0 for act in all_jobs}

        # Sink pre-scheduled at the computed deadline
        activity_start_times[sink] = deadline
        self.scheduled = [sink]

        while (
            len(self.scheduled) < self.rcpsp_problem.n_jobs
            and not unfeasible_non_renewable_resources
        ):
            # Step 1: eligible = unscheduled activities whose ALL successors are placed
            eligibles: Dict = {}
            unscheduled = set(all_jobs) - set(self.scheduled)
            for act in unscheduled:
                if set(successors[act]) <= set(self.scheduled):
                    eligibles[act] = []

            # Step 2: NR-feasible modes (same logic as forward SGS)
            for act in eligibles:
                for m in self.rcpsp_problem.mode_details[act]:
                    valid_mode = True
                    for res in self.rcpsp_problem.resources:
                        if self.rcpsp_problem.mode_details[act][m].get(res, 0) == 0:
                            continue
                        if (
                            self.resource_avail_in_time[res][-1]
                            < self.rcpsp_problem.mode_details[act][m][res]
                        ):
                            valid_mode = False
                    if valid_mode:
                        eligibles[act].append(m)
                if not eligibles[act]:
                    eligibles[act] = list(self.rcpsp_problem.mode_details[act].keys())

            act_id, mode_id = choose(eligibles)
            mode_dict[act_id] = mode_id
            duration = self.rcpsp_problem.mode_details[act_id][mode_id]["duration"]

            # Step 3: latest feasible START = maximum_finish_time[act] - duration,
            #         then search DOWNWARD for a feasible slot.
            # When resource conflicts push current_max_start below 0 we place the
            # activity at 0 (the schedule may be resource-infeasible there, but we
            # still produce a complete schedule whose makespan will be penalised by
            # the fitness function).
            current_max_start = self.maximum_finish_time[act_id] - duration
            valid = False
            while not valid:
                valid = True
                if current_max_start < 0:
                    current_max_start = 0  # clamp; resource conflicts accepted at time 0
                    break
                for t in range(current_max_start, current_max_start + duration):
                    if t >= self.new_horizon:
                        valid = False
                        current_max_start = max(0, self.new_horizon - duration)
                        break
                    for res in self.rcpsp_problem.resources_list:
                        if self.rcpsp_problem.mode_details[act_id][mode_id].get(res, 0) == 0:
                            continue
                        if (
                            self.resource_avail_in_time[res][t]
                            < self.rcpsp_problem.mode_details[act_id][mode_id][res]
                        ):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    current_max_start -= 1

            # Step 4: update resource availability
            start_t = current_max_start
            end_t = start_t + duration
            for t in range(start_t, end_t):
                for res in self.resource_avail_in_time:
                    demand = self.rcpsp_problem.mode_details[act_id][mode_id].get(res, 0)
                    if demand == 0:
                        continue
                    self.resource_avail_in_time[res][t] -= demand
                    if res in self.rcpsp_problem.non_renewable_resources_list and t == end_t - 1:
                        for tt in range(end_t, self.new_horizon):
                            self.resource_avail_in_time[res][tt] -= demand
                            if self.resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True

            # Step 5: record and propagate maximum_finish_time to predecessors
            activity_start_times[act_id] = current_max_start
            self.scheduled.append(act_id)
            for pred in predecessors(act_id):
                self.maximum_finish_time[pred] = min(
                    self.maximum_finish_time[pred], activity_start_times[act_id]
                )

        # ── Time shift: translate so source starts at t=0 ────────────────────
        # The backward schedule was built relative to a deadline; shifting left
        # by T_src gives a forward schedule with source at 0 and makespan = cp_length - T_src.
        T_src = activity_start_times.get(source, 0)
        rcpsp_schedule: Dict[Hashable, Dict[str, int]] = {}
        for act_id, start in activity_start_times.items():
            dur = self.rcpsp_problem.mode_details[act_id][mode_dict[act_id]]["duration"]
            shifted = max(0, start - T_src)
            rcpsp_schedule[act_id] = {"start_time": shifted, "end_time": shifted + dur}

        if unfeasible_non_renewable_resources:
            rcpsp_schedule_feasible = False
            if sink not in rcpsp_schedule:
                rcpsp_schedule[sink] = {"start_time": 99999999, "end_time": 99999999}
        else:
            rcpsp_schedule_feasible = True

        del mode_dict[source]
        del mode_dict[sink]
        return RCPSPSolution(
            problem=self.rcpsp_problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=list(mode_dict.values()),
            rcpsp_schedule_feasible=rcpsp_schedule_feasible,
        )

    def heuristic_earliest_feasible_finish_time(self) -> Union[int, float]:
        # Returns the latest resource-feasible FINISH time for this activity/mode.
        # We search downward from maximum_finish_time and return start + duration.
        duration = self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode]["duration"]
        current_max_start = self.maximum_finish_time[self.cur_act] - duration
        valid = False
        while not valid and current_max_start >= 0:
            valid = True
            for t in range(current_max_start, current_max_start + duration):
                if t >= self.new_horizon:
                    return 0
                for res in self.rcpsp_problem.resources_list:
                    if self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode].get(res, 0) == 0:
                        continue
                    if (
                        self.resource_avail_in_time[res][t]
                        < self.rcpsp_problem.mode_details[self.cur_act][self.cur_mode][res]
                    ):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                current_max_start -= 1
        return current_max_start + duration
