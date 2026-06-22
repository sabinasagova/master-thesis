"""
Modifications to the GPHH baseline (Yuan Tian, Mei, Zhang — CEC 2024).
Each modification is registered at the bottom; gphh_solver.py picks them up
via the use_* flags on ParametersGPHH.

Intentionally thin: just the registries wiring each modification into
gphh_solver.py's primitive sets, plus if_then_else_operator (the one bit of
logic not already in rcpsp_simulation.py / gp_algorithms.py). No hot loop to
optimise here — registry lookups are O(1) setup-time, not per-individual.
"""

from typing import Callable, Dict, List

from yuantian.rcpsp_simulation import FeatureEnum
from yuantian.multitreegp import TerminalTypeEnum
from yuantian.gp_algorithms import mutCriticalPathPreserving


# Modification 1 — lazy if_then_else operator
#
# Yuan's original if_then_else is a plain Python function: it evaluates all
# three arguments before branching, so short-circuiting never happens and
# the truthy check on continuous GP values is ambiguous (a slack of 0.0
# incorrectly branches to the else arm). Every other operator in the codebase
# returns a closure so the tree evaluator can call branches lazily; this
# operator is fixed to follow the same convention. The threshold is changed
# from Python-truthy to > 0, which gives a well-defined zero-crossing and
# pairs cleanly with IS_ON_CRITICAL_PATH (1 on the path, 0 off it).

def if_then_else_operator(
    cond: Callable[[], float], out1: Callable[[], float], out2: Callable[[], float]
) -> Callable[[], float]:
    def if_then_else() -> float:
        return out1() if cond() > 0 else out2()
    return if_then_else


# Modification 2 — nonrenewable-resource terminals
#
# The original terminal set has no signal about NR resources. All resource
# terminals (AVG_RESOURCE_REQUIREMENT etc.) refer to renewable capacity only;
# the SGS tracks NR budgets for feasibility but the priority rule is blind to
# them. Two new terminals expose NR state to the GP:
#
#   NR_STOCK_RATIO (activity tree): mean remaining NR budget as a fraction of
#   total capacity across all NR resources. When NR stock is low the GP can
#   learn to rush activities that have cheap NR modes, or deprioritise those
#   that consume a lot.
#
#   NR_MODE_DEMAND_RATIO (mode tree): demand of this (activity, mode) pair
#   divided by the remaining stock, averaged over NR resources. Values near
#   1 mean this mode would nearly exhaust the remaining budget; values > 1
#   signal infeasibility. Lets the GP steer mode selection away from budget-
#   busting choices when stock is tight.
#
# Both terminals return neutral values (1.0 / 0.0) on instances that have no
# NR resources, so they degrade gracefully on standard MRCPSP benchmarks.
# Implementation: rcpsp_simulation.feature_nr_stock_ratio /
#                 rcpsp_simulation.feature_nr_mode_demand_ratio


# Modification 3 — scheduling-state and mode-flexibility terminals
#
# A set of terminals that give the GP information about the current state of
# the schedule and the structure of the activity being considered, which the
# original terminal set lacks:
#
#   SCHEDULED_FRACTION (activity tree): fraction of non-dummy activities
#   already placed. Lets the GP behave differently early vs. late in the
#   schedule — e.g. be more aggressive about resource use at the end when
#   contention is lower.
#
#   NUM_MODES (activity tree): how many execution modes this activity has.
#   Activities with more choices are less urgent in terms of feasibility, so
#   the GP can safely defer them when resources are tight.
#
#   DURATION_FLEXIBILITY (activity tree): ratio of max to min duration across
#   modes. A high ratio means there is a big trade-off between speed and
#   resource use; a ratio of 1 means all modes take the same time.
#
#   BOTTLENECK_RENEWABLE_RATIO (activity tree): demand on the most-constrained
#   renewable resource (highest demand / capacity ratio) across all modes.
#   Identifies activities that are hard to fit regardless of mode choice.
#
#   RENEWABLE_DEMAND_VS_AVAILABILITY (mode tree): demand of this mode on each
#   renewable resource divided by current availability, summed over resources.
#   A direct measure of how much this (activity, mode) would strain the
#   resource pool right now, as opposed to in isolation.
# Implementation: rcpsp_simulation (one feature_* method per terminal)


# Modification 4 — critical-path-preserving mutation
#
# The standard mutBiased operator picks any subtree uniformly and replaces it,
# so a CP-aware branch like if_then_else(IS_ON_CRITICAL_PATH, ...) is just as
# likely to be destroyed as random noise. This matters because Modification 1
# makes CP-branching rules semantically correct for the first time, but the
# standard mutation immediately erases them. mutCriticalPathPreserving scans
# the tree before choosing a mutation point, marks every node whose subtree
# contains IS_ON_CRITICAL_PATH, Slack, or Dynamic_Slack as protected, and
# picks from the remaining nodes only. If the whole tree is CP-related it
# falls back to mutBiased so evolution never gets stuck.
# Implementation: gp_algorithms.mutCriticalPathPreserving


# Modification 5 — backward serial SGS
#
# The original code only implements forward SGS: activities are scheduled
# source-to-sink, each placed at its earliest resource-feasible start.
# BackwardSerialSimulator reverses this: eligible activities are those whose
# successors are all placed, and each is assigned the latest resource-feasible
# slot before its successors begin. After the full backward pass the schedule
# is time-shifted so the source lands at t=0, giving makespan = deadline - T_src.
# A GP rule evolved for backward scheduling uses LF/LS/Dynamic_LFD as primary
# signals rather than EF/ES, exploring a different region of the solution
# space. The motivation is the double-pass idea from Valls et al. (2005),
# applied here to the hyper-heuristic level.
# Implementation: rcpsp_simulation.BackwardSerialSimulator


# Modification 6 — CP_EXTENSION_IF_SCHEDULED terminal
#
# IS_ON_CRITICAL_PATH answers "is this activity already on the critical path?"
# but not "would scheduling it now, in this mode, move the project end date?"
# These are different questions. An activity on the CP can still have zero
# extension if resources happen to be free; an activity off it can have
# positive extension if resource contention pushes its finish past its LFD.
# CP_Ext = max(0, EFFT - LFD) quantifies this directly: it is the number of
# time units by which committing to this (activity, mode) right now would
# delay the project end. The GP can use it to prefer modes that keep CP_Ext
# at zero even when duration is longer, or to urgently schedule activities
# with large CP_Ext before the delay compounds.
# Implementation: rcpsp_simulation.feature_cp_extension_if_scheduled


# ── Registries ────────────────────────────────────────────────────────────

ACTIVE_MODIFICATIONS: Dict[str, Callable[..., Callable[[], float]]] = {
    "if_else": if_then_else_operator,
}

MUTATION_MODIFICATIONS: Dict[str, Callable] = {
    "mutate": mutCriticalPathPreserving,
}

# Modification 2 — NR budget terminals (use_nr_terminals=True)
NR_TERMINALS: Dict[str, List[FeatureEnum]] = {
    TerminalTypeEnum.ACTIVITY.value: [
        FeatureEnum.NR_STOCK_RATIO,
    ],
    TerminalTypeEnum.MODE.value: [
        FeatureEnum.NR_MODE_DEMAND_RATIO,
    ],
}

# Modifications 3 + 6 — scheduling-state terminals (use_scheduling_state_terminals=True)
SCHEDULING_STATE_TERMINALS: Dict[str, List[FeatureEnum]] = {
    TerminalTypeEnum.ACTIVITY.value: [
        FeatureEnum.SCHEDULED_FRACTION,
        FeatureEnum.NUM_MODES,
        FeatureEnum.DURATION_FLEXIBILITY,
        FeatureEnum.BOTTLENECK_RENEWABLE_RATIO,
    ],
    TerminalTypeEnum.MODE.value: [
        FeatureEnum.RENEWABLE_DEMAND_VS_AVAILABILITY,
        FeatureEnum.CP_EXTENSION_IF_SCHEDULED,
    ],
}

# Modification 7 (exploratory) — dynamic urgency and mode-regret terminals
# (use_opportunity_terminals=True). URGENCY_SCORE reuses the dynamic CPM pass
# every simulator already runs each decision step (see Simulator._compute_dynamic_cpm),
# so it costs nothing extra; MODE_DURATION_REGRET is a static per-mode duration
# comparison, not a full tentative-CPM-recompute opportunity cost.
OPPORTUNITY_TERMINALS: Dict[str, List[FeatureEnum]] = {
    TerminalTypeEnum.ACTIVITY.value: [
        FeatureEnum.URGENCY_SCORE,
    ],
    TerminalTypeEnum.MODE.value: [
        FeatureEnum.MODE_DURATION_REGRET,
    ],
}

NEW_TERMINALS: Dict[str, List[FeatureEnum]] = NR_TERMINALS  # backward-compatible alias
