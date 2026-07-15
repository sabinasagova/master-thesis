"""
Resource-constrained critical path (RCCP) terminals: RCCP_BOTTLENECK_UTIL,
RCCP_CANDIDATE_CONTENTION, RCCP_SLACK, RCCP_PRESSURE_TREND and
RCCP_RESOURCE_CONCENTRATION.

Unlike the CPM-based terminals (which depend on the precedence graph
only), these read the live renewable-resource state of the partial
schedule at each decision point; shared state is cached per decision step
in rcpsp_simulation.py.
"""
from yuantian.rcpsp_simulation import FeatureEnum

# Goes wherever the NR schedule-state terminals go (activity tree for the
# two-step decision types, integrated tree for SIMULTANEOUS).
RCCP_ACTIVITY_FEATURES = [
    FeatureEnum.RCCP_BOTTLENECK_UTIL,
    FeatureEnum.RCCP_SLACK,
    FeatureEnum.RCCP_PRESSURE_TREND,
]

# Mode-specific, goes on the mode tree (integrated for SIMULTANEOUS).
RCCP_MODE_FEATURES = [
    FeatureEnum.RCCP_CANDIDATE_CONTENTION,
    FeatureEnum.RCCP_RESOURCE_CONCENTRATION,
]
