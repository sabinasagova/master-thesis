"""
Mode-interaction terminals: MI_CONSTRAINT_TIGHTENING and
MI_RECIPROCAL_SCARCITY (mode tree) and MI_ACTIVITY_PRESSURE (activity
tree). They measure how much committing a candidate (activity, mode) pair
narrows the feasible mode choices of the other currently eligible
activities, complementing the budget signal of nr_terminals.py and the
contention signal of rccp_terminals.py.
"""
from yuantian.rcpsp_simulation import FeatureEnum

# Same scoping as RCCP_BOTTLENECK_UTIL etc: activity tree (integrated for
# SIMULTANEOUS), no mode dependency.
MODE_INTERACTION_ACTIVITY_FEATURES = [
    FeatureEnum.MI_ACTIVITY_PRESSURE,
]

# Same scoping as RCCP_CANDIDATE_CONTENTION: mode tree (integrated for
# SIMULTANEOUS).
MODE_INTERACTION_MODE_FEATURES = [
    FeatureEnum.MI_CONSTRAINT_TIGHTENING,
    FeatureEnum.MI_RECIPROCAL_SCARCITY,
]
