"""
Critical path propagation features for MRCPSP activities.

Computes, once per problem instance, a forward/backward propagation score for
every activity based on the precedence graph. Results are cached on the
problem instance (`problem.cp_features`) so GP terminal evaluation never
recomputes them.
"""
from typing import Any, Dict

from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP


def compute_cp_propagation_features(problem: ANY_RCPSP) -> Dict[Any, Dict[str, float]]:
    """Compute critical path propagation features for every activity in `problem`.

    Requires `problem.cpm_esd` (project makespan estimate) to already be set,
    e.g. via `compute_cpm`.

    For each activity i:
        cp_forward(i)  = duration(i) + max(cp_forward(succ) for succ in successors(i))
        cp_backward(i) = min(cp_backward(succ) for succ in successors(i)) - duration(i)
        cp_slack_score(i) = cp_backward(i) - cp_forward(i)   (using the normalized values below)
        cp_prob(i) = cp_forward_raw(i) / max(cp_forward_raw over all activities)

    cp_forward and cp_backward are normalized by the project makespan estimate.

    Returns:
        Dict mapping activity id to {"forward", "backward", "slack_score", "prob"}.
    """
    import networkx as nx

    graph_nx = problem.graph.graph_nx
    order = list(nx.topological_sort(graph_nx))  # source -> sink

    min_duration = {
        node: min(problem.mode_details[node][m]["duration"] for m in problem.mode_details[node])
        for node in problem.tasks_list
    }

    makespan_estimate = problem.cpm_esd
    cp_forward_raw: Dict[Any, float] = {}
    cp_backward_raw: Dict[Any, float] = {}
    for node in reversed(order):  # sink -> source: successors are always processed first
        succs = problem.successors[node]
        if succs:
            cp_forward_raw[node] = min_duration[node] + max(cp_forward_raw[s] for s in succs)
            cp_backward_raw[node] = min(cp_backward_raw[s] for s in succs) - min_duration[node]
        else:
            cp_forward_raw[node] = min_duration[node]
            cp_backward_raw[node] = makespan_estimate - min_duration[node]

    max_forward_raw = max(cp_forward_raw.values()) if cp_forward_raw else 0.0

    features: Dict[Any, Dict[str, float]] = {}
    for node in problem.tasks_list:
        forward_norm = cp_forward_raw[node] / makespan_estimate if makespan_estimate else 0.0
        backward_norm = cp_backward_raw[node] / makespan_estimate if makespan_estimate else 0.0
        features[node] = {
            "forward": forward_norm,
            "backward": backward_norm,
            "slack_score": backward_norm - forward_norm,
            "prob": cp_forward_raw[node] / max_forward_raw if max_forward_raw else 0.0,
        }
    return features
