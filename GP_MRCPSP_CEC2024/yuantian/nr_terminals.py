"""
Non-renewable (NR) resource features for the GP terminal sets:
NR_STOCK_RATIO and NR_BUDGET_PRESSURE (activity/integrated trees) and
NR_MODE_DEMAND_RATIO (mode tree). Definitions match the thesis text.

Precondition: instances must be loaded with keep_non_renewable=True in
gphh_solver.read_instances(); on instances without NR resources every
terminal returns a neutral constant. Static per-instance quantities
(initial stock, per-activity cheapest-mode demand) are precomputed here
and cached on the problem object.
"""
from typing import Any, Dict

from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP


def compute_nr_static_features(problem: ANY_RCPSP) -> Dict[str, Any]:
    """Precompute the per-instance, schedule-independent half of the NR
    signal: initial stock per NR resource type, and each task's minimum NR
    demand across its own modes, per resource type.

    Returns `{"initial_stock": {}, "min_mode_demand": {}}` (empty dicts) if
    `problem.non_renewable_resources_list` is empty, e.g. for instances
    loaded through the renewable-only `read_instances()` path -- harmless,
    since `Simulator.feature_nr_*` checks that list first and falls back to
    a neutral default before ever consulting this cache.
    """
    non_renewable = problem.non_renewable_resources_list
    initial_stock: Dict[str, float] = {
        res: problem.resources[res] for res in non_renewable
    }
    min_mode_demand: Dict[Any, Dict[str, float]] = {}
    for task in problem.tasks_list:
        modes = problem.mode_details[task]
        min_mode_demand[task] = {
            res: min(modes[m].get(res, 0) for m in modes) for res in non_renewable
        }
    return {"initial_stock": initial_stock, "min_mode_demand": min_mode_demand}
