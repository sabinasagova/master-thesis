"""
Non-renewable (NR) resource-aware features for MRCPSP activity/mode GP trees.

The baseline's terminal set (ES/EF/LS/LF/GRD/min_RReq and similar) gives the
GP no direct signal about non-renewable resource budget pressure: NR
resources are consumed once per activity and never replenished, and they
become the binding constraint specifically under tight NR settings (e.g.
MMLIB+'s NR50-style configurations). Mode choice is exactly what trades NR
consumption against duration/renewable-resource use, so a GP rule with no
visibility into "how much NR budget is left" or "how exposed is this mode's
NR demand" has to discover NR-awareness indirectly (if at all) from
terminals that were never designed to carry that signal.

IMPORTANT PRECONDITION: `gphh_solver.read_instances()` calls
`to_renewable_only_rcpsp_model`, which permanently deletes every
non-renewable resource from the model before the GP ever sees it -- the
baseline and the other two extensions (cp_propagation, hybrid_gp/
local_search) all run on instances with zero NR resources. These terminals
are therefore only meaningful for instances loaded with
`read_instances(..., keep_non_renewable=True)`, which skips that
conversion. Enabling `--nr_terminals` without that loader leaves
`non_renewable_resources_list` empty and every terminal below degrades to
its documented neutral default (no crash, but no signal either).

This module mirrors cp_propagation.py's structure: it precomputes, once per
problem instance, the *static* half of the NR signal (initial stock per NR
resource type, and each task's minimum NR demand across its own modes, per
resource type -- the same quantity NR_BUDGET_PRESSURE needs at every
decision point, so it's computed once here instead of on every call) and
caches it on the problem instance (`problem.nr_static_features`) so it's
never recomputed during a GP run.

Unlike CP propagation's features, the *live* half of the NR signal (how
much stock actually remains, given the schedule built so far) is inherently
schedule-state-dependent and can't be precomputed -- exactly like the
dynamic CPM terminals (ES_d/EF_d/LS_d/LF_d). That live lookup and the
terminal functions themselves therefore live in `rcpsp_simulation.py`
(`Simulator._nr_remaining_stock`, `Simulator.feature_nr_*`), following the
same split already used for the dynamic CPM terminals: this module supplies
the static half, the simulator supplies the dynamic half.

Terminals (added to `FeatureEnum`, registered via `ParametersGPHH` when
`nr_terminals_feature=True` / CLI `--nr_terminals`):

  - NR_STOCK_RATIO (activity + integrated tree): min, across NR resource
    types, of (remaining stock / initial stock). Backward-looking: how much
    of the project's NR budget is left right now. One aggregated scalar
    rather than one terminal per resource type, since MMLIB/MMLIB+
    instances don't all have the same number of NR resource types and a GP
    tree must stay well-defined across every instance it's evaluated on
    (train/validation/test) -- a per-type terminal would be meaningless on
    an instance with fewer NR types than the tree references.
  - NR_MODE_DEMAND_RATIO (mode tree only): max, across NR resource types,
    of (this candidate mode's NR demand / remaining stock). The most
    directly useful NR terminal for mode choice, since mode is what trades
    off NR consumption against duration. Demand-over-*remaining*-stock
    (not initial stock) was chosen because "can I still afford this mode
    right now" is the decision-relevant question -- it matches how the
    existing eligible-mode filter in `Simulator.buildSolution` already
    gates modes on remaining (not initial) resource availability.
  - NR_BUDGET_PRESSURE (activity + integrated tree): the most novel
    terminal. A forward-looking feasibility-risk proxy: min, across NR
    resource types, of (remaining stock - sum of every not-yet-scheduled
    activity's minimum-demand mode for that resource), normalized by
    initial stock. This is the simple-sum proxy the spec sanctions in place
    of a tight bound: a tight bound on NR feasibility risk would need to
    consider which mode combination is jointly optimal across all
    remaining activities, which is itself an optimization problem. Summing
    each remaining task's own best case independently is an
    *optimistic* approximation -- it can overstate how much slack truly
    remains when several tasks would compete for the same scarce NR
    resource, since in reality not every task can simultaneously get its
    own minimum-demand mode if that mode is also resource-hungry elsewhere.
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
