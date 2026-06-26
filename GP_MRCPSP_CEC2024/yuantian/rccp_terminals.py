"""
Resource-Constrained Critical Path (RCCP) terminals for the activity/mode
GP trees.

cp_propagation.py already gives the GP some "criticality" signal
(CP_FORWARD/CP_BACKWARD/CP_SLACK_SCORE/CP_PROB), but that experiment came
back as basically a null result (see readme.md, extension #1). Turns out
those terminals are just linear combinations of ES/EF/LS/LF that the GP
already had -- CP_BACKWARD is literally the same number as LS. Makes sense
in hindsight: CPM dates only look at the precedence graph, resources never
enter the picture. So there was nothing new there for the GP to use.

This module tries the obvious follow-up: what if the new terminal actually
depends on live resource state instead of precedence alone? Two activities
with identical ES/EF/LS/LF can be in completely different situations
resource-wise depending on what else is competing for capacity right now,
and that's just not something you can reconstruct from CPM numbers no
matter how you combine them.

Unlike cp_propagation.py / nr_terminals.py, there's no static half to
precompute here. "Which resource is the bottleneck right now" and "how
much is left" only make sense once you're partway through building a
schedule -- there's nothing about the raw instance to cache ahead of time.
So everything actually lives in rcpsp_simulation.py
(Simulator._current_resource_avail, Simulator._rccp_bottleneck_resource,
Simulator.feature_rccp_*); this file is really just the FeatureEnum/pset
wiring plus this explanation.

Representing "right now": SerialSimulator and ParallelSimulator track
state differently, so each has its own _current_resource_avail override.
ParallelSimulator already has a single clock (self.current_time) and a
scalar resource_avail dict kept in sync with it, so that's just the
answer directly. SerialSimulator doesn't have one global "now" -- instead
I use the earliest start time among the currently-eligible activities as a
stand-in "now" and read remaining capacity at that point from
resource_avail_in_time.

Also added self.eligibles to both buildSolution implementations (it was
declared in __init__ but never actually filled in before). Purely
additive, nothing else reads it, just needed it so RCCP_PRESSURE_TREND can
see what's currently eligible without recomputing it.

The terminals:

- RCCP_BOTTLENECK_UTIL (activity/integrated): how full the most-contended
  renewable resource is right now. Not activity-specific, same idea as
  NR_STOCK_RATIO.
- RCCP_CANDIDATE_CONTENTION (mode only): how much of that same bottleneck
  resource the candidate mode would use, relative to what's left. Probably
  the most useful one of the bunch -- a plain CP terminal has no way to
  express "pick the mode that's lighter on the thing that's actually
  scarce right now." Only looks at the bottleneck resource, not all of
  them, since that's the one actually limiting progress.
- RCCP_SLACK (activity): CP_SLACK_SCORE discounted by how busy the
  bottleneck is (cp_slack_score * (1 - bottleneck_utilization)). This is
  a rough approximation, not real resource-leveling -- doing that properly
  would mean solving a sub-problem over the rest of the schedule. It just
  shrinks precedence slack toward 0 as contention goes up, which is the
  right direction even if not exact.
- RCCP_PRESSURE_TREND (activity/integrated, added last): forward-looking
  version of RCCP_BOTTLENECK_UTIL. Sums the bottleneck-resource demand of
  everything currently eligible (not the whole rest of the project, just
  the activities about to be decided), divided by what's left. It's an
  optimistic estimate since it ignores that those activities are also
  competing with each other, so it can underestimate how tight things
  will get.
- RCCP_RESOURCE_CONCENTRATION (mode only, small bundled addition): the
  other terminals look at resource types more or less individually, but
  none of them distinguish "8 units from one resource" from "2 units each
  from four resources" -- the first is way more exposed if that one
  resource becomes scarce. This is 1 minus the normalized Shannon entropy
  of the candidate's demand vector across renewable resources (so it's on
  the same scale regardless of how many resource types the instance has).
  Higher = more concentrated = more exposed, matching the sign convention
  everywhere else here except RCCP_SLACK (which just inherits
  CP_SLACK_SCORE's sign). Cheap, no look-ahead. k=1 (instance only has one
  renewable resource type) returns 1.0 to avoid a 0/log(1) blowup; a mode
  that draws nothing at all returns 0.0.

Wiring: off by default, turned on with rccp_terminals_feature=True /
--rccp_terminals. Stacks fine with --cp_propagation and --nr_terminals
(different terminal names, no collisions), works under both serial and
parallel SGS, and doesn't need the keep_non_renewable instance loading
nr_terminals.py needs since this is all about renewable resources.
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
