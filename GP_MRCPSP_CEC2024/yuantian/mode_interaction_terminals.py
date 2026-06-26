"""
Mode-interaction terminals for the mode (and, for one terminal, activity)
GP trees.

This is meant to be the third piece of a pattern across this repo's
terminal extensions:

- nr_terminals.py: how much resource is left overall -- a budget signal.
- rccp_terminals.py: who's fighting over the bottleneck right now -- a
  contention signal.
- this module: if I pick this mode for this activity, how much does that
  squeeze the other activities that share a resource window with it -- a
  consequence signal. None of the timing or current-state terminals from
  the other extensions can express this, since it needs reasoning about
  more than one activity's options at once.

Worth being upfront about: this is the least literature-backed extension
in the repo. NR/RCCP terminals are basically standard resource-leveling
ideas from RCPSP heuristics research; this "constraint propagation as a GP
terminal" idea doesn't have much direct precedent I could find to check
the implementation against. It's a reasonable idea, not an established
technique I'm porting in, and if mode_interaction_experiment.py shows
nothing, that's still a useful, honest result (same as cp_propagation.py's
null result elsewhere in this repo).

Eligibility/neighbors: Simulator.eligibles (added for rccp_terminals.py,
see that module) is the "who's about to be decided" set this looks ahead
over. _mode_interaction_neighbors(candidate_resources) narrows that down
to activities (other than the one currently being scored) that have at
least one mode demanding something from candidate_resources.

Renewable resources only. Two activities sharing a non-renewable resource
aren't really competing the way two overlapping-in-time activities on a
renewable resource are -- NR budget is a whole-project pool, so it doesn't
matter much if A uses it now and B uses it later, they're not fighting
over the same window. So candidate_resources always comes from
renewable_resources_list.

The terminals:

- MI_CONSTRAINT_TIGHTENING (mode tree, the main one): for the candidate
  mode, look at every neighbor and work out what fraction of that
  neighbor's own modes would stop being feasible once the candidate's
  demand is committed. Averaged across neighbors, not maxed -- a max would
  let one badly-affected neighbor dominate even if everyone else is fine,
  the mean reflects the typical effect better. (Max instead of mean would
  be a reasonable thing to try later, just didn't implement it.)
- MI_RECIPROCAL_SCARCITY (mode tree, cheaper version): instead of
  recomputing feasibility for every neighbor mode, just checks whether the
  neighbors' combined cheapest-mode demand on a resource already exceeds
  what's available. About 2.6x cheaper per call (see below) -- kept as a
  fallback in case the full terminal gets too expensive at larger scale.
- MI_ACTIVITY_PRESSURE (activity/integrated tree): same idea as
  MI_CONSTRAINT_TIGHTENING but averaged over the candidate activity's own
  modes, since mode isn't picked yet at the activity-tree stage. Just a
  small extra loop over that activity's 2-4 modes, not expensive.

Cost: I actually measured this instead of guessing. Per-call timing on
J501_1.mm (SerialSimulator, ~14 eligible activities per decision on
average): feature_early_start (an existing terminal, basically a dict
lookup) is about 0.10us. MI_RECIPROCAL_SCARCITY is 10.69us,
MI_CONSTRAINT_TIGHTENING is 27.76us, MI_ACTIVITY_PRESSURE is 87.19us
(roughly 3x the constraint-tightening number, which tracks since it
averages over ~3 modes).

Running full population evaluations with realistic random trees (not
trees that just force the terminal once) showed about a 3x slowdown in
the evolution loop with --mode_interaction_terminals on. That's real, not
negligible, but still keeps a full experiment run in the same few-minutes
ballpark as the other extensions at this scale, so I added two cheap
mitigations rather than anything more drastic:

1. _current_resource_avail() is cached per decision (keyed on
   Simulator._decision_counter) instead of recomputed every single
   terminal call. This doesn't change any values, it's just avoiding
   redundant work, and it speeds up rccp_terminals.py's terminals too
   since they use the same method.
2. _mode_interaction_neighbors caps how many neighbors it actually looks
   at (Simulator._MI_MAX_NEIGHBORS, default 5), picking the
   biggest-demand ones. This one IS an approximation when there are more
   than 5 eligible neighbors sharing a resource (which is most of the
   time on this instance, average eligible set is ~14) -- it can miss
   tightening effects on smaller neighbors past the cap. Worth revisiting
   if a bigger run shows it actually matters.

Wiring: off by default, --mode_interaction_terminals to turn it on. Plays
fine with --cp_propagation/--nr_terminals/--rccp_terminals (different
terminal names), works under serial and parallel SGS since both populate
eligibles and both have a _current_resource_avail override.
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
