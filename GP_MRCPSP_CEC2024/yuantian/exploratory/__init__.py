"""
Phase 0: the exploratory evolutionary-algorithm sweep that preceded and
motivated the three refined extensions (cp_propagation.py, hybrid_gp.py +
local_search.py, nr_terminals.py). See README.md in this directory for what
each strategy is, what it found, and why only "lexicase" was carried
forward.

Restored from yuantian/custom_ea.py, deleted in commit b595a2d5 when the
repo was restructured -- see README.md's "Restoration notes" section.

Self-contained: nothing here is imported by gphh_solver.py or by the three
kept extensions. Run via yuantian/experiments/exploratory_sweep_experiment.py.
"""
from yuantian.exploratory.adaptive_ops import adaptive_operator_gp
from yuantian.exploratory.diagnostic_graft import (
    decision_trace_illumination_gp,
    install_graft_terminals,
    modification_integrated_gp,
    trace_directed_gp,
)
from yuantian.exploratory.diverse_partner import diverse_partner_gp
from yuantian.exploratory.multi_sgs import multi_sgs_gp
from yuantian.exploratory.quality_diversity import map_elites_gp
from yuantian.exploratory.selection import lexicase_gp
from yuantian.exploratory.surrogate import surrogate_assisted_gp

# name -> driver. Matches yuantian/custom_ea.py's EA_REGISTRY exactly (9
# strategies; "standard" baseline is gp_algorithms.standard_gp, not
# repeated here). See README.md for why dmge_mega_gp and strategy_isolated_gp
# (mentioned in some accounts of this sweep) are not restored: no trace of
# either was found anywhere in git history.
EXPLORATORY_REGISTRY = {
    "mod_integrated": modification_integrated_gp,  # DMGE (flagship)
    "trace_directed": trace_directed_gp,  # TDRE
    "decision_trace": decision_trace_illumination_gp,
    "map_elites": map_elites_gp,
    "adaptive": adaptive_operator_gp,
    "surrogate": surrogate_assisted_gp,
    "diverse": diverse_partner_gp,
    "lexicase": lexicase_gp,
    "multi_sgs": multi_sgs_gp,
}

# Drivers that need yuantian.exploratory.diagnostic_graft.install_graft_terminals
# called on the GPHH instance's (pset, simulator) before running, or their
# graft operator is a structural no-op.
GRAFT_DEPENDENT_STRATEGIES = {"mod_integrated", "trace_directed"}
