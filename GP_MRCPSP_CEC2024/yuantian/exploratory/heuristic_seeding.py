"""
Heuristic seeding for the GPHH initial population.

Seeds part of generation 0 with hand-crafted priority rules (earliest
start, latest finish, minimum slack, shortest duration, and similar)
instead of purely random ramped half-and-half trees, gated by
ParametersGPHH.seeding_strategy. Its dedicated before/after comparison
(experiments/heuristic_seeding_experiment.py) was negative: seeding does
not beat random initialisation on mean fitness and increases variance.
gphh_solver.py imports seed_population from here for the
--seeding_strategy CLI flag.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Union

from deap import gp

from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_simulation import DecisionTypeEnum

# Each rule maps to a list of equivalent expressions, tried in order: the
# first one whose terminals are all present in the running pset is used.
# The non-first variants exist to cope with dynamic-CPM terminal names
# (e.g. "ES_d" instead of "ES" when `dynamic_CPM_feature=True`).
# All of these are monotonically-increasing-is-worse, i.e. they match the
# "smallest value scheduled/chosen first" convention GPHH always evaluates
# with (`priority_extre="min"` in `evaluate_heuristic`); rules that are
# naturally "largest first" (e.g. GRPW) are deliberately left out since
# negating a single terminal isn't expressible without adding a new
# primitive to the pset, which would change the search space for every
# individual, not just the seeds.
ACTIVITY_RULES: Dict[str, List[str]] = {
    "EST": ["ES", "ES_d"],  # earliest start time first
    "EFT": ["EF", "EF_d"],  # earliest finish time first
    "LFT": ["LF", "LF_d"],  # latest finish time first
    "MSLK": ["sub(LS, ES)", "sub(LS_d, ES_d)"],  # minimum slack first
}

MODE_RULES: Dict[str, List[str]] = {
    "SPT_MODE": ["task_duration"],  # shortest mode duration first
    "MIN_RES_MODE": ["min_RReq"],  # least resource-hungry mode first
}

# `SIMULTANEOUS` makes one decision per step that folds activity and mode
# together into a single tree, so its seeds also include composites of an
# activity rule and a mode rule.
INTEGRATED_COMPOSITE_RULES: Dict[str, List[str]] = {
    "MSLK+SPT": [
        "add(sub(LS, ES), task_duration)",
        "add(sub(LS_d, ES_d), task_duration)",
    ],
    "EST+MIN_RES": ["add(ES, min_RReq)", "add(ES_d, min_RReq)"],
}


def _first_valid_expr(variants: List[str], pset: gp.PrimitiveSet) -> Optional[str]:
    """Return the first expression in `variants` that parses against `pset`,
    or None if none of them do (e.g. the feature was disabled)."""
    for expr in variants:
        try:
            gp.PrimitiveTree.from_string(expr, pset)
        except Exception:
            continue
        return expr
    return None


def _build_trees(
    rules: Dict[str, List[str]], pset: gp.PrimitiveSet
) -> Dict[str, gp.PrimitiveTree]:
    trees = {}
    for name, variants in rules.items():
        expr = _first_valid_expr(variants, pset)
        if expr is not None:
            trees[name] = gp.PrimitiveTree.from_string(expr, pset)
    return trees


def build_heuristic_trees(
    decision_type: DecisionTypeEnum, pset
) -> Dict[str, Union[gp.PrimitiveTree, Dict[str, gp.PrimitiveTree]]]:
    """Return every textbook rule that can be expressed with the
    terminals/primitives currently in `pset`, keyed by rule name.

    For `SIMULTANEOUS`, each value is a single tree for the INTEGRATED
    pset. For the two-step decision types, each value is a
    `{TerminalTypeEnum.ACTIVITY.value: tree, TerminalTypeEnum.MODE.value:
    tree}` dict pairing every usable activity rule with every usable mode
    rule, matching the shape `MultiPrimitiveTree` expects.
    """
    if decision_type == DecisionTypeEnum.SIMULTANEOUS:
        integrated_pset = pset[TerminalTypeEnum.INTEGRATED.value]
        seeds: Dict[str, gp.PrimitiveTree] = dict(
            _build_trees(ACTIVITY_RULES, integrated_pset)
        )
        for name, variants in INTEGRATED_COMPOSITE_RULES.items():
            expr = _first_valid_expr(variants, integrated_pset)
            if expr is not None:
                seeds[name] = gp.PrimitiveTree.from_string(expr, integrated_pset)
        return seeds

    activity_pset = pset[TerminalTypeEnum.ACTIVITY.value]
    mode_pset = pset[TerminalTypeEnum.MODE.value]
    activity_trees = _build_trees(ACTIVITY_RULES, activity_pset)
    mode_trees = _build_trees(MODE_RULES, mode_pset)
    if not activity_trees or not mode_trees:
        return {}
    seeds = {}
    for a_name, a_tree in activity_trees.items():
        for m_name, m_tree in mode_trees.items():
            seeds[f"{a_name}+{m_name}"] = {
                TerminalTypeEnum.ACTIVITY.value: a_tree,
                TerminalTypeEnum.MODE.value: m_tree,
            }
    return seeds


def seed_population(
    toolbox,
    individual_class,
    pop_size: int,
    decision_type: DecisionTypeEnum,
    pset,
    n_mutated_clones: int = 0,
    mutate: Optional[Callable] = None,
    rng: random.Random = random,
) -> list:
    """Build a generation-0 population where a handful of individuals are
    hand-built textbook priority rules instead of random ramped
    half-and-half trees; the remainder of `pop_size` is filled with the
    baseline's own `toolbox.individual()`, so diversity for crossover and
    mutation in later generations is unaffected.

    `n_mutated_clones`: for each heuristic seed, also add this many mutated
    copies of it (via `mutate`, typically `toolbox.mutate`) so the seeded
    region of the search space is a neighborhood rather than a handful of
    isolated points. Pass 0 (default) for the plain "direct seeding"
    strategy.

    If pop_size is smaller than the number of available rules, a random
    subset of the rules is used and no random individuals are added. If no
    rule is expressible with the current `pset` (e.g. every feature was
    disabled), this falls back to the same all-random population the
    baseline would build.
    """
    heuristic_trees = build_heuristic_trees(decision_type, pset)
    seeds = []
    for tree in heuristic_trees.values():
        ind = individual_class(tree)
        seeds.append(ind)
        if n_mutated_clones and mutate is not None:
            for _ in range(n_mutated_clones):
                clone = toolbox.clone(ind)
                (clone,) = mutate(clone)
                seeds.append(clone)

    if len(seeds) > pop_size:
        seeds = rng.sample(seeds, pop_size)

    n_random = pop_size - len(seeds)
    return seeds + [toolbox.individual() for _ in range(n_random)]


def seed_then_run(
    driver: Callable,
    toolbox,
    individual_class=None,
    pop_size: int = None,
    decision_type=None,
    pset=None,
    n_mutated_clones: int = 0,
    mutate=None,
    rng=None,
    **driver_kwargs,
):
    """Build a heuristic-seeded gen-0 population (`seed_population` above)
    and run it through ``driver`` (any of this package's strategies, or
    gp_algorithms.standard_gp) instead of driver's usual
    toolbox.population(n=...) random init.

    ``driver_kwargs`` must include every other positional/keyword argument
    ``driver`` needs except ``population`` (e.g. cxpb, mutpb, n_elite, ngen,
    training_data_provider, validation_data_provider, stats, halloffame,
    pop_archive).
    """
    from deap import creator

    population = seed_population(
        toolbox=toolbox,
        individual_class=individual_class or creator.Individual,
        pop_size=pop_size,
        decision_type=decision_type,
        pset=pset,
        n_mutated_clones=n_mutated_clones,
        mutate=mutate,
        rng=rng or random,
    )
    return driver(population, toolbox, **driver_kwargs)
