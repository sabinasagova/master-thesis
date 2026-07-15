"""
Epsilon-lexicase selection + ERCs + mini-batch rotation (exploratory
strategy "lexicase"; provenance in README.md).

This strategy motivated yuantian/hybrid_gp.py's epsilon-lexicase +
local-search extension; this module only changes selection/evaluation (no
local search on elites), and its lexicase mechanism is the ancestor of the one in
hybrid_gp.epsilon_lexicase_selection.

Three independent improvements from the GP literature, combined into one driver:

(1) Epsilon-lexicase selection (La Cava et al., GECCO 2016). The baseline
    uses tournament selection on mean deviation -- collapsing per-instance
    scores into a single number throws away information and causes
    premature convergence onto generalists. Lexicase treats each training
    instance as a separate selection "case": for each selection event,
    cases are shuffled and candidates filtered to within epsilon of best on
    each successive case. Adaptive epsilon = epsilon_factor * std(scores)
    keeps specialists alive for free.

(2) Ephemeral random constants (ERCs). The primitive set contains only
    features and arithmetic -- no numeric constants. Rules cannot express
    things like 2*GRPW - Duration without building constants out of feature
    ratios, wasting tree depth. Adding numeric constants fills this
    expressiveness gap.

(3) Mini-batch rotation. Static training sets cause overfitting to the
    fixed sample. Evaluating each generation on a different random subset
    of size batch_fraction * |train| acts as regularisation and cuts
    per-generation cost by (1 - batch_fraction).

Heuristic seeding (yuantian/exploratory/heuristic_seeding.py) is a separate,
orthogonal strategy: it changes gen-0 population construction, not
selection/evaluation, so it composes with this driver simply by seeding
``population`` before calling ``lexicase_gp`` -- see
exploratory_sweep_experiment.py's "lexicase_seeded" condition.

Cite: La Cava et al. (2016) for (1), Koza (1992) for (2), Hildebrandt &
Branke (2015) for (3) in the context of GPHH.
"""
import random
from typing import Optional

import numpy as np
from deap import tools

from yuantian.exploratory.shared import (
    DataProvider,
    Individual,
    Toolbox,
    _evaluate_cases,
    _new_logbook,
    _record,
    _Timer,
)
from yuantian.gp_algorithms import load_elites

# Numeric constants (callable-wrapped for thunk-based primitives in gphh_solver.py).
# gphh_solver's add/sub/mul/div all call left() and right(), so every terminal must
# be a zero-argument callable. Not true ephemeral-random constants (DEAP's
# addEphemeralConstant produces raw floats, breaking that thunk contract here) --
# a fixed grid spanning the useful range instead.
_NUMERIC_CONSTANTS = {
    "K_neg1": -1.0,
    "K_neg05": -0.5,
    "K_025": 0.25,
    "K_05": 0.5,
    "K_075": 0.75,
    "K_1": 1.0,
    "K_2": 2.0,
}


def _add_ercs_to_psets(pset_dict) -> None:
    """Add lambda-wrapped numeric constants to all psets (idempotent)."""
    psets = pset_dict.values() if isinstance(pset_dict, dict) else [pset_dict]
    for pset in psets:
        for name, val in _NUMERIC_CONSTANTS.items():
            if name not in pset.mapping:
                pset.addTerminal((lambda v=val: lambda: v)(), name)


def _epsilon_lexicase_select(population: list, k: int, epsilon_factor: float = 0.1) -> list:
    """Epsilon-lexicase selection (La Cava et al. 2016).

    For each of k selections, randomly orders training cases and filters the
    candidate pool to within epsilon = epsilon_factor * std(case_scores) of
    the best performer on each successive case. Falls back to tournament(7)
    when ``ind.cases`` is not set.
    """
    selected = []
    fallback_k = min(7, len(population))
    for _ in range(k):
        pool = list(population)
        if not hasattr(pool[0], "cases") or not pool[0].cases:
            aspirants = random.sample(pool, fallback_k)
            selected.append(min(aspirants, key=lambda i: i.fitness.values[0]))
            continue
        n_cases = len(pool[0].cases)
        case_order = list(range(n_cases))
        random.shuffle(case_order)
        for c in case_order:
            if len(pool) <= 1:
                break
            scores = np.array([ind.cases[c] for ind in pool], dtype=float)
            valid = ~np.isnan(scores)
            if not valid.any():
                continue
            best = scores[valid].min()
            eps = float(scores[valid].std()) * epsilon_factor if valid.sum() > 1 else 0.0
            pool = [
                ind
                for ind, sc, ok in zip(pool, scores, valid)
                if ok and sc <= best + max(eps, 1e-9)
            ]
            if not pool:
                pool = list(population)  # safety: restart from full pool
                break
        selected.append(random.choice(pool))
    return selected


def lexicase_gp(
    population: list,
    toolbox: Toolbox,
    cxpb: float,
    mutpb: float,
    n_elite: int,
    ngen: int,
    training_data_provider: DataProvider,
    validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None,
    halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None,
    verbose: bool = __debug__,
    epsilon_factor: float = 0.1,
    batch_fraction: float = 0.5,
    use_ercs: bool = True,
) -> tuple:
    """GP with epsilon-lexicase selection, ERCs, and mini-batch rotation.

    Combines three independent improvements (see module docstring) into a
    single drop-in driver, directly comparable to gp_algorithms.standard_gp.
    """
    kw = toolbox.evaluate.keywords
    pset = kw["pset"]
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    if use_ercs:
        _add_ercs_to_psets(pset)

    # Obtain the full training set once; sample subsets each generation.
    full_training = training_data_provider.next()
    batch_size = max(1, int(len(full_training) * batch_fraction))

    def get_batch():
        return (
            full_training
            if batch_size >= len(full_training)
            else random.sample(full_training, batch_size)
        )

    import time

    t0 = time.time()
    training = get_batch()
    _evaluate_cases(population, training, toolbox)
    _record(
        0,
        len(population),
        population,
        toolbox,
        halloffame,
        stats,
        logbook,
        validation_data_provider,
        pop_archive,
        timer,
        t0,
        extra=f"  batch={len(training)}/{len(full_training)}",
    )

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = _epsilon_lexicase_select(population, 2 * pop_size, epsilon_factor)
        offspring, idx, target = [], 0, pop_size - n_elite
        while len(offspring) < target and idx + 1 < len(selected):
            r = random.random()
            if r < cxpb:
                c1, c2 = toolbox.mate(
                    toolbox.clone(selected[idx]), toolbox.clone(selected[idx + 1])
                )
                offspring.append(c1)
                if len(offspring) < target:
                    offspring.append(c2)
                idx += 2
            elif r < cxpb + mutpb:
                (c,) = toolbox.mutate(toolbox.clone(selected[idx]))
                offspring.append(c)
                idx += 1
            else:
                offspring.append(toolbox.clone(selected[idx]))
                idx += 1

        next_pop = offspring[:target] + load_elites(population, n_elite)
        training = get_batch()
        _evaluate_cases(next_pop, training, toolbox)
        population[:] = next_pop
        _record(
            gen,
            len(next_pop),
            population,
            toolbox,
            halloffame,
            stats,
            logbook,
            validation_data_provider,
            pop_archive,
            timer,
            t0,
            extra=f"  batch={len(training)}/{len(full_training)}",
        )

    return population, logbook
