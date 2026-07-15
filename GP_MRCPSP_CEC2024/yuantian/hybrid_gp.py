"""
Hybrid evolutionary loop: epsilon-lexicase selection + critical-path local
search applied to elites, layered on top of the unmodified GPHH
representation, mate/mutate operators, and evaluation function.

This is a parallel loop to `yuantian.gp_algorithms.standard_gp`, not a
replacement for it: the baseline condition keeps using `standard_gp`
(tournament selection, no local search) completely untouched. The two
conditions therefore only differ in (a) the selection operator and (b) the
elite local-search step, as required for a fair comparison.

A gap-aware early-stopping variant lives in
exploratory/gap_aware_stopping.py (null result in both validations).
"""
import random
from functools import partial
from typing import Optional

import numpy as np
from deap import tools

from yuantian.gp_algorithms import evaluate_population, load_elites, varOr
from yuantian.local_search import RefinementStrategyEnum, apply_local_search_to_elite
from yuantian.rcpsp_dataset import DatasetProvider


def epsilon_lexicase_selection(individuals, k, rng=random):
    """Epsilon-lexicase selection for minimization objectives.

    Step 1: shuffle the training cases.
    Step 2: iteratively filter the candidate pool down to individuals within
    an adaptive epsilon (`0.01 * std(case_scores)`) of the best score on each
    case, case by case, until either one candidate remains or all cases have
    been used; then pick uniformly at random among the survivors.
    Repeats `k` times (with replacement) and returns the k selected
    individuals, as required by a DEAP `toolbox.select` operator.
    """
    n_cases = len(individuals[0].case_fitness)
    selected = []
    for _ in range(k):
        candidates = list(individuals)
        case_order = list(range(n_cases))
        rng.shuffle(case_order)
        for case in case_order:
            if len(candidates) <= 1:
                break
            case_scores = [ind.case_fitness[case] for ind in candidates]
            best = min(case_scores)
            epsilon = 0.01 * float(np.std(case_scores))
            candidates = [
                ind
                for ind, score in zip(candidates, case_scores)
                if score <= best + epsilon
            ]
        selected.append(rng.choice(candidates))
    return selected


def _local_search_elites(
    population,
    elite_fraction,
    toolbox,
    training_data,
    decision_type,
    simulator,
    pset,
    local_search_iters,
    rng,
    refinement_strategy=RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP,
):
    """Part 2, Steps 1+4: pick the top `elite_fraction` of the population by
    mean fitness, and overwrite each one's fitness/case_fitness with the
    refined values (the GP tree itself is left untouched).

    `refinement_strategy=RefinementStrategyEnum.BASELINE` means "no
    refinement": this is the single place that decision is made, so every
    caller of `lexicase_memetic_gp` gets a true no-op (elites keep whatever
    fitness they were already evaluated with) rather than each caller having
    to special-case it.
    """
    if refinement_strategy == RefinementStrategyEnum.BASELINE:
        return []
    n_elite_ls = max(1, int(round(elite_fraction * len(population))))
    elites = sorted(population, key=lambda ind: ind.fitness.values[0])[:n_elite_ls]
    for ind in elites:
        apply_local_search_to_elite(
            ind,
            training_data,
            toolbox.compile,
            pset,
            decision_type,
            simulator,
            max_iters=local_search_iters,
            rng=rng,
            strategy=refinement_strategy,
        )
    return elites


def _accumulate_move_stats(elites, move_stats: Optional[dict]):
    """Sum each refined elite's `local_search_moves` (and any
    `critical_path_construct_failed` flag) into the caller-owned `move_stats`
    accumulator, so the GP-loop caller can report, across the whole run, how
    many proposed moves were attempted vs. accepted (hill-climbing
    strategies) or how many one-shot constructions failed
    (CRITICAL_PATH_ONLY) -- without `lexicase_memetic_gp` itself needing to
    know which strategy is in use. No-op if `move_stats` is None (the
    default for callers that don't care, e.g. the pre-existing experiments)."""
    if move_stats is None:
        return
    for ind in elites:
        moves = getattr(ind, "local_search_moves", None)
        if moves:
            move_stats["attempted"] = move_stats.get("attempted", 0) + moves["attempted"]
            move_stats["accepted"] = move_stats.get("accepted", 0) + moves["accepted"]
        if getattr(ind, "critical_path_construct_failed", False):
            move_stats["construct_failures"] = move_stats.get("construct_failures", 0) + 1


def lexicase_memetic_gp(
    population,
    toolbox,
    cxpb: float,
    mutpb: float,
    n_elite: int,
    ngen: int,
    training_data_provider: DatasetProvider,
    validation_data_provider,
    decision_type,
    simulator,
    pset,
    elite_fraction: float = 0.08,
    local_search_iters: int = 10,
    refinement_strategy=RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP,
    stats: tools.Statistics = None,
    halloffame: tools.HallOfFame = None,
    pop_archive: list = None,
    move_stats: Optional[dict] = None,
    rng=random,
    verbose=__debug__,
):
    """Same generational loop as `gp_algorithms.standard_gp`, with the elite
    local-search step inserted right after each fitness evaluation. Selection
    itself is whatever `toolbox.select` is registered to (epsilon-lexicase
    for the proposed condition).
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    training_data = training_data_provider.next()
    evaluate = partial(toolbox.evaluate, domains=training_data)
    evaluate_population(toolbox, evaluate, invalid_ind)

    elites = _local_search_elites(
        population,
        elite_fraction,
        toolbox,
        training_data,
        decision_type,
        simulator,
        pset,
        local_search_iters,
        rng,
        refinement_strategy,
    )
    _accumulate_move_stats(elites, move_stats)

    pop_archive.append([toolbox.clone(ind) for ind in population])

    if halloffame is not None:
        halloffame.update(population)
    best_ind_record = {
        "fitness": halloffame[0].fitness.values[0],
        "tree": str(halloffame[0]),
    }
    if validation_data_provider:
        validation_set = validation_data_provider.next()
        validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
        best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

    record = stats.compile(population) if stats else {}
    record["generation_best"] = best_ind_record
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population) - n_elite)
        offspring = varOr(offspring, toolbox, cxpb, mutpb)
        offspring += load_elites(population, n_elite)

        training_data = training_data_provider.next()
        evaluate = partial(toolbox.evaluate, domains=training_data)
        evaluate_population(toolbox, evaluate, offspring)

        elites = _local_search_elites(
            offspring,
            elite_fraction,
            toolbox,
            training_data,
            decision_type,
            simulator,
            pset,
            local_search_iters,
            rng,
            refinement_strategy,
        )
        _accumulate_move_stats(elites, move_stats)

        if halloffame is not None:
            halloffame.update(offspring)
        best_ind_record = {
            "fitness": halloffame[0].fitness.values[0],
            "tree": str(halloffame[0]),
        }
        if validation_data_provider:
            validation_set = validation_data_provider.next()
            validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
            best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

        population[:] = offspring
        pop_archive.append([toolbox.clone(ind) for ind in population])

        record = stats.compile(population) if stats else {}
        record["generation_best"] = best_ind_record
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
