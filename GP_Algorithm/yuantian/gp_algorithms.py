"""Genetic Programming Algorithms.These algorithms are built upon DEAP's GP module."""

from deap import tools, base, gp
import random
import time
from typing import Union, Iterable
import sys
from yuantian.rcpsp_dataset import DatasetProvider
from functools import partial


def pick_node(
    individual: gp.PrimitiveTree,
    nonterminal_prob: float = 0.5,
    terminal_prob: float = 0.5,
    root_prob: float = 0,
) -> int:
    """
    * KozaNodeSelector is a GPNodeSelector which picks nodes in trees a-la Koza I,
    * with the addition of having a probability of always picking the root.
    * The method divides the range 0.0...1.0 into four probability areas:

    <ul>
    <li>One area specifies that the selector must pick a terminal.
    <li>Another area specifies that the selector must pick a nonterminal (if there is one, else a terminal).
    <li>The third area specifies that the selector pick the root node.
    <li>The fourth area specifies that the selector pick any random node.
    </ul>

    * <p>The KozaNodeSelector chooses by probability between these four situations.
    * Then, based on the situation it has picked, it selects either a random
    * terminal, nonterminal, root, or arbitrary node from the tree and returns it.

    This code is modified from DEAP's gp module.

    """
    rnd: float = random.random()
    terminals: list[int] = []
    non_terminals: list[int] = []
    for idx, node in enumerate(individual):
        if isinstance(node, gp.Primitive):
            non_terminals.append(idx)
        elif isinstance(node, gp.Terminal):
            terminals.append(idx)
    if rnd > nonterminal_prob + terminal_prob + root_prob:  # pick anyone
        return random.randrange(len(individual))
    elif rnd > nonterminal_prob + terminal_prob:  # pick the root
        return 0
    elif rnd > nonterminal_prob:  # pick terminals
        return random.choice(terminals)
    else:  # pick non-terminals if you can
        if non_terminals:
            return random.choice(non_terminals)
        else:
            return random.choice(terminals)


CP_TERMINAL_NAMES = {"Is_On_Critical_Path", "Slack", "Dynamic_Slack"}


def _cp_protected_indices(individual: gp.PrimitiveTree) -> set:
    """Return the set of indices whose subtree contains at least one critical-path terminal.

    Any node whose subtree includes IS_ON_CRITICAL_PATH, Slack, or Dynamic_Slack is
    considered 'protected' — mutating it would risk destroying a useful CP-aware
    subexpression that evolution has already discovered.
    """
    protected = set()
    for i in range(len(individual)):
        sl = individual.searchSubtree(i)
        for node in individual[sl]:
            if isinstance(node, gp.Terminal) and node.name in CP_TERMINAL_NAMES:
                protected.add(i)
                break
    return protected


def mutCriticalPathPreserving(
    individual: gp.PrimitiveTree,
    expr,
    pset,
    nonterminal_prob: float = 0.5,
    terminal_prob: float = 0.5,
    root_prob: float = 0,
):
    """Subtree mutation that avoids replacing any subtree that already contains a
    critical-path terminal (IS_ON_CRITICAL_PATH, Slack, Dynamic_Slack).

    This preserves CP-aware subexpressions that evolution has discovered, while
    still allowing the rest of the tree to be modified freely.  If the entire tree
    is CP-related (e.g. if_then_else(IS_ON_CRITICAL_PATH, ...) at the root), falls
    back to ordinary biased mutation so the operator never gets stuck.
    """
    protected = _cp_protected_indices(individual)
    candidates = [i for i in range(len(individual)) if i not in protected]

    if not candidates:
        # whole tree is CP-related — fall back to regular biased mutation
        return mutBiased(individual, expr, pset, nonterminal_prob, terminal_prob, root_prob)

    index = random.choice(candidates)
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return (individual,)


def mutBiased(
    individual,
    expr,
    pset,
    nonterminal_prob: float = 0.5,
    terminal_prob: float = 0.5,
    root_prob: float = 0,
):
    index = pick_node(
        individual=individual,
        nonterminal_prob=nonterminal_prob,
        terminal_prob=terminal_prob,
        root_prob=root_prob,
    )
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return (individual,)


def varOr(population, toolbox, cxpb: float, mutpb: float):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover, mutation and reproduction probabilities must be smaller "
        "or equal to 1.0."
    )

    offspring = [toolbox.clone(ind) for ind in population]
    max_idx = len(population) - 1
    cur: int = 0
    while cur != max_idx:
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            if cur == max_idx - 1:
                continue
            offspring[cur], offspring[cur + 1] = toolbox.mate(
                offspring[cur], offspring[cur + 1]
            )
            del offspring[cur].fitness.values, offspring[cur + 1].fitness.values
            cur += 2
        elif op_choice < cxpb + mutpb:  # Apply mutation
            (offspring[cur],) = toolbox.mutate(offspring[cur])
            del offspring[cur].fitness.values
            cur += 1
        else:  # Apply reproduction
            cur += 1

    return offspring


def load_elites(population, n_elites: int) -> list:
    """
    Return the top n_elites individuals from the population.
    Elitism strategy to retain the best individuals. Quite useful in GP.
    """
    assert (
        len(population) > n_elites
    ), f"The number of elites for popluation ({n_elites}) exceeds the actual size of the population({len(population)})."
    sorted_population = sorted(
        population, key=lambda ind: ind.fitness.values[0], reverse=False
    )
    return sorted_population[0:n_elites]


def standard_gp(
    population: Iterable,
    toolbox: base.Toolbox,
    cxpb: float,
    mutpb: float,
    n_elite: int,
    ngen: int,
    training_data_provider: DatasetProvider,
    validation_data_provider: DatasetProvider,
    stats: tools.Statistics = None,
    halloffame: tools.HallOfFame = None,
    pop_archive: list = None,
    verbose=__debug__,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    run_start = time.time()
    gen_times = []

    def _log(gen: int, best_fit: float, elapsed_gen: float):
        gen_times.append(elapsed_gen)
        avg_gen = sum(gen_times) / len(gen_times)
        remaining = avg_gen * (ngen - gen)
        elapsed_total = time.time() - run_start
        print(
            f"  gen {gen:>3}/{ngen}  best={best_fit:>8.4f}  "
            f"gen_time={elapsed_gen:>5.1f}s  "
            f"elapsed={elapsed_total:>6.0f}s  "
            f"ETA={remaining:>6.0f}s",
            flush=True,
        )

    # Gen 0: evaluate initial population
    t0 = time.time()
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    training_data = training_data_provider.next()
    evaluate = partial(toolbox.evaluate, domains=training_data)
    fitnesses = toolbox.map(evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop_archive.append([toolbox.clone(ind) for ind in population])

    if halloffame is not None:
        halloffame.update(population)
    best_ind_record = {}
    best_ind_record["fitness"] = halloffame[0].fitness.values[0]
    best_ind_record["tree"] = str(halloffame[0])

    if validation_data_provider:
        validation_set = validation_data_provider.next()
        validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
        best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

    record = stats.compile(population) if stats else {}
    record["generation_best"] = best_ind_record
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    _log(0, best_ind_record["fitness"], time.time() - t0)

    # Generational loop
    for gen in range(1, ngen + 1):
        t0 = time.time()

        offspring = toolbox.select(population, len(population) - n_elite)
        offspring = varOr(offspring, toolbox, cxpb, mutpb)
        offspring += load_elites(population, n_elite)

        training_data = training_data_provider.next()
        evaluate = partial(toolbox.evaluate, domains=training_data)
        fitnesses = toolbox.map(evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)
        best_ind_record = {}
        best_ind_record["fitness"] = halloffame[0].fitness.values[0]
        best_ind_record["tree"] = str(halloffame[0])

        if validation_data_provider:
            validation_set = validation_data_provider.next()
            validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
            best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

        population[:] = offspring
        pop_archive.append([toolbox.clone(ind) for ind in population])

        record = stats.compile(population) if stats else {}
        record["generation_best"] = best_ind_record
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        _log(gen, best_ind_record["fitness"], time.time() - t0)

    return population, logbook

