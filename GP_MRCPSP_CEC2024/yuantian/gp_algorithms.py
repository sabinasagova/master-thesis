"""Genetic Programming Algorithms.These algorithms are built upon DEAP's GP module."""

from deap import tools, base, gp
import random
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


def evaluate_and_capture_case_data(individual, evaluate):
    """Run `evaluate(individual=individual)` and return (fitness,
    case_fitness, case_records, case_feasible) together.

    gphh_solver.evaluate_heuristic sets case_fitness/case_records/
    case_feasible as side-effect attributes on `individual`, not as part of
    its return value. That's invisible to the caller once toolbox.map
    routes through a multiprocessing.Pool: each worker mutates its own
    unpickled copy of `individual`, and pool.map only sends the function's
    *return value* back, not the mutated argument -- so `ind.case_fitness`
    on the original population member is simply never set.
    epsilon_lexicase_selection (hybrid_gp.py) reads `case_fitness` straight
    off population members and crashes with AttributeError if it's missing,
    so this wrapper exists to round-trip that data through the one channel
    that does survive pool.map: the return value. Module-level (not a
    lambda/nested def) so it stays picklable under multiprocessing.
    """
    fitness = evaluate(individual=individual)
    return (
        fitness,
        getattr(individual, "case_fitness", None),
        getattr(individual, "case_records", None),
        getattr(individual, "case_feasible", None),
    )


def evaluate_population(toolbox, evaluate, individuals) -> None:
    """Evaluate `individuals` via `toolbox.map` and set fitness.values AND
    case_fitness/case_records/case_feasible on each, surviving
    multiprocessing (see evaluate_and_capture_case_data) -- shared by
    standard_gp and hybrid_gp.lexicase_memetic_gp so both feed
    epsilon_lexicase_selection valid per-case data regardless of
    --multiprocess."""
    results = toolbox.map(partial(evaluate_and_capture_case_data, evaluate=evaluate), individuals)
    for ind, (fit, case_fitness, case_records, case_feasible) in zip(individuals, results):
        ind.fitness.values = fit
        ind.case_fitness = case_fitness
        ind.case_records = case_records
        ind.case_feasible = case_feasible


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

    # Evaluate the individuals with an invalid fitness under training set
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # update training data set each generation
    training_data = training_data_provider.next()
    evaluate = partial(toolbox.evaluate, domains=training_data)
    evaluate_population(toolbox, evaluate, invalid_ind)
    # Add current pop into pop_archive
    pop_archive.append([toolbox.clone(ind) for ind in population])

    # Update hall of fame and do the validation test
    if halloffame is not None:
        halloffame.update(population)
    best_ind_record = {}
    best_ind_record["fitness"] = halloffame[0].fitness.values[0]
    best_ind_record["tree"] = str(halloffame[0])

    # if validation set is specified, run the validation test
    if validation_data_provider:
       validation_set = validation_data_provider.next()
       validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
       best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

    record = stats.compile(population) if stats else {}
    # Add other records to logbook
    record["generation_best"] = best_ind_record
    # Write record to logbook
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - n_elite)
        # Vary the pool of individuals
        offspring = varOr(offspring, toolbox, cxpb, mutpb)
        offspring += load_elites(population, n_elite)

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # update training data set each generation
        training_data = training_data_provider.next()
        evaluate = partial(toolbox.evaluate, domains=training_data)
        evaluate_population(toolbox, evaluate, offspring)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        best_ind_record = {}
        best_ind_record["fitness"] = halloffame[0].fitness.values[0]
        best_ind_record["tree"] = str(halloffame[0])

        # if validation set is specified, run the validation test
        if validation_data_provider:
           validation_set = validation_data_provider.next()
           validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
           best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

        # Replace the current population by the offspring
        population[:] = offspring
        # Add current pop into pop_archive
        pop_archive.append([toolbox.clone(ind) for ind in population])

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        # Add other records to logbook
        record["generation_best"] = best_ind_record

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

