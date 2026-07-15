"""
Diverse-partner crossover (exploratory strategy "diverse"; provenance in README.md).

Adapted-method baseline: crossover mates a tournament winner with the most
behaviourally distant candidate from a small random pool (RMS difference of
per-instance deviations), instead of the baseline's behaviour-blind partner
choice.
"""
import random
import time
from typing import Optional

from deap import tools

from yuantian.exploratory.shared import (
    DataProvider,
    Toolbox,
    _behavioural_distance,
    _evaluate_cases,
    _new_logbook,
    _record,
    _Timer,
    _tournament,
)
from yuantian.gp_algorithms import load_elites


def diverse_partner_gp(
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
    tournsize: int = 7,
    partner_pool: int = 5,
) -> tuple:
    """GP whose crossover mixes behaviourally distant parents."""
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    t0 = time.time()
    training = training_data_provider.next()
    _evaluate_cases(population, training, toolbox)
    _record(
        0, len(population), population, toolbox, halloffame, stats, logbook,
        validation_data_provider, pop_archive, timer, t0,
    )

    for gen in range(1, ngen + 1):
        t0 = time.time()
        offspring: list = []
        target = pop_size - n_elite
        while len(offspring) < target:
            r = random.random()
            if r < cxpb:
                p1 = _tournament(population, tournsize)
                pool = random.sample(population, min(partner_pool, len(population)))
                p2 = max(pool, key=lambda c: _behavioural_distance(p1.cases, c.cases))
                c1, c2 = toolbox.mate(toolbox.clone(p1), toolbox.clone(p2))
                offspring.append(c1)
                if len(offspring) < target:
                    offspring.append(c2)
            elif r < cxpb + mutpb:
                (c,) = toolbox.mutate(toolbox.clone(_tournament(population, tournsize)))
                offspring.append(c)
            else:
                offspring.append(toolbox.clone(_tournament(population, tournsize)))

        next_pop = offspring[:target] + load_elites(population, n_elite)
        training = training_data_provider.next()
        _evaluate_cases(next_pop, training, toolbox)
        population[:] = next_pop
        _record(
            gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
        )

    return population, logbook
