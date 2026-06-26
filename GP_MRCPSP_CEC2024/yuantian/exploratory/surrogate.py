"""
Phenotypic-characterisation surrogate (Phase 0, strategy "surrogate").
Restored from yuantian/custom_ea.py's ``surrogate_assisted_gp`` (deleted in
commit b595a2d5; see yuantian/exploratory/README.md).

Adapted-method baseline (Hildebrandt & Branke, 2015): breeds more offspring
than the population needs, ranks them with a cheap k-NN surrogate fitted on
a small subset of training instances, and only fully evaluates the
predicted-best survivors -- trading exact fitness for evaluation count.
"""
import time
from typing import Optional

import numpy as np
from deap import tools

from yuantian.exploratory.shared import (
    DataProvider,
    Toolbox,
    _build_heuristic,
    _eval_full,
    _new_logbook,
    _record,
    _Timer,
)
from yuantian.gp_algorithms import load_elites, varOr


def _knn_predict(query: list, hist_desc: list, hist_fit: list, k: int) -> Optional[float]:
    if not hist_desc:
        return None
    D = np.asarray(hist_desc, dtype=float)
    q = np.asarray(query, dtype=float)
    dist = np.sqrt(((D - q) ** 2).sum(axis=1))
    idx = np.argsort(dist)[:k]
    return float(np.mean([hist_fit[j] for j in idx]))


def surrogate_assisted_gp(
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
    breeding_multiplier: int = 3,
    surrogate_size: int = 2,
    k_neighbors: int = 3,
    history_cap: int = 2000,
) -> tuple:
    """Surrogate-assisted GP with a k-NN phenotypic-characterisation model."""
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)
    hist_desc: list = []
    hist_fit: list = []

    def cheap_descriptor(individuals: list, surrogate_domains: list) -> list:
        """Per-instance dev vector on the surrogate domains -> k-NN descriptor."""
        descriptors = []
        for ind in individuals:
            simulator, heuristic = _build_heuristic(ind, toolbox)
            devs = []
            for d in surrogate_domains:
                sol = simulator.buildSolution(domain=d, choose=heuristic)
                mk = sol.get_end_time(d.sink_task)
                devs.append((mk - d.cpm_esd) * 100 / d.cpm_esd)
            descriptors.append(devs)
        return descriptors

    def remember(descs, individuals):
        for d, ind in zip(descs, individuals):
            hist_desc.append(d)
            hist_fit.append(ind.fitness.values[0])
        if len(hist_desc) > history_cap:
            del hist_desc[:-history_cap]
            del hist_fit[:-history_cap]

    t0 = time.time()
    training = training_data_provider.next()
    surrogate_domains = training[: max(1, surrogate_size)]
    _eval_full(population, training, toolbox)
    remember(cheap_descriptor(population, surrogate_domains), population)
    _record(
        0, len(population), population, toolbox, halloffame, stats, logbook,
        validation_data_provider, pop_archive, timer, t0,
    )

    for gen in range(1, ngen + 1):
        t0 = time.time()
        training = training_data_provider.next()
        surrogate_domains = training[: max(1, surrogate_size)]

        pool: list = []
        for _ in range(max(1, breeding_multiplier)):
            pool += varOr(toolbox.select(population, pop_size), toolbox, cxpb, mutpb)

        descs = cheap_descriptor(pool, surrogate_domains)
        predicted = [_knn_predict(d, hist_desc, hist_fit, k_neighbors) for d in descs]
        predicted = [p if p is not None else float(np.mean(d)) for p, d in zip(predicted, descs)]
        order = sorted(range(len(pool)), key=lambda j: predicted[j])
        survivors = [pool[j] for j in order[: pop_size - n_elite]]
        survivor_descs = [descs[j] for j in order[: pop_size - n_elite]]

        next_pop = survivors + load_elites(population, n_elite)
        _eval_full(next_pop, training, toolbox)
        remember(survivor_descs, survivors)
        population[:] = next_pop
        _record(
            gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0, extra=f"  hist={len(hist_desc)}",
        )

    return population, logbook
