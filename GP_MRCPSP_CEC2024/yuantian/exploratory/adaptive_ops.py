"""
Adaptive Operator Selection (Phase 0, strategy "adaptive"). Restored from
yuantian/custom_ea.py's ``adaptive_operator_gp`` (deleted in commit
b595a2d5; see yuantian/exploratory/README.md).

Adapted-method baseline: probability matching on credit (reward = fitness
improvement over the parent) adaptively reweights crossover / mutation /
reproduction rates instead of using the fixed cxpb/mutpb the baseline holds
constant for the whole run.
"""
import random
import time
from typing import Optional

import numpy as np
from deap import tools

from yuantian.exploratory.shared import DataProvider, Toolbox, _eval_full, _new_logbook, _record, _Timer
from yuantian.gp_algorithms import load_elites


def adaptive_operator_gp(
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
    learning_rate: float = 0.3,
    p_min: float = 0.1,
) -> tuple:
    """GP with adaptive operator selection (probability matching on credit)."""
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)
    ops = ["cx", "mut", "repro"]
    probs = {"cx": cxpb, "mut": mutpb, "repro": max(0.0, 1.0 - cxpb - mutpb)}
    reward_ema = {op: 0.0 for op in ops}

    t0 = time.time()
    training = training_data_provider.next()
    _eval_full(population, training, toolbox)
    _record(
        0, len(population), population, toolbox, halloffame, stats, logbook,
        validation_data_provider, pop_archive, timer, t0,
    )

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = toolbox.select(population, pop_size)
        pending = []  # (child, op, parent_fitness)
        i = 0
        target = pop_size - n_elite
        while len(pending) < target:
            r = random.random()
            op = "cx" if r < probs["cx"] else "mut" if r < probs["cx"] + probs["mut"] else "repro"
            if op == "cx" and i + 1 < len(selected):
                a, b = toolbox.clone(selected[i]), toolbox.clone(selected[i + 1])
                i = (i + 2) % len(selected)
                c1, c2 = toolbox.mate(a, b)
                pfit = min(
                    selected[i - 2].fitness.values[0], selected[i - 1].fitness.values[0]
                )
                pending.append((c1, "cx", pfit))
                if len(pending) < target:
                    pending.append((c2, "cx", pfit))
            elif op == "mut":
                p = selected[i]
                i = (i + 1) % len(selected)
                (c,) = toolbox.mutate(toolbox.clone(p))
                pending.append((c, "mut", p.fitness.values[0]))
            else:
                p = selected[i]
                i = (i + 1) % len(selected)
                pending.append((toolbox.clone(p), "repro", p.fitness.values[0]))

        offspring = [c for c, _, _ in pending]
        next_pop = offspring + load_elites(population, n_elite)
        training = training_data_provider.next()
        _eval_full(next_pop, training, toolbox)

        gains = {op: [] for op in ops}
        for child, op, pfit in pending:
            gains[op].append(max(0.0, pfit - child.fitness.values[0]))
        for op in ops:
            if gains[op]:
                reward_ema[op] = (1 - learning_rate) * reward_ema[op] + learning_rate * float(
                    np.mean(gains[op])
                )
        total = sum(reward_ema.values())
        if total > 0:
            for op in ops:
                probs[op] = p_min + (1 - len(ops) * p_min) * (reward_ema[op] / total)

        population[:] = next_pop
        extra = "  p=" + "/".join(f"{op}:{probs[op]:.2f}" for op in ops)
        _record(
            gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0, extra=extra,
        )

    return population, logbook
