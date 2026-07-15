"""
Shared bookkeeping for the exploratory strategies.

Originally in yuantian/custom_ea.py (removed when the
repo was restructured into cp_propagation.py / hybrid_gp.py / local_search.py
-- see exploratory/README.md for provenance).
Every driver in this package reproduces ``gp_algorithms.standard_gp``'s
bookkeeping contract: update the hall-of-fame, append a clone snapshot to
``pop_archive`` each generation, return ``(population, logbook)`` with the
``fitness`` and ``generation_best`` chapters, so ``GPHH.write_result`` keeps
working unchanged on any of these drivers' output.

Nothing here is imported by gphh_solver.py or by cp_propagation.py /
hybrid_gp.py / local_search.py / nr_terminals.py; this package is invoked
only from yuantian/experiments/exploratory_sweep_experiment.py.
"""
import random
import time
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
from deap import gp, tools

Individual = Any
Toolbox = Any
DataProvider = Any  # .next() -> list[RCPSPModel]

from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_simulation import DecisionTypeEnum


class _Timer:
    """Per-generation ETA logger, matching standard_gp's console output."""

    def __init__(self, ngen: int):
        self.ngen = ngen
        self.start = time.time()
        self.gen_times: list[float] = []

    def log(self, gen: int, best_fit: float, elapsed_gen: float, extra: str = ""):
        self.gen_times.append(elapsed_gen)
        avg = sum(self.gen_times) / len(self.gen_times)
        print(
            f"  gen {gen:>3}/{self.ngen}  best={best_fit:>8.4f}  "
            f"gen_time={elapsed_gen:>5.1f}s  "
            f"elapsed={time.time() - self.start:>6.0f}s  "
            f"ETA={avg * (self.ngen - gen):>6.0f}s{extra}",
            flush=True,
        )


def _new_logbook(stats: Optional[tools.Statistics]) -> tools.Logbook:
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    return logbook


def _eval_full(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """Full-fidelity evaluation (mean deviation over all instances), in place."""
    evaluate = partial(toolbox.evaluate, domains=domains)
    for ind, fit in zip(individuals, toolbox.map(evaluate, individuals)):
        ind.fitness.values = fit


def _build_heuristic(individual: Individual, toolbox: Toolbox, simulator: Any = None) -> tuple:
    """Reconstruct the SGS decision function for one individual, reusing the
    simulator / pset / decision-type bound into ``toolbox.evaluate``."""
    kw = toolbox.evaluate.keywords
    compile_func, pset = kw["compile_func"], kw["pset"]
    decision_type = kw["decision_type"]
    simulator = simulator or kw["simulator"]
    if decision_type == DecisionTypeEnum.SIMULTANEOUS:
        heuristic = partial(
            simulator.together,
            priority_func=compile_func(
                expr=individual, pset=pset[TerminalTypeEnum.INTEGRATED.value]
            ),
            mode_func=None,
            priority_extre="min",
            mode_extre="min",
        )
    else:
        chooser = (
            simulator.activity_first_choose
            if decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE
            else simulator.mode_first_choose
        )
        heuristic = partial(
            chooser,
            priority_func=compile_func(
                expr=individual[TerminalTypeEnum.ACTIVITY.value],
                pset=pset[TerminalTypeEnum.ACTIVITY.value],
            ),
            mode_func=compile_func(
                expr=individual[TerminalTypeEnum.MODE.value],
                pset=pset[TerminalTypeEnum.MODE.value],
            ),
            priority_extre="min",
            mode_extre="min",
        )
    return simulator, heuristic


def _evaluate_cases(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """Sets ``ind.cases`` (per-instance dev) and ``ind.fitness`` (mean), for
    ``diverse_partner_gp``'s behaviour distance and ``selection.lexicase_gp``.

    Compiles once per individual via ``_build_heuristic`` instead of once per
    domain via ``toolbox.evaluate`` (drops the cross-individual ``toolbox.map``
    parallelism, cpu_cores>1 only).
    """
    if not domains:
        for ind in individuals:
            ind.cases = []
            ind.fitness.values = (0.0,)
        return
    for ind in individuals:
        simulator, heuristic = _build_heuristic(ind, toolbox)
        cases = []
        for domain in domains:
            sol = simulator.buildSolution(domain=domain, choose=heuristic)
            mk = sol.get_end_time(domain.sink_task)
            cases.append((mk - domain.cpm_esd) * 100 / domain.cpm_esd)
        ind.cases = cases
        ind.fitness.values = (float(np.mean(cases)),)


def _record(
    gen: int,
    nevals: int,
    population: list,
    toolbox: Toolbox,
    halloffame: Optional[tools.HallOfFame],
    stats: Optional[tools.Statistics],
    logbook: tools.Logbook,
    validation_provider: Optional[DataProvider],
    pop_archive: Optional[list],
    timer: "_Timer",
    t0: float,
    extra: str = "",
) -> None:
    """Update HOF, snapshot the population, compile stats, log one generation."""
    if halloffame is not None:
        halloffame.update(population)
    best = halloffame[0]
    best_record = {"fitness": best.fitness.values[0], "tree": str(best)}
    if validation_provider:
        validation_set = validation_provider.next()
        best_record["validation_fitness"] = partial(toolbox.evaluate, domains=validation_set)(
            best
        )[0]

    pop_archive.append([toolbox.clone(ind) for ind in population])

    compiled = stats.compile(population) if stats else {}
    compiled["generation_best"] = best_record
    logbook.record(gen=gen, nevals=nevals, **compiled)
    timer.log(gen, best_record["fitness"], time.time() - t0, extra)


def _iter_trees(individual: Individual) -> list:
    """Yield the PrimitiveTree(s) of an individual (multi-tree dict or single)."""
    if isinstance(individual, dict):
        return list(individual.values())
    return [individual]


def _terminal_reliance(individual: Individual, names: set) -> float:
    """Fraction of an individual's terminals whose name is in ``names``."""
    hits = total = 0
    for tree in _iter_trees(individual):
        for node in tree:
            if isinstance(node, gp.Terminal):
                total += 1
                if getattr(node, "name", None) in names:
                    hits += 1
    return hits / total if total else 0.0


def _tournament(population: list, tournsize: int) -> Individual:
    aspirants = random.sample(population, min(tournsize, len(population)))
    return min(aspirants, key=lambda i: i.fitness.values[0])


def _behavioural_distance(a: list, b: list) -> float:
    """RMS difference of per-instance deviations."""
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((av - bv) ** 2))) if av.size else 0.0


def _rank(a: list) -> np.ndarray:
    order = np.argsort(np.asarray(a, dtype=float), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def _spearman(x, y) -> float:
    if len(x) < 2:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _decision_trace(solution: Any, domain: Any) -> tuple:
    """(CP-respect rho, NR-frugality phi) of one schedule. phi is None on
    renewable-only instances so the NR axis collapses gracefully."""
    source, sink = domain.source_task, domain.sink_task

    starts, slacks = [], []
    for act, sched in solution.rcpsp_schedule.items():
        if act in (source, sink) or sched["start_time"] > 1e7:
            continue
        node = domain.cpm[act]
        starts.append(sched["start_time"])
        slacks.append(node._LSD - node._ESD)
    rho = _spearman(starts, slacks)

    nr_list = domain.non_renewable_resources_list
    if not nr_list:
        return rho, None
    tasks = [t for t in domain.tasks_list if t not in (source, sink)]
    frugal = 0
    for act, mode in zip(tasks, solution.rcpsp_modes):
        nr_cost = {
            m: sum(domain.mode_details[act][m].get(r, 0) for r in nr_list)
            for m in domain.mode_details[act]
        }
        if nr_cost[mode] <= min(nr_cost.values()):
            frugal += 1
    phi = frugal / len(tasks) if tasks else 0.0
    return rho, phi
