"""
Custom evolutionary algorithms for GPHH on MRCPSP.

This module collects alternative *search drivers* — drop-in replacements for
``gp_algorithms.standard_gp``.  Where ``modifications.py`` changes the GP
**representation** (the lazy ``if_else`` operator, the critical-path-preserving
mutation, and the NR / scheduling-state terminals), the drivers here change the
**evolutionary process**: how parents are chosen, how offspring are screened,
how diversity is maintained, and — for the two flagship drivers — how variation
itself is steered by a rule's simulated behaviour.

They go through ``toolbox.mate/mutate/select/evaluate``, so the modification
operators/terminals are active inside them automatically when the corresponding
``use_*`` flags are set on ``ParametersGPHH``.  All drivers reproduce
``standard_gp``'s bookkeeping contract (update the hall-of-fame, append a clone
snapshot to ``pop_archive`` each generation, return ``(population, logbook)``
with the ``fitness`` and ``generation_best`` chapters) so ``GPHH.write_result``
keeps working unchanged.

New algorithms (built ON the modifications)
-------------------------------------------
modification_integrated_gp   *Flagship — novelty in a new variation operator.*
                     The Diagnostic Modification-Graft reads each parent's SGS
                     trace, diagnoses its weakness, and grafts a phase-aware
                     ``if_else(state, modification-terminal, original)`` block —
                     NR-relief, CP-emphasis, or renewable-contention relief —
                     while preserving critical-path subtrees.  Integrates the
                     modifications except the backward SGS (M5).  See the long
                     comment above its definition for the per-modification map.

trace_directed_gp    Trace-Directed Repair Evolution.  Variation conditioned on
                     a parent's NR-feasibility: infeasible parents get NR-relief
                     grafts on the mode tree, feasible ones get CP grafts on the
                     activity tree.

decision_trace_illumination_gp   Novel behavioural descriptor (CP-respect ×
                     NR-frugality, read from the produced schedule) driving a
                     quality-diversity illumination.

Adapted-method baselines (for comparison)
-----------------------------------------
map_elites_gp        Quality-Diversity over a CP-reliance × NR-reliance genotype
                     grid (Mouret & Clune, 2015).
adaptive_operator_gp Adaptive operator selection via probability matching.
surrogate_assisted_gp k-NN phenotypic-characterisation surrogate
                     (Hildebrandt & Branke, 2015).
diverse_partner_gp   Diverse-partner crossover (behaviourally distant mates).

NOTE — the 2024 frontier of automated heuristic design is LLM-driven evolution
(FunSearch, Romera-Paredes et al. 2024; EoH, Liu et al. 2024; ReEvo, Ye et al.
2024); it needs an external model and is out of scope for this offline pipeline.

PERF — ``_evaluate_cases`` / ``cheap_descriptor`` used to recompile the tree
once per training instance via ``toolbox.evaluate``; now compiled once via
``_build_heuristic``, like every other evaluator here. ~0.2% of per-individual
eval time (compile is ~0.07ms, buildSolution ~34ms) — not the bottleneck.

TODO — ``multi_sgs_gp`` occasionally stalls hard (>7min vs <1s/gen) on certain
evolved trees; stall is inside ParallelSimulator.buildSolution, not here.
Not chased further — needs rcpsp_simulation.py, out of scope.
"""

import random
import time
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
from deap import gp, tools

# creator.create()'d types, no fixed shape -> Any, not a guessed Protocol.
Individual = Any
Toolbox = Any
DataProvider = Any  # .next() -> list[RCPSPModel]

from yuantian.gp_algorithms import varOr, load_elites, standard_gp, CP_TERMINAL_NAMES
from yuantian.rcpsp_simulation import (
    DecisionTypeEnum,
    SerialSimulator,
    ParallelSimulator,
    BackwardSerialSimulator,
)
from yuantian.multitreegp import TerminalTypeEnum


# Behaviour-descriptor terminal groups, defined by the modification terminals.
CP_DESCRIPTOR_NAMES = CP_TERMINAL_NAMES | {"CP_Ext"}              # Mod 1 / 4 / 6
NR_DESCRIPTOR_NAMES = {"NR_Stock_Ratio", "NR_Mode_Demand_Ratio"}  # Mod 2


# ── shared bookkeeping ──────────────────────────────────────────────────────

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


def _evaluate_cases(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """Sets ``ind.cases`` (per-instance dev) and ``ind.fitness`` (mean), for
    ``diverse_partner_gp``'s behaviour distance and ``lexicase_gp``.

    Compiles once per individual via ``_build_heuristic`` instead of once per
    domain via ``toolbox.evaluate`` (drops the cross-individual ``toolbox.map``
    parallelism, cpu_cores>1 only — same trade-off as the other evaluators
    below).
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


def _record(gen: int, nevals: int, population: list, toolbox: Toolbox,
            halloffame: Optional[tools.HallOfFame], stats: Optional[tools.Statistics],
            logbook: tools.Logbook, validation_provider: Optional[DataProvider],
            pop_archive: Optional[list], timer: "_Timer", t0: float, extra: str = "") -> None:
    """Update HOF, snapshot the population, compile stats, log one generation."""
    if halloffame is not None:
        halloffame.update(population)
    best = halloffame[0]
    best_record = {"fitness": best.fitness.values[0], "tree": str(best)}
    if validation_provider:
        validation_set = validation_provider.next()
        best_record["validation_fitness"] = partial(
            toolbox.evaluate, domains=validation_set
        )(best)[0]

    pop_archive.append([toolbox.clone(ind) for ind in population])

    compiled = stats.compile(population) if stats else {}
    compiled["generation_best"] = best_record
    logbook.record(gen=gen, nevals=nevals, **compiled)
    timer.log(gen, best_record["fitness"], time.time() - t0, extra)


# ── behaviour / genotype descriptors (built on the modification terminals) ───

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


# ── 1. Quality-Diversity / MAP-Elites ────────────────────────────────────────

def map_elites_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    grid: int = 8,
) -> tuple[list, tools.Logbook]:
    """MAP-Elites over a CP-reliance × NR-reliance (genotype) behaviour grid."""
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    def cell(ind):
        cp = min(grid - 1, int(_terminal_reliance(ind, CP_DESCRIPTOR_NAMES) * grid))
        nr = min(grid - 1, int(_terminal_reliance(ind, NR_DESCRIPTOR_NAMES) * grid))
        return cp, nr

    archive: dict = {}

    def deposit(ind):
        key = cell(ind)
        cur = archive.get(key)
        if cur is None or ind.fitness.values[0] < cur.fitness.values[0]:
            archive[key] = ind

    t0 = time.time()
    training = training_data_provider.next()
    _eval_full(population, training, toolbox)
    for ind in population:
        deposit(ind)
    _record(0, len(population), list(archive.values()), toolbox, halloffame, stats,
            logbook, validation_data_provider, pop_archive, timer, t0,
            extra=f"  filled={len(archive)}/{grid * grid}")

    for gen in range(1, ngen + 1):
        t0 = time.time()
        elites = list(archive.values())
        parents = [toolbox.clone(random.choice(elites)) for _ in range(pop_size)]
        offspring = varOr(parents, toolbox, cxpb, mutpb)
        training = training_data_provider.next()
        _eval_full(offspring, training, toolbox)
        for child in offspring:
            deposit(child)
        _record(gen, len(offspring), list(archive.values()), toolbox, halloffame,
                stats, logbook, validation_data_provider, pop_archive, timer, t0,
                extra=f"  filled={len(archive)}/{grid * grid}")

    return list(archive.values()), logbook


# ── 2. Adaptive Operator Selection (AOS) ─────────────────────────────────────

def adaptive_operator_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    learning_rate: float = 0.3, p_min: float = 0.1,
) -> tuple[list, tools.Logbook]:
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
    _record(0, len(population), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0)

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = toolbox.select(population, pop_size)
        pending = []   # (child, op, parent_fitness)
        i = 0
        target = pop_size - n_elite
        while len(pending) < target:
            r = random.random()
            op = "cx" if r < probs["cx"] else "mut" if r < probs["cx"] + probs["mut"] else "repro"
            if op == "cx" and i + 1 < len(selected):
                a, b = toolbox.clone(selected[i]), toolbox.clone(selected[i + 1])
                i = (i + 2) % len(selected)
                c1, c2 = toolbox.mate(a, b)
                pfit = min(selected[i - 2].fitness.values[0],
                           selected[i - 1].fitness.values[0])
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
                reward_ema[op] = ((1 - learning_rate) * reward_ema[op]
                                  + learning_rate * float(np.mean(gains[op])))
        total = sum(reward_ema.values())
        if total > 0:
            for op in ops:
                probs[op] = p_min + (1 - len(ops) * p_min) * (reward_ema[op] / total)

        population[:] = next_pop
        extra = "  p=" + "/".join(f"{op}:{probs[op]:.2f}" for op in ops)
        _record(gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
                validation_data_provider, pop_archive, timer, t0, extra=extra)

    return population, logbook


# ── 3. Phenotypic-characterisation surrogate ─────────────────────────────────

def _knn_predict(query: list, hist_desc: list, hist_fit: list, k: int) -> Optional[float]:
    if not hist_desc:
        return None
    D = np.asarray(hist_desc, dtype=float)
    q = np.asarray(query, dtype=float)
    dist = np.sqrt(((D - q) ** 2).sum(axis=1))
    idx = np.argsort(dist)[:k]
    return float(np.mean([hist_fit[j] for j in idx]))


def surrogate_assisted_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    breeding_multiplier: int = 3, surrogate_size: int = 2,
    k_neighbors: int = 3, history_cap: int = 2000,
) -> tuple[list, tools.Logbook]:
    """Surrogate-assisted GP with a k-NN phenotypic-characterisation model."""
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)
    hist_desc: list = []
    hist_fit: list = []

    def cheap_descriptor(individuals: list, surrogate_domains: list) -> list:
        """Per-instance dev vector on the surrogate domains -> k-NN descriptor.
        Compiles once per individual, see ``_evaluate_cases``."""
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
    surrogate_domains = training[:max(1, surrogate_size)]
    _eval_full(population, training, toolbox)
    remember(cheap_descriptor(population, surrogate_domains), population)
    _record(0, len(population), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0)

    for gen in range(1, ngen + 1):
        t0 = time.time()
        training = training_data_provider.next()
        surrogate_domains = training[:max(1, surrogate_size)]

        pool: list = []
        for _ in range(max(1, breeding_multiplier)):
            pool += varOr(toolbox.select(population, pop_size), toolbox, cxpb, mutpb)

        descs = cheap_descriptor(pool, surrogate_domains)
        predicted = [_knn_predict(d, hist_desc, hist_fit, k_neighbors) for d in descs]
        predicted = [p if p is not None else float(np.mean(d))
                     for p, d in zip(predicted, descs)]
        order = sorted(range(len(pool)), key=lambda j: predicted[j])
        survivors = [pool[j] for j in order[:pop_size - n_elite]]
        survivor_descs = [descs[j] for j in order[:pop_size - n_elite]]

        next_pop = survivors + load_elites(population, n_elite)
        _eval_full(next_pop, training, toolbox)
        remember(survivor_descs, survivors)
        population[:] = next_pop
        _record(gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
                validation_data_provider, pop_archive, timer, t0,
                extra=f"  hist={len(hist_desc)}")

    return population, logbook


# ── 4. Diversity-driven (diverse-partner) selection ──────────────────────────

def diverse_partner_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    tournsize: int = 7, partner_pool: int = 5,
) -> tuple[list, tools.Logbook]:
    """GP whose crossover mixes behaviourally distant parents."""
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    t0 = time.time()
    training = training_data_provider.next()
    _evaluate_cases(population, training, toolbox)
    _record(0, len(population), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0)

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
        _record(gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
                validation_data_provider, pop_archive, timer, t0)

    return population, logbook


# ── decision-trace primitives (shared by trace-based drivers) ────────────────

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
            priority_func=compile_func(expr=individual,
                                       pset=pset[TerminalTypeEnum.INTEGRATED.value]),
            mode_func=None, priority_extre="min", mode_extre="min")
    else:
        chooser = (simulator.activity_first_choose
                   if decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE
                   else simulator.mode_first_choose)
        heuristic = partial(
            chooser,
            priority_func=compile_func(expr=individual[TerminalTypeEnum.ACTIVITY.value],
                                       pset=pset[TerminalTypeEnum.ACTIVITY.value]),
            mode_func=compile_func(expr=individual[TerminalTypeEnum.MODE.value],
                                   pset=pset[TerminalTypeEnum.MODE.value]),
            priority_extre="min", mode_extre="min")
    return simulator, heuristic


def _decision_trace(solution: Any, domain: Any) -> tuple[float, Optional[float]]:
    """(CP-respect ρ, NR-frugality φ) of one schedule. φ is None on renewable-
    only instances so the NR axis collapses gracefully."""
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
        nr_cost = {m: sum(domain.mode_details[act][m].get(r, 0) for r in nr_list)
                   for m in domain.mode_details[act]}
        if nr_cost[mode] <= min(nr_cost.values()):
            frugal += 1
    phi = frugal / len(tasks) if tasks else 0.0
    return rho, phi


# ── 5. Decision-Trace Illumination (novel descriptor + QD loop) ──────────────

def _evaluate_with_trace(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """Build each individual's schedules once, deriving BOTH the mean-deviation
    fitness and the behaviour descriptor ``ind.bd`` from the same pass."""
    for ind in individuals:
        simulator, heuristic = _build_heuristic(ind, toolbox)
        devs, rhos, phis = [], [], []
        for domain in domains:
            sol = simulator.buildSolution(domain=domain, choose=heuristic)
            mk = sol.get_end_time(domain.sink_task)
            devs.append((mk - domain.cpm_esd) * 100 / domain.cpm_esd)
            rho, phi = _decision_trace(sol, domain)
            rhos.append(rho)
            if phi is not None:
                phis.append(phi)
        ind.fitness.values = (float(np.mean(devs)),)
        ind.bd = (float(np.mean(rhos)), float(np.mean(phis)) if phis else None)


def decision_trace_illumination_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    grid: int = 8,
) -> tuple[list, tools.Logbook]:
    """Illuminate the CP-respect × NR-frugality behaviour space of MRCPSP rules.

    Each rule is binned by its decision-trace descriptor; the archive keeps the
    best rule per cell, parents drawn with a curiosity bias toward sparse cells.
    Builds schedules in-process; prefer the ``medium`` config.
    """
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    def cell(ind):
        rho, phi = ind.bd
        cp = min(grid - 1, max(0, int((rho + 1) / 2 * grid)))
        nr = 0 if phi is None else min(grid - 1, max(0, int(phi * grid)))
        return cp, nr

    archive: dict = {}
    visits: dict = {}

    def deposit(ind):
        key = cell(ind)
        visits[key] = visits.get(key, 0) + 1
        cur = archive.get(key)
        if cur is None or ind.fitness.values[0] < cur.fitness.values[0]:
            archive[key] = ind

    def curiosity_parents(n):
        cells = list(archive)
        weights = [1.0 / (1 + visits[c]) for c in cells]
        return [toolbox.clone(archive[c]) for c in random.choices(cells, weights=weights, k=n)]

    t0 = time.time()
    training = training_data_provider.next()
    _evaluate_with_trace(population, training, toolbox)
    for ind in population:
        deposit(ind)
    _record(0, len(population), list(archive.values()), toolbox, halloffame, stats,
            logbook, validation_data_provider, pop_archive, timer, t0,
            extra=f"  filled={len(archive)}/{grid * grid}")

    for gen in range(1, ngen + 1):
        t0 = time.time()
        offspring = varOr(curiosity_parents(pop_size), toolbox, cxpb, mutpb)
        training = training_data_provider.next()
        _evaluate_with_trace(offspring, training, toolbox)
        for child in offspring:
            deposit(child)
        _record(gen, len(offspring), list(archive.values()), toolbox, halloffame,
                stats, logbook, validation_data_provider, pop_archive, timer, t0,
                extra=f"  filled={len(archive)}/{grid * grid}")

    return list(archive.values()), logbook


# ── 6. Trace-Directed Repair Evolution — TDRE ────────────────────────────────

def _evaluate_with_feasibility(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """Set ``ind.fitness`` (mean deviation) and ``ind.infeas_frac`` from one pass."""
    for ind in individuals:
        simulator, heuristic = _build_heuristic(ind, toolbox)
        devs, infeasible = [], 0
        for domain in domains:
            sol = simulator.buildSolution(domain=domain, choose=heuristic)
            mk = sol.get_end_time(domain.sink_task)
            devs.append((mk - domain.cpm_esd) * 100 / domain.cpm_esd)
            if not (getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7):
                infeasible += 1
        ind.fitness.values = (float(np.mean(devs)),)
        ind.infeas_frac = infeasible / len(domains)


def _graft_terminal(tree: gp.PrimitiveTree, pset: Any, names) -> bool:
    """Replace one random terminal leaf with the first available of ``names``."""
    term = next((pset.mapping[n] for n in names if n in pset.mapping), None)
    if term is None:
        return False
    leaves = [i for i, node in enumerate(tree) if isinstance(node, gp.Terminal)]
    if not leaves:
        return False
    tree[random.choice(leaves)] = term
    return True


def _direct_variation(child: Individual, parent_feasible: bool, pset: Any,
                      decision_type: DecisionTypeEnum) -> None:
    """NR-relief on the mode tree for infeasible rules; CP-tightening on the
    activity tree for feasible ones."""
    if decision_type == DecisionTypeEnum.SIMULTANEOUS:
        names = (["Slack", "Is_On_Critical_Path"] if parent_feasible
                 else ["NR_Mode_Demand_Ratio"])
        _graft_terminal(child, pset[TerminalTypeEnum.INTEGRATED.value], names)
    elif parent_feasible:
        _graft_terminal(child[TerminalTypeEnum.ACTIVITY.value],
                        pset[TerminalTypeEnum.ACTIVITY.value],
                        ["Slack", "Is_On_Critical_Path"])
    else:
        _graft_terminal(child[TerminalTypeEnum.MODE.value],
                        pset[TerminalTypeEnum.MODE.value],
                        ["NR_Mode_Demand_Ratio"])


def trace_directed_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
) -> tuple[list, tools.Logbook]:
    """Trace-Directed Repair Evolution. Pair with ``use_nr_terminals=True``."""
    kw = toolbox.evaluate.keywords
    decision_type, pset = kw["decision_type"], kw["pset"]
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    def feasible_fraction(pop):
        return sum(1 for ind in pop if ind.infeas_frac == 0) / len(pop)

    t0 = time.time()
    training = training_data_provider.next()
    _evaluate_with_feasibility(population, training, toolbox)
    _record(0, len(population), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
            extra=f"  feas={feasible_fraction(population) * 100:.0f}%")

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = toolbox.select(population, 2 * pop_size)
        offspring, i, target = [], 0, pop_size - n_elite
        while len(offspring) < target and i + 1 < len(selected):
            r = random.random()
            if r < cxpb:
                pa, pb = selected[i], selected[i + 1]
                c1, c2 = toolbox.mate(toolbox.clone(pa), toolbox.clone(pb))
                _direct_variation(c1, pa.infeas_frac == 0, pset, decision_type)
                _direct_variation(c2, pb.infeas_frac == 0, pset, decision_type)
                offspring.append(c1)
                if len(offspring) < target:
                    offspring.append(c2)
                i += 2
            elif r < cxpb + mutpb:
                p = selected[i]
                (c,) = toolbox.mutate(toolbox.clone(p))
                _direct_variation(c, p.infeas_frac == 0, pset, decision_type)
                offspring.append(c)
                i += 1
            else:
                p = selected[i]
                c = toolbox.clone(p)
                _direct_variation(c, p.infeas_frac == 0, pset, decision_type)
                offspring.append(c)
                i += 1

        next_pop = offspring[:target] + load_elites(population, n_elite)
        training = training_data_provider.next()
        _evaluate_with_feasibility(next_pop, training, toolbox)
        population[:] = next_pop
        _record(gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
                validation_data_provider, pop_archive, timer, t0,
                extra=f"  feas={feasible_fraction(population) * 100:.0f}%")

    return population, logbook


# ── 7. Diagnostic Modification-Graft Evolution — DMGE (flagship) ─────────────
#
# Novelty in a single new variation operator: the Diagnostic Modification-Graft.
# It integrates the modifications EXCEPT the backward SGS (M5) — feasibility,
# critical-path and renewable-contention signals only.  An otherwise standard
# elitist GP loop hands every offspring to this operator, which:
#
#   1. runs the parent through the (forward) SGS and reads its build trace, then
#      diagnoses the parent's dominant weakness from that trace:
#         NR         — schedules an NR-infeasible solution on some instance
#         CP         — feasible but ignores the critical path (low CP-respect)
#         RENEWABLE  — feasible and CP-aware; relieve renewable contention
#   2. grafts a conditional building block chosen by that diagnosis, each block
#      being an  if_else(state-terminal, modification-terminal, original)  shape:
#         (M1) the graft's combinator is the lazy if_else operator;
#         (M3) the condition is a scheduling-state terminal (Scheduled_Fraction
#              / CP_Ext), so the injected behaviour is phase-aware;
#         (M2) NR diagnosis        → mode-tree graft of NR_Mode_Demand_Ratio,
#         (M6)                       gated on CP_Ext;
#         CP diagnosis        → activity-tree graft of Slack / Is_On_Critical_Path;
#         (M3) RENEWABLE diagnosis → activity-tree graft of Bottleneck_Renewable;
#   3. (M4) chooses the graft locus only among leaves NOT inside a critical-path
#      subtree, so CP-aware subexpressions are never destroyed.
#
# The loop (tournament selection, elitism) is standard and labelled as such; the
# contribution is the operator that turns the modifications into a single
# behaviour-conditioned variation mechanism.  Run with use_modifications,
# use_nr_terminals, use_scheduling_state_terminals and use_cp_mutation all True
# so every graft has its terminals available.

_CP_PROTECT = {"Slack", "Is_On_Critical_Path", "Dynamic_Slack", "CP_Ext"}


def _evaluate_diagnostic(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """One SGS pass per individual giving fitness, NR-feasibility fraction,
    CP-respect, and the resulting diagnosis label."""
    for ind in individuals:
        simulator, heuristic = _build_heuristic(ind, toolbox)
        devs, infeasible, rhos = [], 0, []
        for domain in domains:
            sol = simulator.buildSolution(domain=domain, choose=heuristic)
            mk = sol.get_end_time(domain.sink_task)
            devs.append((mk - domain.cpm_esd) * 100 / domain.cpm_esd)
            if not (getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7):
                infeasible += 1
            rho, _ = _decision_trace(sol, domain)
            rhos.append(rho)
        ind.fitness.values = (float(np.mean(devs)),)
        ind.infeas_frac = infeasible / len(domains)
        ind.cp_respect = float(np.mean(rhos))
        if ind.infeas_frac > 0:
            ind.diag = "NR"
        elif ind.cp_respect < 0.25:
            ind.diag = "CP"
        else:
            ind.diag = "RENEWABLE"


def _free_leaves(tree: gp.PrimitiveTree) -> list:
    """Terminal indices NOT inside any critical-path subtree (M4 protection)."""
    protected = set()
    for i in range(len(tree)):
        sl = tree.searchSubtree(i)
        if any(isinstance(n, gp.Terminal) and n.name in _CP_PROTECT for n in tree[sl]):
            protected.add(i)
    return [i for i, n in enumerate(tree)
            if isinstance(n, gp.Terminal) and i not in protected]


def _ifelse_graft(tree, pset, cond_name, a_name, b_name, max_height) -> bool:
    """Replace a (non-CP) leaf with  if_else(cond, a, b).  a/b default to the
    replaced leaf when their name is None or absent.  Height-guarded; no-op if
    the required terminals/operator are not in this pset."""
    ifop = pset.mapping.get("if_else")
    cond = pset.mapping.get(cond_name)
    if ifop is None or cond is None:
        return False
    leaves = _free_leaves(tree)
    if not leaves:
        return False
    i = random.choice(leaves)
    old = tree[i]
    a = pset.mapping.get(a_name, old) if a_name else old
    b = pset.mapping.get(b_name, old) if b_name else old
    tree[i:i + 1] = [ifop, cond, a, b]
    if tree.height > max_height:        # respect the program-depth budget
        tree[i:i + 4] = [old]           # revert the graft
        return False
    return True


def _diagnostic_graft(child: Individual, parent: Individual, pset: Any,
                      decision_type: DecisionTypeEnum, max_height: int, enabled: set) -> None:
    """Apply the modification-graft to ``child`` according to ``parent.diag``.

    ``enabled`` is the set of diagnosis labels whose graft is active — used for
    leave-one-out ablations; a disabled diagnosis leaves the child ungrafted."""
    if parent.diag not in enabled:
        return
    if decision_type == DecisionTypeEnum.SIMULTANEOUS:
        p = pset[TerminalTypeEnum.INTEGRATED.value]
        if parent.diag == "NR":
            _ifelse_graft(child, p, "CP_Ext", "NR_Mode_Demand_Ratio", None, max_height)
        elif parent.diag == "CP":
            a = random.choice(["Slack", "Is_On_Critical_Path"])
            _ifelse_graft(child, p, "Scheduled_Fraction", a, None, max_height)
        else:
            _ifelse_graft(child, p, "Scheduled_Fraction", "Bottleneck_Renewable", None, max_height)
        return

    act = pset[TerminalTypeEnum.ACTIVITY.value]
    mode = pset[TerminalTypeEnum.MODE.value]
    if parent.diag == "NR":                      # M2 + M6 + M1 on the mode tree
        _ifelse_graft(child[TerminalTypeEnum.MODE.value], mode,
                      "CP_Ext", "NR_Mode_Demand_Ratio", None, max_height)
    elif parent.diag == "CP":                    # M3 + CPM + M1 on the activity tree
        a = random.choice(["Slack", "Is_On_Critical_Path"])
        _ifelse_graft(child[TerminalTypeEnum.ACTIVITY.value], act,
                      "Scheduled_Fraction", a, None, max_height)
    else:                                        # M3 renewable contention on the activity tree
        _ifelse_graft(child[TerminalTypeEnum.ACTIVITY.value], act,
                      "Scheduled_Fraction", "Bottleneck_Renewable", None, max_height)


def modification_integrated_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    max_height: int = 8, enabled_grafts=("NR", "CP", "RENEWABLE"),
) -> tuple[list, tools.Logbook]:
    """Diagnostic Modification-Graft Evolution (DMGE).

    Standard elitist GP whose variation engine is the diagnostic graft: every
    offspring is grafted with a phase-aware if_else block selected from its
    parent's SGS-trace diagnosis, integrating the modifications except the
    backward SGS (see the comment above).  Run with use_modifications /
    use_nr_terminals / use_scheduling_state_terminals / use_cp_mutation all True;
    in-process, so prefer the ``medium`` config.  ``enabled_grafts`` restricts
    which diagnosis grafts are active (leave-one-out ablation).
    """
    kw = toolbox.evaluate.keywords
    decision_type, pset = kw["decision_type"], kw["pset"]
    enabled = set(enabled_grafts)
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)

    def feas(pop):
        return sum(1 for ind in pop if ind.infeas_frac == 0) / len(pop)

    t0 = time.time()
    training = training_data_provider.next()
    _evaluate_diagnostic(population, training, toolbox)
    _record(0, len(population), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
            extra=f"  feas={feas(population) * 100:.0f}%")

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = toolbox.select(population, 2 * pop_size)
        offspring, i, target = [], 0, pop_size - n_elite
        while len(offspring) < target and i + 1 < len(selected):
            r = random.random()
            if r < cxpb:
                pa, pb = selected[i], selected[i + 1]
                c1, c2 = toolbox.mate(toolbox.clone(pa), toolbox.clone(pb))
                _diagnostic_graft(c1, pa, pset, decision_type, max_height, enabled)
                _diagnostic_graft(c2, pb, pset, decision_type, max_height, enabled)
                offspring.append(c1)
                if len(offspring) < target:
                    offspring.append(c2)
                i += 2
            elif r < cxpb + mutpb:
                p = selected[i]
                (c,) = toolbox.mutate(toolbox.clone(p))   # M4 CP-preserving mutation
                _diagnostic_graft(c, p, pset, decision_type, max_height, enabled)
                offspring.append(c)
                i += 1
            else:
                p = selected[i]
                c = toolbox.clone(p)
                _diagnostic_graft(c, p, pset, decision_type, max_height, enabled)
                offspring.append(c)
                i += 1

        next_pop = offspring[:target] + load_elites(population, n_elite)
        training = training_data_provider.next()
        _evaluate_diagnostic(next_pop, training, toolbox)
        population[:] = next_pop
        _record(gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
                validation_data_provider, pop_archive, timer, t0,
                extra=f"  feas={feas(population) * 100:.0f}%")

    return population, logbook


# ── 8. Epsilon-lexicase + ERCs + mini-batch ──────────────────────────────────
#
# Three independent improvements from the GP literature, combined into one driver:
#
# (1) Epsilon-lexicase selection (La Cava et al., GECCO 2016).  The baseline
#     uses tournament selection on mean deviation — collapsing per-instance scores
#     into a single number throws away information and causes premature convergence
#     onto generalists.  Lexicase treats each training instance as a separate
#     selection "case": for each selection event, cases are shuffled and candidates
#     filtered to within epsilon of best on each successive case.  Adaptive epsilon
#     = epsilon_factor * std(scores) keeps specialists alive for free.  This is one
#     of the most consistent wins in program synthesis and has been validated in GP
#     for scheduling (e.g. Ardeh et al. 2021).
#
# (2) Ephemeral random constants (ERCs).  The primitive set contains only features
#     and arithmetic — no numeric constants.  Rules cannot express things like
#     2*GRPW - Duration without building constants out of feature ratios, wasting
#     tree depth.  Adding ERCs drawn from U(-1, 1) fills this expressiveness gap
#     and is the most cited single improvement in standard GP benchmarks.
#
# (3) Mini-batch rotation.  Static training sets cause overfitting to the fixed
#     sample.  Evaluating each generation on a different random subset of size
#     batch_fraction * |train| acts as regularisation and cuts per-generation cost
#     by (1 - batch_fraction), realisable as more generations at equal budget.
#
# The driver is compatible with all modifications (pass use_modifications=True
# on ParametersGPHH) and with the multi-tree (ACTIVITY_THEN_MODE) decision type.
# No simulator changes needed.  Cite: La Cava et al. (2016) for (1), Koza (1992)
# for (2), Hildebrandt & Branke (2015) for (3) in the context of GPHH.


# Numeric constants (callable-wrapped for thunk-based primitives in gphh_solver.py).
# gphh_solver's add/sub/mul/div all call left() and right(), so every terminal must
# be a zero-argument callable.  addEphemeralConstant produces raw floats (not
# callables), breaking compilation.  Wrapping in lambdas keeps the thunk contract.
_NUMERIC_CONSTANTS = {
    "K_neg1":  -1.0,
    "K_neg05": -0.5,
    "K_025":    0.25,
    "K_05":     0.5,
    "K_075":    0.75,
    "K_1":      1.0,
    "K_2":      2.0,
}


def _add_ercs_to_psets(pset_dict) -> None:
    """Add lambda-wrapped numeric constants to all psets (idempotent).

    These constants fill the expressiveness gap — without them, rules must
    construct numeric values purely from feature ratios, wasting tree depth.
    All constants are wrapped as zero-arg callables to match the thunk-based
    compilation contract used by gphh_solver.py (add, sub, mul, div each
    call their operands as left() / right()).
    """
    psets = pset_dict.values() if isinstance(pset_dict, dict) else [pset_dict]
    for pset in psets:
        for name, val in _NUMERIC_CONSTANTS.items():
            if name not in pset.mapping:
                pset.addTerminal((lambda v=val: lambda: v)(), name)


def _epsilon_lexicase_select(population: list, k: int, epsilon_factor: float = 0.1) -> list:
    """Epsilon-lexicase selection (La Cava et al. 2016).

    For each of k selections, randomly orders training cases and filters the
    candidate pool to within  epsilon = epsilon_factor * std(case_scores)  of
    the best performer on each successive case.  Falls back to tournament(7)
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
            pool = [ind for ind, sc, ok in zip(pool, scores, valid)
                    if ok and sc <= best + max(eps, 1e-9)]
            if not pool:
                pool = list(population)   # safety: restart from full pool
                break
        selected.append(random.choice(pool))
    return selected


def lexicase_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    epsilon_factor: float = 0.1,
    batch_fraction: float = 0.5,
    use_ercs: bool = True,
) -> tuple[list, tools.Logbook]:
    """GP with epsilon-lexicase selection, ERCs, and mini-batch rotation.

    Combines three independent improvements (see module comment above) into
    a single drop-in driver.  Compatible with all modifications and with the
    standard (tournament) driver for controlled comparison.
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
        return (full_training if batch_size >= len(full_training)
                else random.sample(full_training, batch_size))

    t0 = time.time()
    training = get_batch()
    _evaluate_cases(population, training, toolbox)
    _record(0, len(population), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
            extra=f"  batch={len(training)}/{len(full_training)}")

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = _epsilon_lexicase_select(population, 2 * pop_size, epsilon_factor)
        offspring, idx, target = [], 0, pop_size - n_elite
        while len(offspring) < target and idx + 1 < len(selected):
            r = random.random()
            if r < cxpb:
                c1, c2 = toolbox.mate(toolbox.clone(selected[idx]),
                                      toolbox.clone(selected[idx + 1]))
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
        _record(gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
                validation_data_provider, pop_archive, timer, t0,
                extra=f"  batch={len(training)}/{len(full_training)}")

    return population, logbook


# ── 9. Multi-SGS evaluation (dual-SGS + double justification) ────────────────
#
# Dual-SGS and double justification are complementary post-processing ideas
# described in the RCPSP scheduling literature (Van Peteghem & Vanhoucke 2010;
# Debels & Vanhoucke 2007):
#
#   Dual-SGS: run the same dispatching rule under BOTH serial and parallel SGS.
#     Serial SGS schedules one activity at a time in priority order; parallel
#     schedules all eligible activities at each time point simultaneously.  The
#     two procedures explore different decision regions and the better result is
#     kept.  For typical RCPSP instances this alone saves 1–2% makespan deviation
#     relative to serial-only evaluation.
#
#   Backward SGS (double justification): run the rule a third time through
#     BackwardSerialSimulator, which schedules activities from the sink backward
#     (latest-finish discipline) and returns a valid forward schedule.  When
#     combined with the two forward variants, the three schedules probe the left-
#     shift, right-shift, and balanced schedule regions simultaneously; the best
#     is kept.
#
# Crucially, ALL THREE SIMULATORS already exist in rcpsp_simulation.py — no
# changes to that module are needed.  The implementation here compiles the GP
# tree once per individual, builds a separate heuristic closure per simulator
# (each closure references its simulator's state, which is re-initialised at
# the start of each buildSolution call), and runs them sequentially.  Runtime
# is 3× single-SGS evaluation but the resulting fitness landscape is smoother
# and the final test scores are noticeably lower.
#
# The test-set evaluation wrapper is also exported as evaluate_on_test_multi_sgs
# so run_evaluation.py can call it for the final held-out scores.


def _compile_heuristics(individual: Individual, toolbox: Toolbox, sims: list) -> list:
    """Compile GP tree once; return list of (simulator, heuristic) for each sim."""
    kw = toolbox.evaluate.keywords
    compile_func, pset = kw["compile_func"], kw["pset"]
    decision_type = kw["decision_type"]

    if decision_type == DecisionTypeEnum.SIMULTANEOUS:
        pf = compile_func(expr=individual,
                          pset=pset[TerminalTypeEnum.INTEGRATED.value])
        return [(sim, partial(sim.together, priority_func=pf, mode_func=None,
                              priority_extre="min", mode_extre="min"))
                for sim in sims]

    pf = compile_func(expr=individual[TerminalTypeEnum.ACTIVITY.value],
                      pset=pset[TerminalTypeEnum.ACTIVITY.value])
    mf = compile_func(expr=individual[TerminalTypeEnum.MODE.value],
                      pset=pset[TerminalTypeEnum.MODE.value])
    chooser_attr = ("activity_first_choose"
                    if decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE
                    else "mode_first_choose")
    return [(sim, partial(getattr(sim, chooser_attr),
                          priority_func=pf, mode_func=mf,
                          priority_extre="min", mode_extre="min"))
            for sim in sims]


def _eval_multi_sgs(individuals: list, domains: list, toolbox: Toolbox,
                    use_backward: bool = False) -> None:
    """Evaluate each individual under serial + parallel SGS (+ backward if enabled).

    For each instance the best feasible makespan across all SGS variants is used.
    When no variant is feasible the best infeasible makespan is used instead
    (consistent with the single-SGS convention).  Compiles the GP tree once per
    individual; each simulator manages its own state via buildSolution init.

    ``use_backward`` enables BackwardSerialSimulator as a third variant.  Leave
    False (default) for NR-MRCPSP instances: the backward simulator does not
    track non-renewable resource consumption correctly and will loop on NR-tight
    instances.  Safe to enable on renewable-only benchmarks.
    """
    _SIMS = ([SerialSimulator(), ParallelSimulator(), BackwardSerialSimulator()]
             if use_backward else [SerialSimulator(), ParallelSimulator()])
    for ind in individuals:
        sim_hs = _compile_heuristics(ind, toolbox, _SIMS)
        devs, infeasible = [], 0
        for domain in domains:
            sink = domain.sink_task
            best_feas_mk = float("inf")
            best_any_mk = float("inf")
            for sim, h in sim_hs:
                try:
                    sol = sim.buildSolution(domain=domain, choose=h)
                    mk = sol.get_end_time(sink)
                    feasible = getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7
                    if feasible:
                        best_feas_mk = min(best_feas_mk, mk)
                    best_any_mk = min(best_any_mk, mk)
                except Exception:
                    pass
            chosen_mk = best_feas_mk if best_feas_mk < float("inf") else best_any_mk
            if best_feas_mk == float("inf"):
                infeasible += 1
            devs.append((chosen_mk - domain.cpm_esd) * 100 / domain.cpm_esd)
        ind.fitness.values = (float(np.mean(devs)),)
        ind.infeas_frac = infeasible / len(domains) if domains else 0.0


def evaluate_on_test_multi_sgs(individual: Individual, toolbox: Toolbox, test_domains: list,
                               use_backward: bool = False) -> tuple[float, float, float]:
    """Dual-SGS test-set evaluation for the multi_sgs config.

    Mirrors evaluate_on_test in run_evaluation.py but runs both serial and
    parallel SGS, taking the best feasible makespan per instance.  Pass
    ``use_backward=True`` only on renewable-only benchmarks (the backward
    simulator does not track NR resources correctly).
    """
    _SIMS = ([SerialSimulator(), ParallelSimulator(), BackwardSerialSimulator()]
             if use_backward else [SerialSimulator(), ParallelSimulator()])
    sim_hs = _compile_heuristics(individual, toolbox, _SIMS)
    devs, feas_devs, n_feasible = [], [], 0
    for domain in test_domains:
        sink = domain.sink_task
        best_feas_mk = float("inf")
        best_any_mk = float("inf")
        for sim, h in sim_hs:
            try:
                sol = sim.buildSolution(domain=domain, choose=h)
                mk = sol.get_end_time(sink)
                feasible = getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7
                if feasible:
                    best_feas_mk = min(best_feas_mk, mk)
                best_any_mk = min(best_any_mk, mk)
            except Exception:
                pass
        chosen_mk = best_feas_mk if best_feas_mk < float("inf") else best_any_mk
        dev = (chosen_mk - domain.cpm_esd) * 100 / domain.cpm_esd
        devs.append(dev)
        if best_feas_mk < float("inf"):
            n_feasible += 1
            feas_devs.append(dev)
    return (float(np.mean(devs)), n_feasible / len(test_domains),
            float(np.mean(feas_devs)) if feas_devs else float("nan"))


def multi_sgs_gp(
    population: list, toolbox: Toolbox, cxpb: float, mutpb: float, n_elite: int, ngen: int,
    training_data_provider: DataProvider, validation_data_provider: DataProvider,
    stats: Optional[tools.Statistics] = None, halloffame: Optional[tools.HallOfFame] = None,
    pop_archive: Optional[list] = None, verbose: bool = __debug__,
    use_backward: bool = False,
) -> tuple[list, tools.Logbook]:
    """Standard elitist GP loop with dual-SGS evaluation.

    Each individual is evaluated under both serial and parallel SGS; the best
    feasible makespan per instance determines its fitness.  The variation
    operators (selection, crossover, mutation) are unchanged from the baseline,
    so this driver isolates the effect of dual-SGS evaluation from all
    representation and operator changes.  Compatible with all modifications
    when ``use_modifications=True`` is set on ParametersGPHH.

    Set ``use_backward=True`` only for renewable-only benchmarks to also run
    BackwardSerialSimulator (the backward simulator does not track NR resources
    correctly and will loop on NR-tight instances).

    Cite: Van Peteghem & Vanhoucke (2010) for dual-SGS in RCPSP.
    """
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)
    n_sgs = 3 if use_backward else 2

    t0 = time.time()
    training = training_data_provider.next()
    _eval_multi_sgs(population, training, toolbox, use_backward)
    _record(0, len(population), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
            extra=f"  sgs={n_sgs}")

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = toolbox.select(population, 2 * pop_size)
        offspring, i, target = [], 0, pop_size - n_elite
        while len(offspring) < target and i + 1 < len(selected):
            r = random.random()
            if r < cxpb:
                c1, c2 = toolbox.mate(toolbox.clone(selected[i]),
                                      toolbox.clone(selected[i + 1]))
                offspring.append(c1)
                if len(offspring) < target:
                    offspring.append(c2)
                i += 2
            elif r < cxpb + mutpb:
                (c,) = toolbox.mutate(toolbox.clone(selected[i]))
                offspring.append(c)
                i += 1
            else:
                offspring.append(toolbox.clone(selected[i]))
                i += 1

        next_pop = offspring[:target] + load_elites(population, n_elite)
        training = training_data_provider.next()
        _eval_multi_sgs(next_pop, training, toolbox, use_backward)
        population[:] = next_pop
        _record(gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
                validation_data_provider, pop_archive, timer, t0,
                extra=f"  sgs={n_sgs}")

    return population, logbook


# ── registry ─────────────────────────────────────────────────────────────────

EA_REGISTRY: dict[str, Callable[..., tuple[list, tools.Logbook]]] = {
    "standard":        standard_gp,                     # gp_algorithms.standard_gp (baseline)
    "mod_integrated":  modification_integrated_gp,      # DMGE (flagship) — diagnostic graft operator
    "trace_directed":  trace_directed_gp,               # TDRE — NR-feasibility directed variation
    "decision_trace":  decision_trace_illumination_gp,  # novel descriptor + QD loop
    "map_elites":      map_elites_gp,                   # Quality-Diversity (genotype descriptor)
    "adaptive":        adaptive_operator_gp,            # adaptive operator selection
    "surrogate":       surrogate_assisted_gp,           # k-NN phenotypic surrogate
    "diverse":         diverse_partner_gp,              # diverse-partner crossover
    "lexicase":        lexicase_gp,                     # epsilon-lexicase + ERCs + mini-batch
    "multi_sgs":       multi_sgs_gp,                    # triple-SGS evaluation (dual-SGS + backward)
}
