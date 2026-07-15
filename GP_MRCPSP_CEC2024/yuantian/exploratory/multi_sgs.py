"""
Multi-SGS evaluation (dual-SGS + double justification). Restored from
yuantian/custom_ea.py (see
yuantian/exploratory/README.md).

Dual-SGS and double justification are complementary post-processing ideas
from the RCPSP scheduling literature (Van Peteghem & Vanhoucke 2010; Debels
& Vanhoucke 2007):

  Dual-SGS: run the same dispatching rule under BOTH serial and parallel
    SGS and keep the better result. Serial schedules one activity at a time
    in priority order; parallel schedules all eligible activities at each
    time point simultaneously -- the two explore different decision
    regions.

  Backward SGS (double justification): run the rule a third time through
    BackwardSerialSimulator (restored in yuantian/rcpsp_simulation.py for
    this driver), which schedules from the sink backward and returns a
    valid forward schedule. Combined with the two forward variants, the
    three schedules probe the left-shift, right-shift, and balanced
    schedule regions simultaneously; the best is kept.

Correction made during this restoration (see exploratory/README.md): the
original ``_eval_multi_sgs`` / ``evaluate_on_test_multi_sgs`` took an
unconditional ``min()`` across simulators' makespans even when *no* variant
produced a feasible schedule -- ``best_any_mk = min(best_any_mk, mk)`` ran
on every iteration regardless of the ``feasible`` flag, so an individual
infeasible under all simulators got the minimum of several infeasible
sentinel makespans rather than one consistent penalty, optimistically
biasing its fitness relative to an individual infeasible under only one
simulator. Fixed below by always falling back to the *serial* simulator's
makespan (the first entry in ``sim_hs``) when no variant is feasible,
instead of taking a cross-simulator min of infeasible results.

Second bug, found only once I actually ran this for real (the restoration
smoke test didn't catch it): ``_compile_heuristics`` was compiling each
tree against the toolbox's own pset, but that pset's terminals are bound
methods of the toolbox's simulator specifically (baked in at pset build
time, see ``ParametersGPHH``). This module builds its own fresh
``SerialSimulator()``/``ParallelSimulator()`` instances to run
``buildSolution`` on though, so every terminal call ended up reading state
off the wrong simulator -- one whose ``rcpsp_problem`` was never set,
i.e. always ``None``. That crashed on literally every call, and the
``except Exception: pass`` below (which is there for other legitimate
reasons) just silently swallowed it, so it looked like every individual
was simply infeasible instead of erroring out. ``_rebind_pset_to_simulator``
fixes this by rebuilding a pset with the same terminal names, just bound
to whichever simulator is actually going to run the schedule.
"""
import random
import time
from typing import Optional

import numpy as np
from deap import tools
from deap.gp import PrimitiveSet

from yuantian.exploratory.shared import (
    DataProvider,
    Individual,
    Toolbox,
    _new_logbook,
    _record,
    _Timer,
)
from yuantian.gp_algorithms import load_elites
from yuantian.gphh_solver import (
    add_operator,
    max_operator,
    min_operator,
    mul_operator,
    protected_div_operator,
    sub_operator,
)
from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_simulation import (
    BackwardSerialSimulator,
    DecisionTypeEnum,
    FeatureEnum,
    ParallelSimulator,
    SerialSimulator,
)


def _rebind_pset_to_simulator(orig_pset: dict, decision_type, simulator) -> dict:
    """Same terminal names as orig_pset, just rebound to simulator's
    feature_function_map instead of whatever simulator orig_pset was
    originally built for (see the module docstring for why that matters).
    Terminal names come from pset.terminals -- deap.gp actually stores the
    bound callable in pset.context[name], not on the Terminal object, which
    is why we can't just reuse orig_pset as-is for a different
    simulator."""
    new_pset = {}
    for terminal_type, pset in orig_pset.items():
        new_pset[terminal_type] = PrimitiveSet(decision_type, 0)
        for terms in pset.terminals.values():
            for term in terms:
                feature = FeatureEnum(term.name)
                new_pset[terminal_type].addTerminal(
                    simulator.feature_function_map[feature], term.name
                )
        new_pset[terminal_type].addPrimitive(add_operator, 2, name="add")
        new_pset[terminal_type].addPrimitive(sub_operator, 2, name="sub")
        new_pset[terminal_type].addPrimitive(mul_operator, 2, name="mul")
        new_pset[terminal_type].addPrimitive(protected_div_operator, 2, name="div")
        new_pset[terminal_type].addPrimitive(min_operator, 2, name="min")
        new_pset[terminal_type].addPrimitive(max_operator, 2, name="max")
    return new_pset


def _compile_heuristics(individual: Individual, toolbox: Toolbox, sims: list) -> list:
    """Compile GP tree once per simulator in `sims`, each against its OWN
    pset (see _rebind_pset_to_simulator's docstring for why a shared pset
    silently breaks this -- every terminal call would read the wrong
    simulator's state instead of the one actually running buildSolution)."""
    from functools import partial

    kw = toolbox.evaluate.keywords
    compile_func, orig_pset = kw["compile_func"], kw["pset"]
    decision_type = kw["decision_type"]

    result = []
    for sim in sims:
        pset = _rebind_pset_to_simulator(orig_pset, decision_type, sim)
        if decision_type == DecisionTypeEnum.SIMULTANEOUS:
            pf = compile_func(expr=individual, pset=pset[TerminalTypeEnum.INTEGRATED.value])
            result.append(
                (sim, partial(sim.together, priority_func=pf, mode_func=None, priority_extre="min", mode_extre="min"))
            )
            continue

        pf = compile_func(
            expr=individual[TerminalTypeEnum.ACTIVITY.value], pset=pset[TerminalTypeEnum.ACTIVITY.value]
        )
        mf = compile_func(
            expr=individual[TerminalTypeEnum.MODE.value], pset=pset[TerminalTypeEnum.MODE.value]
        )
        chooser_attr = (
            "activity_first_choose"
            if decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE
            else "mode_first_choose"
        )
        result.append(
            (
                sim,
                partial(
                    getattr(sim, chooser_attr), priority_func=pf, mode_func=mf, priority_extre="min", mode_extre="min"
                ),
            )
        )
    return result


def _eval_multi_sgs(
    individuals: list, domains: list, toolbox: Toolbox, use_backward: bool = False
) -> None:
    """Evaluate each individual under serial + parallel SGS (+ backward if
    enabled). For each instance the best FEASIBLE makespan across all SGS
    variants is used. When no variant is feasible, the *serial* simulator's
    makespan is used as the penalty value -- a single consistent fallback,
    not a min-across-variants (see module docstring: the original took an
    unconditional min even across infeasible results, biasing fitness
    optimistically).

    ``use_backward`` enables BackwardSerialSimulator as a third variant.
    Leave False (default) for NR-MRCPSP instances: the backward simulator
    does not track non-renewable resource consumption correctly and will
    loop on NR-tight instances. Safe to enable on renewable-only benchmarks.
    """
    _SIMS = (
        [SerialSimulator(), ParallelSimulator(), BackwardSerialSimulator()]
        if use_backward
        else [SerialSimulator(), ParallelSimulator()]
    )
    for ind in individuals:
        sim_hs = _compile_heuristics(ind, toolbox, _SIMS)
        devs, infeasible = [], 0
        for domain in domains:
            sink = domain.sink_task
            best_feas_mk = float("inf")
            # Fallback penalty when no variant is feasible: the *serial*
            # simulator's own makespan (mk computed even if infeasible), not
            # a min across variants -- see module docstring. Stays inf (not
            # None) if even the serial buildSolution call itself raises, so
            # this degrades the same way the pre-fix code did for that
            # doubly-degenerate case rather than introducing a new crash.
            serial_mk = float("inf")
            for j, (sim, h) in enumerate(sim_hs):
                try:
                    sol = sim.buildSolution(domain=domain, choose=h)
                    mk = sol.get_end_time(sink)
                    feasible = getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7
                    if feasible:
                        best_feas_mk = min(best_feas_mk, mk)
                    if j == 0:
                        serial_mk = mk  # consistent fallback penalty, see docstring
                except Exception:
                    pass
            chosen_mk = best_feas_mk if best_feas_mk < float("inf") else serial_mk
            if best_feas_mk == float("inf"):
                infeasible += 1
            devs.append((chosen_mk - domain.cpm_esd) * 100 / domain.cpm_esd)
        ind.fitness.values = (float(np.mean(devs)),)
        ind.infeas_frac = infeasible / len(domains) if domains else 0.0


def evaluate_on_test_multi_sgs(
    individual: Individual, toolbox: Toolbox, test_domains: list, use_backward: bool = False
) -> tuple:
    """Dual-SGS test-set evaluation for the multi_sgs config. Mirrors the
    fitness convention in gphh_solver.py's evaluate_heuristic but runs both
    serial and parallel SGS (+ backward if enabled), taking the best
    feasible makespan per instance, with the same single-consistent-
    fallback fix as ``_eval_multi_sgs``. Pass ``use_backward=True`` only on
    renewable-only benchmarks (the backward simulator does not track NR
    resources correctly)."""
    _SIMS = (
        [SerialSimulator(), ParallelSimulator(), BackwardSerialSimulator()]
        if use_backward
        else [SerialSimulator(), ParallelSimulator()]
    )
    sim_hs = _compile_heuristics(individual, toolbox, _SIMS)
    devs, feas_devs, n_feasible = [], [], 0
    for domain in test_domains:
        sink = domain.sink_task
        best_feas_mk = float("inf")
        serial_mk = float("inf")  # see _eval_multi_sgs's comment on this fallback
        for j, (sim, h) in enumerate(sim_hs):
            try:
                sol = sim.buildSolution(domain=domain, choose=h)
                mk = sol.get_end_time(sink)
                feasible = getattr(sol, "rcpsp_schedule_feasible", True) and mk < 1e7
                if feasible:
                    best_feas_mk = min(best_feas_mk, mk)
                if j == 0:
                    serial_mk = mk
            except Exception:
                pass
        chosen_mk = best_feas_mk if best_feas_mk < float("inf") else serial_mk
        dev = (chosen_mk - domain.cpm_esd) * 100 / domain.cpm_esd
        devs.append(dev)
        if best_feas_mk < float("inf"):
            n_feasible += 1
            feas_devs.append(dev)
    return (
        float(np.mean(devs)),
        n_feasible / len(test_domains),
        float(np.mean(feas_devs)) if feas_devs else float("nan"),
    )


def multi_sgs_gp(
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
    use_backward: bool = False,
) -> tuple:
    """Standard elitist GP loop with dual-SGS evaluation. Each individual is
    evaluated under both serial and parallel SGS; the best feasible
    makespan per instance determines its fitness. Selection/crossover/
    mutation are unchanged from the baseline, so this driver isolates the
    effect of dual-SGS evaluation from all representation/operator changes.

    Set ``use_backward=True`` only for renewable-only benchmarks (see
    ``_eval_multi_sgs``).

    Cite: Van Peteghem & Vanhoucke (2010) for dual-SGS in RCPSP.
    """
    logbook = _new_logbook(stats)
    timer = _Timer(ngen)
    pop_size = len(population)
    n_sgs = 3 if use_backward else 2

    t0 = time.time()
    training = training_data_provider.next()
    _eval_multi_sgs(population, training, toolbox, use_backward)
    _record(
        0, len(population), population, toolbox, halloffame, stats, logbook,
        validation_data_provider, pop_archive, timer, t0, extra=f"  sgs={n_sgs}",
    )

    for gen in range(1, ngen + 1):
        t0 = time.time()
        selected = toolbox.select(population, 2 * pop_size)
        offspring, i, target = [], 0, pop_size - n_elite
        while len(offspring) < target and i + 1 < len(selected):
            r = random.random()
            if r < cxpb:
                c1, c2 = toolbox.mate(toolbox.clone(selected[i]), toolbox.clone(selected[i + 1]))
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
        _record(
            gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0, extra=f"  sgs={n_sgs}",
        )

    return population, logbook
