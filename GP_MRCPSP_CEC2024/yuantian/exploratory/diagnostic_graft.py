"""
Diagnostic Modification-Graft Evolution (DMGE) and its two dependent drivers.
The three diagnosis-driven variation schemes (see
yuantian/exploratory/README.md):

  modification_integrated_gp   strategy "mod_integrated" -- the main
                     scheme. Every offspring is grafted with a phase-aware
                     if_else block selected from its parent's SGS-trace
                     diagnosis (NR-infeasible / CP-blind / renewable-
                     contention-blind).
  trace_directed_gp strategy "trace_directed" -- variation
                     conditioned on a parent's NR-feasibility only (a
                     simpler precursor to the diagnostic graft above).
  decision_trace_illumination_gp  strategy "decision_trace" -- a
                     CP-respect x NR-frugality behavioural descriptor
                     driving a quality-diversity illumination loop (does
                     not use the graft operator; grouped here because it
                     shares ``_decision_trace``/``_build_heuristic`` with
                     the other two and the original module grouped them
                     together).

Restored dependency note (see exploratory/README.md "Restoration notes"):
the graft operator needs an ``if_else`` primitive and several terminals
(Slack, Is_On_Critical_Path, Dynamic_Slack, Scheduled_Fraction,
Bottleneck_Renewable, CP_Ext, NR_Stock_Ratio, NR_Mode_Demand_Ratio) that
lived in yuantian/modifications.py -- also deleted in the same commit, and
not otherwise needed by this restoration. ``install_graft_terminals``
below ports those definitions verbatim, scoped *only* to the GPHH instance
passed to it (it mutates that instance's pset dict and monkey-patches that
instance's simulator object, not the shared yuantian/rcpsp_simulation.py
classes or yuantian/gphh_solver.py), so calling it has no effect outside
these three drivers.
"""
import random
import time
from typing import Any, Callable, Optional

from deap import gp, tools

from yuantian.exploratory.shared import (
    DataProvider,
    Individual,
    Toolbox,
    _build_heuristic,
    _decision_trace,
    _new_logbook,
    _record,
    _Timer,
)
from yuantian.gp_algorithms import load_elites, varOr
from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_simulation import DecisionTypeEnum

# ── restored modifications.py terminals/primitive (scoped to this module) ───


def if_then_else_operator(
    cond: Callable[[], float], out1: Callable[[], float], out2: Callable[[], float]
) -> Callable[[], float]:
    """Lazy if/else: evaluates only the taken branch. Threshold is cond() > 0
    (not Python-truthy), giving a well-defined zero-crossing that pairs
    cleanly with Is_On_Critical_Path (1 on the path, 0 off it)."""

    def if_then_else() -> float:
        return out1() if cond() > 0 else out2()

    return if_then_else


def _term_slack(simulator):
    node = simulator.rcpsp_problem.cpm[simulator.cur_act]
    return node._LSD - node._ESD


def _term_is_on_critical_path(simulator):
    return 1.0 if _term_slack(simulator) == 0 else 0.0


def _term_dynamic_slack(simulator):
    node = simulator.dynamic_cpm[simulator.cur_act]
    return node._LSD - node._ESD


def _term_scheduled_fraction(simulator):
    return len(simulator.scheduled) / simulator.rcpsp_problem.n_jobs


def _term_bottleneck_renewable(simulator):
    """Min over renewable resources of (available / capacity) at this
    activity's earliest start. Requires simulator.resource_avail_in_time
    (SerialSimulator only -- see exploratory/README.md restoration notes)."""
    nr = set(simulator.rcpsp_problem.non_renewable_resources_list)
    renewable = [r for r in simulator.rcpsp_problem.resources_list if r not in nr]
    if not renewable:
        return 1.0
    t = simulator.minimum_starting_time[simulator.cur_act]
    ratios = [
        simulator.resource_avail_in_time[r][min(t, len(simulator.resource_avail_in_time[r]) - 1)]
        / max(1, simulator.rcpsp_problem.resources[r])
        for r in renewable
    ]
    return min(ratios)


def _term_cp_ext(simulator):
    """max(0, EFFT - LFD): how many time units scheduling this (activity,
    mode) right now would push the project end past its dynamic CPM
    deadline."""
    if not hasattr(simulator, "dynamic_cpm") or simulator.dynamic_cpm is None:
        return 0.0
    node = simulator.dynamic_cpm.get(simulator.cur_act)
    if node is None or node._LFD is None:
        return 0.0
    efft = simulator.heuristic_earliest_feasible_finish_time()
    return max(0.0, float(efft) - float(node._LFD))


def _term_nr_stock_ratio(simulator):
    nr_list = simulator.rcpsp_problem.non_renewable_resources_list
    if not nr_list:
        return 1.0
    ratios = [
        max(0, simulator.resource_avail_in_time[r][-1]) / max(1, simulator.rcpsp_problem.resources[r])
        for r in nr_list
    ]
    return sum(ratios) / len(ratios)


def _term_nr_mode_demand_ratio(simulator):
    if simulator.cur_mode is None:
        raise ValueError("Mode is not specified.")
    nr_list = simulator.rcpsp_problem.non_renewable_resources_list
    if not nr_list:
        return 0.0
    ratios = [
        simulator.rcpsp_problem.mode_details[simulator.cur_act][simulator.cur_mode].get(r, 0)
        / max(1, simulator.resource_avail_in_time[r][-1])
        for r in nr_list
    ]
    return sum(ratios) / len(ratios)


def install_graft_terminals(pset_dict, simulator) -> None:
    """Idempotently add the if_else primitive and the 8 modification-era
    terminals (closed over ``simulator``) to ``pset_dict``'s psets, scoped
    the same way yuantian/modifications.py originally scoped them:
    ACTIVITY/INTEGRATED get the CP + scheduling-state terminals, MODE/
    INTEGRATED get the NR-mode-demand and CP_Ext terminals, ACTIVITY/
    INTEGRATED get NR-stock-ratio. Mutates ``pset_dict`` in place; has no
    effect on any pset not passed to it (in particular, never touches
    gphh_solver.py's pset construction or yuantian/rcpsp_simulation.py)."""
    act_key, mode_key, int_key = (
        TerminalTypeEnum.ACTIVITY.value,
        TerminalTypeEnum.MODE.value,
        TerminalTypeEnum.INTEGRATED.value,
    )
    act_mode_int = [k for k in (act_key, mode_key, int_key) if k in pset_dict]
    act_int = [k for k in (act_key, int_key) if k in pset_dict]
    mode_int = [k for k in (mode_key, int_key) if k in pset_dict]

    for key in act_mode_int:
        pset = pset_dict[key]
        if "if_else" not in pset.mapping:
            pset.addPrimitive(if_then_else_operator, 3, name="if_else")

    def _add_terminal(keys, name, func):
        for key in keys:
            pset = pset_dict[key]
            if name not in pset.mapping:
                pset.addTerminal((lambda f=func: lambda: f(simulator))(), name)

    _add_terminal(act_int, "Slack", _term_slack)
    _add_terminal(act_int, "Is_On_Critical_Path", _term_is_on_critical_path)
    _add_terminal(act_int, "Dynamic_Slack", _term_dynamic_slack)
    _add_terminal(act_int, "Scheduled_Fraction", _term_scheduled_fraction)
    _add_terminal(act_int, "Bottleneck_Renewable", _term_bottleneck_renewable)
    _add_terminal(act_int, "NR_Stock_Ratio", _term_nr_stock_ratio)
    _add_terminal(mode_int, "CP_Ext", _term_cp_ext)
    _add_terminal(mode_int, "NR_Mode_Demand_Ratio", _term_nr_mode_demand_ratio)


_CP_PROTECT = {"Slack", "Is_On_Critical_Path", "Dynamic_Slack", "CP_Ext"}


# ── shared evaluators ────────────────────────────────────────────────────────


def _evaluate_with_feasibility(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """Set ``ind.fitness`` (mean deviation) and ``ind.infeas_frac`` from one pass."""
    import numpy as np

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


def _evaluate_diagnostic(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """One SGS pass per individual giving fitness, NR-feasibility fraction,
    CP-respect, and the resulting diagnosis label."""
    import numpy as np

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


def _evaluate_with_trace(individuals: list, domains: list, toolbox: Toolbox) -> None:
    """Build each individual's schedules once, deriving BOTH the
    mean-deviation fitness and the behaviour descriptor ``ind.bd``."""
    import numpy as np

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


# ── graft primitives ─────────────────────────────────────────────────────────


def _free_leaves(tree: gp.PrimitiveTree) -> list:
    """Terminal indices NOT inside any critical-path subtree."""
    protected = set()
    for i in range(len(tree)):
        sl = tree.searchSubtree(i)
        if any(isinstance(n, gp.Terminal) and n.name in _CP_PROTECT for n in tree[sl]):
            protected.add(i)
    return [
        i for i, n in enumerate(tree) if isinstance(n, gp.Terminal) and i not in protected
    ]


def _ifelse_graft(tree, pset, cond_name, a_name, b_name, max_height) -> bool:
    """Replace a (non-CP) leaf with if_else(cond, a, b). Height-guarded;
    no-op if the required terminals/operator are not in this pset."""
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
    tree[i : i + 1] = [ifop, cond, a, b]
    if tree.height > max_height:  # respect the program-depth budget
        tree[i : i + 4] = [old]  # revert the graft
        return False
    return True


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


def _direct_variation(
    child: Individual, parent_feasible: bool, pset: Any, decision_type: DecisionTypeEnum
) -> None:
    """NR-relief on the mode tree for infeasible rules; CP-tightening on the
    activity tree for feasible ones."""
    if decision_type == DecisionTypeEnum.SIMULTANEOUS:
        names = ["Slack", "Is_On_Critical_Path"] if parent_feasible else ["NR_Mode_Demand_Ratio"]
        _graft_terminal(child, pset[TerminalTypeEnum.INTEGRATED.value], names)
    elif parent_feasible:
        _graft_terminal(
            child[TerminalTypeEnum.ACTIVITY.value],
            pset[TerminalTypeEnum.ACTIVITY.value],
            ["Slack", "Is_On_Critical_Path"],
        )
    else:
        _graft_terminal(
            child[TerminalTypeEnum.MODE.value],
            pset[TerminalTypeEnum.MODE.value],
            ["NR_Mode_Demand_Ratio"],
        )


def _diagnostic_graft(
    child: Individual,
    parent: Individual,
    pset: Any,
    decision_type: DecisionTypeEnum,
    max_height: int,
    enabled: set,
) -> None:
    """Apply the modification-graft to ``child`` according to ``parent.diag``.

    ``enabled`` is the set of diagnosis labels whose graft is active -- used
    for leave-one-out ablations; a disabled diagnosis leaves the child
    ungrafted."""
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
    if parent.diag == "NR":  # NR-relief graft on the mode tree
        _ifelse_graft(
            child[TerminalTypeEnum.MODE.value], mode, "CP_Ext", "NR_Mode_Demand_Ratio", None, max_height
        )
    elif parent.diag == "CP":  # CP-tightening graft on the activity tree
        a = random.choice(["Slack", "Is_On_Critical_Path"])
        _ifelse_graft(
            child[TerminalTypeEnum.ACTIVITY.value], act, "Scheduled_Fraction", a, None, max_height
        )
    else:  # renewable-contention-relief graft on the activity tree
        _ifelse_graft(
            child[TerminalTypeEnum.ACTIVITY.value],
            act,
            "Scheduled_Fraction",
            "Bottleneck_Renewable",
            None,
            max_height,
        )


# ── drivers ───────────────────────────────────────────────────────────────────


def trace_directed_gp(
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
) -> tuple:
    """Trace-Directed Repair Evolution. Call ``install_graft_terminals`` on
    this GPHH instance's pset/simulator first, or the grafts are no-ops."""
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
    _record(
        0, len(population), population, toolbox, halloffame, stats, logbook,
        validation_data_provider, pop_archive, timer, t0,
        extra=f"  feas={feasible_fraction(population) * 100:.0f}%",
    )

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
        _record(
            gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
            extra=f"  feas={feasible_fraction(population) * 100:.0f}%",
        )

    return population, logbook


def modification_integrated_gp(
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
    max_height: int = 8,
    enabled_grafts=("NR", "CP", "RENEWABLE"),
) -> tuple:
    """Diagnostic Modification-Graft Evolution (DMGE), the main graft-based
    strategy. Call ``install_graft_terminals`` on this GPHH instance's
    pset/simulator first, or the grafts are no-ops. ``enabled_grafts``
    restricts which diagnosis grafts are active (leave-one-out ablation)."""
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
    _record(
        0, len(population), population, toolbox, halloffame, stats, logbook,
        validation_data_provider, pop_archive, timer, t0,
        extra=f"  feas={feas(population) * 100:.0f}%",
    )

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
                (c,) = toolbox.mutate(toolbox.clone(p))
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
        _record(
            gen, len(next_pop), population, toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
            extra=f"  feas={feas(population) * 100:.0f}%",
        )

    return population, logbook


def decision_trace_illumination_gp(
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
    grid: int = 8,
) -> tuple:
    """Illuminate the CP-respect x NR-frugality behaviour space of MRCPSP
    rules. Each rule is binned by its decision-trace descriptor; the archive
    keeps the best rule per cell, parents drawn with a curiosity bias toward
    sparse cells. Does not need install_graft_terminals (no graft operator;
    the descriptor is read off the produced schedule, not off terminal
    names)."""
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
    _record(
        0, len(population), list(archive.values()), toolbox, halloffame, stats, logbook,
        validation_data_provider, pop_archive, timer, t0,
        extra=f"  filled={len(archive)}/{grid * grid}",
    )

    for gen in range(1, ngen + 1):
        t0 = time.time()
        offspring = varOr(curiosity_parents(pop_size), toolbox, cxpb, mutpb)
        training = training_data_provider.next()
        _evaluate_with_trace(offspring, training, toolbox)
        for child in offspring:
            deposit(child)
        _record(
            gen, len(offspring), list(archive.values()), toolbox, halloffame, stats, logbook,
            validation_data_provider, pop_archive, timer, t0,
            extra=f"  filled={len(archive)}/{grid * grid}",
        )

    return list(archive.values()), logbook
