"""
Unit tests for yuantian/mode_interaction_terminals.py + the MI feature
functions it documents (implemented in rcpsp_simulation.py, same split as
rccp_terminals.py).

Run from the GP_MRCPSP_CEC2024 repo root:

    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/test/test_mode_interaction_terminals.py

Do NOT run this with `-O`: these tests rely on plain `assert` statements.
"""
import random

import numpy as np

from yuantian.gp_algorithms import standard_gp
from yuantian.gphh_solver import GPHH, ParametersGPHH, RefreshHallOfFame, read_instances
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import (
    DecisionTypeEnum,
    ParallelSimulator,
    SerialSimulator,
    SimulatorTypeEnum,
)
from yuantian.utils import PopulationArchive
from deap import tools


def _training_instance():
    return read_instances([RCPSPDatabase.MMLIB_50_DIR + "J501_1.mm"])[0]


def test_constraint_tightening_requires_mode():
    training = _training_instance()
    sim = SerialSimulator()

    def choose(eligibles):
        act = next(iter(eligibles))
        sim.cur_act, sim.cur_mode = act, None
        try:
            sim.feature_mi_constraint_tightening()
            raise AssertionError("expected ValueError when cur_mode is None")
        except ValueError:
            pass
        try:
            sim.feature_mi_reciprocal_scarcity()
            raise AssertionError("expected ValueError when cur_mode is None")
        except ValueError:
            pass
        return act, eligibles[act][0]

    sim.buildSolution(domain=training, choose=choose)
    print("test_constraint_tightening_requires_mode passed")


def test_activity_pressure_does_not_require_mode():
    """The activity-tree variant must work with cur_mode unset, since
    activity-priority scoring happens before mode is chosen."""
    training = _training_instance()
    sim = SerialSimulator()
    checked = {"n": 0}

    def choose(eligibles):
        act = next(iter(eligibles))
        sim.cur_act, sim.cur_mode = act, None
        value = sim.feature_mi_activity_pressure()
        assert value == value and value >= 0.0
        checked["n"] += 1
        return act, eligibles[act][0]

    sim.buildSolution(domain=training, choose=choose)
    assert checked["n"] > 5
    print(f"test_activity_pressure_does_not_require_mode passed ({checked['n']} checks)")


def test_neighbors_exclude_self_and_respect_cap():
    training = _training_instance()
    for SimCls in (SerialSimulator, ParallelSimulator):
        sim = SimCls()
        checked = {"n": 0}

        def choose(eligibles):
            act = next(iter(eligibles))
            mode = eligibles[act][0]
            sim.cur_act, sim.cur_mode = act, mode
            demand = sim.rcpsp_problem.mode_details[act][mode]
            renewable = sim.rcpsp_problem.renewable_resources_list
            candidate_resources = {res for res in renewable if demand.get(res, 0) > 0}
            neighbors = sim._mode_interaction_neighbors(candidate_resources)
            assert act not in neighbors, f"{SimCls.__name__}: candidate appeared in its own neighbor set"
            assert len(neighbors) <= sim._MI_MAX_NEIGHBORS, (
                f"{SimCls.__name__}: neighbor set exceeded cap: {len(neighbors)} > {sim._MI_MAX_NEIGHBORS}"
            )
            checked["n"] += 1
            return act, mode

        sim.buildSolution(domain=training, choose=choose)
        assert checked["n"] > 5
    print("test_neighbors_exclude_self_and_respect_cap passed")


def test_constraint_tightening_in_unit_interval():
    training = _training_instance()
    sim = SerialSimulator()
    values = []

    def choose(eligibles):
        act = list(eligibles)[-1]
        mode = eligibles[act][-1]
        sim.cur_act, sim.cur_mode = act, mode
        values.append(sim.feature_mi_constraint_tightening())
        values.append(sim.feature_mi_activity_pressure())
        return act, mode

    sim.buildSolution(domain=training, choose=choose)
    assert len(values) > 10
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in values), f"out-of-range MI value in {values}"
    print("test_constraint_tightening_in_unit_interval passed")


def test_resource_avail_cache_invalidates_across_decisions_not_within():
    """`_current_resource_avail` should return the identical dict object on
    repeated calls within one decision (cache hit), and recompute
    (different dict object, though possibly equal values) once
    `_decision_counter` advances to the next decision."""
    training = _training_instance()
    sim = SerialSimulator()
    seen_within_decision = []
    seen_across_decisions = []

    def choose(eligibles):
        act = next(iter(eligibles))
        mode = eligibles[act][0]
        sim.cur_act, sim.cur_mode = act, mode
        first = sim._current_resource_avail()
        second = sim._current_resource_avail()
        seen_within_decision.append(first is second)
        seen_across_decisions.append(id(first))
        return act, mode

    sim.buildSolution(domain=training, choose=choose)
    assert all(seen_within_decision), "expected a cache hit (same object) within one decision"
    assert len(set(seen_across_decisions)) > 1, "expected the cache to recompute across decisions"
    print("test_resource_avail_cache_invalidates_across_decisions_not_within passed")


def _build_solver(simulator_type, decision_type):
    training = [_training_instance()]
    params = ParametersGPHH.fast(
        simulator_type=simulator_type,
        decision_type=decision_type,
        mode_interaction_terminals_feature=True,
    )
    params.pop_size = 12
    params.n_gen = 2
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    return solver, training


def test_mi_terminals_registered_and_evaluable_all_combinations():
    mi_names = {"MI_CONSTRAINT_TIGHTENING", "MI_RECIPROCAL_SCARCITY", "MI_ACTIVITY_PRESSURE"}
    for simulator_type in (SimulatorTypeEnum.SERIAL_SGS, SimulatorTypeEnum.PARALLEL_SGS):
        for decision_type in (
            DecisionTypeEnum.ACTIVITY_THEN_MODE,
            DecisionTypeEnum.MODE_THEN_ACTIVITY,
            DecisionTypeEnum.SIMULTANEOUS,
        ):
            random.seed(0)
            solver, training = _build_solver(simulator_type, decision_type)
            registered = set()
            for pset in solver.params_gphh.set_primitves.values():
                registered |= set(pset.mapping.keys()) & mi_names
            assert registered, f"no MI terminal registered for {simulator_type}/{decision_type}"

            pop = solver.toolbox.population(n=solver.params_gphh.pop_size)
            for ind in pop:
                fitness = solver.toolbox.evaluate(individual=ind, domains=training)
                value = fitness[0]
                assert value == value, f"NaN fitness for {simulator_type}/{decision_type}"
                assert value >= 0
            print(f"  ok: {simulator_type.value}/{decision_type.value}, registered={registered}")
    print("test_mi_terminals_registered_and_evaluable_all_combinations passed")


def test_mi_combines_with_other_extensions_without_collisions():
    training = read_instances(
        [RCPSPDatabase.MMLIB_50_DIR + "J501_1.mm"], keep_non_renewable=True
    )
    random.seed(0)
    params = ParametersGPHH.fast(
        simulator_type=SimulatorTypeEnum.SERIAL_SGS,
        decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE,
        cp_propagation_feature=True,
        nr_terminals_feature=True,
        rccp_terminals_feature=True,
        mode_interaction_terminals_feature=True,
    )
    params.pop_size = 12
    params.n_gen = 1
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()

    all_names = set()
    for pset in params.set_primitves.values():
        all_names |= set(pset.mapping.keys())
    expected = {
        "CP_FORWARD", "NR_STOCK_RATIO", "RCCP_BOTTLENECK_UTIL",
        "MI_CONSTRAINT_TIGHTENING", "MI_RECIPROCAL_SCARCITY", "MI_ACTIVITY_PRESSURE",
    }
    missing = expected - all_names
    assert not missing, f"missing terminals when all four extensions combined: {missing}"

    pop = solver.toolbox.population(n=params.pop_size)
    for ind in pop:
        fitness = solver.toolbox.evaluate(individual=ind, domains=training)
        assert fitness[0] == fitness[0] and fitness[0] >= 0
    print("test_mi_combines_with_other_extensions_without_collisions passed")


def test_mi_terminal_actually_used_by_evolution():
    solver, training = _build_solver(
        SimulatorTypeEnum.SERIAL_SGS, DecisionTypeEnum.ACTIVITY_THEN_MODE
    )
    mi_names = ("MI_CONSTRAINT_TIGHTENING", "MI_RECIPROCAL_SCARCITY", "MI_ACTIVITY_PRESSURE")
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", np.mean)

    found = False
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        solver.init_model()
        pop = solver.toolbox.population(n=solver.params_gphh.pop_size)
        standard_gp(
            pop, solver.toolbox, cxpb=0.8, mutpb=0.15, n_elite=2, ngen=3,
            training_data_provider=StaticDatasetProvider(training),
            validation_data_provider=None, stats=mstats,
            halloffame=RefreshHallOfFame(1), pop_archive=PopulationArchive(),
            verbose=False,
        )
        if any(any(name in str(ind) for name in mi_names) for ind in pop):
            found = True
            break
    assert found, "no MI terminal appeared in any individual across 5 seeds x 3 generations"
    print("test_mi_terminal_actually_used_by_evolution passed")


if __name__ == "__main__":
    test_constraint_tightening_requires_mode()
    test_activity_pressure_does_not_require_mode()
    test_neighbors_exclude_self_and_respect_cap()
    test_constraint_tightening_in_unit_interval()
    test_resource_avail_cache_invalidates_across_decisions_not_within()
    test_mi_terminals_registered_and_evaluable_all_combinations()
    test_mi_combines_with_other_extensions_without_collisions()
    test_mi_terminal_actually_used_by_evolution()
    print("All mode_interaction_terminals tests passed")
