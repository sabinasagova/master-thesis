"""
Unit tests for yuantian/rccp_terminals.py + the RCCP feature functions it
documents (implemented in rcpsp_simulation.py, since RCCP has no static
half to precompute -- see rccp_terminals.py's module docstring).

Run from the GP_MRCPSP_CEC2024 repo root (matches the other tests in this
folder):

    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/test/test_rccp_terminals.py

Do NOT run this with `-O`: these tests rely on plain `assert` statements,
and `-O` strips those out, turning every test into a silent no-op. (The
*solver* itself should still be run with `-O` in real experiments, per the
existing note in GP_MRCPSP_CEC2024/readme.md about ParallelSimulator's
debug prints -- this file just can't be.)
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


def test_bottleneck_util_zero_before_anything_scheduled():
    """At the very first decision of a fresh build, nothing has been
    committed to any resource yet, so the bottleneck resource's utilization
    must be exactly 0 for both simulators, regardless of which activity
    ends up chosen."""
    training = _training_instance()
    for SimCls in (SerialSimulator, ParallelSimulator):
        sim = SimCls()
        first_util = []

        def choose(eligibles):
            if not first_util:
                first_util.append(sim.feature_rccp_bottleneck_util())
            act = next(iter(eligibles))
            return act, eligibles[act][0]

        sim.buildSolution(domain=training, choose=choose)
        assert first_util[0] == 0.0, f"{SimCls.__name__}: expected 0.0, got {first_util[0]}"
    print("test_bottleneck_util_zero_before_anything_scheduled passed")


def test_bottleneck_util_varies_and_stays_in_unit_interval():
    """Over the course of a real build, utilization of whatever resource is
    currently most contended should move (not be stuck at a constant) and
    always stay within [0, 1] (renewable resources are never oversubscribed
    by a feasible schedule)."""
    training = _training_instance()
    for SimCls in (SerialSimulator, ParallelSimulator):
        sim = SimCls()
        values = []

        def choose(eligibles):
            values.append(sim.feature_rccp_bottleneck_util())
            act = list(eligibles)[-1]
            return act, eligibles[act][-1]

        sim.buildSolution(domain=training, choose=choose)
        assert len(values) > 5, f"{SimCls.__name__}: too few decisions to be meaningful"
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in values), f"{SimCls.__name__}: out-of-range util in {values}"
        assert max(values) > min(values), f"{SimCls.__name__}: utilization never changed: {values}"
    print("test_bottleneck_util_varies_and_stays_in_unit_interval passed")


def test_candidate_contention_requires_mode():
    training = _training_instance()
    sim = SerialSimulator()

    def choose(eligibles):
        act = next(iter(eligibles))
        sim.cur_act, sim.cur_mode = act, None
        try:
            sim.feature_rccp_candidate_contention()
            raise AssertionError("expected ValueError when cur_mode is None")
        except ValueError:
            pass
        return act, eligibles[act][0]

    sim.buildSolution(domain=training, choose=choose)
    print("test_candidate_contention_requires_mode passed")


def test_slack_matches_discounted_cp_slack_score():
    """feature_rccp_slack should always equal
    feature_cp_slack_score() * (1 - bottleneck_utilization) at the exact
    moment it's read -- this is a direct re-check of the documented formula,
    not just an end-to-end sanity check."""
    training = _training_instance()
    sim = SerialSimulator()
    sim.rcpsp_problem = training
    checked = {"n": 0}

    def choose(eligibles):
        act = next(iter(eligibles))
        mode = eligibles[act][0]
        sim.cur_act, sim.cur_mode = act, mode
        expected = sim.feature_cp_slack_score() * (1.0 - sim.feature_rccp_bottleneck_util())
        actual = sim.feature_rccp_slack()
        assert abs(actual - expected) < 1e-9, f"{actual} != {expected}"
        checked["n"] += 1
        return act, mode

    sim.buildSolution(domain=training, choose=choose)
    assert checked["n"] > 5
    print(f"test_slack_matches_discounted_cp_slack_score passed ({checked['n']} checks)")


def test_pressure_trend_zero_when_no_eligibles():
    sim = SerialSimulator()
    sim.rcpsp_problem = _training_instance()
    sim.eligibles = {}
    assert sim.feature_rccp_pressure_trend() == 0.0
    sim.eligibles = None
    assert sim.feature_rccp_pressure_trend() == 0.0
    print("test_pressure_trend_zero_when_no_eligibles passed")


def test_resource_concentration_edge_cases_and_real_instance():
    sim = SerialSimulator()

    class _FakeProblem:
        renewable_resources_list = ["R1", "R2", "R3"]
        mode_details = {
            1: {
                1: {"R1": 8, "R2": 0, "R3": 0},  # fully concentrated
                2: {"R1": 0, "R2": 0, "R3": 0},  # zero demand
                3: {"R1": 2, "R2": 2, "R3": 4},  # spread
            }
        }

    sim.rcpsp_problem = _FakeProblem()
    sim.cur_act = 1
    sim.cur_mode = 1
    assert sim.feature_rccp_resource_concentration() == 1.0
    sim.cur_mode = 2
    assert sim.feature_rccp_resource_concentration() == 0.0
    sim.cur_mode = 3
    spread_value = sim.feature_rccp_resource_concentration()
    assert 0.0 < spread_value < 1.0

    class _FakeProblemK1:
        renewable_resources_list = ["R1"]
        mode_details = {1: {1: {"R1": 5}}}

    sim.rcpsp_problem = _FakeProblemK1()
    sim.cur_act, sim.cur_mode = 1, 1
    assert sim.feature_rccp_resource_concentration() == 1.0, "k=1 must be the fixed boundary value, not 0/0"

    sim.cur_mode = None
    try:
        sim.feature_rccp_resource_concentration()
        raise AssertionError("expected ValueError when cur_mode is None")
    except ValueError:
        pass

    # real instance, multiple resource types, just confirm no errors and a valid range
    training = _training_instance()
    sim2 = SerialSimulator()
    values = []

    def choose(eligibles):
        act = next(iter(eligibles))
        mode = eligibles[act][0]
        sim2.cur_act, sim2.cur_mode = act, mode
        values.append(sim2.feature_rccp_resource_concentration())
        return act, mode

    sim2.buildSolution(domain=training, choose=choose)
    assert len(values) > 5
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in values)
    print("test_resource_concentration_edge_cases_and_real_instance passed")


def _build_solver(simulator_type, decision_type, **extra_features):
    training = [_training_instance()]
    params = ParametersGPHH.fast(
        simulator_type=simulator_type,
        decision_type=decision_type,
        rccp_terminals_feature=True,
        **extra_features,
    )
    params.pop_size = 12
    params.n_gen = 2
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    return solver, training


def test_rccp_terminals_registered_and_evaluable_all_combinations():
    """Smoke test across every (simulator, decision_type) combination the
    default CLI supports: --rccp_terminals must produce a usable pset (at
    least one RCCP terminal registered somewhere) and every individual in a
    population must evaluate to a finite, non-negative fitness."""
    for simulator_type in (SimulatorTypeEnum.SERIAL_SGS, SimulatorTypeEnum.PARALLEL_SGS):
        for decision_type in (
            DecisionTypeEnum.ACTIVITY_THEN_MODE,
            DecisionTypeEnum.MODE_THEN_ACTIVITY,
            DecisionTypeEnum.SIMULTANEOUS,
        ):
            random.seed(0)
            solver, training = _build_solver(simulator_type, decision_type)
            rccp_names = {
                "RCCP_BOTTLENECK_UTIL",
                "RCCP_CANDIDATE_CONTENTION",
                "RCCP_SLACK",
                "RCCP_PRESSURE_TREND",
                "RCCP_RESOURCE_CONCENTRATION",
            }
            registered = set()
            for pset in solver.params_gphh.set_primitves.values():
                registered |= set(pset.mapping.keys()) & rccp_names
            assert registered, f"no RCCP terminal registered for {simulator_type}/{decision_type}"

            pop = solver.toolbox.population(n=solver.params_gphh.pop_size)
            for ind in pop:
                fitness = solver.toolbox.evaluate(individual=ind, domains=training)
                value = fitness[0]
                assert value == value, f"NaN fitness for {simulator_type}/{decision_type}"
                assert value >= 0, f"negative fitness for {simulator_type}/{decision_type}: {value}"
            print(f"  ok: {simulator_type.value}/{decision_type.value}, registered={registered}")
    print("test_rccp_terminals_registered_and_evaluable_all_combinations passed")


def test_rccp_combines_with_cp_propagation_and_nr_terminals_without_collisions():
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
    )
    params.pop_size = 12
    params.n_gen = 1
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()

    all_names = set()
    for pset in params.set_primitves.values():
        all_names |= set(pset.mapping.keys())
    expected = {
        "CP_FORWARD", "CP_BACKWARD", "CP_SLACK_SCORE", "CP_PROB",
        "NR_STOCK_RATIO", "NR_MODE_DEMAND_RATIO", "NR_BUDGET_PRESSURE",
        "RCCP_BOTTLENECK_UTIL", "RCCP_CANDIDATE_CONTENTION", "RCCP_SLACK",
        "RCCP_PRESSURE_TREND", "RCCP_RESOURCE_CONCENTRATION",
    }
    missing = expected - all_names
    assert not missing, f"missing terminals when all three extensions combined: {missing}"

    pop = solver.toolbox.population(n=params.pop_size)
    for ind in pop:
        fitness = solver.toolbox.evaluate(individual=ind, domains=training)
        assert fitness[0] == fitness[0] and fitness[0] >= 0
    print("test_rccp_combines_with_cp_propagation_and_nr_terminals_without_collisions passed")


def test_rccp_terminal_actually_used_by_evolution():
    """Not just registered -- confirm at least one individual across a
    handful of seeds/generations actually contains an RCCP terminal in its
    tree, i.e. the terminal is reachable and selectable by ramped
    half-and-half initialization, not dead weight in the pset."""
    solver, training = _build_solver(
        SimulatorTypeEnum.SERIAL_SGS, DecisionTypeEnum.ACTIVITY_THEN_MODE
    )
    rccp_names = (
        "RCCP_BOTTLENECK_UTIL", "RCCP_CANDIDATE_CONTENTION", "RCCP_SLACK", "RCCP_PRESSURE_TREND",
    )
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
        if any(any(name in str(ind) for name in rccp_names) for ind in pop):
            found = True
            break
    assert found, "no RCCP terminal appeared in any individual across 5 seeds x 3 generations"
    print("test_rccp_terminal_actually_used_by_evolution passed")


if __name__ == "__main__":
    test_bottleneck_util_zero_before_anything_scheduled()
    test_bottleneck_util_varies_and_stays_in_unit_interval()
    test_candidate_contention_requires_mode()
    test_slack_matches_discounted_cp_slack_score()
    test_pressure_trend_zero_when_no_eligibles()
    test_resource_concentration_edge_cases_and_real_instance()
    test_rccp_terminals_registered_and_evaluable_all_combinations()
    test_rccp_combines_with_cp_propagation_and_nr_terminals_without_collisions()
    test_rccp_terminal_actually_used_by_evolution()
    print("All rccp_terminals tests passed")
