"""
Unit tests for the refinement-strategy split in yuantian/local_search.py
(LOCAL_SEARCH_NO_CP, CRITICAL_PATH_ONLY, LOCAL_SEARCH_WITH_CP) and the
BASELINE no-op wired through yuantian/hybrid_gp.py's `_local_search_elites`.

Run from the GP_MRCPSP_CEC2024 repo root (matches test_dynamic_cpm.py and
test_heuristic_seeding.py: a __main__ script with assertions, since pytest
isn't a project dependency):

    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/test/test_local_search_variants.py

Do NOT run this with `-O`: these tests rely on plain `assert` statements,
and `-O` strips those out, turning every test into a silent no-op.
"""
import random

import yuantian.local_search as local_search
from yuantian.gphh_solver import GPHH, ParametersGPHH, read_instances
from yuantian.hybrid_gp import _local_search_elites
from yuantian.local_search import (
    RefinementStrategyEnum,
    _ALL_MOVE_TYPES,
    _decode,
    _hill_climb,
    apply_local_search_to_elite,
    build_heuristic_func,
    critical_path_construct,
    critical_path_repair,
)
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum


def _reference_critical_path_repair(domain, base_solution, max_iters=10, rng=random):
    """Verbatim copy of the pre-refactor `critical_path_repair` (mode + swap
    + resource_shift, no shared `_hill_climb` core), used only to check that
    refactoring into `_hill_climb` didn't change LOCAL_SEARCH_WITH_CP's
    behavior for a given RNG seed."""
    from yuantian.local_search import _extract_permutation_and_modes

    permutation, modes = _extract_permutation_and_modes(domain, base_solution)
    tasks_non_dummy = domain.tasks_list_non_dummy
    current_solution, current_makespan = _decode(domain, permutation, modes)
    if current_solution is None:
        return base_solution

    for _ in range(max_iters):
        move_type = rng.choice(["mode", "swap", "resource_shift"])
        candidate_perm, candidate_modes = list(permutation), list(modes)

        if move_type == "mode":
            idx = rng.randrange(len(tasks_non_dummy))
            task = tasks_non_dummy[idx]
            alt_modes = [m for m in domain.mode_details[task] if m != modes[idx]]
            if not alt_modes:
                continue
            candidate_modes[idx] = rng.choice(alt_modes)
        elif move_type == "swap":
            i, j = rng.sample(range(len(permutation)), 2)
            task_i = tasks_non_dummy[permutation[i]]
            task_j = tasks_non_dummy[permutation[j]]
            successors_i = domain.graph.full_successors.get(task_i, set())
            successors_j = domain.graph.full_successors.get(task_j, set())
            if task_j in successors_i or task_i in successors_j:
                continue
            candidate_perm[i], candidate_perm[j] = candidate_perm[j], candidate_perm[i]
        else:
            idx = rng.randrange(len(permutation))
            task = tasks_non_dummy[permutation[idx]]
            slack = domain.cpm[task]._LSD - domain.cpm[task]._ESD
            if slack <= 0 or idx >= len(permutation) - 1:
                continue
            shift = rng.randint(1, min(3, len(permutation) - 1 - idx))
            candidate_perm.insert(idx + shift, candidate_perm.pop(idx))

        candidate_solution, candidate_makespan = _decode(domain, candidate_perm, candidate_modes)
        if candidate_solution is not None and candidate_makespan < current_makespan:
            permutation, modes = candidate_perm, candidate_modes
            current_solution, current_makespan = candidate_solution, candidate_makespan

    return current_solution


class _ChoiceTrackingRandom(random.Random):
    """Records only the move-type draws (`rng.choice(move_types)`), not the
    other `.choice()` calls `_hill_climb` makes inside the "mode" branch to
    pick an alternate mode value -- those are always ints, move types are
    always one of "mode"/"swap"/"resource_shift", so the two are
    distinguishable by the candidate sequence's contents."""

    _MOVE_TYPE_NAMES = {"mode", "swap", "resource_shift"}

    def __init__(self, seed):
        super().__init__(seed)
        self.choices_made = []

    def choice(self, seq):
        picked = super().choice(seq)
        if set(seq) <= self._MOVE_TYPE_NAMES:
            self.choices_made.append(picked)
        return picked


def _build_solver_and_domain(decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE):
    # J5020_1.mm: the (permutation, modes) round-trip through
    # _extract_permutation_and_modes + _decode is only feasible for *some*
    # individuals on a given instance (non-renewable resource feasibility in
    # discrete_optimization's serial-SGS decoder is schedule-order
    # sensitive); this instance/seed combination reliably gives a feasible
    # round-trip so the hill-climbing loop actually runs instead of
    # short-circuiting at iteration 0 -- a property of the decoder, not of
    # the refinement-strategy split under test here.
    training = read_instances([RCPSPDatabase.MMLIB_50_DIR + "J5020_1.mm"])
    params = ParametersGPHH.fast(decision_type=decision_type)
    params.pop_size = 8
    params.n_gen = 1
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    return solver, training[0]


def _build_base_solution(solver, domain, seed=4):
    from deap import creator

    random.seed(seed)
    ind = solver.toolbox.individual()
    ind.fitness.values = solver.toolbox.evaluate(individual=ind, domains=[domain])
    heuristic_func = build_heuristic_func(
        ind, solver.toolbox.compile, solver.pset, solver.decision_type, solver.simulator
    )
    base_solution = solver.simulator.buildSolution(domain=domain, choose=heuristic_func)
    return ind, base_solution


def test_decode_uses_fast_false_not_the_buggy_default():
    """Regression test for a real bug found while building this module:
    `RCPSPSolution` defaults to `fast=True`, which routes through
    discrete_optimization's numba-accelerated SGS (`problem.func_sgs`). That
    decoder disagrees with the reference pure-Python one
    (`generate_schedule_from_permutation_serial_sgs`, `fast=False`) on this
    renewable-only-converted model: round-tripping a demonstrably feasible
    schedule through fast=True came back infeasible 99.6% of the time
    (1/250 sampled elites) in manual testing, vs. 100% (250/250) with
    fast=False. `_decode` must keep passing `fast=False` explicitly, or
    every refinement strategy quietly degrades to a near-total no-op (it
    always falls back to the unrefined base schedule when the round-trip
    looks infeasible).

    J501_1.mm with this exact seed/individual is one of the cases that
    failed under fast=True in that sampling, so it pins the fix directly
    rather than relying on a fixture that happened to be in the lucky 0.4%.
    """
    from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution

    training = read_instances([RCPSPDatabase.MMLIB_50_DIR + "J501_1.mm"])
    domain = training[0]
    params = ParametersGPHH.fast(decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE)
    params.pop_size = 8
    params.n_gen = 1
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    random.seed(0)
    ind = solver.toolbox.individual()
    heuristic_func = build_heuristic_func(
        ind, solver.toolbox.compile, solver.pset, solver.decision_type, solver.simulator
    )
    base_solution = solver.simulator.buildSolution(domain=domain, choose=heuristic_func)
    assert base_solution.rcpsp_schedule_feasible

    from yuantian.local_search import _extract_permutation_and_modes

    permutation, modes = _extract_permutation_and_modes(domain, base_solution)
    fast_true = RCPSPSolution(
        problem=domain, rcpsp_permutation=permutation, rcpsp_modes=modes, fast=True
    )
    assert not fast_true.rcpsp_schedule_feasible, (
        "expected this known case to still expose the fast=True/fast=False mismatch; "
        "if discrete_optimization fixed it upstream, this assertion (not the fix in "
        "_decode) is now the stale part"
    )

    solution, makespan = _decode(domain, permutation, modes)
    assert solution is not None and solution.rcpsp_schedule_feasible
    assert makespan == base_solution.get_end_time(domain.sink_task)
    print("test_decode_uses_fast_false_not_the_buggy_default passed")


def test_hill_climb_no_cp_never_uses_resource_shift():
    solver, domain = _build_solver_and_domain()
    _, base_solution = _build_base_solution(solver, domain)
    tracking_rng = _ChoiceTrackingRandom(0)
    solution, n_attempted, n_accepted = _hill_climb(
        domain, base_solution, move_types=("mode", "swap"), max_iters=200, rng=tracking_rng
    )
    assert solution is not None
    assert solution.rcpsp_schedule_feasible
    assert "resource_shift" not in tracking_rng.choices_made
    assert set(tracking_rng.choices_made) <= {"mode", "swap"}
    assert n_accepted <= n_attempted <= 200
    print("test_hill_climb_no_cp_never_uses_resource_shift passed")


def test_hill_climb_with_cp_can_use_resource_shift():
    """Sanity check that the full move set is actually reachable (otherwise
    test_hill_climb_no_cp_never_uses_resource_shift would pass vacuously for
    the wrong reason)."""
    solver, domain = _build_solver_and_domain()
    _, base_solution = _build_base_solution(solver, domain)
    tracking_rng = _ChoiceTrackingRandom(0)
    _hill_climb(
        domain, base_solution, move_types=_ALL_MOVE_TYPES, max_iters=200, rng=tracking_rng
    )
    assert "resource_shift" in tracking_rng.choices_made
    print("test_hill_climb_with_cp_can_use_resource_shift passed")


def test_local_search_with_cp_matches_pre_refactor_reference():
    """Refactor regression check: critical_path_repair (now backed by the
    shared _hill_climb core) must produce the exact same result as the
    verbatim pre-refactor implementation, for the same seed."""
    solver, domain = _build_solver_and_domain()
    _, base_solution = _build_base_solution(solver, domain)

    refactored = critical_path_repair(domain, base_solution, max_iters=50, rng=random.Random(42))
    reference = _reference_critical_path_repair(
        domain, base_solution, max_iters=50, rng=random.Random(42)
    )
    assert refactored.get_end_time(domain.sink_task) == reference.get_end_time(domain.sink_task)
    assert refactored.rcpsp_permutation == reference.rcpsp_permutation
    assert refactored.rcpsp_modes == reference.rcpsp_modes
    print("test_local_search_with_cp_matches_pre_refactor_reference passed")


def test_critical_path_construct_is_deterministic_and_feasible():
    solver, domain = _build_solver_and_domain()
    first = critical_path_construct(domain)
    second = critical_path_construct(domain)
    assert first is not None and second is not None
    assert first.rcpsp_schedule_feasible
    assert first.rcpsp_permutation == second.rcpsp_permutation
    assert first.rcpsp_modes == second.rcpsp_modes
    # ordering must actually be ascending CPM slack
    ordered_tasks = [domain.tasks_list_non_dummy[i] for i in first.rcpsp_permutation]
    slacks = [domain.cpm[t]._LSD - domain.cpm[t]._ESD for t in ordered_tasks]
    assert slacks == sorted(slacks)
    print("test_critical_path_construct_is_deterministic_and_feasible passed")


def test_apply_local_search_to_elite_all_strategies_feasible():
    solver, domain = _build_solver_and_domain()
    for strategy in (
        RefinementStrategyEnum.LOCAL_SEARCH_NO_CP,
        RefinementStrategyEnum.CRITICAL_PATH_ONLY,
        RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP,
    ):
        ind, _ = _build_base_solution(solver, domain)
        apply_local_search_to_elite(
            ind,
            [domain],
            solver.toolbox.compile,
            solver.pset,
            solver.decision_type,
            solver.simulator,
            max_iters=20,
            rng=random.Random(1),
            strategy=strategy,
        )
        assert len(ind.case_fitness) == 1
        assert ind.case_fitness[0] >= -1e-6, (strategy, ind.case_fitness)
        assert ind.fitness.values[0] >= -1e-6
        assert "attempted" in ind.local_search_moves and "accepted" in ind.local_search_moves
        if strategy == RefinementStrategyEnum.CRITICAL_PATH_ONLY:
            assert ind.local_search_moves == {"attempted": 0, "accepted": 0}
    print("test_apply_local_search_to_elite_all_strategies_feasible passed")


def test_apply_local_search_to_elite_rejects_baseline():
    solver, domain = _build_solver_and_domain()
    ind, _ = _build_base_solution(solver, domain)
    try:
        apply_local_search_to_elite(
            ind,
            [domain],
            solver.toolbox.compile,
            solver.pset,
            solver.decision_type,
            solver.simulator,
            strategy=RefinementStrategyEnum.BASELINE,
        )
        raise AssertionError("expected ValueError for strategy=BASELINE")
    except ValueError:
        pass
    print("test_apply_local_search_to_elite_rejects_baseline passed")


def test_local_search_elites_baseline_is_noop():
    solver, domain = _build_solver_and_domain()
    population = [solver.toolbox.individual() for _ in range(4)]
    for ind in population:
        ind.fitness.values = solver.toolbox.evaluate(individual=ind, domains=[domain])
    fitness_before = [ind.fitness.values[0] for ind in population]

    elites = _local_search_elites(
        population,
        elite_fraction=0.5,
        toolbox=solver.toolbox,
        training_data=[domain],
        decision_type=solver.decision_type,
        simulator=solver.simulator,
        pset=solver.pset,
        local_search_iters=20,
        rng=random.Random(0),
        refinement_strategy=RefinementStrategyEnum.BASELINE,
    )
    assert elites == []
    assert [ind.fitness.values[0] for ind in population] == fitness_before
    print("test_local_search_elites_baseline_is_noop passed")


def test_critical_path_construct_failure_is_flagged_not_silent():
    """If the one-shot construction can't find a feasible schedule, the
    elite must fall back to its own (unrefined) GP-rule schedule and the
    failure must be recorded on the individual, not swallowed."""
    solver, domain = _build_solver_and_domain()
    ind, _ = _build_base_solution(solver, domain)
    assert not hasattr(ind, "critical_path_construct_failed")

    original = local_search.critical_path_construct
    local_search.critical_path_construct = lambda domain: None
    try:
        apply_local_search_to_elite(
            ind,
            [domain],
            solver.toolbox.compile,
            solver.pset,
            solver.decision_type,
            solver.simulator,
            strategy=RefinementStrategyEnum.CRITICAL_PATH_ONLY,
        )
    finally:
        local_search.critical_path_construct = original

    assert ind.critical_path_construct_failed is True
    assert len(ind.case_fitness) == 1
    print("test_critical_path_construct_failure_is_flagged_not_silent passed")


if __name__ == "__main__":
    test_decode_uses_fast_false_not_the_buggy_default()
    test_hill_climb_no_cp_never_uses_resource_shift()
    test_hill_climb_with_cp_can_use_resource_shift()
    test_local_search_with_cp_matches_pre_refactor_reference()
    test_critical_path_construct_is_deterministic_and_feasible()
    test_apply_local_search_to_elite_all_strategies_feasible()
    test_apply_local_search_to_elite_rejects_baseline()
    test_local_search_elites_baseline_is_noop()
    test_critical_path_construct_failure_is_flagged_not_silent()
    print("All local_search variant tests passed")
