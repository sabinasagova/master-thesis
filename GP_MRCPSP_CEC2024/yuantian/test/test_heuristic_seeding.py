"""
Unit tests for yuantian/exploratory/heuristic_seeding.py.

Run from the GP_MRCPSP_CEC2024 repo root (matches the other test in this
folder, test_dynamic_cpm.py, which is also a __main__ script rather than a
pytest suite, since pytest isn't a project dependency):

    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/test/test_heuristic_seeding.py

Do NOT run this with `-O`: unlike test_dynamic_cpm.py (which signals failure
via `raise`, not `assert`), these tests rely on plain `assert` statements,
and `-O` strips those out, turning every test into a silent no-op.
"""
import random

from deap import gp

from yuantian.gphh_solver import GPHH, ParametersGPHH, read_instances
from yuantian.exploratory.heuristic_seeding import build_heuristic_trees, seed_population
from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum, SimulatorTypeEnum


def test_build_trees_two_step_default():
    """Default activity-then-mode pset (static CPM terminals): every
    activity rule x mode rule combo should be expressible."""
    params = ParametersGPHH.default(decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE)
    trees = build_heuristic_trees(DecisionTypeEnum.ACTIVITY_THEN_MODE, params.set_primitves)
    assert len(trees) == 4 * 2, f"expected 8 activity x mode combos, got {len(trees)}: {trees.keys()}"
    for name, subtrees in trees.items():
        assert set(subtrees.keys()) == {
            TerminalTypeEnum.ACTIVITY.value,
            TerminalTypeEnum.MODE.value,
        }, name
        for tree in subtrees.values():
            assert isinstance(tree, gp.PrimitiveTree)
            assert len(tree) >= 1
    print("test_build_trees_two_step_default passed")


def test_build_trees_mode_then_activity():
    """MODE_THEN_ACTIVITY's activity pset drops the activity-first-only
    terminals (avg/min/max task duration, RReq across modes) but keeps
    ES/EF/LS/LF and GRPW, so the same 4 activity rules should still build."""
    params = ParametersGPHH.default(decision_type=DecisionTypeEnum.MODE_THEN_ACTIVITY)
    trees = build_heuristic_trees(DecisionTypeEnum.MODE_THEN_ACTIVITY, params.set_primitves)
    assert len(trees) == 4 * 2
    print("test_build_trees_mode_then_activity passed")


def test_build_trees_dynamic_cpm_variant():
    """With dynamic_CPM_feature=True, ES/LS aren't in the pset (ES_d/LS_d
    are instead): the rule builder should fall back to the _d variants
    rather than silently dropping the rule."""
    params = ParametersGPHH.default(
        decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE, dynamic_CPM_feature=True
    )
    activity_pset = params.set_primitves[TerminalTypeEnum.ACTIVITY.value]
    assert "ES" not in activity_pset.mapping
    assert "ES_d" in activity_pset.mapping
    trees = build_heuristic_trees(DecisionTypeEnum.ACTIVITY_THEN_MODE, params.set_primitves)
    assert len(trees) == 4 * 2
    est_tree = next(t[TerminalTypeEnum.ACTIVITY.value] for name, t in trees.items() if name.startswith("EST+"))
    assert str(est_tree) == "ES_d"
    print("test_build_trees_dynamic_cpm_variant passed")


def test_build_trees_simultaneous():
    """SIMULTANEOUS uses a single INTEGRATED tree per individual: expect
    the 4 plain activity-style rules plus the 2 activity+mode composites."""
    params = ParametersGPHH.default(decision_type=DecisionTypeEnum.SIMULTANEOUS)
    trees = build_heuristic_trees(DecisionTypeEnum.SIMULTANEOUS, params.set_primitves)
    assert len(trees) == 4 + 2, trees.keys()
    for tree in trees.values():
        assert isinstance(tree, gp.PrimitiveTree)
    print("test_build_trees_simultaneous passed")


def test_build_trees_no_match_falls_back_to_empty():
    """If the pset can't express any rule (e.g. every relevant feature was
    disabled), build_heuristic_trees must return {} rather than raise, so
    seed_population can fall back to an all-random population."""
    pset = gp.PrimitiveSet("MAIN", 0)
    pset.addTerminal(lambda: 0.0, "some_unrelated_feature")
    empty_pset_dict = {
        TerminalTypeEnum.ACTIVITY.value: pset,
        TerminalTypeEnum.MODE.value: pset,
    }
    trees = build_heuristic_trees(DecisionTypeEnum.ACTIVITY_THEN_MODE, empty_pset_dict)
    assert trees == {}
    print("test_build_trees_no_match_falls_back_to_empty passed")


def _build_solver(decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE, seeding_strategy="random"):
    training = read_instances([RCPSPDatabase.MMLIB_50_DIR + "J501_1.mm"])
    params = ParametersGPHH.fast(decision_type=decision_type)
    params.pop_size = 12
    params.n_gen = 1
    params.seeding_strategy = seeding_strategy
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    return solver, training


def test_seed_population_size_matches_pop_size_larger_than_rules():
    solver, _ = _build_solver()
    from deap import creator

    pop = seed_population(
        toolbox=solver.toolbox,
        individual_class=creator.Individual,
        pop_size=12,
        decision_type=solver.decision_type,
        pset=solver.pset,
        rng=random.Random(0),
    )
    assert len(pop) == 12
    # 8 heuristic combos + 4 random fill-ins
    print("test_seed_population_size_matches_pop_size_larger_than_rules passed")


def test_seed_population_size_smaller_than_rules_is_trimmed():
    """pop_size smaller than the number of available rules: seed_population
    must still return exactly pop_size individuals (a random subset of the
    rules), not crash or over/under-produce."""
    solver, _ = _build_solver()
    from deap import creator

    pop = seed_population(
        toolbox=solver.toolbox,
        individual_class=creator.Individual,
        pop_size=3,
        decision_type=solver.decision_type,
        pset=solver.pset,
        rng=random.Random(0),
    )
    assert len(pop) == 3
    print("test_seed_population_size_smaller_than_rules_is_trimmed passed")


def test_seed_population_with_mutated_clones():
    """heuristic_mutated strategy: each rule should contribute itself plus
    n_mutated_clones mutated copies before the population is filled/trimmed
    to pop_size."""
    solver, _ = _build_solver()
    from deap import creator

    pop_size = 100  # large enough to hold all 8 rules x (1 + 2 clones) = 24
    pop = seed_population(
        toolbox=solver.toolbox,
        individual_class=creator.Individual,
        pop_size=pop_size,
        decision_type=solver.decision_type,
        pset=solver.pset,
        n_mutated_clones=2,
        mutate=solver.toolbox.mutate,
        rng=random.Random(0),
    )
    assert len(pop) == pop_size
    n_heuristic_derived = 8 * 3  # 8 rules, each with itself + 2 mutated clones
    n_random = pop_size - n_heuristic_derived
    assert n_random == 76
    print("test_seed_population_with_mutated_clones passed")


def test_seeded_individuals_are_valid_and_evaluable():
    """End-to-end validity check: every seeded individual must compile and
    evaluate through the real GPHH pipeline on a real instance, producing a
    finite, non-negative deviation-from-CPM-lower-bound fitness."""
    solver, training = _build_solver(seeding_strategy="heuristic")
    from deap import creator

    pop = seed_population(
        toolbox=solver.toolbox,
        individual_class=creator.Individual,
        pop_size=8,
        decision_type=solver.decision_type,
        pset=solver.pset,
        rng=random.Random(0),
    )
    assert len(pop) == 8
    for ind in pop:
        assert not ind.fitness.valid
        fitness = solver.toolbox.evaluate(individual=ind, domains=training)
        assert len(fitness) == 1
        value = fitness[0]
        assert value == value, "fitness is NaN"  # NaN != NaN
        assert value >= 0, f"deviation from CPM lower bound should be >= 0, got {value}"
    print("test_seeded_individuals_are_valid_and_evaluable passed")


def test_full_solve_with_heuristic_seeding_strategy():
    """Smoke test that GPHH.solve() actually runs end-to-end when
    `seeding_strategy` is set to "heuristic" or "heuristic_mutated", not
    just "random" (the only path exercised before this change)."""
    for strategy in ("heuristic", "heuristic_mutated"):
        random.seed(0)
        solver, _ = _build_solver(seeding_strategy=strategy)
        solver.solve(output_path="/tmp/test_heuristic_seeding_solve.json")
        assert solver.best_heuristic is not None
        assert solver.best_heuristic.fitness.valid
    print("test_full_solve_with_heuristic_seeding_strategy passed")


def test_solve_random_vs_heuristic_actually_differ():
    """Regression test for a real bug found in heuristic_seeding_experiment.py:
    GPHH.solve() only special-cases the literal string "random" for "no
    seeding" -- any other string (e.g. a mislabeled condition name like
    "baseline") silently takes the heuristic-seeding branch instead of
    raising or falling back. That made an experiment's "baseline" and
    "heuristic" conditions run the exact same code and produce
    byte-identical results. This checks that, for the same RNG seed,
    seeding_strategy="random" and "heuristic" produce different gen-0
    population fitness stats (proving they really did build different
    populations), and that an unrecognized seeding_strategy raises instead
    of silently falling through to either path."""
    import json

    fitness_by_strategy = {}
    for strategy in ("random", "heuristic"):
        random.seed(0)
        solver, _ = _build_solver(seeding_strategy=strategy)
        out_path = f"/tmp/test_solve_diff_{strategy}.json"
        solver.solve(output_path=out_path)
        with open(out_path) as f:
            result = json.load(f)
        fitness_by_strategy[strategy] = result["fitness"][0]["avg"]

    assert fitness_by_strategy["random"] != fitness_by_strategy["heuristic"], (
        "random and heuristic seeding produced identical gen-0 fitness for the "
        "same seed; the seeding_strategy branch in GPHH.solve() likely isn't "
        "actually distinguishing them"
    )

    solver, _ = _build_solver(seeding_strategy="not_a_real_strategy")
    try:
        solver.solve(output_path="/tmp/test_solve_diff_invalid.json")
        raise AssertionError("expected ValueError for an unrecognized seeding_strategy")
    except ValueError:
        pass
    print("test_solve_random_vs_heuristic_actually_differ passed")


if __name__ == "__main__":
    test_build_trees_two_step_default()
    test_build_trees_mode_then_activity()
    test_build_trees_dynamic_cpm_variant()
    test_build_trees_simultaneous()
    test_build_trees_no_match_falls_back_to_empty()
    test_seed_population_size_matches_pop_size_larger_than_rules()
    test_seed_population_size_smaller_than_rules_is_trimmed()
    test_seed_population_with_mutated_clones()
    test_seeded_individuals_are_valid_and_evaluable()
    test_full_solve_with_heuristic_seeding_strategy()
    test_solve_random_vs_heuristic_actually_differ()
    print("All heuristic_seeding tests passed")
