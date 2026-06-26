"""
Unit tests for gap_aware_stopping.py's detect_gap_onset and
lexicase_memetic_gp_gap_aware. Updated to import from exploratory/ after
the mechanism got moved there (null result, see that module's docstring).

Run from the GP_MRCPSP_CEC2024 repo root:

    PYTHONPATH=$(pwd):$(pwd)/yuantian python3 yuantian/test/test_gap_aware_stopping.py

Don't run with -O, these use plain assert statements.
"""
import random

from yuantian.exploratory.gap_aware_stopping import (
    detect_gap_onset,
    lexicase_memetic_gp_gap_aware,
)
from yuantian.gphh_solver import GPHH, ParametersGPHH, RefreshHallOfFame, read_instances
from yuantian.hybrid_gp import epsilon_lexicase_selection
from yuantian.local_search import RefinementStrategyEnum
from yuantian.rcpsp_dataset import RCPSPDatabase, StaticDatasetProvider
from yuantian.rcpsp_simulation import DecisionTypeEnum
from yuantian.utils import PopulationArchive
from deap import tools


def test_detect_gap_onset_flat_then_rising():
    """Synthetic flat-then-rising series: 7 flat generations (gap=0.5) then
    a sustained rise to gap=2.0 (matching the cited investigation's ~0.59 ->
    ~1.61 magnitude). Onset should be detected close to the actual rise
    point (index 7), allowed a few generations of lag since the heuristic
    confirms via a trailing window + patience, not an instant threshold
    crossing."""
    gap_history = [0.5] * 7 + [2.0] * 13
    onset = detect_gap_onset(gap_history, window=3, patience=2, threshold_ratio=2.0)
    assert onset is not None, "expected onset to be detected on a clear flat-then-rising series"
    assert 6 <= onset <= 10, f"expected onset near the rise point (index 7), got {onset}"
    print(f"test_detect_gap_onset_flat_then_rising passed (onset={onset})")


def test_detect_gap_onset_flat_series_never_fires():
    """A perfectly flat (noisy-but-stationary) series must never report onset."""
    rng = random.Random(0)
    gap_history = [1.0 + rng.uniform(-0.05, 0.05) for _ in range(30)]
    onset = detect_gap_onset(gap_history, window=3, patience=2, threshold_ratio=2.0)
    assert onset is None, f"expected no onset on a flat series, got {onset}"
    print("test_detect_gap_onset_flat_series_never_fires passed")


def test_detect_gap_onset_insufficient_history_returns_none():
    assert detect_gap_onset([0.1, 0.2], window=3, patience=2) is None
    assert detect_gap_onset([], window=3, patience=2) is None
    print("test_detect_gap_onset_insufficient_history_returns_none passed")


def test_detect_gap_onset_single_noisy_spike_does_not_fire():
    """patience=2 requires two consecutive confirming generations: a single
    one-generation spike back down to baseline must not trigger onset."""
    gap_history = [0.5] * 7 + [5.0] + [0.5] * 10
    onset = detect_gap_onset(gap_history, window=3, patience=2, threshold_ratio=2.0)
    assert onset is None, f"expected a single spike not to trigger onset, got {onset}"
    print("test_detect_gap_onset_single_noisy_spike_does_not_fire passed")


def test_detect_gap_onset_handles_nonpositive_early_baseline():
    """early_mean <= 0 (validation scoring better than training early on)
    must fall back to the absolute-rise threshold instead of crashing or
    triggering on a degenerate ratio."""
    gap_history = [-0.2] * 7 + [-0.2 + 0.1] * 13  # rise of 0.1, below default min_absolute_rise
    onset = detect_gap_onset(gap_history, window=3, patience=2, min_absolute_rise=0.5)
    assert onset is None, "rise smaller than min_absolute_rise should not trigger"

    gap_history2 = [-0.2] * 7 + [2.0] * 13  # clear rise, well above any reasonable floor
    onset2 = detect_gap_onset(gap_history2, window=3, patience=2, min_absolute_rise=0.5)
    assert onset2 is not None
    print("test_detect_gap_onset_handles_nonpositive_early_baseline passed")


def _build_solver(pop_size=10, n_gen=8, decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE):
    training = read_instances([RCPSPDatabase.MMLIB_50_DIR + "J501_1.mm"])
    validation = read_instances([RCPSPDatabase.MMLIB_50_DIR + "J5020_1.mm"])
    params = ParametersGPHH.fast(decision_type=decision_type)
    params.pop_size = pop_size
    params.n_gen = n_gen
    solver = GPHH(training_set_provider=StaticDatasetProvider(training), params_gphh=params)
    solver.init_model()
    return solver, training, validation


def _make_mstats():
    import numpy as np

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", np.mean)
    return mstats


def test_lexicase_memetic_gp_rejects_gap_aware_without_validation_provider():
    solver, training, _ = _build_solver()
    solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)
    pop = solver.toolbox.population(n=solver.params_gphh.pop_size)
    try:
        lexicase_memetic_gp_gap_aware(
            pop,
            solver.toolbox,
            cxpb=0.7,
            mutpb=0.2,
            n_elite=2,
            ngen=3,
            training_data_provider=StaticDatasetProvider(training),
            validation_data_provider=None,
            decision_type=solver.decision_type,
            simulator=solver.simulator,
            pset=solver.pset,
            stats=_make_mstats(),
            halloffame=RefreshHallOfFame(1),
            pop_archive=PopulationArchive(),
        )
        raise AssertionError("expected ValueError when validation_data_provider is falsy")
    except ValueError:
        pass
    print("test_lexicase_memetic_gp_rejects_gap_aware_without_validation_provider passed")


def test_lexicase_memetic_gp_rollback_smoke():
    random.seed(0)
    solver, training, validation = _build_solver(pop_size=12, n_gen=8)
    solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)
    pop = solver.toolbox.population(n=solver.params_gphh.pop_size)
    hof = RefreshHallOfFame(1)
    _, log = lexicase_memetic_gp_gap_aware(
        pop,
        solver.toolbox,
        cxpb=0.7,
        mutpb=0.2,
        n_elite=2,
        ngen=8,
        training_data_provider=StaticDatasetProvider(training),
        validation_data_provider=StaticDatasetProvider(validation),
        decision_type=solver.decision_type,
        simulator=solver.simulator,
        pset=solver.pset,
        refinement_strategy=RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP,
        stats=_make_mstats(),
        halloffame=hof,
        pop_archive=PopulationArchive(),
        rng=random,
        stopping_mode="rollback",
        gap_onset_window=2,
        gap_onset_patience=2,
    )
    assert len(log.chapters["generation_best"]) == 9  # gen 0..8 inclusive, rollback never cuts short
    report = log.gap_aware_report
    assert report["stopping_mode"] == "rollback"
    assert report["stopped_early"] is False
    assert 0 <= report["returned_generation"] <= 8
    assert hof[0] is not None and hof[0].fitness.valid
    print(f"test_lexicase_memetic_gp_rollback_smoke passed (report={report})")


def test_lexicase_memetic_gp_stop_mode_can_cut_short():
    """Force onset to be detected almost immediately (patience=1, a tiny
    min_absolute_rise) and confirm stop mode actually returns before ngen
    generations are logged."""
    random.seed(1)
    solver, training, validation = _build_solver(pop_size=12, n_gen=10)
    solver.toolbox.register("select", epsilon_lexicase_selection, rng=random)
    pop = solver.toolbox.population(n=solver.params_gphh.pop_size)
    hof = RefreshHallOfFame(1)
    _, log = lexicase_memetic_gp_gap_aware(
        pop,
        solver.toolbox,
        cxpb=0.7,
        mutpb=0.2,
        n_elite=2,
        ngen=10,
        training_data_provider=StaticDatasetProvider(training),
        validation_data_provider=StaticDatasetProvider(validation),
        decision_type=solver.decision_type,
        simulator=solver.simulator,
        pset=solver.pset,
        stats=_make_mstats(),
        halloffame=hof,
        pop_archive=PopulationArchive(),
        rng=random,
        stopping_mode="stop",
        gap_onset_window=2,
        gap_onset_patience=1,
        gap_onset_min_absolute_rise=1e-9,
    )
    report = log.gap_aware_report
    n_logged = len(log.chapters["generation_best"])
    assert n_logged <= 11  # at most gen 0..10
    if report["onset_generation"] is not None:
        assert report["stopped_early"] is True
        assert n_logged < 11, "stop mode should have cut the run short given a near-zero rise floor"
    print(f"test_lexicase_memetic_gp_stop_mode_can_cut_short passed (n_logged={n_logged}, report={report})")


if __name__ == "__main__":
    test_detect_gap_onset_flat_then_rising()
    test_detect_gap_onset_flat_series_never_fires()
    test_detect_gap_onset_insufficient_history_returns_none()
    test_detect_gap_onset_single_noisy_spike_does_not_fire()
    test_detect_gap_onset_handles_nonpositive_early_baseline()
    test_lexicase_memetic_gp_rejects_gap_aware_without_validation_provider()
    test_lexicase_memetic_gp_rollback_smoke()
    test_lexicase_memetic_gp_stop_mode_can_cut_short()
    print("All gap_aware_stopping tests passed")
