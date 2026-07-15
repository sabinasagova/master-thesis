"""
Gap-aware early stopping / rollback for the lexicase + local-search loop.

detect_gap_onset watches the run's own train/validation trajectory for the
onset of a sustained generalisation gap; lexicase_memetic_gp_gap_aware
then either rolls back to the best-on-validation generation or stops the
run early. Evaluated twice with negative results: once on a split with no
real train/test gap (proposed vs proposed_gap_aware on test, p=0.109) and
once on the known-gap split of full_mmlib_experiment.py, where onset was
detected and rollback happened (to ~generation 7 of 20 on average) but
test fitness did not improve (p=0.734) and the gap barely moved
(2.40 vs 2.37). The validation signal (10-25 instances) is too noisy to
select the generation that generalises best.

Used only by lexicase_local_search_experiment.py's "proposed_gap_aware"
condition and its own tests.
"""
import random
from functools import partial
from typing import List, Literal, Optional, Tuple

from deap import tools

from yuantian.gp_algorithms import load_elites, varOr
from yuantian.hybrid_gp import _accumulate_move_stats, _local_search_elites
from yuantian.local_search import RefinementStrategyEnum
from yuantian.rcpsp_dataset import DatasetProvider


def detect_gap_onset(
    gap_history: List[float],
    window: int = 3,
    patience: int = 2,
    threshold_ratio: float = 2.0,
    min_absolute_rise: float = 0.1,
) -> Optional[int]:
    """Try to spot the point where the train/val gap stops being flat and
    starts climbing for good.

    The gap doesn't just spike randomly, it tends to sit flat for a while
    and then jump to a clearly higher plateau, so a single generation
    crossing some threshold isn't really enough evidence on its own (these
    curves are noisy). Instead this compares the trailing mean gap over the
    last `window` generations against the mean from the first `window`
    generations of the run, and only confirms onset once that holds for
    `patience` generations in a row.

    `threshold_ratio` (2.0 by default, so roughly "doubled") is how much
    the gap has to rise above the early baseline to count -- picked from a
    rise I measured earlier (~0.59 -> ~1.61) but rounded down since I'd
    expect the exact number to depend on pop/gen size. `min_absolute_rise`
    is a fallback floor for when the early baseline is at or below zero
    (val scoring better than train early on is possible, and a pure ratio
    doesn't really make sense there).

    Uses non-overlapping blocks of size `window` rather than a sliding
    window for the confirmation check. With a sliding window a single bad
    generation can sit inside the trailing average for `window` checks in a
    row, which would let one noisy spike alone satisfy `patience` and
    defeat the whole point of requiring patience. Non-overlapping blocks
    mean one weird generation can only pollute one block.

    Returns the generation index where the confirmed rise starts, or None
    if nothing's been confirmed yet (including just not having enough
    history -- need at least `window * patience` points).
    """
    n = len(gap_history)
    if n < window * patience:
        return None
    early_mean = float(sum(gap_history[:window]) / window)
    rise = early_mean * (threshold_ratio - 1.0) if early_mean > 0 else 0.0
    cutoff = early_mean + max(rise, min_absolute_rise)

    run_start: Optional[int] = None
    run_len = 0
    for block_start in range(0, n - window + 1, window):
        block = gap_history[block_start : block_start + window]
        block_mean = float(sum(block) / window)
        if block_mean > cutoff:
            if run_len == 0:
                run_start = block_start
            run_len += 1
            if run_len >= patience:
                return run_start
        else:
            run_len = 0
            run_start = None
    return None


def lexicase_memetic_gp_gap_aware(
    population,
    toolbox,
    cxpb: float,
    mutpb: float,
    n_elite: int,
    ngen: int,
    training_data_provider: DatasetProvider,
    validation_data_provider,
    decision_type,
    simulator,
    pset,
    elite_fraction: float = 0.08,
    local_search_iters: int = 10,
    refinement_strategy=RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP,
    stats: tools.Statistics = None,
    halloffame: tools.HallOfFame = None,
    pop_archive: list = None,
    move_stats: Optional[dict] = None,
    rng=random,
    verbose=__debug__,
    stopping_mode: Literal["stop", "rollback"] = "rollback",
    gap_onset_window: int = 3,
    gap_onset_patience: int = 2,
    gap_onset_threshold_ratio: float = 2.0,
    gap_onset_min_absolute_rise: float = 0.1,
):
    """Same generational loop as hybrid_gp.lexicase_memetic_gp, but tracks
    the train/val gap every generation and uses detect_gap_onset to either
    roll back to the best-on-validation individual or stop early.

    stopping_mode:
      - "rollback" (default): run the full ngen budget like normal, but at
        the end return whichever generation's individual was best on
        validation instead of the last generation's.
      - "stop": once onset is confirmed, stop right there and return the
        best-on-validation individual, saving the rest of the budget.

    Either way attaches logbook.gap_aware_report with onset_generation,
    validation_fitness_at_onset, validation_fitness_final,
    returned_generation, stopping_mode, and stopped_early.

    Needs a validation_data_provider -- raises ValueError without one,
    since there's no gap to track without a validation signal.
    """
    if not validation_data_provider:
        raise ValueError(
            "lexicase_memetic_gp_gap_aware requires a validation_data_provider "
            "(the gap trajectory is validation_fitness - fitness; there is "
            "no validation signal to track without it)."
        )

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    gap_history: List[float] = []
    best_on_validation: Optional[Tuple[int, object, float]] = None  # (gen, clone, val_fitness)
    onset_generation: Optional[int] = None
    stopped_early = False

    def _update_gap_tracking(gen: int, best_ind_record: dict) -> bool:
        """Append this generation's gap, update the best-on-validation
        clone, and check for confirmed onset. Returns True if "stop" mode
        should halt the loop now."""
        nonlocal best_on_validation, onset_generation
        val_fitness = best_ind_record["validation_fitness"]
        gap_history.append(val_fitness - best_ind_record["fitness"])
        if best_on_validation is None or val_fitness < best_on_validation[2]:
            best_on_validation = (gen, toolbox.clone(halloffame[0]), val_fitness)
        if onset_generation is None:
            onset_generation = detect_gap_onset(
                gap_history,
                window=gap_onset_window,
                patience=gap_onset_patience,
                threshold_ratio=gap_onset_threshold_ratio,
                min_absolute_rise=gap_onset_min_absolute_rise,
            )
        return onset_generation is not None and stopping_mode == "stop"

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    training_data = training_data_provider.next()
    evaluate = partial(toolbox.evaluate, domains=training_data)
    fitnesses = toolbox.map(evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    elites = _local_search_elites(
        population,
        elite_fraction,
        toolbox,
        training_data,
        decision_type,
        simulator,
        pset,
        local_search_iters,
        rng,
        refinement_strategy,
    )
    _accumulate_move_stats(elites, move_stats)

    pop_archive.append([toolbox.clone(ind) for ind in population])

    if halloffame is not None:
        halloffame.update(population)
    best_ind_record = {
        "fitness": halloffame[0].fitness.values[0],
        "tree": str(halloffame[0]),
    }
    validation_set = validation_data_provider.next()
    validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
    best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

    record = stats.compile(population) if stats else {}
    record["generation_best"] = best_ind_record
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    should_stop = _update_gap_tracking(0, best_ind_record)

    for gen in range(1, ngen + 1):
        if should_stop:
            stopped_early = True
            break

        offspring = toolbox.select(population, len(population) - n_elite)
        offspring = varOr(offspring, toolbox, cxpb, mutpb)
        offspring += load_elites(population, n_elite)

        training_data = training_data_provider.next()
        evaluate = partial(toolbox.evaluate, domains=training_data)
        fitnesses = toolbox.map(evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        elites = _local_search_elites(
            offspring,
            elite_fraction,
            toolbox,
            training_data,
            decision_type,
            simulator,
            pset,
            local_search_iters,
            rng,
            refinement_strategy,
        )
        _accumulate_move_stats(elites, move_stats)

        if halloffame is not None:
            halloffame.update(offspring)
        best_ind_record = {
            "fitness": halloffame[0].fitness.values[0],
            "tree": str(halloffame[0]),
        }
        validation_set = validation_data_provider.next()
        validation_evaluate = partial(toolbox.evaluate, domains=validation_set)
        best_ind_record["validation_fitness"] = validation_evaluate(halloffame[0])[0]

        population[:] = offspring
        pop_archive.append([toolbox.clone(ind) for ind in population])

        record = stats.compile(population) if stats else {}
        record["generation_best"] = best_ind_record
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        should_stop = _update_gap_tracking(gen, best_ind_record)

    returned_gen, best_clone, _ = best_on_validation
    if halloffame is not None:
        halloffame.update([best_clone])
    gen_best_log = logbook.chapters["generation_best"]
    logbook.gap_aware_report = {
        "onset_generation": onset_generation,
        "validation_fitness_at_onset": (
            gen_best_log[onset_generation]["validation_fitness"]
            if onset_generation is not None
            else None
        ),
        "validation_fitness_final": gen_best_log[-1]["validation_fitness"],
        "returned_generation": returned_gen,
        "stopping_mode": stopping_mode,
        "stopped_early": stopped_early,
    }

    return population, logbook
