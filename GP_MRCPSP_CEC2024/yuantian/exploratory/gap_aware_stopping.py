"""
Gap-aware early stopping, moved here after it didn't actually help (same
deal as heuristic_seeding.py -- a real idea I tried, not one of the
restored Phase 0 strategies, that got moved to exploratory/ once its own
before/after comparison came back negative).

Why I tried this: while looking at lexicase's train/val curves I noticed
the train->validation gap stays flat for a while and then jumps up to a
higher plateau, and training fitness keeps improving past that point
without any of it showing up on held-out data. detect_gap_onset below
tries to catch that flat-to-rising transition as it happens (from the
run's own validation curve, not some fixed generation number), and
lexicase_memetic_gp_gap_aware either rolls back to whichever generation
was best on validation, or stops the run early once it's confirmed.

Tested this twice and both times it didn't help:

1. At lexicase_local_search_experiment.py's normal settings (25 classes,
   1 train instance per class), test fitness actually came out better than
   training fitness for every condition -- so there wasn't really a gap to
   catch in the first place. proposed vs proposed_gap_aware on test
   fitness: p=0.109, not significant (if anything plain proposed was
   slightly better).

2. Reran with --known_gap_split (the split from full_mmlib_experiment.py
   that's known to produce an actual gap, ~18-19 test fitness). This time
   there really was a gap (test clearly worse than train for everyone).
   The mechanism did detect onset in most runs and rolled back to roughly
   generation 7 out of 20 on average. Still no significant difference on
   test fitness though (p=0.734), and the generalization gap barely
   moved (2.40 vs 2.37).

So the detector itself seems to work fine -- it picks up on a real,
sustained gap when one exists -- but rolling back to whatever generation
looked best on a small validation set doesn't reliably pick the
generation that'll actually do best on a separate held-out test set.
Validation fitness with only 10-25 instances is just noisy enough that
the signal isn't precise enough to act on.

Not used by gphh_solver.py or hybrid_gp.py anymore, only by
lexicase_local_search_experiment.py's "proposed_gap_aware" condition and
its own tests.
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
