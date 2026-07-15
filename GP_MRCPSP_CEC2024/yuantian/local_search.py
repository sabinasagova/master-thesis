"""
Elite-refinement strategies for GPHH individuals.

Refinement is a Baldwinian, fitness-level step: it improves the *schedule*
an elite individual's priority rule produces, then writes the improved
fitness back onto that individual. The GP tree (genotype) is never
modified, satisfying the "representation stays the same" constraint.

Candidate schedules are decoded with discrete_optimization's own
(permutation, modes) serial-SGS decoder (`RCPSPSolution`), so feasibility and
makespan are always computed by that single canonical decoder, independent
of which simulator (serial/parallel) produced the starting schedule.

Four refinement strategies (`RefinementStrategyEnum`), selectable with no
other code changes to the GA:
  - BASELINE: no refinement; elites keep whatever fitness the GP rule's own
    schedule already got. Handled by callers (`hybrid_gp._local_search_elites`)
    skipping refinement entirely, not by anything in this module.
  - LOCAL_SEARCH_NO_CP: greedy hill-climbing with the mode and swap moves
    only. No CPM/slack dependency.
  - CRITICAL_PATH_ONLY: one-shot constructive heuristic, not a search. Orders
    activities by ascending CPM slack (minimum-slack-first) and decodes that
    priority list exactly once. Reuses `domain.cpm`, computed once per
    instance in `gphh_solver.read_instances`, rather than recomputing slack.
  - LOCAL_SEARCH_WITH_CP: the original behavior (mode + swap + resource_shift,
    where resource_shift is the CPM-aware move using `domain.cpm[task]._LSD/
    _ESD`), unchanged.

LOCAL_SEARCH_NO_CP and LOCAL_SEARCH_WITH_CP share one hill-climbing core
(`_hill_climb`) parameterized by which moves are eligible, so the
accept/reject and decoding logic can't drift apart between them; only their
`move_types` differ.
"""
import random
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple

from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_simulation import DecisionTypeEnum


class RefinementStrategyEnum(Enum):
    BASELINE = "baseline"
    LOCAL_SEARCH_NO_CP = "local_search_no_cp"
    CRITICAL_PATH_ONLY = "critical_path_only"
    LOCAL_SEARCH_WITH_CP = "local_search_with_cp"


def build_heuristic_func(individual, compile_func, pset, decision_type, simulator):
    """Mirrors the heuristic_func construction in gphh_solver.evaluate_heuristic."""
    if decision_type == DecisionTypeEnum.ACTIVITY_THEN_MODE:
        return partial(
            simulator.activity_first_choose,
            priority_func=compile_func(
                expr=individual[TerminalTypeEnum.ACTIVITY.value],
                pset=pset[TerminalTypeEnum.ACTIVITY.value],
            ),
            mode_func=compile_func(
                expr=individual[TerminalTypeEnum.MODE.value],
                pset=pset[TerminalTypeEnum.MODE.value],
            ),
            priority_extre="min",
            mode_extre="min",
        )
    elif decision_type == DecisionTypeEnum.MODE_THEN_ACTIVITY:
        return partial(
            simulator.mode_first_choose,
            priority_func=compile_func(
                expr=individual[TerminalTypeEnum.ACTIVITY.value],
                pset=pset[TerminalTypeEnum.ACTIVITY.value],
            ),
            mode_func=compile_func(
                expr=individual[TerminalTypeEnum.MODE.value],
                pset=pset[TerminalTypeEnum.MODE.value],
            ),
            priority_extre="min",
            mode_extre="min",
        )
    else:
        return partial(
            simulator.together,
            priority_func=compile_func(
                expr=individual, pset=pset[TerminalTypeEnum.INTEGRATED.value]
            ),
            mode_func=None,
            priority_extre="min",
            mode_extre="min",
        )


def _extract_permutation_and_modes(domain, solution: RCPSPSolution):
    """(A) Schedule extraction: read off a (permutation, modes) representation
    from the schedule the GP rule produced."""
    sorted_tasks = sorted(
        (t for t in solution.rcpsp_schedule if t in domain.index_task_non_dummy),
        key=lambda t: solution.rcpsp_schedule[t]["start_time"],
    )
    permutation = [domain.index_task_non_dummy[t] for t in sorted_tasks]
    modes = list(solution.rcpsp_modes)
    if len(modes) != domain.n_jobs_non_dummy:
        modes = [1 for _ in range(domain.n_jobs_non_dummy)]
    return permutation, modes


def _decode(domain, permutation, modes):
    # fast=False: RCPSPSolution's default fast=True path routes through
    # discrete_optimization's numba-accelerated SGS (`problem.func_sgs`),
    # which disagrees with the reference pure-Python decoder on this
    # renewable-only-converted model -- empirically, round-tripping a
    # demonstrably feasible schedule through fast=True came back infeasible
    # 99.6% of the time (1/250 sampled elites), vs. 100% (250/250) with
    # fast=False. fast=False is slower but correct; every call site in this
    # module goes through here, so the fix applies uniformly.
    solution = RCPSPSolution(
        problem=domain, rcpsp_permutation=permutation, rcpsp_modes=modes, fast=False
    )
    if not solution.rcpsp_schedule_feasible:
        return None, None
    return solution, solution.get_end_time(domain.sink_task)


_ALL_MOVE_TYPES = ("mode", "swap", "resource_shift")


def _hill_climb(
    domain,
    base_solution: RCPSPSolution,
    move_types: Sequence[str],
    max_iters: int = 10,
    rng=random,
) -> Tuple[RCPSPSolution, int, int]:
    """(B)+(C) Greedy hill-climbing local search on the (permutation, modes)
    representation, shared by LOCAL_SEARCH_NO_CP and LOCAL_SEARCH_WITH_CP.
    At each iteration one of `move_types` is proposed:
      - "mode": try an alternate execution mode for one activity
      - "swap": swap two activities not linked by a precedence relation
      - "resource_shift" (CPM-aware; only valid if included in `move_types`):
        shift a slack-rich activity later in priority order, to relieve
        resource contention with its neighbors
    A move is accepted only if it strictly reduces the decoded makespan.

    Returns (best_solution_found, n_attempted, n_accepted), where
    "attempted" counts proposals that passed their validity check and were
    actually decoded (excludes e.g. a "mode" proposal with no alternate mode,
    or a "swap" of a precedence-linked pair, which are skipped before
    decoding), and "accepted" counts those that strictly improved the
    makespan.
    """
    permutation, modes = _extract_permutation_and_modes(domain, base_solution)
    tasks_non_dummy = domain.tasks_list_non_dummy
    current_solution, current_makespan = _decode(domain, permutation, modes)
    if current_solution is None:
        return base_solution, 0, 0

    n_attempted = 0
    n_accepted = 0
    for _ in range(max_iters):
        move_type = rng.choice(move_types)
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
                continue  # precedence-constrained pair: not a valid swap
            candidate_perm[i], candidate_perm[j] = candidate_perm[j], candidate_perm[i]

        else:  # resource_shift (CPM-aware; only reachable if move_types includes it)
            idx = rng.randrange(len(permutation))
            task = tasks_non_dummy[permutation[idx]]
            slack = domain.cpm[task]._LSD - domain.cpm[task]._ESD
            if slack <= 0 or idx >= len(permutation) - 1:
                continue  # only shift activities with positive total slack
            shift = rng.randint(1, min(3, len(permutation) - 1 - idx))
            candidate_perm.insert(idx + shift, candidate_perm.pop(idx))

        n_attempted += 1
        candidate_solution, candidate_makespan = _decode(
            domain, candidate_perm, candidate_modes
        )
        if candidate_solution is not None and candidate_makespan < current_makespan:
            permutation, modes = candidate_perm, candidate_modes
            current_solution, current_makespan = candidate_solution, candidate_makespan
            n_accepted += 1

    return current_solution, n_attempted, n_accepted


def critical_path_repair(
    domain, base_solution: RCPSPSolution, max_iters: int = 10, rng=random
) -> RCPSPSolution:
    """LOCAL_SEARCH_WITH_CP as a standalone call, kept for backward
    compatibility with the pre-refactor signature: mode + swap +
    resource_shift, dropping the move-count bookkeeping `_hill_climb`
    exposes to callers that don't need it."""
    solution, _, _ = _hill_climb(
        domain, base_solution, move_types=_ALL_MOVE_TYPES, max_iters=max_iters, rng=rng
    )
    return solution


def critical_path_construct(domain) -> Optional[RCPSPSolution]:
    """CRITICAL_PATH_ONLY: a one-shot constructive heuristic, not a search.

    Orders every non-dummy activity by ascending total slack (LSD - ESD)
    from `domain.cpm` -- the same CPM computation `gphh_solver.read_instances`
    already attaches to the instance once, reused here rather than
    recomputed -- giving a minimum-slack-first priority list. CPM slack
    doesn't speak to mode choice, so each activity separately gets the mode
    minimizing its total resource requirement (ties broken by shortest
    duration): picking every activity's fastest mode instead reliably
    overloads shared resources and blows the schedule horizon (empirically,
    0/6 sampled MMLIB50 instances decoded feasibly with mode 1 everywhere,
    vs. 4/6 with this rule). That priority list is decoded exactly once with
    the same canonical (permutation, modes) decoder the hill-climbing
    variants use; there is no iterative improvement loop and no retry.

    Returns None if that single decode is infeasible. Callers must surface
    this rather than silently substituting another schedule, since this
    strategy has no repair step of its own.
    """
    tasks_non_dummy: List = domain.tasks_list_non_dummy
    ordered_tasks = sorted(
        tasks_non_dummy, key=lambda t: domain.cpm[t]._LSD - domain.cpm[t]._ESD
    )
    permutation = [domain.index_task_non_dummy[t] for t in ordered_tasks]

    def _total_resource_requirement(task, mode):
        return sum(
            domain.mode_details[task][mode].get(res, 0) for res in domain.resources_list
        )

    modes = [
        min(
            domain.mode_details[task],
            key=lambda m: (_total_resource_requirement(task, m), domain.mode_details[task][m]["duration"]),
        )
        for task in tasks_non_dummy
    ]
    solution, _ = _decode(domain, permutation, modes)
    return solution


def apply_local_search_to_elite(
    individual,
    domains,
    compile_func,
    pset,
    decision_type,
    simulator,
    max_iters: int = 10,
    rng=random,
    strategy: RefinementStrategyEnum = RefinementStrategyEnum.LOCAL_SEARCH_WITH_CP,
):
    """(D) Reinsert improved individuals: rebuild the elite's GP-rule schedule
    per training instance, refine it according to `strategy`, and overwrite
    the individual's fitness/case_fitness with the (non-increasing, for the
    hill-climbing strategies) resulting deviation scores. The GP tree itself
    is never touched.

    Also records, on the individual:
      - `local_search_moves`: {"attempted": int, "accepted": int} move
        counts, summed across `domains`. Always 0/0 for CRITICAL_PATH_ONLY
        (it has no move loop).
      - `critical_path_construct_failed`: True if, for CRITICAL_PATH_ONLY,
        any domain's one-shot construction was infeasible and this elite's
        schedule for that domain fell back to the GP rule's own (unrefined)
        schedule instead. Not set otherwise.

    `strategy` must not be RefinementStrategyEnum.BASELINE: "no refinement"
    means not calling this function at all (see `hybrid_gp._local_search_elites`),
    not a code path within it.
    """
    if strategy == RefinementStrategyEnum.BASELINE:
        raise ValueError(
            "apply_local_search_to_elite should not be called for BASELINE; "
            "callers should skip refinement entirely instead."
        )
    heuristic_func = build_heuristic_func(
        individual, compile_func, pset, decision_type, simulator
    )
    case_values = []
    n_attempted_total = 0
    n_accepted_total = 0
    for domain in domains:
        base_solution = simulator.buildSolution(domain=domain, choose=heuristic_func)
        if strategy == RefinementStrategyEnum.CRITICAL_PATH_ONLY:
            constructed = critical_path_construct(domain)
            if constructed is None:
                individual.critical_path_construct_failed = True
                improved_solution = base_solution
            else:
                improved_solution = constructed
        else:
            move_types = (
                ("mode", "swap")
                if strategy == RefinementStrategyEnum.LOCAL_SEARCH_NO_CP
                else _ALL_MOVE_TYPES
            )
            improved_solution, n_attempted, n_accepted = _hill_climb(
                domain, base_solution, move_types=move_types, max_iters=max_iters, rng=rng
            )
            n_attempted_total += n_attempted
            n_accepted_total += n_accepted
        makespan = improved_solution.get_end_time(domain.sink_task)
        case_values.append((makespan - domain.cpm_esd) * 100 / domain.cpm_esd)
    individual.case_fitness = case_values
    individual.fitness.values = (sum(case_values) / len(case_values),)
    individual.local_search_moves = {
        "attempted": n_attempted_total,
        "accepted": n_accepted_total,
    }
    return individual
