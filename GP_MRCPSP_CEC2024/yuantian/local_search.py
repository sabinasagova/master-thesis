"""
Critical-path-based local search for elite GPHH individuals.

Local search is a Lamarckian, fitness-level refinement: it improves the
*schedule* an elite individual's priority rule produces, then writes the
improved fitness back onto that individual. The GP tree (genotype) is never
modified, satisfying the "representation stays the same" constraint.

Candidate schedules are decoded with discrete_optimization's own
(permutation, modes) serial-SGS decoder (`RCPSPSolution`), so feasibility and
makespan are always computed by that single canonical decoder, independent
of which simulator (serial/parallel) produced the starting schedule.
"""
import random
from functools import partial

from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from yuantian.multitreegp import TerminalTypeEnum
from yuantian.rcpsp_simulation import DecisionTypeEnum


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
    solution = RCPSPSolution(
        problem=domain, rcpsp_permutation=permutation, rcpsp_modes=modes
    )
    if not solution.rcpsp_schedule_feasible:
        return None, None
    return solution, solution.get_end_time(domain.sink_task)


def critical_path_repair(
    domain, base_solution: RCPSPSolution, max_iters: int = 10, rng=random
) -> RCPSPSolution:
    """(B)+(C) Greedy hill-climbing local search on the (permutation, modes)
    representation. At each iteration one of three moves is proposed:
      - mode improvement: try an alternate execution mode for one activity
      - local swap: swap two activities not linked by a precedence relation
      - resource smoothing: shift a slack-rich activity later in priority
        order, to relieve resource contention with its neighbors
    A move is accepted only if it strictly reduces the decoded makespan.
    """
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
                continue  # precedence-constrained pair: not a valid swap
            candidate_perm[i], candidate_perm[j] = candidate_perm[j], candidate_perm[i]

        else:  # resource_shift
            idx = rng.randrange(len(permutation))
            task = tasks_non_dummy[permutation[idx]]
            slack = domain.cpm[task]._LSD - domain.cpm[task]._ESD
            if slack <= 0 or idx >= len(permutation) - 1:
                continue  # only shift activities with positive total slack
            shift = rng.randint(1, min(3, len(permutation) - 1 - idx))
            candidate_perm.insert(idx + shift, candidate_perm.pop(idx))

        candidate_solution, candidate_makespan = _decode(
            domain, candidate_perm, candidate_modes
        )
        if candidate_solution is not None and candidate_makespan < current_makespan:
            permutation, modes = candidate_perm, candidate_modes
            current_solution, current_makespan = candidate_solution, candidate_makespan

    return current_solution


def apply_local_search_to_elite(
    individual,
    domains,
    compile_func,
    pset,
    decision_type,
    simulator,
    max_iters: int = 10,
    rng=random,
):
    """(D) Reinsert improved individuals: rebuild the elite's GP-rule schedule
    per training instance, locally improve it, and overwrite the individual's
    fitness/case_fitness with the (non-increasing) resulting deviation scores.
    The GP tree itself is never touched.
    """
    heuristic_func = build_heuristic_func(
        individual, compile_func, pset, decision_type, simulator
    )
    case_values = []
    for domain in domains:
        base_solution = simulator.buildSolution(domain=domain, choose=heuristic_func)
        improved_solution = critical_path_repair(
            domain, base_solution, max_iters=max_iters, rng=rng
        )
        makespan = improved_solution.get_end_time(domain.sink_task)
        case_values.append((makespan - domain.cpm_esd) * 100 / domain.cpm_esd)
    individual.case_fitness = case_values
    individual.fitness.values = (sum(case_values) / len(case_values),)
    return individual
