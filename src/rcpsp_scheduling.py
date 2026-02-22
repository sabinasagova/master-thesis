from typing import Dict, List, Tuple, Set, Optional
import random
import sys

from psplib import parse, ProjectInstance


def build_predecessors(instance: ProjectInstance) -> Dict[int, Set[int]]:
    """
    Build a predecessor set for each activity from a psplib.ProjectInstance.

    Returns:
        predecessors[j]: set of activity indices that are direct predecessors of j.
    """
    n = instance.num_activities
    predecessors: Dict[int, Set[int]] = {j: set() for j in range(n)}

    for j, activity in enumerate(instance.activities):
        for succ in activity.successors:
            predecessors[succ].add(j)

    return predecessors


def random_topological_order(
    instance: ProjectInstance, seed: Optional[int] = None
) -> List[int]:
    """
    Generate a random topological order of activities based on the precedence graph
    stored in a psplib.ProjectInstance.

    This respects precedence constraints: each activity appears after all its
    predecessors. Among currently available activities, one is chosen at random.

    Args:
        instance: psplib.ProjectInstance object (e.g., from psplib.parse).
        seed: Optional random seed for reproducibility.

    Returns:
        A list of activity indices in a random topological order.

    Raises:
        ValueError: if the precedence graph is not acyclic or something goes wrong.
    """
    rng = random.Random(seed)
    n = instance.num_activities

    predecessors = build_predecessors(instance)
    in_degree = {j: len(predecessors[j]) for j in range(n)}
    available = [j for j in range(n) if in_degree[j] == 0]

    order: List[int] = []
    while available:
        j = rng.choice(available)
        available.remove(j)
        order.append(j)

        for succ in instance.activities[j].successors:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                available.append(succ)

    if len(order) != n:
        raise ValueError("Precedence graph is not acyclic or topological sort failed.")
    return order


def serial_sgs(
    instance: ProjectInstance,
    priority_list: List[int],
) -> Dict[int, Tuple[int, int]]:
    """
    Serial Scheduling Scheme (SSS) for RCPSP using a psplib.ProjectInstance.

    The algorithm goes through activities in 'priority_list' order and schedules each
    activity at its earliest feasible start time that satisfies:
      - all precedence constraints (all predecessors finished), and
      - all renewable resource capacities in each time period.

    It constructs an active schedule.

    Args:
        instance: psplib.ProjectInstance (e.g. from psplib.parse(...)).
        priority_list: A permutation of activity indices specifying the order in which
                       activities will be scheduled.

    Returns:
        schedule: dict mapping activity index -> (start_time, finish_time).
    """
    n = instance.num_activities
    R = instance.num_resources

    # For single-mode RCPSP, each activity typically has exactly one mode.
    durations = [activity.modes[0].duration for activity in instance.activities]
    demands = [list(activity.modes[0].demands) for activity in instance.activities]
    capacities = [resource.capacity for resource in instance.resources]

    # Crude upper bound on horizon: sum of all durations.
    horizon = sum(durations)

    # Resource usage profile: usage[r][t] = units of resource r used at time t.
    usage: List[List[int]] = [[0] * horizon for _ in range(R)]

    predecessors = build_predecessors(instance)
    start_times = [-1] * n
    finish_times = [-1] * n

    for j in priority_list:
        duration_j = durations[j]
        demand_j = demands[j]

        # Earliest start based on precedence (all predecessors must be finished).
        if predecessors[j]:
            earliest_start = max(finish_times[p] for p in predecessors[j])
        else:
            earliest_start = 0

        # Activities with zero duration can be placed at earliest_start directly.
        if duration_j == 0:
            start_times[j] = earliest_start
            finish_times[j] = earliest_start
            continue

        t = earliest_start
        while True:
            # Extend horizon and usage if we run out of time slots.
            if t + duration_j > horizon:
                extra = (t + duration_j) - horizon
                horizon += extra
                for r in range(R):
                    usage[r].extend([0] * extra)

            feasible = True

            # Check resource feasibility in [t, t + duration_j - 1]
            for tau in range(t, t + duration_j):
                for r in range(R):
                    if usage[r][tau] + demand_j[r] > capacities[r]:
                        feasible = False
                        break
                if not feasible:
                    break

            if feasible:
                # Reserve resources.
                for tau in range(t, t + duration_j):
                    for r in range(R):
                        usage[r][tau] += demand_j[r]
                start_times[j] = t
                finish_times[j] = t + duration_j
                break
            else:
                # Try the next time step.
                t += 1

    schedule = {j: (start_times[j], finish_times[j]) for j in range(n)}
    return schedule


def makespan(schedule: Dict[int, Tuple[int, int]]) -> int:
    """
    Compute the makespan (project completion time) from a schedule.

    Args:
        schedule: dict mapping activity index -> (start_time, finish_time).

    Returns:
        The maximum finish_time over all activities.
    """
    return max(f for _, f in schedule.values())


def main(argv: List[str]) -> None:
    """
    Simple CLI entry point.

    Usage:
        python rcpsp_scheduling.py path/to/instance.sm
    """
    if len(argv) < 2:
        print("Usage: python rcpsp_scheduling.py path/to/instance.sm")
        return

    instance_path = argv[1]
    instance = parse(instance_path, instance_format="psplib")

    print(f"Loaded instance with {instance.num_activities} activities "
          f"and {instance.num_resources} resources.")

    # Generate a random topological order as the priority list.
    priority_list = random_topological_order(instance, seed=42)
    print("Random topological order of activities:")
    print(priority_list)

    # Optionally, you might want to remove dummy start/end activities here,
    # depending on how the instance is defined.

    # Build a schedule using Serial Scheduling Scheme.
    schedule = serial_sgs(instance, priority_list)

    # Print schedule and makespan.
    print("\nSchedule (activity: start -> finish):")
    for j in sorted(schedule):
        s, f = schedule[j]
        print(f"Activity {j:2d}: {s:3d} -> {f:3d}")

    print(f"\nProject makespan: {makespan(schedule)}")


if __name__ == "__main__":
    main(sys.argv)

# python src/rcpsp_scheduling.py data/data-explanation/j301_1.sm