# Vanilla Genetic Algorithm with Local Search for Multi-Mode Project Scheduling (MRCPSP)

This document describes a basic genetic algorithm (GA) framework for the multi-mode resource-constrained project scheduling problem (MRCPSP), extended with a simple local search (GA+LS).

------------------------------------------------------------

## 0. Problem Setup

Given:

- A set of activities \( j = 1, \dots, J \) with precedence constraints.
- Each activity \( j \) has multiple execution modes \( m = 1, \dots, L_j \).
  Each mode \( m \) is characterized by:
  - Duration \( d_{jm} \)
  - Renewable resource requirements
  - Nonrenewable resource requirements
- Renewable resource capacities for each time period.
- Nonrenewable resource capacities over the project horizon.
- Objective: usually to minimize the project makespan, but the framework can be extended
  to include other criteria such as cost, risk, or quality.

------------------------------------------------------------

## 1. Chromosome Structure

We use a two-part representation:

```text
Individual = (AL[1..J], ML[1..J])
```

- `AL` (Activity List): a precedence-feasible permutation or priority sequence of activities.
- `ML` (Mode List): the chosen mode for each activity \( j \), with `ML[j]` in `{1, ..., L_j}`.

------------------------------------------------------------

## 2. Decoding via Serial Schedule Generation Scheme (SSGS)

The chromosome does not store exact start times. Instead, a decoding procedure constructs
a feasible schedule from `(AL, ML)` using a serial schedule generation scheme (SSGS).

### SSGS Decoding Procedure (Conceptual)

```text
function Decode(AL, ML):
    Scheduled = {}
    while |Scheduled| < J:
        Eligible = { j not in Scheduled | all predecessors of j are in Scheduled }
        Choose j* in Eligible with the smallest position in AL
        mode = ML[j*]
        t' = earliest feasible start time for j* in mode, such that:
             - all predecessors of j* finish by t'
             - renewable resource capacities not violated during [t', t' + d_{j*,mode})
             - nonrenewable capacities remain within limits

        Set start[j*]  = t'
        Set finish[j*] = t' + d_{j*,mode}
        Add j* to Scheduled

    Return schedule, makespan
```

From the final schedule we can compute:

- Makespan \( C_{\max} = \max_j \text{finish}[j] \)
- Any other performance metrics needed (cost, risk, etc.).

------------------------------------------------------------

## 3. Fitness Function

A basic fitness function uses the makespan only:

```text
Fitness(individual) = -Makespan(individual)
```

because we typically maximize fitness in GA. Smaller makespan implies larger fitness.

For multiple criteria (e.g., time and cost), one may define:

```text
Fitness(individual) = -( α * Makespan + β * Cost + γ * Risk )
```

where \( \alpha, \beta, \gamma \) are nonnegative weights chosen by the decision-maker.

Infeasible schedules (e.g., exceeding nonrenewable resources) can:

- Be penalized by adding a large penalty term to the objective, or
- Be repaired by a problem-specific repair procedure before evaluation.

------------------------------------------------------------

## 4. Population Initialization

Create an initial population of size `P` as follows:

1. Generate precedence-feasible activity lists `AL`:
   - Repeatedly pick a random activity from the set of precedence-eligible activities
     until all activities are selected.

2. Generate mode lists `ML`:
   - For each activity `j`, randomly assign a mode in `{1, ..., L_j}`, or
   - Use a heuristic rule (e.g., shortest duration mode or cheapest mode).

Optionally, include high-quality solutions from a construction heuristic along with
random individuals to speed up convergence.

------------------------------------------------------------

## 5. Selection

Use a standard selection mechanism to pick parents:

- Tournament selection (e.g., of size 2 or 3),
- Roulette-wheel (fitness-proportional) selection,
- Rank-based selection.

For multi-objective problems, use a Pareto-based selection such as NSGA-II,
but the rest of the GA pipeline remains similar.

------------------------------------------------------------

## 6. Crossover Operators

Apply crossover separately to `AL` and `ML` to produce offspring.

### 6.1 Activity List (AL) Crossover

Since `AL` is a permutation, we need a permutation-preserving operator, such as:

- Order Crossover (OX),
- Position-based crossover,
- Partially-mapped crossover (PMX).

These operators take two parent permutations and generate children that are also
permutations of the activities.

### 6.2 Mode List (ML) Crossover

The mode list is just an integer vector of fixed length. Simple crossover operators can be used:

- One-point crossover,
- Two-point crossover,
- Uniform crossover.

Example (one-point):

```text
Choose cut index k randomly in {1, ..., J-1}
Child1.ML[1..k]   = Parent1.ML[1..k]
Child1.ML[k+1..J] = Parent2.ML[k+1..J]
Child2.ML[1..k]   = Parent2.ML[1..k]
Child2.ML[k+1..J] = Parent1.ML[k+1..J]
```

------------------------------------------------------------

## 7. Mutation Operators

Mutate AL and ML separately to maintain diversity in the population.

### 7.1 Activity List (AL) Mutation

Common operators:

- Swap mutation: choose two positions and swap their activities.
- Insertion mutation: remove an activity from one position and insert it at another.

If necessary, ensure precedence feasibility or rely on the decoder to handle it via
repair or penalties.

### 7.2 Mode List (ML) Mutation

- Random mode mutation: choose activity `j` and assign a different mode.
- Mode-swap: swap the modes for two activities.

Optionally, heuristics can bias mutation to favor shorter duration or lower-cost modes,
depending on the objective.

------------------------------------------------------------

## 8. Local Search (LS) Integration

To form a GA+LS (memetic) algorithm, we apply a local search procedure to some
or all offspring before inserting them into the next generation. The local search
attempts to improve the fitness by making small modifications to the individual.

### 8.1 Neighborhood Structures

Typical neighborhoods include:

1. Activity-order neighborhood:
   - Swap the positions of two activities in `AL` (while preserving feasibility),
   - Move (insert) one activity to a different position.

2. Mode-change neighborhood:
   - Change the mode of one activity `j` to another feasible mode.

### 8.2 Simple Local Search Procedure

```text
function LocalSearch(individual, max_neighbors):
    best = individual
    best_fitness = Fitness(best)

    neighbors_explored = 0

    while neighbors_explored < max_neighbors:
        candidate = GenerateNeighbor(best)
        Decode(candidate)
        f = Fitness(candidate)

        if f > best_fitness:
            best = candidate
            best_fitness = f
            neighbors_explored = 0   # restart search from improved solution
        else:
            neighbors_explored += 1

    return best
```

`GenerateNeighbor` can randomly choose one of:

- Swap two activities in `AL` (subject to precedence feasibility),
- Change one activity's mode in `ML`.

### 8.3 When to Apply Local Search

Local search can be applied:

- To every offspring, or
- To a subset of the best offspring in each generation, or
- With some probability `p_ls`.

There is a trade-off between solution quality and computational effort.

------------------------------------------------------------

## 9. Full GA+LS Procedure

Pseudocode for the full GA with local search:

```text
Initialize a population Pop of size P
for each individual in Pop:
    Decode(individual)
    individual.fitness = Fitness(individual)

repeat until termination condition is met:

    NewPop = {}

    while |NewPop| < P:
        parent1 = Select(Pop)
        parent2 = Select(Pop)

        child1, child2 = parent1, parent2

        if random() < pc:
            child1, child2 = Crossover(parent1, parent2)

        if random() < pm:
            Mutate(child1)
        if random() < pm:
            Mutate(child2)

        Decode(child1)
        child1 = LocalSearch(child1, max_neighbors)
        child1.fitness = Fitness(child1)

        Decode(child2)
        child2 = LocalSearch(child2, max_neighbors)
        child2.fitness = Fitness(child2)

        Add child1 and child2 to NewPop (or only the better one if near capacity)

    Pop = ApplyElitism(Pop, NewPop)

end repeat

Return the best individual in Pop
```

Where:

- `pc` is the crossover probability.
- `pm` is the mutation probability.
- `max_neighbors` controls the intensity of local search.
- `ApplyElitism` preserves a few top individuals from the old population.

------------------------------------------------------------

## 10. Termination Criteria

Common stopping rules:

- Maximum number of generations reached.
- No improvement in best fitness for a given number of generations.
- Time limit reached.

------------------------------------------------------------

## 11. Summary

- The solution is represented by an activity list (AL) and a mode list (ML).
- A serial schedule generation scheme (SSGS) decodes the chromosome to a feasible schedule.
- The GA evolves both activity ordering and mode choices.
- A local search procedure refines selected individuals by exploring small neighborhoods
  (swapping activities or changing modes).
- The framework is extensible to multiple objectives such as cost, risk, quality, or
  energy-related criteria.

This Markdown file can be used as design documentation for implementing a GA+LS solver
for the multi-mode resource-constrained project scheduling problem.
