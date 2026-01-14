# Scheduling Problem Definitions (Simple Explanation)

## 1. Resource-Constrained Project Scheduling Problem (RCPSP)
This is the basic version of the scheduling problem.

**Goal:** Finish the project as early as possible.  
**Constraints:**  
- Tasks depend on each other (some must wait for others).  
- Resources are limited (e.g., workers, machines).  
- Each task has a fixed duration.

Example:  
Job 2 starts only after Job 1 finishes, and we cannot use more workers than we have.

---

## 2. RCPSP/max – With Time Lags (Minimum and Maximum Delay)
This is RCPSP with extra timing rules.

Some jobs must start:
- Not too early (minimum lag),  
- Not too late (maximum lag).

Example:  
Job 2 must start at least 3 days after Job 1.  
Job 3 must start no more than 5 days after Job 1.

---

## 3. MRCPSP – Multi-Mode RCPSP
Each job can be done in different modes.

- Mode 1 → shorter time but uses many resources  
- Mode 2 → longer time but uses fewer resources  

Goal: Choose the best mode for each job while scheduling the project.

Example:

| Mode | Duration | Workers Needed |
|------|----------|----------------|
| 1    | 2 days   | 5 workers      |
| 2    | 4 days   | 2 workers      |

---

## 4. MRCPSP/max – Multi-Mode with Time Lags
This combines both features:
- Multiple execution modes per job  
- Minimum and maximum time lags between jobs  

This version is more difficult to solve.

---

## 5. RIP/max – Resource Investment Problem with Time Lags
Here the goal changes.

**Goal:** Use as few total resources as possible (minimize cost), even if the project takes longer.  
Time lags are also considered.

This applies when saving resources/money is more important than finishing quickly.

---

## 6. RCPSP with Time-Dependent Resource Capacities
In this version, resource availability changes over time.  
The schedule must adapt to this.

Example:

| Time Period | Available Workers |
|-------------|-------------------|
| Day 1–3     | 5 workers         |
| Day 4–6     | 10 workers        |
| Day 7–10    | 4 workers         |

---

## Comparison Table

| Problem Type | Multiple Modes | Time Lags | Changing Resources | Goal |
|--------------|----------------|-----------|---------------------|------|
| RCPSP | No | No | No | Minimize project duration |
| RCPSP/max | No | Yes | No | Respect time lags and minimize duration |
| MRCPSP | Yes | No | No | Choose best mode and minimize duration |
| MRCPSP/max | Yes | Yes | No | Harder scheduling with both constraints |
| RIP/max | No | Yes | No | Minimize resource cost |
| RCPSP with time-dependent capacities | No | No | Yes | Adapt schedule to changing resources |

