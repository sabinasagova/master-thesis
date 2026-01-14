# RCPSP Scheduling – Master Thesis

This repository contains code and data used in my master thesis for solving the **Resource-Constrained Project Scheduling Problem (RCPSP)** using the **Serial Scheduling Scheme (SSS)** and instances from **PSPLIB**.

The main script is:

- `src/rcpsp_scheduling.py` – loads PSPLIB instances, generates a priority list of activities, and builds a feasible schedule using SSS.

Data files (e.g. `j301_1.sm`) are stored in:

- `data/data-explanation/`

---

## What is RCPSP?

The **Resource-Constrained Project Scheduling Problem (RCPSP)** is a classic combinatorial optimization problem:

- We have a set of activities (jobs).
- Each activity has:
  - a processing time (duration),
  - resource demands,
  - and precedence constraints (some activities must finish before others start).
- Resources have **limited capacities**.
- The goal is typically to **minimize the project makespan** (total completion time), while respecting all resource and precedence constraints.

---

## What is Serial Scheduling Scheme (SSS)?

The **Serial Scheduling Scheme (SSS)** is a constructive algorithm that turns a **priority list of activities** into a **feasible schedule**:

1. Take a **priority list**: a permutation of all activities.
2. Process activities one by one in that order.
3. For each activity:
   - compute the **earliest start time** allowed by **precedence constraints** (all predecessors must have finished),
   - then find the earliest time where there is enough **resource capacity** for its entire duration.
4. Fix the activity at that time and **never move it again**.

The result is an **active schedule**: no activity can be moved earlier without violating some precedence or resource constraint.

SSS is the standard way to use **priority rules** (like shortest processing time, most successors, etc.) to generate RCPSP schedules.

---

## Repository Structure

```text
master-thesis/
│
├─ README.md
│
├─ src/
│   └─ rcpsp_scheduling.py    # SSS implementation + CLI
│
└─ data/
    └─ data-explanation/
        ├─ j301_1.sm 
       