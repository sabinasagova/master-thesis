# RCPSP Dataset Short Explanation

This dataset represents a **Resource-Constrained Project Scheduling Problem (RCPSP)** with **32 jobs** (including start and end jobs). The objective is to schedule all jobs while respecting **resource limits** and **precedence constraints**, and **minimizing the total project time**.

---

##  Basic Info
| Item | Value |
|------|------|
| Jobs | 32 (job 1 = start, job 32 = end) |
| Time horizon | 158 |
| Renewable resources | 4 |
| Nonrenewable resources | 0 |

---
### Horizon vs Due Date

- **Due Date (38)** → Soft constraint (we want to finish by this time).
- **Tardcost (26)** → Penalty if we exceed 38.
- **Horizon (158)** → Hard limit.  
  - If > 158 → Solution is invalid.
  - If ≤ 158 → Solution is valid, but may be late.

#### Priority:
1. Valid schedule (≤ 158)
2. Try to be ≤ 38 (avoid penalty)
3. Minimize total time (makespan)

| Field | Explanation |
|-------|-------------|
| `#jobs = 30` | Number of real tasks (jobs 1 and 32 are start/end dummy nodes) |
| `rel.date = 0` | Project can start at time 0 |
| `duedate = 38` | Desired finish time – finishing later gives penalty (tardcost) |
| `tardcost = 26` | Penalty if project finishes after the due date |
| `MPM-Time = 38` | Minimum possible finish time **based only on dependencies** |
| `horizon = 158` | Hard upper limit – schedules **cannot exceed this** |

---
## Precedence
Each job lists its **successors**, meaning those jobs can only start after it finishes.  
This forms a **directed acyclic graph** (DAG) of dependencies.

---

## Job Data
For every job:
- **Duration** (processing time)
- **Resource usage** per time unit (R1–R4)

Example:
| Job | Duration | R1 | R2 | R3 | R4 |
|-----|----------|----|----|----|----|
| 6 | 8 | 0 | 0 | 0 | 8 |

---

## Resource Limits
| Resource | Capacity |
|----------|----------|
| R1 | 12 |
| R2 | 13 |
| R3 | 4 |
| R4 | 12 |

A valid schedule must **never exceed capacity at any time**.

---

## Goal
Find start times for all jobs such that:
Precedence is respected  
Resource limits are not violated  
Total completion time is minimal (makespan)
