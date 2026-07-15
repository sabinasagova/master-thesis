# Recovered artifacts: the trail that led to selecting lexicase

Files recovered from git history on 2026-07-12 (they no longer exist in the
working tree). These are the oldest surviving artifacts documenting why
epsilon-lexicase selection was singled out during the exploratory screening.

| File | Origin | What it is |
|---|---|---|
| `custom_ea__pre-b595a2d5.py` | `git show b595a2d5^:GP_Algorithm/yuantian/custom_ea.py` | The original exploratory sweep module from the old `GP_Algorithm/` codebase. Section 8 (`Epsilon-lexicase + ERCs + mini-batch`, lines ~929–1100) contains the written rationale for trying lexicase: tournament selection on mean deviation collapses per-instance scores and converges onto generalists; lexicase treats each training instance as a separate selection case and keeps specialists alive (citing La Cava et al. 2016; Ardeh et al. 2021 for GP scheduling). |
| `run_evaluation__fe611f00.py` | `git show fe611f00:GP_Algorithm/yuantian/run_evaluation.py` | The DMGE evaluation harness of the old codebase, whose configs (`baseline,dmge_full,tdre,tdre_mods,lexicase,lexicase_mods`) included lexicase as a comparison arm. |
| `GP_Algorithm_readme__fe611f00.md` | `git show fe611f00:GP_Algorithm/readme.md` | The old codebase's README documenting how the evaluation was run. |

## What does NOT exist

The *numeric outputs* of the original sweep (the run where lexicase was first
observed to be the promising direction) were never committed to git and are
not on disk; `GP_MRCPSP_CEC2024/yuantian/exploratory/README.md` states this
explicitly ("results pending re-run"). The earliest surviving *numbers* are:

- the first-pass pilot of the follow-up (lexicase+LS train 11.87 vs baseline
  13.48, p=0.0059, r=0.93), cited in thesis Section 4.2.4;
- `yuantian/experiments/results/full_mmlib_experiment/serial_lexicase_power_followup.json`
  (62 per-seed records, n=31 per method), from which the stage 2–4 numbers in
  thesis Table 4.x are computed exactly (subsets n=10/16/31 reproduce the
  published p-values);
- the deterministic sweep re-runs in
  `yuantian/experiments/results/exploratory_sweep_experiment*/all_runs.json`.

## Recreation of the first numeric output (2026-07-12)

Because the original pilot's output was never persisted, the experiment was
re-executed in the original configuration (pop 60, gen 25, 5 MMLIB50
instances, 10 seeds, serial SGS, activity-first; baseline vs
epsilon-lexicase + local search). Result:
`yuantian/experiments/results/lexicase_local_search_experiment/initial_pilot_recreation_2026-07-12.json`.
Train: lexicase+LS 11.88±1.10 vs baseline 12.69±1.03 (p=0.19, r=0.51);
test: 8.18±1.10 vs 8.96±1.50 (p=0.41). The direction of the original finding
reproduces (lexicase+LS better on train and test), but not its significance;
the original's exact instance split and seeds were unrecorded, so the
recreated baseline landed better than the originally reported 13.48.
