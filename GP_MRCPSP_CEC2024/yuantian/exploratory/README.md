# Phase 0: exploratory evolutionary-algorithm sweep

Before narrowing down to the refined extensions documented in the top-level
[`GP_MRCPSP_CEC2024/readme.md`](../../readme.md) (critical-path propagation
terminals, epsilon-lexicase selection + memetic local search, non-renewable
resource terminals, resource-constrained critical path terminals,
mode-interaction terminals), this thesis ran a broader exploratory sweep:
nine evolutionary-algorithm variants against the published GPHH baseline
(Tian, Mei & Zhang, CEC 2024), most of which did not outperform it. That
sweep is real, methodologically valuable work -- it is what motivated
narrowing down to the kept extensions -- and is presented here as the
preliminary phase that precedes them, not mixed into their results.

This package is otherwise self-contained: nothing here is imported by
`cp_propagation.py` / `hybrid_gp.py` / `local_search.py` / `nr_terminals.py`
/ `rccp_terminals.py` / `mode_interaction_terminals.py`, and the nine
restored strategy drivers are invoked only from
`yuantian/experiments/exploratory_sweep_experiment.py`. Two modules here
are real, independently-conceived extensions (not part of the historical
nine-strategy sweep), each relocated here after its own dedicated
before/after comparison came back negative:

- `heuristic_seeding.py`: `gphh_solver.py` imports `seed_population` from
  it for the `--seeding_strategy`/`ParametersGPHH.seeding_strategy` CLI
  feature -- see its module docstring and the "Strategies" section below.
- `gap_aware_stopping.py`: `yuantian/experiments/lexicase_local_search_
  experiment.py` imports `lexicase_memetic_gp_gap_aware` from it for the
  "proposed_gap_aware" condition (not part of the main solver CLI, since
  the mechanism only ever existed as an experimental variant of
  `hybrid_gp.lexicase_memetic_gp`) -- see its module docstring and
  GP_MRCPSP_CEC2024/readme.md's extension #2 for what was tried and what
  the result was.

## Restoration notes

This code was not written from scratch for this task. It previously existed
in this repository as `GP_Algorithm/yuantian/custom_ea.py` and
`GP_Algorithm/yuantian/modifications.py`, and was deleted (along with
`experiment_runner.py`, `run_evaluation.py`, and several analysis scripts)
in commit `b595a2d5` ("feat: Extend GPHH with epsilon-lexicase selection,
memetic local search, and CP propagation terminal"), the same commit that
introduced the current `cp_propagation.py` / `hybrid_gp.py` /
`local_search.py`. Both deleted files were recovered with
`git show b595a2d5^:GP_Algorithm/yuantian/custom_ea.py` (and the equivalent
for `modifications.py`) and ported here, split into one module per strategy
to match this repo's current convention (one file per extension) instead of
the original single 1314-line file.

The complete original first-phase codebase has since also been restored,
byte-identical to its last pre-deletion state (`b595a2d5^`), under
[`legacy_gp_algorithm/`](legacy_gp_algorithm/) in this package: the full
`yuantian/` tree (including `modifications.py`, `custom_ea.py`, the old
`gphh_solver.py`/`rcpsp_simulation.py` they ran against, the evaluation
harnesses, and the analysis notebooks) plus its original `readme.md` and
`requirements.txt`. It is an archive for provenance and thesis
reproducibility, not an importable part of this package: nothing here or
elsewhere in the active codebase imports from it, and running it requires
its two vendored directories (`discrete_optimization/`,
`discrete_optimization_data/`), which were not duplicated -- symlink or
copy them from the repository root (`GP_MRCPSP_CEC2024/`) into
`legacy_gp_algorithm/` first.

Two adaptations were necessary because the surrounding codebase changed
shape between the original work and now:

- **Terminal-name adaptation (`map_elites_gp`'s behaviour descriptor).** The
  original CP/NR reliance descriptor read terminal names
  `{"Is_On_Critical_Path", "Slack", "Dynamic_Slack", "CP_Ext"}` (CP axis) and
  `{"NR_Stock_Ratio", "NR_Mode_Demand_Ratio"}` (NR axis) from
  `modifications.py`. This port uses the current repo's analogous terminal
  sets instead: `CP_FORWARD`/`CP_BACKWARD`/`CP_SLACK_SCORE`/`CP_PROB`
  (`cp_propagation.py`, `--cp_propagation`) and `NR_STOCK_RATIO`/
  `NR_MODE_DEMAND_RATIO`/`NR_BUDGET_PRESSURE` (`nr_terminals.py`,
  `--nr_terminals`). The grid is only non-degenerate when the GPHH instance
  was built with both flags on.

- **Restoring `modifications.py`'s graft vocabulary for the diagnostic-graft
  drivers.** `modification_integrated_gp` (DMGE), `trace_directed_gp`, and
  their shared graft machinery in `diagnostic_graft.py` graft an `if_else`
  primitive plus terminals (`Slack`, `Is_On_Critical_Path`, `Dynamic_Slack`,
  `Scheduled_Fraction`, `Bottleneck_Renewable`, `CP_Ext`, `NR_Stock_Ratio`,
  `NR_Mode_Demand_Ratio`) that lived in `modifications.py` -- also deleted in
  commit `b595a2d5`, and not needed by anything else in this restoration.
  Without it, the graft is a structural no-op (every `_ifelse_graft`/
  `_graft_terminal` call returns `False` because the names aren't in the
  pset), reducing two of the nine strategies to `standard_gp` with unused
  bookkeeping. `diagnostic_graft.install_graft_terminals(pset, simulator)`
  ports those definitions verbatim from the recovered `modifications.py` and
  old `rcpsp_simulation.py`, but applies them only to the specific GPHH
  instance's `pset`/`simulator` objects passed in -- it mutates that
  instance's pset dict and monkey-patches nothing on the shared
  `rcpsp_simulation.py` classes -- so calling it has no effect on any other
  GPHH run. Call it once after `solver.init_model()` and before running
  `modification_integrated_gp` or `trace_directed_gp`; both have a one-line
  warning in their docstring.

- **Known bug fixed during the port (multi-SGS).** The original
  `_eval_multi_sgs` / `evaluate_on_test_multi_sgs` took an unconditional
  `min()` across simulators' makespans even when *no* SGS variant produced a
  feasible schedule (`best_any_mk = min(best_any_mk, mk)` ran every
  iteration regardless of the `feasible` flag), so an individual infeasible
  under every simulator got the minimum of several infeasible sentinel
  makespans instead of one consistent penalty -- optimistically biasing its
  fitness relative to an individual infeasible under only one simulator.
  This was identified and confirmed dead-code-on-PSPLIB-J20 (every individual
  sampled there was feasible under at least one simulator, so the buggy
  branch never actually executed) during the original investigation, but is
  a real issue in general on tighter instances (e.g. MMLIB50/MMLIB+).
  `multi_sgs.py` fixes this: when no variant is feasible, fitness now falls
  back to the *serial* simulator's own makespan, not a cross-simulator min
  of infeasible results.

What was **not** restored, because no trace of it exists anywhere in this
repository's git history (all commits, all branches, no stash) despite being
named in the restoration request -- `seed_heuristic_individuals` /
`_HEURISTIC_SEEDS` as a standalone Phase 0 driver, `strategy_isolated_gp`,
and `dmge_mega_gp` / `dmge_mega_experiment.py`:

- `EA_REGISTRY` in the recovered `custom_ea.py` has exactly nine entries
  (`mod_integrated`, `trace_directed`, `decision_trace`, `map_elites`,
  `adaptive`, `surrogate`, `diverse`, `lexicase`, `multi_sgs`) plus
  `"standard"` for the baseline -- no separate seeding driver, ever.
  `heuristic_seeding.py` in this package is the real, already-existing
  heuristic-seeding extension (originally `yuantian/heuristic_seeding.py`,
  written for its own before/after comparison, not part of this sweep) --
  relocated here, not restored from custom_ea.py, so the exploratory
  experiment can report where it ranks among these nine strategies without
  inventing a second seeding mechanism under a name that never existed in
  this codebase's history.
- `dmge_mega_gp` (an "all nine strategies combined" driver) and
  `strategy_isolated_gp` (a single flag-dispatched driver) were not found
  either. Per the restoration request, a combined mega-driver would not be
  restored even if found -- the point of Phase 0 in the thesis narrative is
  showing each strategy in isolation against baseline, which is what
  motivated *not* combining them. The combination experiment, when it was
  tried, gave a bimodal result: one mild winner bundled with several
  mild-to-strong losers, not a clear net improvement, which is part of why
  it was abandoned in favour of testing strategies independently.

## Module structure

| File | Strategy key | Contents |
|---|---|---|
| `shared.py` | -- | `_Timer`, `_record`, `_new_logbook`, `_eval_full`, `_evaluate_cases`, `_build_heuristic`, `_iter_trees`, `_terminal_reliance`, `_behavioural_distance`, `_rank`, `_spearman`, `_decision_trace` -- used by 2+ strategies |
| `selection.py` | `lexicase` | Epsilon-lexicase selection + numeric constants + mini-batch rotation |
| `diagnostic_graft.py` | `mod_integrated`, `trace_directed`, `decision_trace` | DMGE diagnostic-graft backbone, TDRE, decision-trace illumination, plus the restored graft vocabulary (`install_graft_terminals`) |
| `quality_diversity.py` | `map_elites` | MAP-Elites over a CP-reliance x NR-reliance grid |
| `adaptive_ops.py` | `adaptive` | Adaptive operator selection (probability matching) |
| `surrogate.py` | `surrogate` | k-NN phenotypic-characterisation surrogate |
| `diverse_partner.py` | `diverse` | Diverse-partner crossover |
| `multi_sgs.py` | `multi_sgs` | Dual/triple-SGS evaluation (serial + parallel + optional backward), with the bug fix above |
| `heuristic_seeding.py` | -- (real extension, see above) | `build_heuristic_trees`, `seed_population` (also imported by `gphh_solver.py`'s `--seeding_strategy`), plus `seed_then_run` for combining seeding with any driver in this package |
| `gap_aware_stopping.py` | -- (real extension, see above) | `detect_gap_onset`, `lexicase_memetic_gp_gap_aware` (also imported by `lexicase_local_search_experiment.py`'s "proposed_gap_aware" condition) |
| `__init__.py` | -- | `EXPLORATORY_REGISTRY` (strategy name -> driver), `GRAFT_DEPENDENT_STRATEGIES` |

## Strategies

For each strategy: what it does, and what it found. Original numeric
results (dev_feas / delta vs baseline / feasible% / p) from the historical
sweep were not found anywhere in this repository -- the recovered commits
contain the sweep's *code* but no committed CSV/JSON output and no notebook
cell output matching this sweep (the one surviving analysis notebook is for
the published baseline's serial/parallel x activity/mode/simultaneous
comparison, a different study). Every entry below therefore says "results
pending re-run via `exploratory_sweep_experiment.py`" rather than citing a
number that cannot be verified.

- **`lexicase` -- epsilon-lexicase + numeric constants + mini-batch.** The
  one strategy promising enough to be developed further: it directly
  motivated `hybrid_gp.py`'s epsilon-lexicase selection + memetic local
  search extension (not the same driver -- this one only changes
  selection/evaluation, with no local-search refinement step on elites).
  Results pending re-run via `exploratory_sweep_experiment.py`.
- **`mod_integrated` -- Diagnostic Modification-Graft Evolution (DMGE),
  flagship.** Every offspring is grafted with a phase-aware `if_else` block
  chosen from its parent's SGS-trace diagnosis (NR-infeasible /
  CP-blind / renewable-contention-blind). Results pending re-run.
- **`trace_directed` -- Trace-Directed Repair Evolution (TDRE).** A simpler
  precursor to DMGE: variation conditioned only on a parent's
  NR-feasibility (infeasible parents get an NR-relief graft on the mode
  tree, feasible ones get a CP-tightening graft on the activity tree).
  Results pending re-run.
- **`decision_trace` -- decision-trace illumination.** A novel behavioural
  descriptor (CP-respect x NR-frugality, read off the produced schedule)
  driving a quality-diversity illumination loop. Results pending re-run.
- **`map_elites` -- MAP-Elites / Quality-Diversity.** Illuminates a
  CP-reliance x NR-reliance genotype grid (Mouret & Clune, 2015) instead of
  optimising mean fitness alone. Results pending re-run.
- **`adaptive` -- adaptive operator selection.** Probability matching on
  credit (fitness improvement over the parent) reweights crossover/
  mutation/reproduction rates instead of holding them fixed. Results
  pending re-run.
- **`surrogate` -- phenotypic-characterisation surrogate.** k-NN model
  (Hildebrandt & Branke, 2015) screens a larger breeding pool before full
  evaluation. Results pending re-run.
- **`diverse` -- diverse-partner crossover.** Mates a tournament winner with
  the most behaviourally distant candidate from a small pool, instead of a
  behaviour-blind partner choice. Results pending re-run.
- **`multi_sgs` -- dual/triple-SGS evaluation.** Runs each rule under serial
  + parallel (+ optional backward) SGS and keeps the best feasible
  makespan (Van Peteghem & Vanhoucke, 2010). Results pending re-run; note
  the bug fix above changes this strategy's numbers relative to whatever
  the original (buggy) run reported, so this is not directly comparable to
  any old number even if one were found.

LLM-driven evolution (FunSearch, EoH, ReEvo) was noted in the original
sweep's scope discussion as the 2024 frontier of automated heuristic design,
but out of scope (needs an external model) for this offline pipeline -- not
restored as a strategy because it was never implemented, only discussed.

## Running it

```bash
PYTHONPATH=$(pwd):$(pwd)/yuantian python3 -O \
  yuantian/experiments/exploratory_sweep_experiment.py
```

See that script's module docstring for the train/val/test split, seed count,
and Wilcoxon convention (matches `cp_propagation_experiment.py` and
`heuristic_seeding_experiment.py`).
