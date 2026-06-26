# Genetic Programming Hyper-heuristics for solving multi-mode resource-constrained project scheduling problem

## Overview
This repository contains the implementation of the paper "Learning Heuristics via Genetic Programming for Multi-Mode Resource-Constrained Project Scheduling" by Yuan Tian, Yi Mei, and Mengjie Zhang, presented at the 2024 IEEE Congress on Evolutionary Computation (CEC 2024).

## Usage

Install the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```
discrete-optimization package is already included in the project. You don't need to install it separately.

Before run any code, please include the project root directory to the `PYTHONPATH`. For example, in Linux or MacOS, you can run:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

The main script is `yuantian/gphh_solver.py`. 

You can run it with the following command:
```bash
python yuantian/gphh_solver.py
````
The result file will be saved in the `results` folder.

### Arguments in gphh_solver.py
You can run the script with different arguments:
- `-s`: specify the schedule generation scheme. Options are `serial` and `parallel`. Default is `serial`.
- `-d`: speificy the decision type. Options are `activity_first`, `mode_first` and `simultaneous`. Default is `activity_first`.
- `--default`: use the parameters in the paper. If not set, the minimum working example parameters will be used.
- `--dataset`: specify the dataset to use. Options are `MMLIB50`, `MMLIB100`, `MMLIBPLUS_50`, `MMLIB_PLUS_100`. Default is a small set.
- `--start_index`: specify the start index of the instance to run. Default is `0`.`
- `-n`: specify the number of runs. Default is `1`.
- `--seed`: specify the random seed. Default is `0`.
- `--log`: if set, the detailed log will be saved in the `logs` folder.
- `--dynamic`: if set, the CPM related terminals will be updated every time the eligible set is updated. Otherwise, the static mode will be used.
- `--split`: Whether the training set is splited into servral subsets, so during the evolution process each generation will be evaluated on a different subset. Default is `False`.
- `--multi_process`: Whether to use multi-process to speed up the evaluation. 90% of logical CPUs will be used. You can change the number in `gphh_solver.py`. Default is `False`.

For example, to run the serial schedule generation scheme with activity-first decision type on the MMLIB50 dataset with default parameters, you can run:
```bash
python yuantian/gphh_solver.py -s serial -d activity_first --default --dataset MMLIB50
```

## Extensions for my thesis

Everything above is Tian et al.'s original code. Below is what I added on top of it.

### Phase 0: exploratory sweep (preliminary)

Before settling on the extensions below, I ran a broader exploratory sweep: nine evolutionary-algorithm variants (epsilon-lexicase selection, a diagnostic modification-graft operator, MAP-Elites, adaptive operator selection, a k-NN surrogate, diverse-partner crossover, dual/triple-SGS evaluation, and two trace-conditioned variation drivers) against the baseline, most of which did not outperform it. That sweep is what narrowed things down to the extensions documented below — in particular, the epsilon-lexicase strategy from this sweep is the one that turned out promising enough to develop further into extension #2. The sweep lives in `yuantian/exploratory/` (one module per strategy) with its own [README](yuantian/exploratory/README.md) explaining what each strategy does and found, and is run via `yuantian/experiments/exploratory_sweep_experiment.py`. It's otherwise self-contained — nothing in it is imported by the main CLI or by the extensions below, except `heuristic_seeding.py`, which the CLI's `--seeding_strategy` flag (used by extension #2's seeding comparison) imports from directly. That module lives here because its own before/after result came back negative, same as most of this sweep, even though it isn't one of the nine restored drivers.

### 1. Critical path propagation terminals

The baseline already has ES/EF/LS/LF (classic CPM dates) as terminals, but I wanted to try giving the GP a more "ready-made" critical-path signal instead of making it rediscover slack-like quantities by combining ES/EF/LS/LF itself. So I added `yuantian/cp_propagation.py`, which computes four extra per-activity features once per instance (cached on `problem.cp_features`, never recomputed during a GP run):

- `CP_FORWARD`: longest path from the activity to the sink, normalized by the project makespan estimate
- `CP_BACKWARD`: basically the same idea as LS but written explicitly as a propagation
- `CP_SLACK_SCORE`: backward minus forward
- `CP_PROB`: how close the activity's forward value is to the most "critical" activity in the instance

These are toggled with `cp_propagation_feature=True` in `ParametersGPHH`, or `--cp_propagation` on the CLI. The terminals themselves are wired into `rcpsp_simulation.py`'s `FeatureEnum`/`feature_function_map`.

**Result so far** (`yuantian/experiments/cp_propagation_experiment.py`, pop=60, gen=25, 5 MMLIB50 instances, 10 seeds): adding these terminals sped up convergence quite a bit (8.7 vs 14.4 generations to reach the best solution) and produced the single best run overall, but didn't move the mean or the held-out test fitness in any meaningful way (Wilcoxon p=0.88, not significant). My read on this: `CP_BACKWARD` is literally the same formula as the existing `LS` terminal, and the other two are linear combos of terminals the GP already has, so there isn't really new information here, just a shortcut to something the tree could already build. Explains the convergence-speed bump without a quality bump — and is exactly the diagnosis that motivated extension #3 (resource-driven, not precedence-only, criticality information) below.

### 2. Epsilon-lexicase selection + critical path local search

The bigger change: I swapped tournament selection for epsilon-lexicase selection (`yuantian/hybrid_gp.py`), and added a memetic local search step (`yuantian/local_search.py`) that runs critical-path repair (mode swaps, precedence-respecting activity swaps, slack-based resource shifting) on the top ~8% of the population every generation. Important detail: the local search only ever touches an individual's recorded fitness, never the GP tree itself, so the representation stays exactly what the baseline uses. It's a Lamarckian/memetic trick, not a different genotype.

`evaluate_heuristic` in `gphh_solver.py` now also stores `individual.case_fitness` (the per-instance deviation scores), which is what the lexicase selection needs to filter candidates case by case.

I ran this in stages, getting more rigorous (and more expensive) each time:

- **First pass**, small scale (pop=60, gen=25, 5 instances, 10 seeds): lexicase+localsearch beat the baseline mean (11.87 vs 13.48) and best-of-run (10.07 vs 12.35), and it was actually significant (Wilcoxon p=0.0059, r=0.93). Convergence was slower though (17.5 vs 12.0 generations), which matches what you'd expect from lexicase keeping more diverse "specialist" individuals around instead of letting tournament selection collapse onto one strong generalist early.

- **Bigger pass** on real MMLIB50 (`yuantian/experiments/full_mmlib_experiment.py`), using the official 60/20/20 train/val/test split logic but on a stratified subset of 10 classes (out of 108) to keep it runnable on my laptop instead of the ~200+ days the literal pop=1000/gen=50/full-split spec would have taken. pop=60, gen=20, 10 seeds, both serial and parallel SGS, 3 conditions (baseline / lexicase / lexicase+localsearch). On serial SGS, lexicase and the full hybrid both significantly beat baseline on training fitness (p=0.0098 and p=0.027), but the same comparison on held-out test fitness landed at p=0.109 (lexicase), same direction, just not under 0.05 yet. Also found something I wasn't expecting: serial SGS is dramatically better than parallel SGS for every method here (test fitness ~18-19 vs ~30-33), so SGS choice matters way more than which selection/search method you use, at least at this scale.

- Did a quick power calculation on that 0.109 result (r=0.667 at n=10 suggested ~14-16 seeds might be enough to flip it), so I added 6 more seeds just for that one comparison (serial, baseline vs lexicase) instead of rerunning everything. At n=16 it moved to p=0.074, r=0.543, still not significant, train fitness comparison was already significant and got more so (p=0.00058). Train fitness clearly favors lexicase either way, it's specifically the test-set number that's taking more seeds to pin down, probably because the held-out set is only 10 instances so it's noisier.

- **Update, pushed to n=31** (`serial_lexicase_power_followup.py`): this is where it got interesting, and not in the direction the power calculation predicted. Instead of the test-fitness effect strengthening toward significance, it went the other way: p=0.325, r=0.209 at n=31, weaker than the p=0.074/r=0.543 at n=16. So the n=10 estimate (r=0.667) was just small-sample noise, not a real trend, classic effect-size inflation from a small pilot. Lesson for myself: don't trust a power calculation built on one early effect-size estimate, especially with only 10 held-out instances to average over, it can easily point the wrong way once more data comes in. Meanwhile training fitness stayed exactly where it was supposed to: p=0.0002, r=0.72, clearly significant, lexicase mean 16.68 vs baseline 17.60. So the honest conclusion at this scale is that lexicase selection reliably finds GP trees that fit the training instances better, but that advantage isn't showing up as significantly better generalization to held-out instances. Possibly lexicase's case-by-case selection pressure is fitting the training set's specific quirks rather than learning something that transfers, i.e. mild overfitting, but I'd want to look at the per-seed train/val fitness curves before claiming that for sure.

- Also added a proper 2x2 ablation to `full_mmlib_experiment.py` (baseline / lexicase only / local-search only / lexicase+localsearch) so I can actually tell whether gains come from the selection change, the local search, or both together, instead of just comparing baseline to the combined method. Haven't run the full 80-run version of this yet (would be ~8-9h), code is ready though.

- **Tried and didn't pan out: gap-aware early stopping.** The per-seed train/val curves I'd been meaning to look at (previous bullet) led to a dedicated investigation, which found lexicase's train->validation gap stays flat for several generations then widens to a stable, much higher plateau, with none of the post-onset training-fitness improvement transferring to held-out data. Built `detect_gap_onset` to catch that flat-to-rising transition online (from the run's own validation trajectory, not a hard-coded generation) and used it to either roll back to the best-on-validation generation's individual or stop the run early. Validated twice: once at `lexicase_local_search_experiment.py`'s default scale (25 stratified classes, 1 train instance/class) — no real train-test gap existed there to catch (test fitness came out *better* than training for every condition), so that run alone wasn't conclusive — and once with `--known_gap_split` (full_mmlib_experiment.py's exact 10-class/3-train-case split, known to produce a real ~18-19 test-fitness gap) to rule that out specifically. Both times: the mechanism correctly detected onset and rolled back (8-9/10 seeds, to a mean generation roughly a third to half of the full budget), but proposed-with-gap-aware vs plain proposed on test fitness was never significant (p=0.109 and p=0.734 respectively), and the generalization gap itself barely moved. Diagnosis: validation-fitness "best so far" is too noisy an estimator at this scale (same 10-25 held-out instances) to reliably pick the generation that will actually generalize best to a third, fully held-out set — the gap signal is real, but not precise enough to act on. Moved to `yuantian/exploratory/gap_aware_stopping.py` after this result; `lexicase_memetic_gp` in `hybrid_gp.py` no longer carries this logic.

One thing worth flagging if anyone reruns this: `ParallelSimulator.get_eligibles()` has `if __debug__:` print statements that dump a huge amount of text per call. Doesn't matter for correctness but it'll blow up your terminal/log files unless you run with `python -O` (which strips `__debug__` blocks). I always run the experiment scripts that way.

### 3. Non-renewable resource terminals

Mode choice trades non-renewable (NR) resource consumption against duration/renewable-resource use, but the baseline terminal set has no signal about NR budget — all the existing resource terminals (`avg_RReq`, `GRD`, etc.) only see renewable capacity. `yuantian/nr_terminals.py` adds three terminals exposing NR state to the GP:

- `NR_STOCK_RATIO` (activity + integrated tree): how much of the project's NR budget is left right now, aggregated across NR resource types
- `NR_MODE_DEMAND_RATIO` (mode tree): this candidate mode's NR demand relative to remaining stock — the most directly decision-relevant of the three, since mode choice is what spends NR budget
- `NR_BUDGET_PRESSURE` (activity + integrated tree): a forward-looking, optimistic feasibility-risk proxy — remaining stock minus the sum of every not-yet-scheduled activity's cheapest-mode NR demand

Toggled with `nr_terminals_feature=True` in `ParametersGPHH`, or `--nr_terminals` on the CLI. Unlike the other extensions, these terminals are only meaningful on instances that still have their non-renewable resources — `read_instances(..., keep_non_renewable=True)` skips the conversion that the baseline pipeline normally applies (`to_renewable_only_rcpsp_model`, which deletes every NR resource before the GP ever sees the instance), so this extension's baseline comparison runs on a different instance set than extensions #1 and #2 and isn't numerically comparable to them out of the box.

Implementing this also surfaced a real, previously-unreachable bug in `rcpsp_simulation.py`'s `SerialSimulator.buildSolution`: with NR resources actually enforced, an activity that's precedence-eligible can end up with zero resource-feasible modes (its NR budget exhausted by earlier decisions), and the existing eligible-activity selection assumed every eligible activity has at least one eligible mode, crashing with `min() arg is an empty sequence`. Fixed by dropping zero-mode activities from the eligible set before choosing, falling back to the existing infeasible-schedule path if that empties the set entirely — this can never trigger on the renewable-only baseline, so it doesn't change anything for extensions #1/#2.

**Result** (`yuantian/experiments/nr_terminals_experiment.py`, pop=60, gen=25, 5 MMLIB50 NR-preserving instances, n=18 seeds after a follow-up to the initial 10): baseline+nr beats plain baseline on training fitness, 110.62 vs 118.25, and it's significant (p=0.0032, r=0.77, dropping the one infeasible baseline run from the comparison). Adding `cp_propagation`'s terminals on top of NR (baseline+nr+cp) doesn't help any further — 113.25, basically the same as nr alone. So the NR signal itself is doing real work, the CP terminals still aren't adding anything once NR is already there.

Reran the same baseline vs baseline+nr comparison on MMLIB+ instead of MMLIB50 (`nr_terminals_mmlib_plus_experiment.py`, same settings, n=10 seeds) to check it wasn't a MMLIB50 quirk — held up, actually got stronger: 250.10 vs 286.88, p=0.0059, r=0.93. Test fitness on MMLIB+ was a mess for both conditions though (most seeds came back infeasible on the held-out instances). Dug into this properly since it could've meant the whole result was a measurement artifact: checked one of the infeasible test instances by hand, and even the most conservative possible schedule (slowest mode on every activity, fully serial) blows the NR budget on it. So it's a real generalization gap, not a harness bug — the evolved heuristics that do well in training just aren't robust enough to handle MMLIB+'s tighter NR budgets on instances they haven't seen.

### 4. Resource-constrained critical path terminals

Extension #1 (critical path propagation) reported a null result on quality, and the diagnosis was that every one of those terminals is a linear combination of the baseline's own ES/EF/LS/LF — standard CPM slack is computed from the precedence graph alone, resources never enter it, so there was no genuinely new information for the GP to use. `yuantian/rccp_terminals.py` tests the natural follow-up hypothesis: does *resource-driven* criticality information do better, since it's something ES/EF/LS/LF cannot encode no matter how they're combined? Five terminals (four resource-contention terminals plus a resource-concentration/entropy terminal added afterward, see below), all read live from the simulator's own resource-tracking state during schedule construction (there's no static half to precompute, unlike extensions #1 and #3 above — see the module's docstring for why):

- `RCCP_BOTTLENECK_UTIL` (activity + integrated tree): current utilization of whichever renewable resource is most contended right now
- `RCCP_CANDIDATE_CONTENTION` (mode tree): the candidate mode's demand on that same bottleneck resource, relative to how much of it remains right now — the most directly decision-relevant of the original four
- `RCCP_SLACK` (activity tree): precedence-only `CP_SLACK_SCORE` discounted by current bottleneck utilization — a simple multiplicative approximation of "true" resource-constrained slack, not an exact resource-leveling computation
- `RCCP_PRESSURE_TREND` (activity + integrated tree, lower priority, added last): forward-looking complement to `RCCP_BOTTLENECK_UTIL` — sums currently-eligible activities' cheapest-mode demand on the bottleneck resource, relative to what remains, as a short look-ahead over the next few decisions
- `RCCP_RESOURCE_CONCENTRATION` (mode tree; small, bundled addition, not a separate contribution — see below)

Toggled with `rccp_terminals_feature=True` in `ParametersGPHH`, or `--rccp_terminals` on the CLI. Unlike extension #3, no special instance loading is needed — these terminals are about *renewable* resources, which the baseline pipeline's NR-stripping conversion never touches, so this extension runs on the same instances as the baseline and extensions #1/#2 and is directly comparable to them. Implementing it required two small additive changes to `rcpsp_simulation.py`: exposing `self.eligibles` (declared in the base `Simulator.__init__` but never actually populated before this) in both `SerialSimulator` and `ParallelSimulator`'s `buildSolution`, and adding a per-simulator `_current_resource_avail()` (the resource-contention analogue of the existing `_nr_remaining_stock()`) since Serial (no single global "now", a per-activity earliest-start frontier instead) and Parallel (a single global clock with a scalar resource ledger already kept in sync with it) represent "current resource availability" in genuinely different shapes.

**Bundled addition: resource-concentration (entropy) terminal.** Budget and contention terminals both treat resource types individually (or only flag the single worst one); neither distinguishes an activity demanding 8 units all from one resource type from one demanding 2 units each from four types — the first is far more exposed to any one resource becoming a bottleneck, the second is more robust. `RCCP_RESOURCE_CONCENTRATION` is the *complement* of the normalized Shannon entropy of the candidate mode's renewable-resource demand vector (`1 - H/log(k)`, `k` = number of renewable resource types in the instance, so the terminal means the same thing across instances with different resource-type counts) — higher = more concentrated = more constrained, matching this module's other terminals' sign convention. Cheap (one pass over one mode's own demand vector, no look-ahead, no other-activity state), so it rides along with this extension rather than being its own contribution. See `rccp_terminals.py`'s docstring for the k=1 and zero-demand edge cases.

**Result** (`yuantian/experiments/rccp_terminals_experiment.py`, pop=60, gen=25, 5 MMLIB50 instances, n=10 seeds, baseline / cp_propagation / rccp / both): null result across the board. Training fitness: baseline 12.25, cp_propagation 12.39, rccp 12.55, both 12.25 — basically indistinguishable, and none of the comparisons against baseline come close to significant (p=0.82, p=0.46, p=1.0). So the resource-driven-criticality hypothesis didn't pan out, at least not at this scale — giving the GP live resource-contention info instead of precedence-only CP info didn't translate into better trees. Same honest-null-result territory as extension #1.

### 5. Mode-interaction terminals

Extensions #3 and #4 expose *budget* (how much of a resource is left) and *contention* (who's fighting over the bottleneck right now) signals. `yuantian/mode_interaction_terminals.py` is a third, structurally different kind: a *consequence* signal — if I choose this mode for this activity, how much does that narrow down the feasible mode choices for the other, not-yet-decided activities sharing a resource window with it. This requires reasoning jointly about multiple activities' mode options, not any single activity's or the schedule's current state, so it's not expressible by any combination of the other extensions' terminals. It's also the most novel and least literature-established extension here — there's much less direct precedent for it in published GPHH/RCPSP work than for the resource-leveling ideas behind extensions #3/#4, and it's documented as such; a null result here would be exactly as reportable as extension #1's.

- `MI_CONSTRAINT_TIGHTENING` (mode tree): for the candidate mode, the mean fraction of each resource-sharing neighbor activity's own mode options that would become infeasible if this mode is chosen
- `MI_RECIPROCAL_SCARCITY` (mode tree): a cheaper proxy — how many of the candidate's own resource types are already in over-subscribed near-term demand among its neighbors, measured ~2.6x cheaper per call than the terminal above
- `MI_ACTIVITY_PRESSURE` (activity + integrated tree): the same constraint-tightening computation averaged over the candidate activity's own mode options, for use before a mode is chosen

Toggled with `mode_interaction_terminals_feature=True` in `ParametersGPHH`, or `--mode_interaction_terminals` on the CLI. No special instance loading needed, same as extension #4. Measured wall-clock cost (direct per-call timing plus full population-evaluation timing, both on MMLIB50): individual terminal calls run tens of microseconds (vs ~0.1us for an existing terminal like ES), and realistic evolved trees ran the GP loop ~3x slower with this extension enabled — real but, in absolute terms, still in the same low-minutes-per-run range as the other extensions at this experiment's scale. Two mitigations were applied (documented in the module): a per-decision cache for the shared resource-availability lookup (value-preserving, also speeds up extension #4 for free), and a cap on how many resource-sharing neighbors `MI_CONSTRAINT_TIGHTENING`/`MI_ACTIVITY_PRESSURE` examine (an actual approximation once more than 5 eligible activities share a resource with the candidate, which is the common case on MMLIB50).

**Result** (`yuantian/experiments/mode_interaction_experiment.py`, same scale as the others, n=10 seeds, baseline / mode_interaction / rccp / both): mode_interaction alone isn't significant (12.40 vs 13.37, p=0.16), and if anything the mean is worse. Stacking it with rccp ("both") makes things actively worse and that one IS significant — 14.21 vs baseline's 12.40, p=0.037, r=-0.6, meaning baseline beats "both" most of the time. So this extension doesn't help, and combining it with RCCP terminals actively hurts, possibly just adding noisy/redundant dimensions to the search space without giving the GP anything it can reliably exploit. Filed as a negative result, same spirit as extension #1 and #4 — the "least literature-backed" framing in `mode_interaction_terminals.py`'s docstring turned out to be the right level of skepticism.

## Citation

```bibtex
@INPROCEEDINGS{10612172,
  author={Tian, Yuan and Mei, Yi and Zhang, Mengjie},
  booktitle={2024 IEEE Congress on Evolutionary Computation (CEC)}, 
  title={Learning Heuristics via Genetic Programming for Multi-Mode Resource-Constrained Project Scheduling}, 
  year={2024},
  volume={},
  number={},
  pages={01-08},
  keywords={Schedules;Sequential analysis;Operations research;Processor scheduling;Buildings;Project management;Genetic programming;Project Scheduling;Multiple Modes;Hyper-heuristics;Genetic Programming},
  doi={10.1109/CEC60901.2024.10612172}}

```

## Acknowledgements

This code is developed based on the following projects:

- [DEAP](https://github.com/DEAP/deap)
- [discrete-optimization](https://github.com/airbus/discrete-optimization)

The benchmark dataset MMLIB is from [Operations Research and Scheduling (OR&amp;S) Research group](https://www.projectmanagement.ugent.be/research/project_scheduling/mmrcpsp) at Ghent University, Belgium.
