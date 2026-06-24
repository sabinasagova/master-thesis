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

Everything above is Tian et al.'s original code. Below is what I added on top of it, basically two separate ideas I wanted to test against their baseline GPHH.

### 1. Critical path propagation terminals

The baseline already has ES/EF/LS/LF (classic CPM dates) as terminals, but I wanted to try giving the GP a more "ready-made" critical-path signal instead of making it rediscover slack-like quantities by combining ES/EF/LS/LF itself. So I added `yuantian/cp_propagation.py`, which computes four extra per-activity features once per instance (cached on `problem.cp_features`, never recomputed during a GP run):

- `CP_FORWARD`: longest path from the activity to the sink, normalized by the project makespan estimate
- `CP_BACKWARD`: basically the same idea as LS but written explicitly as a propagation
- `CP_SLACK_SCORE`: backward minus forward
- `CP_PROB`: how close the activity's forward value is to the most "critical" activity in the instance

These are toggled with `cp_propagation_feature=True` in `ParametersGPHH`, or `--cp_propagation` on the CLI. The terminals themselves are wired into `rcpsp_simulation.py`'s `FeatureEnum`/`feature_function_map`.

**Result so far** (`yuantian/experiments/cp_propagation_experiment.py`, pop=60, gen=25, 5 MMLIB50 instances, 10 seeds): adding these terminals sped up convergence quite a bit (8.7 vs 14.4 generations to reach the best solution) and produced the single best run overall, but didn't move the mean or the held-out test fitness in any meaningful way (Wilcoxon p=0.88, not significant). My read on this: `CP_BACKWARD` is literally the same formula as the existing `LS` terminal, and the other two are linear combos of terminals the GP already has, so there isn't really new information here, just a shortcut to something the tree could already build. Explains the convergence-speed bump without a quality bump.

### 2. Epsilon-lexicase selection + critical path local search

The bigger change: I swapped tournament selection for epsilon-lexicase selection (`yuantian/hybrid_gp.py`), and added a memetic local search step (`yuantian/local_search.py`) that runs critical-path repair (mode swaps, precedence-respecting activity swaps, slack-based resource shifting) on the top ~8% of the population every generation. Important detail: the local search only ever touches an individual's recorded fitness, never the GP tree itself, so the representation stays exactly what the baseline uses. It's a Lamarckian/memetic trick, not a different genotype.

`evaluate_heuristic` in `gphh_solver.py` now also stores `individual.case_fitness` (the per-instance deviation scores), which is what the lexicase selection needs to filter candidates case by case.

I ran this in stages, getting more rigorous (and more expensive) each time:

- **First pass**, small scale (pop=60, gen=25, 5 instances, 10 seeds): lexicase+localsearch beat the baseline mean (11.87 vs 13.48) and best-of-run (10.07 vs 12.35), and it was actually significant (Wilcoxon p=0.0059, r=0.93). Convergence was slower though (17.5 vs 12.0 generations), which matches what you'd expect from lexicase keeping more diverse "specialist" individuals around instead of letting tournament selection collapse onto one strong generalist early.

- **Bigger pass** on real MMLIB50 (`yuantian/experiments/full_mmlib_experiment.py`), using the official 60/20/20 train/val/test split logic but on a stratified subset of 10 classes (out of 108) to keep it runnable on my laptop instead of the ~200+ days the literal pop=1000/gen=50/full-split spec would have taken. pop=60, gen=20, 10 seeds, both serial and parallel SGS, 3 conditions (baseline / lexicase / lexicase+localsearch). On serial SGS, lexicase and the full hybrid both significantly beat baseline on training fitness (p=0.0098 and p=0.027), but the same comparison on held-out test fitness landed at p=0.109 (lexicase), same direction, just not under 0.05 yet. Also found something I wasn't expecting: serial SGS is dramatically better than parallel SGS for every method here (test fitness ~18-19 vs ~30-33), so SGS choice matters way more than which selection/search method you use, at least at this scale.

- Did a quick power calculation on that 0.109 result (r=0.667 at n=10 suggested ~14-16 seeds might be enough to flip it), so I added 6 more seeds just for that one comparison (serial, baseline vs lexicase) instead of rerunning everything. At n=16 it moved to p=0.074, r=0.543, still not significant, train fitness comparison was already significant and got more so (p=0.00058). Currently running another batch to push to n=26 since the revised effect size estimate suggests that's closer to where it'd actually cross the line. Train fitness clearly favors lexicase either way, it's specifically the test-set number that's taking more seeds to pin down, probably because the held-out set is only 10 instances so it's noisier.

- Also added a proper 2x2 ablation to `full_mmlib_experiment.py` (baseline / lexicase only / local-search only / lexicase+localsearch) so I can actually tell whether gains come from the selection change, the local search, or both together, instead of just comparing baseline to the combined method. Haven't run the full 80-run version of this yet (would be ~8-9h), code is ready though.

One thing worth flagging if anyone reruns this: `ParallelSimulator.get_eligibles()` has `if __debug__:` print statements that dump a huge amount of text per call. Doesn't matter for correctness but it'll blow up your terminal/log files unless you run with `python -O` (which strips `__debug__` blocks). I always run the experiment scripts that way.

All the experiment drivers live in `yuantian/experiments/`, raw per-run JSON results and a convergence plot are saved under `yuantian/experiments/results/<experiment_name>/`.

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
