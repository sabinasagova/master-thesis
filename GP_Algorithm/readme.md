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
