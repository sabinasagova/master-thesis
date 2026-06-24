"""
Evaluate best individuals on a new test set and save the results.
The result structure is like
RESULT_HOME
|-- serial_activity
|   |-- 0.json
|   |-- 1.json
|   `-- ...
|-- serial_mode
|   |-- 0.json
|   |-- 1.json
|   `-- ...

"""
from gphh_solver import ParametersGPHH, evaluate_heuristic
from yuantian.rcpsp_dataset import read_instances
from deap import gp
from multitreegp import MultiPrimitiveTree
from rcpsp_simulation import (
    SerialSimulator,
    Simulator,
    ParallelSimulator,
    DecisionTypeEnum,
    SimulatorTypeEnum,
)

if __name__ == "__main__":
    RESULT_HOME = ""
    NEW_RESULT_HOME = ""
    # Read test problems
    test_set_files: list[str] = [
        "./discrete_optimization_data/mm//MMLIB//MMLIB50/J501_4.mm"
    ]
    from yuantian.rcpsp_dataset import RCPSPDatabase

    test_set_files: list[str] = RCPSPDatabase.get_all_MMLIB_PLUS_files()
    test_set: list = read_instances(test_set_files)
    # Read best individuals in each run and write fitness in new test set
    configs = {
        "serial_activity": [
            SimulatorTypeEnum.SERIAL_SGS,
            DecisionTypeEnum.ACTIVITY_THEN_MODE,
        ],
        "serial_mode": [
            SimulatorTypeEnum.SERIAL_SGS,
            DecisionTypeEnum.MODE_THEN_ACTIVITY,
        ],
        "serial_simultaneous": [
            SimulatorTypeEnum.SERIAL_SGS,
            DecisionTypeEnum.SIMULTANEOUS,
        ],
        "parallel_activity": [
            SimulatorTypeEnum.PARALLEL_SGS,
            DecisionTypeEnum.ACTIVITY_THEN_MODE,
        ],
        "parallel_mode": [
            SimulatorTypeEnum.PARALLEL_SGS,
            DecisionTypeEnum.MODE_THEN_ACTIVITY,
        ],
        "parallel_simultaneous": [
            SimulatorTypeEnum.PARALLEL_SGS,
            DecisionTypeEnum.SIMULTANEOUS,
        ],
    }
    import os
    import json

    for conf in configs:
        conf_dir: str = os.path.join(RESULT_HOME, str(conf))
        log_files: list[str] = os.listdir(conf_dir)
        param = ParametersGPHH.default(
            simulator_type=configs[conf][0], decision_type=configs[conf][1]
        )
        for log_file in log_files:
            log = json.load(open(os.path.join(conf_dir, log_file)))
            individual = eval(log["best_heuristic_validation"]["tree"])
            test_fitness = evaluate_heuristic(
                individual=individual,
                domains=test_set,
                compile_func=gp.compile,
                pset=param.set_primitves,
                decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE,
                simulator=param.simulator,
            )
            result = {
                "best_heuristic_validation": {
                    "tree": str(individual),
                    "test_fitness": test_fitness,
                    "category": RESULT_HOME
                }
            }
