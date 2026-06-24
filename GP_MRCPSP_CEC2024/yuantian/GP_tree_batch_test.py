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
    RESULT_HOME = r"Z:\code\discrete_optimization\results\2023.10.24static"
    NEW_RESULT_HOME = r"Z:\code\discrete_optimization\results\newtestreuslt"
    # Read test problems
    from rcpsp_dataset import RCPSPDatabase

    test_set_files: list[str] = RCPSPDatabase.get_all_MMLIB_PLUS_files()
    test_set_files: list[str] = [
        "./discrete_optimization_data/mm//MMLIB//MMLIB50/J501_4.mm"
    ]
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
        # Read from each configuration
        conf_dir: str = os.path.join(RESULT_HOME, str(conf))
        if not os.path.exists(conf_dir):
            continue
        new_result_dir: str = os.path.join(NEW_RESULT_HOME, str(conf))
        os.makedirs(new_result_dir, exist_ok=False)
        log_files: list[str] = os.listdir(conf_dir)
        param = ParametersGPHH.default(
            simulator_type=configs[conf][0], decision_type=configs[conf][1]
        )
        # Read best GP tree from each run
        for log_file in log_files:
            log = json.load(open(os.path.join(conf_dir, log_file)))
            individual = eval(log["best_heuristic_validation"]["tree"])
            test_fitness = evaluate_heuristic(
                individual=individual,
                domains=test_set,
                compile_func=gp.compile,
                pset=param.set_primitves,
                decision_type=conf[1],
                simulator=param.simulator,
            )
            result = {
                "best_heuristic_validation": {
                    "tree": str(individual),
                    "test_fitness": test_fitness[0]
                }
            }
            json.dump(result, open(os.path.join(new_result_dir, log_file), mode="w"))
