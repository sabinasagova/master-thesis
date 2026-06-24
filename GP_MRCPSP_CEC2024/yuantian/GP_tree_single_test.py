"""
A test script to evaluate a single GP individual on multiple RCPSP instances.
"""
if __name__ == "__main__":
    files = [
        "./discrete_optimization_data/mm//MMLIB//MMLIB50/J5090_3.mm",
        "./discrete_optimization_data/mm//MMLIB//MMLIB50/J5033_1.mm",
        "./discrete_optimization_data/mm//MMLIB//MMLIB50/J5033_3.mm",
        "./discrete_optimization_data/mm//MMLIB//MMLIB50/J5019_3.mm",
    ]
    from yuantian.gphh_solver import ParametersGPHH, evaluate_heuristic,read_instances
    from deap import gp
    from yuantian.rcpsp_simulation import (
        SerialSimulator,
        Simulator,
        ParallelSimulator,
        DecisionTypeEnum,
        SimulatorTypeEnum,
    )

    problems = read_instances(files)
    param = ParametersGPHH.default(
        simulator_type=SimulatorTypeEnum.SERIAL_SGS,
        decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE,
        dynamic_CPM_feature=True,
    )
    individual = {
        'activity': 'max(div(mul(sub(sub(add(LS_d, LF_d), avg_task_duration), min_task_duration), sub(TSC, max_RReq_m)), div(sub(mul(mul(LS_d, LF_d), max(LF_d, ES_d)), LS_d), sub(sub(min(LF_d, SC), add(LF_d, SC)), min(mul(TSC, EF_d), div(LS_d, LF_d))))), max(add(max(ES_d, LF_d), max(ES_d, add(LS_d, LF_d))), LS_d))', 
        'mode': 'add(add(div(sub(div(EFFT, div(mul(avg_RReq, avg_RReq), min(RR, max_RReq))), min(sub(max(EFFT, GRD), mul(min_RReq, task_duration)), min(min(min_RReq, min_RReq), EFFT))), RR), mul(avg_RReq, EFFT)), add(RR, mul(EFFT, EFFT)))'
    }
    fitness = evaluate_heuristic(
        individual=individual,
        domains=problems,
        compile_func=gp.compile,
        pset=param.set_primitves,
        decision_type=DecisionTypeEnum.ACTIVITY_THEN_MODE,
        simulator=param.simulator,
    )
    print(fitness)
