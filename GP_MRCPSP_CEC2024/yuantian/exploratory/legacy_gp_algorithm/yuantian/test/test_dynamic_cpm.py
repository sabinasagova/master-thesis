if __name__ == "__main__":
    from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
    from yuantian.gphh_solver import read_instances
    from yuantian.rcpsp_dataset import RCPSPDatabase
    from yuantian.rcpsp_simulation import SerialSimulator, ParallelSimulator
    files = RCPSPDatabase.get_all_MMLIB_50_files()
    problems:list[RCPSPModel] = read_instances(files)
    simulator = SerialSimulator()
    for problem in problems:
        simulator.rcpsp_problem = problem
        simulator.scheduled=[]
        simulator.minimum_starting_time = {
            act:0 for act in problem.tasks_list
        }
        simulator._compute_dynamic_cpm(eligible=[])
        for task in problem.tasks_list:
            if not all([
            problem.cpm[task]._ESD == simulator.dynamic_cpm[task]._ESD,
            problem.cpm[task]._EFD == simulator.dynamic_cpm[task]._EFD,
            problem.cpm[task]._LSD == simulator.dynamic_cpm[task]._LSD,
            problem.cpm[task]._LFD == simulator.dynamic_cpm[task]._LFD,
            ] ):
                raise ValueError("!!!")
    print("Serial SGS dynamic CPM passed")

    simulator = ParallelSimulator()
    for problem in problems:
        simulator.rcpsp_problem = problem
        simulator.scheduled=[]
        simulator.minimum_starting_time = {
            act:0 for act in problem.tasks_list
        }
        simulator.current_time = 0
        simulator._compute_dynamic_cpm(eligible=[])
        for task in problem.tasks_list:
            if not all([
            problem.cpm[task]._ESD == simulator.dynamic_cpm[task]._ESD,
            problem.cpm[task]._EFD == simulator.dynamic_cpm[task]._EFD,
            problem.cpm[task]._LSD == simulator.dynamic_cpm[task]._LSD,
            problem.cpm[task]._LFD == simulator.dynamic_cpm[task]._LFD,
            ] ):
                raise ValueError("!!!")
    print("Parallel SGS dynamic CPM passed")

