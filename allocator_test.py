import logging
import os
from os.path import dirname

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MP_NUM_THREADS"] = "1"

from smallab.runner_implementations.multiprocessing_runner import MultiprocessingRunner

from sample_sim.memory_mapper_utility import map_memory
from plannin_experiment import PlanningExperiment

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

from smallab.specification_generator import SpecificationGenerator
from smallab.runner.runner import ExperimentRunner
from smallab.runner_implementations.main_process_runner import MainRunner
import os
from os.path import exists, abspath, join, dirname
from os import mkdir

if __name__ == '__main__':
    if "experiments" in os.getcwd():
        os.chdir("../..")

    this_dir = dirname(abspath(__file__))
    for dir_name in ('.cache', '.params'):
        path = join(this_dir, dir_name)
        if not exists(path):
            mkdir(path)
    non_experiment_logger = logging.getLogger("default")
    non_experiment_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    non_experiment_logger.addHandler(handler)
    name = "allocator_test"

    files = ["fn:sbo"]

    generation_specifications = {"plot": False,
                                 "file": files,
                                 "objective_c": 10,
                                 "state_space_dimensionality": [[50,50,200]],
                                 "rollout_number_goal": 30000,
                                 "rollout_allocation_method": ["fixed","beta-ugapeb", "beta","beta-sr"],
                                 "seed": list(range(5)),
                                 "alpha_param": 6,
                                 "beta_param":1,
                                 "epsilon": 10,
                                 "delta": 0.1,
                                 "plan_commitment_algorithm": "n_steps",
                                 "plan_threshold": [1],
                                 "sample_observations": False,
                                 "use_expected_improvement":False,
                                 "planning_steps": 200
                                 }



    ##Create shared memory
    map_memory(generation_specifications["file"], generation_specifications["state_space_dimensionality"])

    specifications = SpecificationGenerator().generate(generation_specifications)
    runner = ExperimentRunner()
    DEBUG = False


    if DEBUG:
        runner.run(name, specifications, PlanningExperiment(), propagate_exceptions=True,
                   specification_runner=MainRunner(), use_dashboard=False, force_pickle=True, context_type="fork")
    else:

        runner.run(name, specifications, PlanningExperiment(), propagate_exceptions=False,
                   specification_runner=MultiprocessingRunner(), context_type="fork", use_dashboard=True,
                   force_pickle=True)
