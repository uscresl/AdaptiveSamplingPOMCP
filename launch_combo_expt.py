import logging
import os
from os.path import exists, abspath, join, dirname
from os import mkdir
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MP_NUM_THREADS"] = "1"

from smallab.runner_implementations.multiprocessing_runner import MultiprocessingRunner

from plannin_experiment import PlanningExperiment

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logging.getLogger("smallab").propogate = False

from smallab.specification_generator import SpecificationGenerator
from smallab.runner.runner import ExperimentRunner
from smallab.runner_implementations.main_process_runner import MainRunner
from itertools import product
from sample_sim.memory_mapper_utility import map_memory
from smallab.file_locations import get_experiment_save_directory
import sys
import numpy as np

if __name__ == '__main__':
    if "experiments" in os.getcwd():
        os.chdir("../..")

    this_dir = dirname(abspath(__file__))
    for dir_name in ('.cache', '.params'):
        path = join(this_dir, dir_name)
        if not exists(path):
            mkdir(path)


    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "IPP_POMCP"
    num_seeds = 5
    num_steps = 200
    base_specs = {
        "plot": False,
        "file": ["fn:sbo"],
        "seed": list(range(num_seeds)),
        "objective_c": 10, # 10 for sbo, 100 for validation envs
        "state_space_dimensionality": [[50,50,200]], # for fn:sbo, [[62, 70, 5]], # for validation envs
        "rollout_number_goal": [num_steps * 150], # z_steps * 150
        "alpha_param": 6,
        "beta_param": 1,
        "epsilon": 10,
        "delta": 0.1,
        "sample_observations": False,
        "use_expected_improvement": False,
        "planning_steps": [num_steps],
        }

    gen_baseline = base_specs.copy()
    gen_baseline.update({
        "plan_commitment_algorithm": "n_steps",
        "plan_threshold": [1],
        "rollout_allocation_method": ["fixed"],
        "waste_unused_rollouts": [False],
        })
    specs_baseline = SpecificationGenerator().generate(gen_baseline)

    gen_our_best = base_specs.copy()
    gen_our_best.update({
        "plan_commitment_algorithm":"tTest",
        "plan_threshold":[0.05],
        "rollout_allocation_method": ["beta-ugapeb"],
        "waste_unused_rollouts": [True],
        })
    specs_our_best = SpecificationGenerator().generate(gen_our_best)

    specifications = []
    specifications += specs_baseline
    specifications += specs_our_best

    print(f"Expt {name}:\t{len(specifications)/num_seeds} specs to run, over {num_seeds} seeds")
    for spec in specifications:
        if spec["seed"] == 0:
            print(spec)

    runner = ExperimentRunner()
    map_memory(base_specs["file"], base_specs["state_space_dimensionality"])
    DEBUG = False

    if DEBUG:
        runner.run(name, specifications, PlanningExperiment(), propagate_exceptions=True,
                   specification_runner=MainRunner(), use_dashboard=False, force_pickle=True, context_type="fork")
    else:
        gpus = 4
        jobs_per_gpu = 2
        resources = list(product(list(range(gpus)), list(range(jobs_per_gpu))))
        runner.run(name, specifications, PlanningExperiment(), propagate_exceptions=False,
                   specification_runner=MultiprocessingRunner(), context_type="fork", use_dashboard=True,
                   force_pickle=True)
