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

import numpy as np

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

    name = "plan_commit_comparison"
    num_seeds = 5
    base_specs = {
        "plot": False,
        "file": ["fn:sbo"],
        "seed": list(range(num_seeds)),
        "objective_c": [10], # 10 for sbo, 100 for validation envs
        "state_space_dimensionality": [[50,50,200]], # for fn:sbo, 
        "rollout_number_goal": [200 * 150], # z_steps * 150
        "alpha_param": 6,
        "beta_param": 1,
        "epsilon": 10,
        "delta": 0.1,
        "sample_observations": False,
        "use_expected_improvement": False,
        "planning_steps": [200],
        "rollout_allocation_method": ["fixed"],
        "waste_unused_rollouts": [False],
        }

    gen_baseline = base_specs.copy()
    gen_baseline.update({
        "plan_commitment_algorithm": "n_steps",
        "plan_threshold": [1],
        })
    specs_baseline = SpecificationGenerator().generate(gen_baseline)

    gen_ugapec = base_specs.copy()
    gen_ugapec.update({
        "plan_commitment_algorithm":"ugapec",
        # 1000 should allow for more points to added to self.current_traj. This means fewer rollouts
        "plan_threshold": [5, 10], # [1, 5, 10, 100, 1000],
        })
    specs_ugapec = SpecificationGenerator().generate(gen_ugapec)

    gen_tTest = base_specs.copy()
    gen_tTest.update({
        "plan_commitment_algorithm":"tTest",
        # p-value
        "plan_threshold":[0.05, 0.1], # [0.0, 0.25, 0.5],
        })
    specs_tTest = SpecificationGenerator().generate(gen_tTest)

    gen_tree_count = base_specs.copy()
    gen_tree_count.update({
        "plan_commitment_algorithm":"tree_count",
        # 0 => all tree counts accepted. So it goes completely to the leaf (same as n_steps inf)
        "plan_threshold":[1, 5, 10],
        })
    specs_tree_count = SpecificationGenerator().generate(gen_tree_count)
    
    specifications = []
    specifications += specs_baseline
    specifications += specs_ugapec
    specifications += specs_tTest

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
