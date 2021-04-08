import logging
import typing

import os
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from smallab.experiment_types.overlapping_output_experiment import (OverlappingOutputCheckpointedExperimentReturnValue,
                                                                    OverlappingOutputCheckpointedExperiment)
from smallab.smallab_types import Specification, ExpProgressTuple

from sample_sim.data_model.loaders.uav_picture_loader import load_from_egret_data
from sample_sim.general_utils import coordinates_to_unit_cube
from sample_sim.memory_mapper_utility import memory_mapped_dict, map_memory
from sample_sim.planning.pomcp_planner import POMCPPlanner
from sample_sim.planning.pomcp_utilities import ActionModel

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# matplotlib.use('TKAgg')

from sample_sim.data_model.data_model import TorchExactGPBackedDataModel, DataModel
from sample_sim.data_model.loaders.ecomapper_loader import load_from_ecomapper_data
from sample_sim.robot_manager import Simulated2dTemporalPoint
from sample_sim.vis.crit_path_vis import CriticalPathVisualizer
import matplotlib.pyplot as plt
import numpy as np
from sample_sim.data_model.loaders.analytical_function_loader import (curved_function, exponential_function,
                                                                      create_data,
                                                                      sbo_function)


class PlanningExperiment(OverlappingOutputCheckpointedExperiment):
    def initialize(self, specification: Specification):

        self.saved_before_first_action = False
        logger = logging.getLogger(self.get_logger_name())
        logger.setLevel(logging.DEBUG)
        self.specification = specification
        logger.info(f"Begin {specification}")
        self.use_heteroskedastic_noise = ["use_heteroskedastic_noise"]
        self.file = specification["file"]
        if "data" in self.file:
            self.action_model = ActionModel.XYZ
        elif "fn" in self.file:
            self.action_model = ActionModel.XYT
        else:
            raise Exception(f"cannot detect action model for file {self.file}")
        self.plot = specification["plot"]
        self.seed = specification["seed"]
        self.objective_c = specification["objective_c"]
        self.used_rollouts = 0
        self.rollout_allocation_method = specification["rollout_allocation_method"]
        self.state_space_dimensionality = specification["state_space_dimensionality"]
        self.rollout_number_goal = specification["rollout_number_goal"]

        # This needs to be done to make the spawn context work so the gpu can work
        map_memory([self.file], [self.state_space_dimensionality])

        # Plan commitment feature
        self.plan_commitment_algorithm = specification["plan_commitment_algorithm"]
        self.plan_threshold = specification["plan_threshold"]

        if "waste_unused_rollouts" in specification:
            self.waste_unused_rollouts = specification["waste_unused_rollouts"]
        else:
            self.waste_unused_rollouts = False

        self.current_traj = []

        self.max_budget = specification["planning_steps"]#self.state_space_dimensionality[2]

        self.alpha_param = specification["alpha_param"]
        self.beta_param = specification["beta_param"]
        self.epsilon = specification["epsilon"]
        self.delta = specification["delta"]
        self.sample_observations = specification["sample_observations"]
        self.use_expected_improvement = specification["use_expected_improvement"]

        self.data_X = []
        self.data_Y = []
        self.noises = []
        self.budgets_that_were_used = []
        if "fn" in self.file:
            oracle_model, workspace = create_data(self.file)
            oracle_model_for_hps = oracle_model
            if self.file == "fn:exp":
                self.chosen_function = exponential_function
            elif self.file == "fn:curved":
                self.chosen_function = curved_function
            elif self.file == "fn:sbo":
                self.chosen_function = sbo_function
            else:
                raise Exception()
            self.workspace = workspace
        elif "drone:" in self.file:
            oracle_model, workspace = load_from_egret_data(self.file.split(":")[1])
            oracle_model_for_hps = oracle_model
        else:
            oracle_model, workspace = load_from_ecomapper_data(self.file, 15, 50)
            oracle_model_for_hps = oracle_model
            self.oracle_model = oracle_model
            self.workspace = workspace
            workspace = self.workspace
        oracle_model.model.eval_model()
        logger.info(
            f"Workspace: X ({self.workspace.xmin} - {self.workspace.xmax}) Y ({self.workspace.ymin} - {self.workspace.ymax}) Z ({self.workspace.zmin} - {self.workspace.zmax})")

        self.used_budget = 0
        # self.planning_iterations = specification["planning_iterations"]

        # This is a dumb way to do this
        initial = [int((workspace.xmin + workspace.xmax) / 2), int((workspace.ymin + workspace.ymax) / 2),
                   workspace.zmin]
        x = initial[0]
        y = initial[1]
        t = initial[2]

        self.auv = Simulated2dTemporalPoint(x, y, t)

        self.planner = POMCPPlanner(workspace, budget=self.max_budget,
                                    logger_name=self.get_logger_name(), seed=self.seed,
                                    objective_c=self.objective_c,
                                    action_model=self.action_model,
                                    total_rollouts=self.rollout_number_goal,
                                    rollout_allocation_method=self.rollout_allocation_method,
                                    state_space_dimensionality=self.state_space_dimensionality,
                                    filename=self.file,
                                    alpha_param=self.alpha_param,
                                    beta_param=self.beta_param, epsilon=self.epsilon, delta=self.delta,
                                    plan_commitment_algorithm=self.plan_commitment_algorithm,
                                    plan_threshold=self.plan_threshold,
                                    waste_unused_rollouts=self.waste_unused_rollouts,
                                    sample_observations=self.sample_observations,
                                    use_expected_improvement=self.use_expected_improvement)

        if self.plot:
            vbounds = (np.min(oracle_model.Ys), np.max(oracle_model.Ys))
            plt.ion()
            # env_view = GeneratingFieldViewVis(oracle_model, workspace, vbounds=vbounds)
            # env_view.update()
            # rig_view = RIGVisualizer(workspace.dimensions(), self.planner)

        X_t = np.array([self.auv.get_current_state()[:3]])
        if "fn" in self.file:
            samples = self.chosen_function(
                coordinates_to_unit_cube(X_t, self.workspace))
        else:
            samples = oracle_model.query_many(X_t, return_std=False)

        self.auv_data_model = TorchExactGPBackedDataModel(X_t, samples, logger=self.logger, workspace=workspace)
        self.auv_data_model.model.copy_hyperparameters_from(oracle_model_for_hps.model.model)
        self.auv_data_model.update(X_t, samples)
        self.auv_data_model.model.eval_model()

        if self.plot:
            crit_path_view = CriticalPathVisualizer(self.auv, self.planner, self.auv_data_model,
                                                    workspace, vbounds=vbounds)
            # crit_path_view.update()
            # self.auv_view = AUVViewVis(self.auv, self.auv_data_model, workspace)  # , vbounds=vbounds)
            # self.visualizers = [env_view, self.auv_view, rig_view]
            # self.visualizers = [self.auv_view]
            self.visualizers = [crit_path_view]
            # self.visualizers = [self.auv_view, rig_view]

        self.data_X.append(deepcopy(self.auv_data_model.Xs))
        self.data_Y.append(deepcopy(self.auv_data_model.Ys))
        # empty_cache()

        self.X_t = memory_mapped_dict[str(self.file) + str(self.state_space_dimensionality) + "Sreal_ndarrays"]
        if "fn" in self.file:
            X_t = coordinates_to_unit_cube(
                self.X_t,
                self.workspace)
            self.gt_Y = self.chosen_function(X_t)
        else:
            neigh = KNeighborsRegressor(n_neighbors=10)
            neigh.fit(self.oracle_model.Xs, self.oracle_model.Ys)
            self.oracle_model = neigh
            self.gt_Y = neigh.predict(self.X_t)

        del oracle_model_for_hps

    def calculate_progress(self) -> ExpProgressTuple:
        return self.used_budget, self.max_budget

    def calculate_return(self) -> OverlappingOutputCheckpointedExperimentReturnValue:

        logger = logging.getLogger(self.logger)
        logger.debug(f"Budget Remaining: {self.planner.budget}")
        should_continue = self.planner.budget > 1
        out_specification = deepcopy(self.specification)
        out_specification["budget"] = self.used_budget

        m = self.auv_data_model.query_many(self.X_t, return_std=False)

        mmae = mean_absolute_error(self.gt_Y, m)
        mmse = np.sqrt(mean_squared_error(self.gt_Y, m))
        logger.debug(
            "AUV Output MAX: {} MIN: {} STD: {} MEAN: {} ".format(np.max(m), np.min(m),
                                                                  np.std(m), np.mean(m)))
        logger.debug(
            "GT Output MAX: {} MIN: {} STD: {} MEAN: {} ".format(np.max(self.gt_Y), np.min(self.gt_Y),
                                                                 np.std(self.gt_Y), np.mean(self.gt_Y)))

        stats = {"MMAE": mmae, "MMSE": mmse}

        rewards = list(sorted(map(np.average, self.planner.get_root_rewards()), reverse=True))
        highest_reward = rewards[0]
        second_highest_reward = rewards[1]

        logging.getLogger(self.get_logger_name()).info("Completed: {stats}".format(stats=stats))
        return_dict = {"Xs": self.data_X,
                       "Ys": self.data_Y,
                       "used_budget": self.used_budget,
                       "total_used_rollouts": self.used_rollouts,
                       "reward_gap": highest_reward - second_highest_reward,
                       "reward": highest_reward,
                       "pomcp_traj_length": len(self.current_traj),
                       "tTest_pval": self.planner.plan_ttest_pval,
                       "reward_kurtosis": self.planner.plan_reward_kurtosis,
                       }
        # return_dict.update(stats)
        return_dict.update(stats)
        progress, outof = self.calculate_progress()
        return_value = OverlappingOutputCheckpointedExperimentReturnValue(should_continue, out_specification,
                                                                          return_dict, progress, outof)

        return return_value

    def step(self) -> typing.Union[ExpProgressTuple, OverlappingOutputCheckpointedExperimentReturnValue]:
        if self.saved_before_first_action == False:
            self.saved_before_first_action = True

            return self.calculate_return()

        logger = logging.getLogger(self.get_logger_name())
        # logger.debug(f"Remaining Budget: {self.planner.budget} / {self.max_budget}")
        self.budgets_that_were_used.append(self.planner.budget)

        if self.current_traj == []:
            self.current_traj = self.planner.next_step(self.auv, self.auv_data_model, self.workspace)
            self.used_rollouts += self.planner.cur_plan.rollouts_used_this_iteration
        next_step = self.current_traj.pop(0)

        if "fn" in self.file:
            logger.debug(next_step)
            next_samples = self.chosen_function(
                coordinates_to_unit_cube(np.array([next_step[:self.workspace.dimensions()]]), self.workspace))
        else:
            if isinstance(self.oracle_model, DataModel):
                next_samples = self.oracle_model.query_many([next_step[:self.workspace.dimensions()]],
                                                            return_std=False)
            else:
                next_samples = self.oracle_model.predict([next_step[:self.workspace.dimensions()]])

        logger.debug(
            "MAX: {} MIN: {} STD: {} MEAN: {} ".format(np.max(next_samples), np.min(next_samples),
                                                       np.std(next_samples), np.mean(next_samples)))
        if self.workspace.dimensions() == 3:
            self.auv.update_state(next_step[:])
        else:
            self.auv.update_state([next_step[0], next_step[1], 0, 0])
        next_step_no_theta = next_step[:self.workspace.dimensions()]
        self.auv_data_model.update([next_step_no_theta], next_samples)

        if self.plot:
            for vis in self.visualizers:
                vis.update()

                plt.draw()

                plt.pause(0.1)

                if not os.path.exists(self.get_logger_name()):
                    os.mkdir(self.get_logger_name())
                vis.save(self.get_logger_name() + "/" + str(self.planner.budget) + ".png")
            # plt.show()
        self.data_X.append(deepcopy(self.auv_data_model.Xs))
        self.data_Y.append(deepcopy(self.auv_data_model.Ys))
        self.noises.append(deepcopy(self.auv_data_model.input_uncertanties))

        self.planner.budget -= 1
        self.used_budget += 1

        logger.debug(f"Remaining Budget: {self.planner.budget}")
        # self.planner.budget = self.planning_horizon
        return self.calculate_return()

    def steps_before_checkpoint(self):
        # Save every 1/8th of the experiments
        return int(self.max_budget / 8)
