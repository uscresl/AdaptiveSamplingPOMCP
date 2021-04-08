import json
import logging
import unittest

import numpy as np
import random
from statistics import mode, variance
import scipy.stats as stats

from pomcp.pomcp import POMCP, average_or_0_on_empty
from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace, RectangularPrismWorkspace
from sample_sim.memory_mapper_utility import memory_mapped_dict
from sample_sim.planning.utils import euc_dist
from sample_sim.planning.planning import PlanningAgent
from sample_sim.planning.pomcp_generator import Generator
from sample_sim.planning.pomcp_rollout_allocation.beta_dist import CumulativeBetaDistributions
from sample_sim.planning.pomcp_rollout_allocation.fixed import FixedRolloutAllocator
from sample_sim.planning.pomcp_rollout_allocation.succesive_rejects import SuccessiveRejects
from sample_sim.planning.pomcp_rollout_allocation.ugap import UGapEb, UGapEc
from sample_sim.planning.pomcp_utilities import (apply_action_to_state, create_adjacency_dict, ActionModel,
                                                 adjacency_dict_to_numpy, get_uct_c, action_enum, get_default_low_param, get_default_hi_param)
from sample_sim.robot_manager import RobotManager


class POMCPPlanner(PlanningAgent):
    def __init__(self, workspace: Workspace, budget: int, logger_name, seed,
                 objective_c, action_model: ActionModel, total_rollouts, rollout_allocation_method, state_space_dimensionality, filename,
                 alpha_param, beta_param, epsilon, delta,
                 plan_commitment_algorithm: str, plan_threshold: float, waste_unused_rollouts: bool,sample_observations,use_expected_improvement):
        self.use_expected_improvement = use_expected_improvement
        self.logger_name = logger_name
        self.budget = budget
        self.state_space_dimensionality = state_space_dimensionality
        self.action_enum = action_enum(action_model)
        self.x_step = round((workspace.xmax - workspace.xmin) / self.state_space_dimensionality[0], 2)
        self.y_step = round((workspace.ymax - workspace.ymin) / self.state_space_dimensionality[1], 2)
        self.z_step = round((workspace.zmax - workspace.zmin) / self.state_space_dimensionality[2], 2)
        self.step_sizes = (self.x_step, self.y_step, self.z_step)
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        self.epsilon = epsilon
        self.delta = delta
        self.waste_unused_rollouts = waste_unused_rollouts
        self.sample_observations = sample_observations

        self.rollout_allocation_method = rollout_allocation_method

        S = memory_mapped_dict[str(filename) + str(state_space_dimensionality) + "S"]
        # plan commitment feature
        # <n> = do <n> steps from POMCP tree (<n> = Inf implies full path to leaf)
        # stddev = stop once σ goes above stddev_threshold
        # k_sigma = stop when mu_h - k*σh < mu_l + k*σl for best arm h and other arms l (mu, sigma computed through gamma-discounted rewards, not immediate reward from sampling)
        self.plan_commitment_algorithm = plan_commitment_algorithm
        self.plan_threshold = plan_threshold
        self.plan_commit_ugapec = None
        self.plan_ttest_pval = -1
        self.plan_reward_kurtosis = -1

        self.A = [a.value for a in self.action_enum]
        if self.sample_observations:
            self.O = range(len(S) * 5)
        else:
            self.O = S[:]
        self.logger_name = logger_name
        self.rs = np.random.RandomState(seed)
        self.objective_c = objective_c
        self.remaining_rollouts = total_rollouts
        self.total_rollouts = total_rollouts

        self.log_sum_stdvs = []
        self.used_budgets = []
        self.used_rollouts = []
        self.reward_stds = []
        self.filename = filename
        self.objectives = []
        self.state_space_dimensionality = state_space_dimensionality

    # take current state (AUV and world). Runs POMCP. Outputs traj of poses (where next to take samples from)
    def next_step(self, auv: RobotManager, data_model: DataModel, workspace: Workspace):
        Sreal_ndarrays = memory_mapped_dict[
            str(self.filename) + str(self.state_space_dimensionality) + "Sreal_ndarrays"]
        Sreal = memory_mapped_dict[str(self.filename) + str(self.state_space_dimensionality) + "Sreal"]
        S_real_pts_to_idxs = memory_mapped_dict[str(self.filename) + str(self.state_space_dimensionality) + "S_real_pts_to_idxs"]
        transition_matrix = memory_mapped_dict[str(self.filename) + str(self.state_space_dimensionality) + "transition_matrix"]
        mean, stdv = data_model.query_many(Sreal_ndarrays)
        try:
            print(auv.get_current_state())
            start_state_idx = S_real_pts_to_idxs[tuple(auv.get_current_state())]
        except KeyError:  # A bit unsure why this happens, maybe floating point error???
            best_idx = None
            best_dist = float("inf")
            best_key = None
            for i, state_space_point in enumerate(S_real_pts_to_idxs.keys()):
                dist = euc_dist(np.array(state_space_point), np.array(auv.get_current_state()[:3]))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_key = state_space_point
            start_state_idx = best_idx
            assert best_dist < 1 # f"closest point {best_key} should be closer than 1 to our auv state {auv.get_current_state()}, but best dist is {best_dist}"
            logging.getLogger(self.logger_name).debug(
                f"Current state not found in state space, using {best_key} for {auv.get_current_state()[:3]}, Dist {best_dist}")

        self.objectives.append(
            self.objective(mean[start_state_idx], stdv[start_state_idx]))
        self.used_budgets.append(self.budget)
        self.used_rollouts.append(self.total_rollouts - self.remaining_rollouts)
        self.log_sum_stdvs.append(np.log(np.sum(stdv)))

        previous_total_rollouts = self.remaining_rollouts

        if self.rollout_allocation_method == "fixed":
            rollout_allocator = FixedRolloutAllocator(int(self.total_rollouts / self.used_budgets[0]))
        elif self.rollout_allocation_method == "beta":
            rollout_allocator = CumulativeBetaDistributions(max_budget=self.used_budgets[0],
                                                            currently_used_budget=self.used_budgets[0] -
                                                                                  self.used_budgets[
                                                                                      -1],
                                                            max_rollout=self.total_rollouts,
                                                            currently_used_rollout=self.total_rollouts - self.remaining_rollouts,
                                                            alpha_param=self.alpha_param, beta_param=self.beta_param,
                                                            logger_name=self.logger_name,
                                                            action_enum=self.action_enum,
                                                            waste_unused_rollouts=self.waste_unused_rollouts)
        elif self.rollout_allocation_method == "beta-ugapeb":
            # rollout_allocator = UCBBestArmIdentification(beta=self.beta_param, epsilon=0.5, delta=0.5,
            #                                              logger_name=self.logger_name, action_enum=self.action_enum)
            rollout_allocator = CumulativeBetaDistributions(max_budget=self.used_budgets[0],
                                                            currently_used_budget=self.used_budgets[0] -
                                                                                  self.used_budgets[
                                                                                      -1],
                                                            max_rollout=self.total_rollouts,
                                                            currently_used_rollout=self.total_rollouts - self.remaining_rollouts,
                                                            alpha_param=self.alpha_param, beta_param=self.beta_param,
                                                            logger_name=self.logger_name,
                                                            action_enum=self.action_enum,
                                                            waste_unused_rollouts=self.waste_unused_rollouts)
            cur_budget = rollout_allocator.allocated_rollouts()
            rollout_allocator = UGapEb(cur_budget, self.epsilon, self.objective_c, self.filename, self.logger_name,action_enum=self.action_enum)
        elif self.rollout_allocation_method == "ugapeb":
            rollout_allocator = UGapEb(int(self.total_rollouts / self.state_space_dimensionality[2]), self.epsilon,
                                       self.objective_c, self.filename, self.logger_name,action_enum=self.action_enum)
        elif self.rollout_allocation_method == "ugapec":
            rollout_allocator = UGapEc(None, self.delta, self.epsilon,
                                       self.objective_c, self.filename, self.logger_name, action_enum=self.action_enum)
        elif self.rollout_allocation_method == "beta-ugapec":
            rollout_allocator = CumulativeBetaDistributions(max_budget=self.used_budgets[0],
                                                            currently_used_budget=self.used_budgets[0] -
                                                                                  self.used_budgets[
                                                                                      -1],
                                                            max_rollout=self.total_rollouts,
                                                            currently_used_rollout=self.total_rollouts - self.remaining_rollouts,
                                                            alpha_param=self.alpha_param, beta_param=self.beta_param,
                                                            logger_name=self.logger_name,
                                                            action_enum=self.action_enum,
                                                            waste_unused_rollouts=self.waste_unused_rollouts)
            cur_budget = rollout_allocator.allocated_rollouts()
            rollout_allocator = UGapEc(cur_budget, self.delta, self.epsilon,
                                       self.objective_c, self.filename, self.logger_name, self.action_enum)
        elif self.rollout_allocation_method == "sr":
            rollout_allocator = SuccessiveRejects(int(self.total_rollouts / self.state_space_dimensionality[2]),
                                                  self.logger_name, self.action_enum)
        elif self.rollout_allocation_method == "beta-sr":
            rollout_allocator = CumulativeBetaDistributions(max_budget=self.used_budgets[0],
                                                            currently_used_budget=self.used_budgets[0] -
                                                                                  self.used_budgets[
                                                                                      -1],
                                                            max_rollout=self.total_rollouts,
                                                            currently_used_rollout=self.total_rollouts - self.remaining_rollouts,
                                                            alpha_param=self.alpha_param, beta_param=self.beta_param,
                                                            logger_name=self.logger_name,
                                                            action_enum=self.action_enum,
                                                            waste_unused_rollouts=self.waste_unused_rollouts)
            cur_budget = rollout_allocator.allocated_rollouts()
            rollout_allocator = SuccessiveRejects(cur_budget, self.logger_name, self.action_enum)
        else:
            raise Exception()
        c_low, c_hi = get_uct_c(self.objective_c, self.filename, self.logger_name)
        cur_plan = POMCP(Generator, logger=self.logger_name, gamma=0.8, start_states=[start_state_idx], c=c_hi - c_low,
                         random_state=self.rs,
                         action_enum=self.action_enum,
                         extra_generator_data=(
                             mean, stdv, transition_matrix, self.objective_c, data_model, self.filename,
                             self.state_space_dimensionality,self.sample_observations,self.use_expected_improvement),
                         rollout_allocator=rollout_allocator
                         )
        cur_plan.initialize(self.O, self.A, self.O)
        self.cur_plan = cur_plan

        cur_plan.Search()
        self.remaining_rollouts -= self.cur_plan.rollouts_used_this_iteration

        # populate best actions from tree
        next_actions = []
        curr_state_tree_idx = -1  # start from root

        logging.getLogger(self.logger_name).debug(self.pomcp_tree_to_string(cur_plan.tree.nodes, Sreal))
        while not cur_plan.tree.isUnvisitedObsNode(curr_state_tree_idx):
            action_idx, action_tree_idx = cur_plan.SearchBest(curr_state_tree_idx, UseUCB=False)
            rewards_best = cur_plan.tree.get_reward_history(action_tree_idx)
            self.plan_reward_kurtosis = stats.kurtosis(rewards_best)
            # logging.getLogger(self.logger_name).debug(
            # f"Optimal Action {action_idx}, Optimal Action Node {cur_plan.tree.nodes[action_tree_idx]}")
            # append reward for node

            # get child node and continue loop
            # should this be uniform sampling? each action has different probabilities for subsequent states
            child_obs_tree_idx = random.choice(
                list(cur_plan.tree.get_children(action_tree_idx).values()))  # sample from children nodes
            # curr_state_tree_idx = child_obs_tree_idx

            # # logging.getLogger(self.logger_name).debug(
            #     # f"Optimal Action {action_idx}, Optimal Action Node {cur_plan.tree.nodes[action_tree_idx]}")
            
            if self.plan_commitment_algorithm == "k_sigma_pop":
                # caveat: take the first action always
                if len(next_actions) == 0:
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                    continue

                u_i = np.array([0] * len(self.action_enum), dtype=float)
                l_i = np.array([0] * len(self.action_enum), dtype=float)
                
                for i in range(len(self.action_enum)):
                    action_i = cur_plan.tree.get_children(curr_state_tree_idx)[i]
                    mu_i = cur_plan.tree.get_value(action_i)
                    sigma_i = cur_plan.tree.get_value_stdv(action_i)
                    n_i = cur_plan.tree.get_visits(action_i)
                    if sigma_i > 0:
                        l_i[i], u_i[i] = stats.norm.interval(self.plan_threshold, loc=mu_i, scale = sigma_i/np.sqrt(n_i))
                    else:
                        l_i[i], u_i[i] = [mu_i, mu_i]
                
                best_l_i = np.max(l_i)
                best_i = np.argmax(l_i)
                u_i.sort()
                second_best_u_i = u_i[-2]

                if best_l_i >= second_best_u_i:
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                    continue
                else:
                    break # stop appending actions

            elif self.plan_commitment_algorithm == "tTest":
                # caveat: take the first action always
                if len(next_actions) == 0:
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                    continue

                if cur_plan.tree.get_visits(curr_state_tree_idx) < len(self.action_enum) + 1:
                    # you haven't gone down each path at least once, cannot apply ugap theory
                    break

                action_idx_2ndBest, action_tree_2ndBest_idx = cur_plan.SearchSecondBest(curr_state_tree_idx, UseUCB=False)
                rewards_best = cur_plan.tree.get_reward_history(action_tree_idx)
                rewards_2ndBest = cur_plan.tree.get_reward_history(action_tree_2ndBest_idx)
                tstat, self.plan_ttest_pval = stats.ttest_ind(rewards_best, rewards_2ndBest, equal_var = False)
                # assert not np.isnan(self.plan_ttest_pval)
                # assert not np.isnan(tstat)
                if np.isnan(self.plan_ttest_pval):
                    # maybe not enough data to run ttest? skip it
                    break

                if self.plan_ttest_pval < self.plan_threshold: # if p(null hypothesis) is small enough
                    # we can reject the null hypothesis
                    # best arm is statistically better than 2nd best. Append action
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                else: # stop appending from this point onwards
                    break

            elif self.plan_commitment_algorithm == "tree_count":
                # caveat: take the first action always
                if len(next_actions) == 0:
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                    continue

                n_child = cur_plan.tree.get_visits(child_obs_tree_idx)
                # if the child has been visited enough and we still want to go there, then append
                if n_child > self.plan_threshold:
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                    continue
                else: # stop appending from this point onwards
                    break

            elif self.plan_commitment_algorithm == "ugapec":
                # caveat: take the first action always
                if len(next_actions) == 0:
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                    continue

                ugap_delta = 0.1
                ugap_eps = self.plan_threshold

                self.plan_commit_ugapec = UGapEc(None, ugap_delta, ugap_eps,
                    self.objective_c, self.filename, self.logger_name, self.action_enum)

                arm_rewards = []

                for i in range(len(self.action_enum)):
                    action_child = cur_plan.tree.get_children(curr_state_tree_idx)[i]
                    rewards_i = cur_plan.tree.nodes[action_child][3]
                    arm_rewards += [rewards_i]

                self.plan_commit_ugapec.set_rewards(arm_rewards)
                Bs = np.array([-1] * len(self.action_enum))
                for i in range(len(self.action_enum)):
                    Bs[i] = self.plan_commit_ugapec.B(self.plan_commit_ugapec.rewards, i)
                print(f"B_i = {Bs}")

                if self.plan_commit_ugapec.should_continue():
                    # we haven't explored enough to be confident of this gap
                    # don't take this action
                    break
                else:
                    action = self.plan_commit_ugapec.get_action()
                    assert action == action_idx
                    # If assertion passes, append
                    next_actions.append(self.action_enum(action_idx))
                    self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                    curr_state_tree_idx = child_obs_tree_idx  # continue loop
                
            else:
                next_actions.append(self.action_enum(action_idx))
                self.reward_stds.append(np.std(cur_plan.tree.nodes[action_tree_idx][3])) # append reward
                curr_state_tree_idx = child_obs_tree_idx  # continue loop

        # logging.getLogger(self.logger_name).debug(self.pomcp_tree_to_string(cur_plan.tree.nodes, Sreal))
        assert len(next_actions) > 0
        # Populate next states based on planners choice of actions
        logging.getLogger(self.logger_name).warning(
                f"Next Actions {next_actions}, current state {auv.get_current_state()[:3]}")
        next_states = []
        curr_state_coord = auv.get_current_state()[:3]
        for action in next_actions:
            next_state_coord = apply_action_to_state(curr_state_coord, action, self.step_sizes)
            if next_state_coord[2] >= workspace.zmax - self.step_sizes[2] and self.action_enum == action_enum(ActionModel.XYT):
                # if it's beyond maximum time of experiment, then stop
                break
            elif not workspace.is_inside(next_state_coord):
                logging.getLogger(self.logger_name).warning(
                    f"Optimal Action attempted to go outside workspace, defaulting to STAY_STILL")
                next_state_coord = apply_action_to_state(curr_state_coord, self.action_enum.STAY_STILL, self.step_sizes)

            if self.plan_commitment_algorithm == "stddev":  # stddev feature on
                # caveat: take the first action always
                if len(next_states) == 0:
                    next_states.append(next_state_coord)
                    curr_state_coord = next_state_coord  # continue loop
                    continue

                # if stddev < threshold, then append
                next_state_tree_idx = S_real_pts_to_idxs[tuple(next_state_coord)]
                if stdv[next_state_tree_idx] < self.plan_threshold:
                    next_states.append(next_state_coord)
                    curr_state_coord = next_state_coord  # continue loop
                else:  # else, stop appending in outer for loop
                    break

            elif self.plan_commitment_algorithm == "k_sigma":
                # caveat: take the first action always
                if len(next_states) == 0:
                    next_states.append(next_state_coord)
                    curr_state_coord = next_state_coord  # continue loop
                    continue

                # let next state from best action = i, and from other actions = j, objective_function = objf
                # if objf(μi - k*σi, σi) > objf(μj + k*σj, σj) ∀ j != i, k > 0
                # then we can say with high confidence that the action leading to i is the best action
                k = self.plan_threshold
                next_state_tree_idx = S_real_pts_to_idxs[tuple(next_state_coord)]
                best_action_objective = self.objective(mean[next_state_tree_idx] - k * stdv[next_state_tree_idx],
                                                       stdv[next_state_tree_idx])

                append_to_plan = True  # until proven false
                for temp_action in self.action_enum:
                    if temp_action == action:  # skip
                        continue

                    temp_coord = apply_action_to_state(curr_state_coord, action, self.step_sizes)
                    if (temp_coord[2] >= workspace.zmax - self.step_sizes[2]) or (not workspace.is_inside(temp_coord)):
                        # if it's outside x, y, or t bounds, then skip it
                        continue

                    temp_tree_idx = S_real_pts_to_idxs[tuple(temp_coord)]
                    temp_objective = self.objective(mean[temp_tree_idx] + k * stdv[temp_tree_idx], stdv[temp_tree_idx])
                    if best_action_objective > temp_objective:
                        continue
                    else:
                        append_to_plan = False
                        break
                
                if append_to_plan:
                    next_states.append(next_state_coord)
                    curr_state_coord = next_state_coord  # continue loop
                else:
                    # no longer appending states to plan. Stop looping
                    break

            else:  # append by default
                next_states.append(next_state_coord)
                curr_state_coord = next_state_coord  # continue loop

        #self.save_uct_c()
        cur_plan.destroy_rollout_allocator()  # this is needed to allow checkpoints since the rollout allocators may have generators

        assert len(next_states) > 0
        if self.action_enum == action_enum(ActionModel.XYT):
            assert auv.get_current_state()[2] + self.step_sizes[2] == next_states[0][
                2], f"AUV time {auv.get_current_state()} + step {self.z_step} time should be next state time {next_states[0]}"

        if self.plan_commitment_algorithm == "n_steps":
            if self.plan_threshold == 1:  # One-step (default POMCP behaviour)
                return [next_states[0]]
            elif self.plan_threshold == np.Inf:  # max steps (go all the way to leaf node)
                return next_states
            else:  # take n steps in tree
                return next_states[:int(self.plan_threshold)]
        elif self.plan_commitment_algorithm == "stddev":
            return next_states
        elif self.plan_commitment_algorithm == "ugapec":
            print(f"\t\t{len(next_states)} steps in ugapec plan. epsilon = {self.plan_threshold}")
            return next_states
        elif self.plan_commitment_algorithm == "k_sigma":
            print(f"\t\t{len(next_states)} steps in k_sigma plan. k = {self.plan_threshold}")
            return next_states
        elif self.plan_commitment_algorithm == "k_sigma_pop":
            print(f"\t\t{len(next_states)} steps in k_sigma_pop plan. k = {self.plan_threshold}")
            return next_states
        elif self.plan_commitment_algorithm == "tree_count":
            return next_states
        elif self.plan_commitment_algorithm == "tTest":
            return next_states        
        else:
            logging.getLogger(self.logger_name).error(f"Unknown plan commitment algo {self.plan_commitment_algorithm}")
            raise NotImplementedError(f"Unknown plan commitment algo {self.plan_commitment_algorithm}")
    def objective(self, mu, sigma):
        return mu + self.objective_c * sigma

    def pomcp_tree_to_string(self, nodes, state_space, depth=2, include_action_nodes=True):
        # I should've just done this recursively, my haskeller is leaving
        out_str = "\n"
        stack = [("", nodes[-1], depth)]
        while stack != []:
            action, node, remaining_depth = stack.pop()
            is_action_node = node[-1] == -1
            if is_action_node:
                #assert len(node[1]) in [0, 1], "I don't know how to interpret non deterministic action nodes"
                if include_action_nodes and node[2] != 0:
                    out_str += ("\t" * (
                            depth - remaining_depth)) + f"{self.action_enum(action)} R - N({round(average_or_0_on_empty(node[3]), 2)},{round(np.std(node[3]), 2)}), N: {node[2]} \n"
                for parent, child in node[1].items():
                    stack.append((action, nodes[child], remaining_depth))
            else:  # Belief node
                if include_action_nodes:
                    out_str += ("\t" * (
                            depth - remaining_depth)) + f"  {node[0]} R: {round(node[3], 2)}, N: {node[2]} B: {node[-1]} \n"
                else:
                    out_str += ("\t" * (
                            depth - remaining_depth)) + f"{self.action_enum(action)} - Key: {node[0]} R: {round(node[3], 2)}, N: {node[2]} B: {set(map(lambda state_index: state_space[state_index], node[-1]))} \n"
                if remaining_depth > 0:
                    for action, child_idx in node[1].items():
                        stack.append((action, nodes[child_idx], remaining_depth - 1))
        return out_str

    def save_uct_c(self):
        cur_low_param, cur_hi_param = get_uct_c(self.objective_c, self.filename, self.logger_name)
        rewards = [reward for arm_reward in self.get_root_rewards() for reward in arm_reward]
        cur_low = min(rewards)
        cur_hi = max(rewards)

        if cur_low_param != get_default_low_param():
            new_low_param = min(cur_low, cur_low_param)
        else:
            new_low_param = cur_low
        if cur_hi_param != get_default_hi_param():
            new_hi_param = max(cur_hi, cur_hi_param)
        else:
            new_hi_param = cur_hi
        if isinstance(new_hi_param, np.ndarray):
            new_hi_param = float(new_hi_param[0])
        if isinstance(new_low_param, np.ndarray):
            new_low_param = float(new_low_param[0])

        with open(f"params/{self.objective_c}-{self.filename.replace('/', '')}.json", "w") as f:
            json.dump({"low": new_low_param, "high": new_hi_param}, f)

    def get_root_rewards(self):
        try:
            return self.cur_plan.get_root_rewards()
        except AttributeError:
            logging.getLogger(self.logger_name).warning("Get Reward called without a computed plan")
            return [0 for _ in range(len(self.action_enum))]

    def get_critical_path(self, coordinates: bool = False):
        """
        Returns the critical path

        :param coordinates: Whether to return the path as a list
                            of (observation node, reward) or (coordinates, reward) tuples
        """
        
        Sreal_ndarrays = memory_mapped_dict[str(self.filename) + str(self.state_space_dimensionality) + "Sreal_ndarrays"]

        # Exit if the plan hasn't been made yet
        if not hasattr(self, "cur_plan") or not self.cur_plan:
            return []

        # First, iterate through current plan to find most recent critical path
        critical_path = []
        tree = self.cur_plan.tree
        work = [tree.nodes[n] for n in tree.get_children(-1).values()]
        best_node = None
        best_reward = None

        while work:
            # Get info about one of the nodes
            cur_node = work.pop()

            if cur_node[3]:  # Some nodes don't have reward assigned yet, just skip them

                cur_reward = tree.get_node_reward(cur_node)

                if not best_node:
                    # Should only happen on first run at each depth
                    best_node = cur_node
                    best_reward = cur_reward

                elif cur_reward > best_reward:
                    # Found a better node
                    best_node = cur_node
                    best_reward = cur_reward

                elif cur_reward == best_reward and len(tree.get_node_value(cur_node)) > 1 and \
                        len(tree.get_node_value(best_node)) > 1 and variance(cur_node[3]) < variance(best_node[3]):
                    # Choose the node with lower variance if rewards are equal
                    best_node = cur_node
                    best_reward = cur_reward

            # Check if we're done with nodes at this depth.
            if not work and best_node:
                # If so, get the child observation node
                obs_node = tree.nodes[list(best_node[1].values())[0]]

                work = [tree.nodes[n] for n in obs_node[1].values()]  # Go to next depth based on chosen node

                if coordinates:
                    if obs_node[4]:
                        # Only add node if it has been visited bc we dont have a position estimate otherwise
                        position = Sreal_ndarrays[mode(obs_node[4])]
                        critical_path.append((position, best_reward))
                else:
                    critical_path.append((obs_node, best_reward))

                # Reset values
                best_node = None
                best_reward = None

        return critical_path

class TestAdjacencyMatrix(unittest.TestCase):
    def test_adjacency_matrix(self):
        workspace = RectangularPrismWorkspace(0, 4, 0, 4, 0, 4)
        adjacency_dict = create_adjacency_dict([0, 0, 0], workspace)
        self.assertIn((0, 0, 0), adjacency_dict)
        self.assertEqual((1, 0, 1), adjacency_dict[(0, 0, 0)][0])
        self.assertEqual((0, 0, 1), adjacency_dict[(0, 0, 0)][4])
        # self.assertEquals((0, 0, 1), adjacency_dict[(0, 0, 0)][1])
        # self.assertEquals((0, 0, 1), adjacency_dict[(0, 0, 0)][3])
        self.assertEqual((0, 1, 1), adjacency_dict[(0, 0, 0)][2])

        self.assertEqual(dict(), adjacency_dict[(0, 0, 4)])

    def test_adjacency_matrix_to_numpy(self):
        workspace = RectangularPrismWorkspace(0, 4, 0, 4, 0, 4)
        adjacency_dict = create_adjacency_dict([0, 0, 0], workspace)
        Sreal = list(adjacency_dict.keys())
        S = list(range(len(Sreal)))
        S_real_pts_to_idxs = {tuple(Sreal[v]): v for v in S}

        matrix = adjacency_dict_to_numpy(adjacency_dict, S_real_pts_to_idxs)
        self.assertIn((0, 0, 0), adjacency_dict)
        self.assertEqual(S_real_pts_to_idxs[(1, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 0])
        self.assertEqual(S_real_pts_to_idxs[(0, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 4])
        self.assertEqual(S_real_pts_to_idxs[(0, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 1])
        self.assertEqual(S_real_pts_to_idxs[(0, 0, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 3])
        self.assertEqual(S_real_pts_to_idxs[(0, 1, 1)], matrix[S_real_pts_to_idxs[(0, 0, 0)], 2])

        print(matrix)


if __name__ == "__main__":
    unittest.main()
