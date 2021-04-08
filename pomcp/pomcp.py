import logging

import numpy as np
from numpy import average
from numpy.random import RandomState
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from tqdm import tqdm

from pomcp.auxilliary import BuildTree, UCB
# @numba.jit(nopython=True)
from sample_sim.data_model.data_model import DataModel, TorchExactGPBackedDataModel
from sample_sim.data_model.gp_wrapper import TorchExactGp
from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import (BaseRolloutAllocator,
                                                                                 AllocatorWhichChoosesRootActionMixin,
                                                                                 PrecalculatedBaseRolloutAllocator)
from sample_sim.planning.pomcp_rollout_allocation.fixed import FixedRolloutAllocator
from sample_sim.planning.pomcp_utilities import ActionXYT, ActionXYZ


def fast_choice(l, length, rs: RandomState):
    i = rs.randint(length)
    return l[i]


def average_or_0_on_empty(l):
    if l != []:
        return average(l)
    else:
        return 0


class POMCP():
    # gamma = discount rate
    # c = higher value to encourage UCB exploration
    # threshold = threshold below which discount is too little
    # timeout = number of runs from node
    def __init__(self, generator, random_state, rollout_allocator: BaseRolloutAllocator,logger, gamma=0.95, c=1,
                 threshold=0.005, fixed_rollout_number=10000, no_particles=1200, action_enum=None,
                 start_states=None, extra_generator_data=None,):
        self.logger = logger
        self.gamma = gamma
        if gamma >= 1:
            raise ValueError("gamma should be less than 1.")
        self.Generator = generator
        self.e = threshold
        self.c = c
        self.fixed_rollout_number = fixed_rollout_number
        self.no_particles = no_particles
        self.action_enum = action_enum
        if start_states is None:
            start_states = []
        self.tree = BuildTree(start_states=start_states)
        self.random_state = random_state
        self.extra_generator_data = extra_generator_data
        self.rollouts_used_this_iteration = 0
        self.rollout_allocator = rollout_allocator

    def destroy_rollout_allocator(self):
        del self.rollout_allocator
        self.rollout_allocator = None

    # give state, action, and observation space
    # @timeit
    def initialize(self, S, A, O):
        self.states = S
        self.actions = A
        self.observations = O
        self.actions_length = len(A)

    # searchBest action to take
    # UseUCB = False to pick best value at end of Search()
    # @timeit
    def SearchBest(self, h, UseUCB=True):
        max_value = None
        result = None
        resulta = None
        if h == -1 and isinstance(self.rollout_allocator, AllocatorWhichChoosesRootActionMixin) and UseUCB:
            optimal_action = self.rollout_allocator.get_action()
            children = self.tree.get_children(h)
            return optimal_action, children[optimal_action]

        if UseUCB:
            if not self.tree.isActionNode(h):
                children = self.tree.get_children(h)
                # UCB for each child node
                for action, child in children.items():
                    # if node is unvisited return it
                    if self.tree.nodes[child][2] == 0:
                        if self.tree.get_visits(h) > len(self.action_enum)+1:
                            raise ValueError(f"node {h} has {self.tree.get_visits(h)}(>{len(self.action_enum)}) visits, yet child {child} has not been visited even once?")
                        return action, child
                    # print(average_or_0_on_empty(self.tree.nodes[child][3]))
                    ucb = UCB(self.tree.nodes[h][2], self.tree.nodes[child][2],
                              average_or_0_on_empty(self.tree.nodes[child][3]), self.c)

                    # Max is kept
                    if max_value is None or max_value < ucb:
                        max_value = ucb
                        result = child
                        resulta = action
            # return action-child_id values
            return resulta, result
        else:
            if not self.tree.isActionNode(h):
                children = self.tree.get_children(h)
                # pick optimal value node for termination
                for action, child in children.items():
                    node_value = average_or_0_on_empty(self.tree.nodes[child][3])
                    # keep max
                    if max_value is None or max_value < node_value:
                        max_value = node_value
                        result = child
                        resulta = action
            return resulta, result

    def SearchSecondBest(self, h, UseUCB=False):
        node_value = np.array([-np.Inf] * len(self.action_enum))

        if UseUCB:
            raise NotImplementedError()
        else:
            if not self.tree.isActionNode(h):
                children = self.tree.get_children(h)
                for i in range(self.actions_length):
                    node_value[i] = self.tree.get_value(children[i])
                
                ind = np.argpartition(node_value, -2)[-2:] # index of 2 largest values (unsorted)
                if node_value[ind[0]] < node_value[ind[1]]:
                    second_best_action = ind[0]
                else:
                    second_best_action = ind[1]
            child_node = self.tree.get_children(h)[second_best_action]
                
            return second_best_action, child_node

    def get_root_rewards(self):
        rewards = [[] for _ in range(len(self.action_enum))]
        for action, child_idx in self.tree.nodes[-1][1].items():
            rewards[action] = self.tree.nodes[child_idx][3]
        return rewards

    def update_rollout_allocator_reward(self):
        self.rollout_allocator.set_rewards(self.get_root_rewards())

    # Search module
    # @timeit
    def Search(self):
        self.rollouts_used_this_iteration = 0
        Bh = self.tree.nodes[-1][4].copy()
        # Repeat Simulations until timeout
        # for _ in range(self.timeout):
        self.update_rollout_allocator_reward()
        if isinstance(self.rollout_allocator, PrecalculatedBaseRolloutAllocator):
            pbar = tqdm(total=self.rollout_allocator.allocated_rollouts(), desc="Rollouts", file=TqdmToLogger(logging.getLogger(self.logger)))
        else:
            pbar = tqdm(desc="Rollouts", file=TqdmToLogger(logging.getLogger(self.logger)))
        while self.rollout_allocator.should_continue():
            pbar.update(1)
            if Bh == []:
                s = self.random_state.choice(self.states)
            else:
                s = self.random_state.choice(Bh)
            self.Simulate(s, -1, 0, [])

            #Reset GP to base GP
            data_model = self.extra_generator_data[4]
            assert isinstance(data_model,TorchExactGPBackedDataModel)
            base_gp = data_model.model
            assert isinstance(base_gp,TorchExactGp)
            base_gp.update_prior(data_model.Xs,data_model.Ys)
            #End reset GP

            self.rollouts_used_this_iteration += 1
            self.update_rollout_allocator_reward()
        pbar.close()
        # Get best action
        action, _ = self.SearchBest(-1, UseUCB=False)
        return action

    # Check if a given observation node has been visited
    # @timeit
    def getObservationNode(self, h, sample_observation):

        if sample_observation not in list(self.tree.nodes[h][1].keys()):
            # If not create the node
            self.tree.ExpandTreeFrom(h, sample_observation)
        # Get the nodes index
        Next_node = self.tree.nodes[h][1][sample_observation]
        return Next_node

    # @timeit
    def Rollout(self, s, depth,s_history):

        # Check significance of update

        if (self.gamma ** depth < self.e or self.gamma == 0) and depth != 0:
            return 0

        cum_reward = 0

        # Pick random action; maybe change this later
        # Need to also add observation in history if this is changed
        # action = choice(self.actions)
        action = fast_choice(self.actions, self.actions_length, self.random_state)

        # Generate states and observations
        pre = len(s_history)
        sample_state, _, r = self.Generator(s, action, s_history, self.extra_generator_data)
        post = len(s_history)
        #print(pre)
        #print(post)
        #assert post == pre + 1
        cum_reward += r + self.gamma * self.Rollout(sample_state, depth + 1, s_history )
        return cum_reward

    # @timeit
    def Simulate(self, s, h, depth, s_history):

        # Check significance of update
        if (self.gamma ** depth < self.e or self.gamma == 0) and depth != 0:
            return 0

        # If leaf node
        if self.tree.isLeafNode(h):
            for action in self.actions:
                self.tree.ExpandTreeFrom(h, action, IsAction=True)
            new_value = self.Rollout(s, depth, s_history )

            self.tree.nodes[h][2] += 1
            # Previously they were erroneously setting the reward for belief nodes, which have undefined reward
            if self.tree.nodes[h][-1] != -1:
                self.tree.nodes[h][3] = -1
            else:
                # self.tree.nodes[h][3] = new_value
                self.tree.nodes[h][3] = [new_value]
            return new_value
        else:
            # Small n next_node is ha
            # Big N next node is hao

            cum_reward = 0
            # Searches best action
            next_action, next_node = self.SearchBest(h)
            # Generate next states etc..
            pre = len(s_history)
            sample_state, sample_observation, reward = self.Generator(s, next_action, s_history,self.extra_generator_data)
            post = len(s_history)
            #assert post == pre + 1
            # Get resulting node index
            Next_node = self.getObservationNode(next_node, sample_observation)
            # Estimate node Value
            cum_reward += reward + self.gamma * self.Simulate(sample_state, Next_node, depth + 1, s_history)
            # Backtrack
            self.tree.nodes[h][4].append(s)
            if len(self.tree.nodes[h][4]) > self.no_particles:
                self.tree.nodes[h][4] = self.tree.nodes[h][4][1:]
            self.tree.nodes[h][2] += 1
            self.tree.nodes[next_node][2] += 1
            # self.tree.nodes[next_node][3] += (cum_reward - self.tree.nodes[next_node][3]) / self.tree.nodes[next_node][2]
            self.tree.nodes[next_node][3].append(cum_reward)
            return cum_reward

    # FIXFIXFIX
    # Samples from posterior after action and observation
    # @timeit
    # def PosteriorSample(self, Bh, action, observation):
    #     if Bh == []:
    #         s = self.random_state.choice(self.states)
    #     else:
    #         s = self.random_state.choice(Bh)
    #     # Sample from transition distribution
    #     s_next, o_next, _ = self.Generator(s, action, self.extra_generator_data)
    #     if o_next == observation:
    #         return s_next
    #     result = self.PosteriorSample(Bh, action, observation)
    #     return result
    #
    # # Updates belief by sampling posterior
    # # @timeit
    # def UpdateBelief(self, action, observation):
    #     prior = self.tree.nodes[-1][4].copy()
    #
    #     self.tree.nodes[-1][4] = []
    #     for _ in range(self.no_particles):
    #         self.tree.nodes[-1][4].append(self.PosteriorSample(prior, action, observation))
