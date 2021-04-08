import logging
from math import sqrt

import abc
import numpy as np

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import (AllocatorWhichChoosesRootActionMixin,
                                                                                 BaseRolloutAllocator)
from sample_sim.planning.pomcp_rollout_allocation.fixed import FixedRolloutAllocator
from sample_sim.planning.pomcp_utilities import get_uct_c, ActionXYT, ActionXYZ


class UGap(abc.ABC):
    def __init__(self, num_arms):
        self.num_arms = num_arms

    @abc.abstractmethod
    def beta(self, num_pulls, total_pulls):
        pass

    def total_pulls(self, all_rewards):
        return sum(map(len, all_rewards))
    def B(self, all_rewards, arm_idx):
        total_pulls = self.total_pulls(all_rewards)
        lower_confidence = np.average(all_rewards[arm_idx]) - self.beta(len(all_rewards[arm_idx]),total_pulls)
        max_gap = float('-inf')
        for upper_arm_idx, reward in enumerate(all_rewards):
            if upper_arm_idx != arm_idx:
                upper_confidence = np.average(reward) + self.beta(len(reward),total_pulls)
                cur_gap = upper_confidence - lower_confidence
                max_gap = max(cur_gap, max_gap)
        return max_gap

    def J(self, all_rewards):
        Bs = list(map(lambda arm_idx: self.B(all_rewards, arm_idx), range(self.num_arms)))
        arm_idx = np.argmin(Bs)
        return arm_idx

    def select_arm(self, all_rewards):
        for arm_idx, reward in enumerate(all_rewards):
            if reward == []:
                return arm_idx
        lt = self.J(all_rewards)
        ut_value = float("-inf")
        ut_arm = None
        for arm_idx, reward in enumerate(all_rewards):
            upper_confidence = np.average(reward) + self.beta(len(reward),self.total_pulls(all_rewards))
            if arm_idx != lt and upper_confidence > ut_value:
                ut_value = upper_confidence
                ut_arm = arm_idx

        if self.beta(len(all_rewards[lt]),self.total_pulls(all_rewards)) > self.beta(len(all_rewards[ut_arm]),self.total_pulls(all_rewards)):
            return lt
        else:
            return ut_arm

class UGapEc(BaseRolloutAllocator, AllocatorWhichChoosesRootActionMixin, UGap):
    '''
    Fixed confidence
    '''
    def __init__(self, budget, delta, epsilon, objective_c, filename, logger_name, action_enum):
        self.budget = budget
        self.objective_c = objective_c
        self.filename = filename
        self.logger_name = logger_name
        b_lo, b_hi = get_uct_c(self.objective_c, self.filename, self.logger_name)
        self.b = b_hi - b_lo
        self.c = 1 / 2
        self.delta = delta
        self.epsilon = epsilon
        self.K = len(action_enum)
        UGap.__init__(self, self.K)

    def beta(self, num_pulls, total_pulls):
        top = self.c * np.log((4 * self.K * total_pulls ** 3) / self.delta)
        return self.b * np.sqrt(top / num_pulls)

    def get_action(self):
        return self.select_arm(self.rewards)

    def should_continue(self):
        if self.total_pulls(self.rewards) < self.K:
            return True
        if self.budget is not None and self.total_pulls(self.rewards) >= self.budget:
            logging.getLogger(self.logger_name).debug("UGapEc stopping due to budget limit")
            return False

        best_arm = self.J(self.rewards)
        best_arm_B = self.B(self.rewards,best_arm)

        total_pulls = self.total_pulls(self.rewards)
        if total_pulls % 100 == 0:
            logging.getLogger(self.logger_name).debug(f"Best arm B {best_arm_B}, pulls {total_pulls}")
        continue_criteria = best_arm_B >= self.epsilon
        if not continue_criteria:
            logging.getLogger(self.logger_name).debug("UGapEc stopping due to confidence limit")
        return continue_criteria


class UGapEb(FixedRolloutAllocator, AllocatorWhichChoosesRootActionMixin, UGap):
    '''
    Fixed budget Ugap
    '''
    def __init__(self, budget, epsilon,objective_c, filename, logger_name, action_enum):
        self.budget = budget
        self.objective_c = objective_c
        self.filename = filename
        self.logger_name = logger_name
        self.K = len(action_enum)
        b_lo, b_hi = get_uct_c(self.objective_c, self.filename, self.logger_name)
        self.b = b_hi - b_lo

        # this uses an under estimate of He
        self.a = (self.budget - self.K) / (4 * sum(map(lambda k: (self.b ** 2) / (epsilon ** 2), range(self.K))))
        logging.getLogger(self.logger_name).debug(f"A estimate {self.a}")
        super().__init__(self.budget)
        UGap.__init__(self, self.K)

    def beta(self, num_pulls, total_pulls):
        return self.b * sqrt(self.a / num_pulls)

    def get_action(self):
        return self.select_arm(self.rewards)
