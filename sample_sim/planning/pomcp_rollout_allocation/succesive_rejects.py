import logging
import math

import numpy as np

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import AllocatorWhichChoosesRootActionMixin
from sample_sim.planning.pomcp_rollout_allocation.fixed import FixedRolloutAllocator
from sample_sim.planning.pomcp_utilities import ActionXYT, ActionXYZ


class SuccessiveRejects(FixedRolloutAllocator, AllocatorWhichChoosesRootActionMixin):
    def __init__(self, budget, logger_name, action_enum):
        self.budget = int(budget)
        self.logger_name = logger_name
        super().__init__(self.budget)
        self.K = len(action_enum)
        self.action_iterator = self.get_iterable()


    def log_bar(self):
        return 1 / 2 + sum(map(lambda i: 1 / i, range(2, self.K + 1)))

    def n_k(self, k):
        if k == 0:
            return 0
        else:
            return math.ceil((1 / self.log_bar()) * ((self.budget - self.K) / (self.K + 1 - k)))
    def get_action(self):
        return next(self.action_iterator)

    def get_iterable(self):
        if self.budget <= self.K+2:
            # If you only have K pulls i guess just uniform
            for arm_idx, reward in enumerate(self.rewards):
                yield arm_idx
            while True:
                yield np.argmax(list(map(sum,self.rewards)))
            
        else:
            arms = list(range(self.K))
            for k in range(1, self.K):
                for arm in arms:
                    if arm is not None:
                        for cur_round in range(self.n_k(k) - self.n_k(k - 1)):
                            yield arm
                worst_arm = None
                worst_reward = float("inf")
                for i, reward in enumerate(self.rewards):
                    avg_reward = np.average(reward)
                    if i in arms and avg_reward < worst_reward:
                        worst_reward = avg_reward
                        worst_arm = i
                assert arms[worst_arm] is not None
                arms[worst_arm] = None
                logging.getLogger(self.logger_name).debug(f"Arms {arms} ")
                if k == self.K-1:
                    assert len(list(filter(lambda x: x is not None, arms))) == 1, "We found the single best arm"

            assert len(list(filter(lambda x: x is not None, arms))) == 1, "We found the single best arm"
            best_arm = list(filter(lambda x: x is not None, arms))[0]
            while True:
                yield best_arm
        
