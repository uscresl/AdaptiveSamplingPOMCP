import math

import numpy as np

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import BaseRolloutAllocator


class ExpectedStandardErrorVsCostRolloutAllocator(BaseRolloutAllocator):
    def __init__(self, currently_used_rollouts, total_rollouts, min_iterations=20, cost_multiple=.1):
        self.currently_used_rollouts = currently_used_rollouts
        self.total_rollouts = total_rollouts
        self.min_iterations = min_iterations
        self.cost_multiple = cost_multiple

    def should_continue(self) -> bool:
        self.currently_used_rollouts += 1
        if len(self.rewards) < 20:
            return True
        else:
            stdv = np.std(self.rewards)
            # This is how much would the standard error decrease if we took one more sample and the std stayed the same
            expected_se_reduction = (stdv / math.sqrt(len(self.rewards))) - (stdv / math.sqrt(len(self.rewards) + 1))
            # This is the cost of taking one more sample
            cost = self.cost_multiple * ((self.currently_used_rollouts) / self.total_rollouts)
            return cost < expected_se_reduction
