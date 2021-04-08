from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import PrecalculatedBaseRolloutAllocator


class FixedRolloutAllocator(PrecalculatedBaseRolloutAllocator):
    def __init__(self,iterations):
        self.iterations = iterations
        super().__init__()

    def allocated_rollouts(self):
        return self.iterations
