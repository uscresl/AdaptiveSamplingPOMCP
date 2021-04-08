from sample_sim.planning.pomcp_rollout_allocation.expected_se_reduction import \
    ExpectedStandardErrorVsCostRolloutAllocator
from sample_sim.planning.pomcp_rollout_allocation.fixed import FixedRolloutAllocator


def dispatch(s,**kwargs,):
    if s == "fixed":
        return FixedRolloutAllocator(**kwargs)
    elif s == "se_cost":
        return ExpectedStandardErrorVsCostRolloutAllocator(**kwargs)
    else:
        raise Exception(f"Rollout Dispatch didn't understand {s}")
