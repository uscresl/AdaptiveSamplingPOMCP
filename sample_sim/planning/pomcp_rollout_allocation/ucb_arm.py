import logging

import numpy as np
from tqdm import tqdm

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import BaseRolloutAllocator


def U(t, delta, epsilon):
    discount = 1 + np.sqrt(epsilon)

    one_plus_epsilon = 1 + epsilon
    top = one_plus_epsilon * t * np.log((np.log(one_plus_epsilon * t) / delta))
    bottom = 2 * t
    return discount * np.sqrt(top / bottom)


# def C(beta, pulls, epsilon, delta):
#     return (1 + beta) * U(pulls, delta / K, epsilon)

def C(total_pulls, pulls):
    return np.sqrt((2 * np.log(total_pulls)) / pulls)

def alpha(beta, total_pulls, delta):
    beta_part = ((2 + beta) / beta) ** 2
    top = np.log(2 * np.log(beta_part * total_pulls / delta))
    bottom = np.log(total_pulls / delta)
    return beta_part * (1 + top / bottom)

def argmax_without(xs,idx):
    max_val = None
    best_idx = None
    for i,x in enumerate(xs):
        if i != idx and (max_val is None or x > max_val):
            max_val = x
            best_idx = i
    return best_idx


class UCBBestArmIdentification(BaseRolloutAllocator):
    def __init__(self, beta, epsilon, delta, logger_name, action_enum):
        #assert 0 < epsilon < 1
        #assert 0 < delta < np.log(1 + epsilon) / np.e
        epsilon = np.exp(np.e * delta) - 1
        self.epsilon = epsilon
        self.beta = beta
        self.delta = delta
        self.logger_name = logger_name
        self.K = len(action_enum)
        self.pbar = tqdm("Pulls")

    def should_continue(self) -> bool:
        total_pulls = sum(map(len, self.rewards))
        confidences = []
        means = []
        # if there are less than K total rewards,
        if total_pulls < self.K:
            return True
        for reward in self.rewards:
            means.append(np.average(reward))
            confidences.append(C(total_pulls,len(reward)))
            #confidences.append(C(self.beta, len(reward), self.epsilon, self.delta))
        means = np.array(means)
        confidences = np.array(confidences)
        print(f"{means} , {confidences}")
        ht = np.argmax(means)
        lt = argmax_without(means + confidences, ht)
        self.pbar.update(1)

        second_best_arm = argmax_without(means,ht)


        # print(f"First Second Gap: {means[ht] - confidences[ht] - means[second_best_arm] + confidences[second_best_arm]}")
        # if (means[ht] - confidences[ht] - means[second_best_arm] + confidences[second_best_arm]) < 0.1:
        #     return False

        # If it's impossible to beat the best arm
        if means[ht] - confidences[ht] > means[lt] + confidences[lt]:
            logging.getLogger(self.logger_name).debug(f"Ending due to confidence in {total_pulls} pulls")
            self.pbar.close()
            return False


        cur_alpha = alpha(self.beta, total_pulls, self.delta)
        for i, reward in enumerate(self.rewards):
            cur_sum = 0
            for j, reward_other in enumerate(self.rewards):
                if j != i:
                    cur_sum += len(reward_other)
            if len(reward) > cur_sum * cur_alpha:
                logging.getLogger(self.logger_name).debug(f"Ending due to alpha in {total_pulls} pulls")
                self.pbar.close()
                return False
        # print(total_pulls)
        return True
