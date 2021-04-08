import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import PrecalculatedBaseRolloutAllocator


class SimpleLogPlanner(PrecalculatedBaseRolloutAllocator):
    def __init__(self, max_budget, cur_budget, total_rollouts, remaining_rollouts, t_step, logger_name, plot=True, use_weighted_model=True):
        self.max_budget = max_budget
        self.cur_budget = cur_budget
        self.total_rollouts = total_rollouts
        self.remaining_rollouts = remaining_rollouts
        self.t_step = t_step
        self.logger_name = logger_name
        self.plot = plot
        self.use_weighted_model = use_weighted_model
        if plot:
            plt.figure(15)

        super().__init__(logger=self.logger_name, pbar=True)

    def allocated_rollouts(self):
        start_point = [self.max_budget, self.total_rollouts]
        mid_point = [self.max_budget / 2, np.exp(self.total_rollouts/2)]
        end_point = [0, 100]

        # points = np.array([start_point,mid_point,end_point])

        points = np.array([start_point,mid_point, end_point])
        X = np.array([points[:, 0]])
        Y = np.array([points[:, 1]])

        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X.T,np.log(Y).T)
        if not self.use_weighted_model:
            rollouts_that_should_be_remaining = np.exp(
                self.model.predict(np.array([self.cur_budget - self.t_step]).reshape(1, -1)))
        #This method is from https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
        #coeffs = np.polyfit(X.ravel(),np.log(Y).ravel(),1,w=np.sqrt(Y).ravel())
        #print(coeffs)
        self.weighted_model = LinearRegression(fit_intercept=True)
        self.weighted_model.fit(X.T,np.log(Y).T,sample_weight=(Y).ravel())
        if self.use_weighted_model:
            #rollouts_that_should_be_remaining = np.exp(coeffs[1]  + coeffs[0] * (self.cur_budget - self.t_step))
            rollouts_that_should_be_remaining = np.exp(
                self.weighted_model.predict(np.array([self.cur_budget - self.t_step]).reshape(1, -1)))
        if self.plot:
            plt.figure(15)
            test_points = np.arange(self.max_budget,0,-self.t_step)
            pred_values_unweighted = np.exp(self.model.predict(np.array(test_points).reshape(-1,1)))
            plt.plot(test_points,pred_values_unweighted,label="Unweighted")

            pred_values_weighted = np.exp(self.weighted_model.predict(np.array(test_points).reshape(-1, 1)))
            plt.plot(test_points,pred_values_weighted,label="Weighted")
            plt.legend()
            plt.show()


        logging.getLogger(self.logger_name).debug(
            f"Remaining rollouts {self.remaining_rollouts}, Rollouts that should be remaining {rollouts_that_should_be_remaining}, Allocated rollouts{self.remaining_rollouts - rollouts_that_should_be_remaining}")
        assert self.remaining_rollouts > rollouts_that_should_be_remaining
        return int(self.remaining_rollouts - rollouts_that_should_be_remaining)
