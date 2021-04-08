import logging

import numpy as np
from sklearn.linear_model import LinearRegression

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import PrecalculatedBaseRolloutAllocator
import matplotlib.pyplot as plt


class LogEntropyLinearPlanner(PrecalculatedBaseRolloutAllocator):
    '''
     fit a log curve to budget->overall objective improvement
    Then we use that to predict how much overall objective improvement there should be based on how much budget change there is
    Then we use that divided by the total expected objective improvement (just use the model to predict at 0 remaining budget) to determine what percent of the rollouts should be used this step
    The reason it's a log curve is bc it's information gathering
    We use objective improvement bc it's a proxy for how important that step will be to the overall solution
    It doesn't yet accommodate a lack of information at the early stages
    '''

    def __init__(self, used_budgets, log_sum_stdvs, t_step, total_rollouts,currently_used_rollouts, logger_name,default_rollouts_percent=0.01,plot=True):
        '''
        log_sum_stdvs is (used_rollouts,used_budget,log_sum_stdv)
        '''

        self.log_sum_stdvs = np.array(log_sum_stdvs)

        assert len(log_sum_stdvs) == len(used_budgets)
        print(log_sum_stdvs)
        self.used_budgets = used_budgets
        if len(log_sum_stdvs) >= 2:
            self.model_1 = LinearRegression(fit_intercept=True)
            self.model_1.fit(np.array([used_budgets]).T, log_sum_stdvs)

        self.default_rollouts_percent = default_rollouts_percent
        self.tstep = t_step
        self.total_rollouts = total_rollouts
        if plot:
            self.budget_figure = plt.figure("log_rollout_allocator")
        self.plot = plot
        self.currently_used_rollouts = currently_used_rollouts
        self.logger_name = logger_name
        super().__init__()

    def allocated_rollouts(self):
        default_rollouts = int(self.total_rollouts * self.default_rollouts_percent)
        if len(self.log_sum_stdvs) < 2:
            logging.info(f"Allocating default rollouts {default_rollouts}")
            return default_rollouts

        estimated_log_sum_stdv = self.model_1.predict(np.array([[self.used_budgets[-1] - self.tstep]]))
        #this is how much improvement we should have when we are done
        estimated_log_sum_stdv_improvement = self.log_sum_stdvs[0] - estimated_log_sum_stdv

        estimated_end_stdv = self.model_1.predict(np.array([[0.0]]))
        estimated_overall_improvement = self.log_sum_stdvs[0] - estimated_end_stdv

        #Now we estimate how much of the total improvement we should have
        fractional_improvment_we_should_be_at = abs(estimated_log_sum_stdv_improvement / estimated_overall_improvement)
        #This is based on the total overall improvement how much effort should we have spent
        number_of_rollouts_we_shouldve_used = fractional_improvment_we_should_be_at * self.total_rollouts

        #Subtracts the already spent rollouts from the rollouts we shouldve used
        number_of_free_rollouts = number_of_rollouts_we_shouldve_used - self.currently_used_rollouts


        logging.getLogger(self.logger_name).debug(f"Expected Current Improvement {estimated_log_sum_stdv_improvement}, Expected Total Improvement {estimated_overall_improvement}, Rollouts this iteration {number_of_free_rollouts}, Estimated log sum stdv {estimated_log_sum_stdv}, Previous log sum stdv {self.log_sum_stdvs[-1]}")
        logging.getLogger(self.logger_name).debug(f"Fractional Improvment {fractional_improvment_we_should_be_at}, Number of rollouts we should've used {number_of_rollouts_we_shouldve_used}")
        logging.getLogger(self.logger_name).debug(f"Model: {self.model_1.coef_}ln(x) + {self.model_1.intercept_}")

        if self.plot:
            self.budget_figure.gca()
            test_points = np.arange(self.used_budgets[0],0,-self.tstep)
            #self.model_1.coef_ = np.array([0.01])
            sum_stdvs = np.exp(self.model_1.predict(np.array([test_points]).T))
            plt.plot(self.used_budgets[0] - test_points,sum_stdvs,label="Budgets vs expected improvement")
            plt.plot(self.used_budgets[0] - np.array(self.used_budgets),np.exp(self.log_sum_stdvs),label="Data so far")
            plt.xlabel("Used Budget")
            plt.ylabel("$\Sigma \sigma$")
            plt.title( f"${self.model_1.coef_}ln(x) + {self.model_1.intercept_}$")

            plt.show()

        assert number_of_free_rollouts > 0
        return int(number_of_free_rollouts)
