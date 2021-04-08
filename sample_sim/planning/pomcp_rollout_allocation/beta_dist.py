import logging

from scipy.stats import beta

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import PrecalculatedBaseRolloutAllocator


class CumulativeBetaDistributions(PrecalculatedBaseRolloutAllocator):
    def __init__(self, max_budget, currently_used_budget, max_rollout, currently_used_rollout, alpha_param,beta_param, logger_name, action_enum, waste_unused_rollouts):
        self.max_budget = max_budget
        self.currently_used_budget = currently_used_budget
        self.max_rollout = max_rollout
        self.currently_used_rollout = currently_used_rollout
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        self.logger_name = logger_name
        self.K = len(action_enum)
        self.waste_unused_rollouts = waste_unused_rollouts
        super().__init__()

    def allocated_rollouts(self):
        fraction_of_budget_used = (self.currently_used_budget + 1) / self.max_budget

        yvals = beta.cdf([fraction_of_budget_used], self.alpha_param, self.beta_param)

        fractional_improvment_we_should_be_at = yvals[0]

        # This is based on the total overall improvement how much effort should we have spent
        number_of_rollouts_we_shouldve_used = fractional_improvment_we_should_be_at * self.max_rollout

        if not self.waste_unused_rollouts:
            # Subtracts the already spent rollouts from the rollouts we shouldve used
            number_of_free_rollouts = number_of_rollouts_we_shouldve_used - self.currently_used_rollout
        else:
        #not wasting rollouts. Thus, number_of_free_rollouts = {cdf (budget+1) - cdf(budget)} * max_rollouts.
        # for a given cdf i.e. fixed alpha and beta
            fractional_improvment_we_are_at = \
            beta.cdf([(self.currently_used_budget) / self.max_budget], self.alpha_param, self.beta_param)[0]
            number_of_rollouts_we_have_used = fractional_improvment_we_are_at * self.max_rollout
            number_of_free_rollouts = number_of_rollouts_we_shouldve_used - number_of_rollouts_we_have_used

        logging.getLogger(self.logger_name).debug(f"Rollouts {number_of_free_rollouts}")
        if number_of_free_rollouts < self.K + 2:
            return self.K + 2
        return int(number_of_free_rollouts)
