import numpy as np
from scipy.special import comb

from sample_sim.planning.pomcp_rollout_allocation.base_rollout_allocator import PrecalculatedBaseRolloutAllocator


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


class ThreePointBezierCurve(PrecalculatedBaseRolloutAllocator):
    def __init__(self, max_budget, currently_used_budget, max_rollout, currently_used_rollout, control_point):
        self.max_budget = max_budget
        self.currently_used_budget = currently_used_budget
        self.max_rollout = max_rollout
        self.currently_used_rollout = currently_used_rollout
        self.control_point = control_point
        super().__init__()

    def allocated_rollouts(self):
        end_point = [1, 1]
        start_point = [0, 0]
        control_point = self.control_point
        points = np.array([start_point, control_point, end_point])
        fraction_of_budget_used = (self.currently_used_budget + 1) / self.max_budget
        xvals,yvals = bezier_curve(points)

        xval_idx = None
        closest_xval_dist = float("inf")
        for i,xval in enumerate(xvals):
            dist = abs(fraction_of_budget_used - xval)
            if dist < closest_xval_dist:
                xval_idx = i
                closest_xval_dist = dist

        fractional_improvment_we_should_be_at = yvals[xval_idx]

        # This is based on the total overall improvement how much effort should we have spent
        number_of_rollouts_we_shouldve_used = fractional_improvment_we_should_be_at * self.max_rollout

        # Subtracts the already spent rollouts from the rollouts we shouldve used
        number_of_free_rollouts = number_of_rollouts_we_shouldve_used - self.currently_used_rollout

        assert number_of_free_rollouts > 0
        return int(number_of_free_rollouts)

