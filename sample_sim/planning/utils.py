import numpy as np


def unit_vector_between_two_points(pi,pj):
    """
    >>> unit_vector_between_two_points(np.array([0,0,0]),np.array([1,0,0]))
    array([1., 0., 0.])
    >>> np.isclose(unit_vector_between_two_points(np.array([5,5,5]),np.array([7,7,7])),unit_vector_between_two_points(np.array([5,5,5]),np.array([8,8,8]))).all()
    True

    :param pi:
    :param pj:
    :return:
    """
    #Unit vector pointing from pi to pj
    #https://math.stackexchange.com/questions/12745/how-do-you-calculate-the-unit-vector-between-two-points
    diff = pj - pi
    return diff / np.linalg.norm(diff,ord=2)

def euc_dist(p1,p2):
    return np.linalg.norm(p2-p1,ord=2)
