import numpy as np

from sample_sim.data_model.workspace import RectangularPrismWorkspace


def is_good_matrix(x):
    """
    >>> is_good_matrix(np.array([float('NaN'),0.0]))
    False
    >>> is_good_matrix(np.array([float('inf'),0.0]))
    False
    >>> is_good_matrix(np.array([float('-inf'),0.0]))
    False
    >>> is_good_matrix(np.random.random((100,100)))
    True

    Returns true if all the elements of x are finite and not NaN
    :param x: test matrix
    :return:
    """
    s = np.sum(x)
    return (not np.isnan(s)) and np.isfinite(s)


def stacked_meshgrid(*args):
    t = np.meshgrid(*args)
    t = tuple(map(lambda x: x.flatten(),t))
    t_X = np.stack(t, axis=-1)
    return t_X

def compute_weights(a):
    out = np.abs(a) / np.sum(np.abs(a))
    assert np.isclose(np.sum(out),1)
    return out

def rwmse(gt,y):
    return np.sqrt(np.sum(compute_weights(gt) * np.square(gt - y)))
def wmae(gt,y):
    return np.sum(compute_weights(gt) * np.abs(gt - y))


def coordinates_to_unit_cube(X_t,workspace:RectangularPrismWorkspace):
    X_t_out = np.zeros(X_t.shape)
    X_t_out[:,0] = (X_t[:,0] - workspace.xmin) / (workspace.xmax - workspace.xmin)
    X_t_out[:,1] = (X_t[:,1] - workspace.ymin) / (workspace.ymax - workspace.ymin)
    X_t_out[:,2] = (X_t[:,2] - workspace.zmin) / (workspace.zmax - workspace.zmin)
    return X_t_out