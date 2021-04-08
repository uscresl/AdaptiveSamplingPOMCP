import enum

import abc
from scipy import interpolate
import numpy as np

class MotionModels(enum.Enum):
    Linear = 1,

def total_distance(path):
    dist = 0
    for p1,p2 in zip(path[:-1],path[1:]):
        dist += np.linalg.norm(p1-p2)
    return dist

class MotionModel(abc.ABC):
    @abc.abstractmethod
    def is_feasible(self,p1,p2):
        pass

    @abc.abstractmethod
    def construct_path(self, nodes):
        pass

class LinearMotionModel(MotionModel):
    def is_feasible(self,p1,p2):
        return True
    def construct_path(self, nodes):
        points = []
        for i in range(nodes.shape[0]-1):
            unit_vector = (nodes[i+1,:] - nodes[i,:])
            unit_vector = unit_vector / np.linalg.norm(unit_vector)
            points.append(nodes[i,:])
            while np.linalg.norm(points[-1] - nodes[i+1,:]) > 1.1:
                points.append(points[-1] + unit_vector)
        out =  np.append(np.array(points),np.zeros((len(points),1)),1)
        assert out.shape[0] >= nodes.shape[0]
        return out

class SplineMotionModel(MotionModel):
    def __init__(self,step_size=1):
        self.step_size = step_size

    def is_feasible(self, p1, p2):
        return True

    def construct_path(self, nodes):
        '''
        https://stackoverflow.com/questions/18962175/spline-interpolation-coefficients-of-a-line-curve-in-3d-space
        :param nodes:
        :return:
        '''
        total_dist = np.sum(nodes)
        num_points = int(round(total_dist * self.step_size))
        #n_steps = total_dist * self.step_size
        tck, u =  interpolate.splprep([nodes[:,0],nodes[:,1],nodes[:,2]], s=2)
        #x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        return np.stack((x_fine,y_fine,z_fine,np.zeros(x_fine.shape)),axis=1)
