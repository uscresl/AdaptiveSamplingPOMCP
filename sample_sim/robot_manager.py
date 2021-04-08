import abc
import numpy as np

from sample_sim.deprecated import deprecated


class RobotManager(abc.ABC):
    @abc.abstractmethod
    def get_current_state(self):
        pass
    @abc.abstractmethod
    def update_state(self,s):
        pass

class SimulatedAuv(RobotManager):
    @deprecated
    def __init__(self,x,y,z,theta=0):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.max_velocity = 1.2
        self.max_bank = np.pi / 4
        self.max_gamma = np.pi / 6
    def get_current_state(self):
        return [self.x,self.y,self.z,self.theta]
    def update_state(self,s):
        self.x,self.y,self.z,self.theta = s

class Simulated2dTemporalPoint(RobotManager):
    def __init__(self,x,y,t):
        self.x = x
        self.y = y
        self.t = t
        self.max_velocity = 1.2
        self.max_bank = np.pi / 4
        self.max_gamma = np.pi / 6

    def get_current_state(self):
        return np.array([self.x,self.y,self.t])

    def update_state(self,s):
        self.x,self.y,self.t = s


