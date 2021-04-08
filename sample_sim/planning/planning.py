import abc
from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace
from sample_sim.robot_manager import RobotManager


class PlanningAgent(abc.ABC):
    @abc.abstractmethod
    def next_step(self,auv:RobotManager,data_model:DataModel,workspace:Workspace):
        pass
