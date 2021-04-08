import matplotlib.pyplot as plt
import numpy as np

from sample_sim.planning.pomcp_planner import POMCPPlanner
from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace
from sample_sim.vis.base_vis import Visualizer


class CriticalPathVisualizer(Visualizer):

    def __init__(self, auv, pomcp_planner: POMCPPlanner, data_model: DataModel, workspace: Workspace, vbounds=None, screen_size=(20, 10), fignum=1):
        if vbounds is not None:
            self.vmin, self.vmax = vbounds
        else:
            self.vmin = self.vmax = None

        self.auv = auv
        self.model = data_model
        self.planner = pomcp_planner
        self.workspace = workspace

        self.fig = plt.figure(fignum, figsize=screen_size)

        super().__init__(self.fig)

        if workspace.dimensions() == 3:
            self.visualizer_ax = self.fig.add_subplot(131, projection="3d")
        else:
            self.visualizer_ax = self.fig.add_subplot(131)

        self.visualizer_ax.set_title("AUV Trajectory and Critical Path")

    def update(self):

        # Clear axes first
        self.visualizer_ax.clear()

        # Plot the AUV trajectory/history
        self.model._flatten_data()
        if self.model.Xs.size:
            self.visualizer_ax.scatter(self.model.Xs[:, 0], self.model.Xs[:, 1], self.model.Xs[:, 2],
                                       s=20, marker="o", c=self.model.Ys, vmin=self.vmin, vmax=self.vmax)

        # Then plot critical path at current time
        rewards = None
        path = None
        
        # Separate critical path into 2 lists
        for position, reward in self.planner.get_critical_path(coordinates=True):
            
            if path is None:
                path = position
            else:
                path = np.vstack((path, position))
            
            if rewards is None:
                rewards = reward
            else:
                rewards = np.append(rewards, reward)
        
        # Convert lists to numpy arrays then plot critical path all at the same time value
        if path is not None and path.size:
            
            path = np.vstack(path)
            
            if rewards.ndim > 1:
                rewards = np.concatenate(rewards)
            
            self.visualizer_ax.scatter(path[:, 0], path[:, 1], path[:, 2], s=20,
                                       marker="x", c=rewards, vmin=self.vmin, vmax=self.vmax)
        
        self.visualizer_ax.set_xlim3d(0, 3)
        self.visualizer_ax.set_ylim3d(0, 3)
        self.visualizer_ax.set_zlim3d(self.workspace.zmin, self.workspace.zmax)

