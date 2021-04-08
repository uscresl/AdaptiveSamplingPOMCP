from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace, RectangularPrismWorkspace
from sample_sim.robot_manager import RobotManager
from sample_sim.vis.base_vis import Visualizer
import matplotlib.pyplot as plt


class AUVView(Visualizer):
    def __init__(self, auv:RobotManager, auv_data:DataModel, workspace:RectangularPrismWorkspace, vbounds=None, axis=None):
        super().__init__(None)
        if vbounds is not None:
            self.vmin, self.vmax = vbounds
        else:
            self.vmin = None
            self.vmax = None
        if axis==None:
            fig = plt.figure()
            if workspace.dimensions() == 3:
                self.axis = fig.add_subplot(111,projection="3d")
            else:
                self.axis = fig.add_subplot(111)
            self.title("AUV")
        else:
            self.axis = axis
            self.title = self.axis.title._text
        self.auv = auv
        self.auv_data = auv_data
        self.workspace = workspace



    def update(self):
        self.auv_data._flatten_data()
        self.axis.clear()
        if self.workspace.dimensions() == 3:
            self.axis.scatter(self.auv_data.Xs[:, 0], self.auv_data.Xs[:, 1], self.auv_data.Xs[:, 2], s=20, marker="o",
                                          c=self.auv_data.Ys.ravel(),vmin=self.vmin,vmax=self.vmax)
            x,y,z = self.auv.get_current_state()
            self.axis.scatter([x],[y],[z],s=40,c="r",marker='x')
            self.axis.set_xlim3d(self.workspace.xmin, self.workspace.xmax)
            self.axis.set_ylim3d(self.workspace.ymin, self.workspace.ymax)
            self.axis.set_zlim3d(self.workspace.zmin, self.workspace.zmax)
        else:
            self.axis.scatter(self.auv_data.Xs[:, 0], self.auv_data.Xs[:, 1], s=20, marker="o",
                              c=self.auv_data.Ys.ravel(), vmin=self.vmin, vmax=self.vmax)
            x, y, z, theta = self.auv.get_current_state()
            self.axis.scatter([x], [y],s=40, c="r", marker='x')
            self.axis.set_xlim(self.workspace.xmin,self.workspace.xmax)
            self.axis.set_ylim(self.workspace.ymin, self.workspace.ymax)


        self.axis.set_title(self.title)


