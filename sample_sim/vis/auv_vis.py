import matplotlib.pyplot as plt

from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace
from sample_sim.robot_manager import RobotManager
from sample_sim.vis.base_vis import Visualizer
from sample_sim.vis.sub.auv_view import AUVView
from sample_sim.vis.sub.field_view import FieldView
from sample_sim.vis.util import link_3d_axes


class AUVViewVis(Visualizer):
    def __init__(self,auv_manager:RobotManager,data_model:DataModel,workspace:Workspace,vbounds=None,screen_size=(20,10),fignum=1):
        if vbounds is not None:
            self.vmin, self.vmax = vbounds
        else:
            self.vmin = None
            self.vmax = None
        self.auv_manager = auv_manager
        self.data_model = data_model
        self.workspace = workspace
        self.fig = plt.figure(fignum,figsize=screen_size,)
        super().__init__(self.fig)
        if workspace.dimensions() == 3:
            self.auv_visualizer_ax = self.fig.add_subplot(131, projection="3d")
            self.auv_gp_mean_ax = self.fig.add_subplot(132, projection="3d")
            self.auv_gp_std_ax = self.fig.add_subplot(133, projection="3d")

            link_3d_axes(self.fig, [self.auv_visualizer_ax, self.auv_gp_std_ax, self.auv_gp_mean_ax])
        else:
            self.auv_visualizer_ax = self.fig.add_subplot(131)
            self.auv_gp_mean_ax = self.fig.add_subplot(132)
            self.auv_gp_std_ax = self.fig.add_subplot(133)
        self.auv_visualizer_ax.set_title("AUV Path Samples")
        self.auv_gp_mean_ax.set_title("AUV Model $\mu$")
        self.auv_gp_std_ax.set_title("AUV Model $\sigma$")
        self.axes = [self.auv_visualizer_ax, self.auv_gp_mean_ax, self.auv_gp_std_ax]

        self.auv_view = AUVView(self.auv_manager,self.data_model,workspace,vbounds,self.auv_visualizer_ax)
        self.mean_field_view = FieldView(workspace.dimensions(), self.auv_gp_mean_ax)
        self.std_field_view=FieldView(workspace.dimensions(), self.auv_gp_std_ax)

        #Link all the axes together
    def update(self):
        for axes in self.axes:
            axes.clear()
        self.auv_view.update()
        if self.workspace.dimensions() == 3:
            t_X = self.workspace.get_meshgrid(10)
        else:
            t_X = self.workspace.get_meshgrid(100)
        mean,std = self.data_model.query_many(t_X)
        self.mean_field_view.update_quantity(t_X,mean,vmin=self.vmin,vmax=self.vmax)
        self.mean_field_view.update()
        self.std_field_view.update_quantity(t_X,std)
        self.std_field_view.update()
