
import matplotlib.pyplot as plt

from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace
from sample_sim.robot_manager import RobotManager
from sample_sim.vis.base_vis import Visualizer
from sample_sim.vis.sub.auv_view import AUVView
from sample_sim.vis.sub.field_view import FieldView
from sample_sim.vis.util import link_3d_axes
import numpy as np


class GeneratingFieldViewVis(Visualizer):
    def __init__(self,data_model:DataModel,workspace:Workspace,screen_size=(20,10),vbounds=None,fignum=2):
        if vbounds is not None:
            self.vmin, self.vmax = vbounds
        else:
            self.vmin = None
            self.vmax = None

        self.data_model = data_model
        self.workspace = workspace
        self.fig = plt.figure(fignum,figsize=screen_size)
        super().__init__(self.fig)
        if workspace.dimensions() == 3:
            self.generating_visualizer_ax = self.fig.add_subplot(131, projection="3d")
            self.auv_gp_mean_ax = self.fig.add_subplot(132, projection="3d")
            self.auv_gp_std_ax = self.fig.add_subplot(133, projection="3d")

            link_3d_axes(self.fig, [self.generating_visualizer_ax, self.auv_gp_mean_ax, self.auv_gp_std_ax])

            t_X = workspace.get_meshgrid(10)
        else:
            self.generating_visualizer_ax = self.fig.add_subplot(131)
            self.auv_gp_mean_ax = self.fig.add_subplot(132)
            self.auv_gp_std_ax = self.fig.add_subplot(133)
            t_X = workspace.get_meshgrid(10)

        self.generating_visualizer_ax.set_title("Loaded Data")
        self.auv_gp_mean_ax.set_title("GP From Data $\mu$")
        self.auv_gp_std_ax.set_title("GP From Data $\sigma$")

        self.original_field_view = FieldView(workspace.dimensions(), self.generating_visualizer_ax)
        self.data_model._flatten_data()
        self.original_field_view.update_quantity(self.data_model.Xs,self.data_model.Ys,vmin=self.vmin,vmax=self.vmax)
        t_X = np.vstack((t_X,self.data_model.Xs))
        mean,std = self.data_model.query_many(t_X)
        self.mean_field_view = FieldView(workspace.dimensions(), self.auv_gp_mean_ax)
        self.mean_field_view.update_quantity(t_X,mean,vmin=self.vmin,vmax=self.vmax)
        self.std_field_view=FieldView(workspace.dimensions(), self.auv_gp_std_ax)
        self.std_field_view.update_quantity(t_X,std)
    def update(self):
        self.fig.clear()
        self.original_field_view.update()
        self.mean_field_view.update()
        self.std_field_view.update()
