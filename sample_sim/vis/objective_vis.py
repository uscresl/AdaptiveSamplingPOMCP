from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.workspace import Workspace
from sample_sim.vis.base_vis import Visualizer
import matplotlib.pyplot as plt
import numpy as np

from sample_sim.vis.sub.field_view import FieldView
from sample_sim.vis.util import link_3d_axes


class ObjectiveVis(Visualizer):
    def __init__(self,fn,data_model:DataModel,workspace:Workspace,screen_size=(20,10),vbounds=None):
        if vbounds is not None:
            self.vmin, self.vmax = vbounds
        else:
            self.vmin = None
            self.vmax = None
        self.data_model = data_model
        self.workspace = workspace
        self.fig = plt.figure(figsize=screen_size)
        super().__init__(self.fig)

        self.object_visualizer_axis = self.fig.add_subplot(131, projection="3d")
        self.auv_gp_mean_ax = self.fig.add_subplot(132, projection="3d")
        self.auv_gp_std_ax = self.fig.add_subplot(133, projection="3d")
        self.object_visualizer_axis.set_title("Objective value")
        self.auv_gp_mean_ax.set_title("Model $\mu$")
        self.auv_gp_std_ax.set_title("Model $\sigma$")

        self.objective_view = FieldView(self.object_visualizer_axis)
        self.mean_field_view = FieldView(self.auv_gp_mean_ax)
        self.std_field_view=FieldView(self.auv_gp_std_ax)
        self.fn = fn

        #Link all the axes together
        link_3d_axes(self.fig,[self.object_visualizer_axis,self.auv_gp_std_ax,self.auv_gp_mean_ax])

    def update(self):
        t_X = self.workspace.get_meshgrid(10)
        means,stds = self.data_model.query_many(t_X)
        objective = []
        for mean, std in zip(means,stds):
            objective.append(self.fn(mean,std))
        self.objective_view.update_quantity(t_X,np.array(objective))
        self.objective_view.update()
        self.mean_field_view.update_quantity(t_X,means,vmin=self.vmin,vmax=self.vmax)
        self.mean_field_view.update()
        self.std_field_view.update_quantity(t_X,stds)
        self.std_field_view.update()