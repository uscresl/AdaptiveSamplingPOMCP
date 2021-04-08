import matplotlib.pyplot as plt

from sample_sim.ocean_currents.simple_ocean_current_model import SwirlyCurrent
from sample_sim.vis.base_vis import Visualizer


class CurrentVis(Visualizer):

    def __init__(self, current_model, spatial_dimensions, screen_size=(20, 10)):
        self.fig = plt.figure(figsize=screen_size)
        self.spatial_dimensions = spatial_dimensions
        if spatial_dimensions == 3:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.current_model = current_model

    def update(self):
        self.ax.quiver(self.current_model.grid[:, 0], self.current_model.grid[:, 1], self.current_model.grid[:, 2],
                       self.current_model.Ys[:, 0], self.current_model.Ys[:, 1], self.current_model.Ys[:, 2])

