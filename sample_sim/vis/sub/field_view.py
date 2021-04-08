from sample_sim.vis.base_vis import Visualizer
import matplotlib.pyplot as plt
import numpy as np


class FieldView(Visualizer):
    def __init__(self,spatial_dimensions,axis=None,below_mean_removal=True):
        super().__init__(None)
        if axis==None:
            fig = plt.figure()
            self.axis = fig.add_subplot(111,projection="3d")
            self.title = "Field View Plot"
        else:
            self.axis = axis
            self.title = self.axis.title._text
        self.below_mean_removal = below_mean_removal
        self.colorbar_plotted = False
        self.spatial_dimensions = spatial_dimensions
    def update_quantity(self,X,Y,std_dev_below=0,vmin=None,vmax=None):
        self.vmin = vmin
        self.vmax = vmax
        self.out_X = []
        self.out_Y = []
        if not self.below_mean_removal:
            self.out_X = X
            self.out_Y = Y
        else:
            y_mean = np.mean(Y)
            y_std = np.std(Y)
            if np.isfinite(y_std) and np.isfinite(y_mean):
                for x, y in zip(X, Y):
                    if y >= y_mean - std_dev_below * y_std:
                        self.out_X.append(x)
                        self.out_Y.append(y)
                self.out_X = np.array(self.out_X)
                # out_Y = np.array(out_Y)
                if len(self.out_Y) == 0:
                    self.update_quantity(X, Y,std_dev_below=std_dev_below + 0.01)  # Plot it but include mean
    def update(self):
        self.axis.clear()
        if self.spatial_dimensions == 3:
            scatter = self.axis.scatter(self.out_X[:, 0], self.out_X[:, 1], self.out_X[:, 2], c=self.out_Y, vmin=self.vmin, vmax=self.vmax)
        else:
            scatter = self.axis.scatter(self.out_X[:, 0], self.out_X[:, 1], c=self.out_Y,
                                        vmin=self.vmin, vmax=self.vmax)
        if not self.colorbar_plotted:
            self.colorbar_plotted = True
            try:
                plt.colorbar(scatter,ax=self.axis)
            except:
                pass
        self.axis.set_title(self.title)


