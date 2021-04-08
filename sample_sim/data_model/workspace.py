import typing

import abc
import numpy as np
from numpy.random.mtrand import RandomState


class Workspace(abc.ABC):
    # @abc.abstractmethod
    # def is_inside(self, x, y, z):
    #     passs
    @abc.abstractmethod
    def dimensions(self) -> typing.SupportsInt:
        pass

    @abc.abstractmethod
    def get_point_inside(self, rs: RandomState = None):
        pass

    @abc.abstractmethod
    def get_meshgrid(self, grid_spacing):
        pass


    @abc.abstractmethod
    def get_meshgrid_with_resolution(self, resolution):
        pass

    @abc.abstractmethod
    def is_inside(self,p):
        pass

class TemporalRectangularPlaneWorkspace(Workspace):
    def __init__(self,xmin,xmax,ymin,ymax,tmin,tmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.tmin = tmin
        self.tmax = tmax
    def dimensions(self) -> typing.SupportsInt:
        return 3

    def get_point_inside(self, rs=None):
        if rs is None:
            rs = np.random.RandomState()
        return rs.uniform(self.xmin, self.xmax), \
               rs.uniform(self.ymin, self.ymax), \
               rs.uniform(self.tmin, self.tmax)

    def get_meshgrid(self, grid_spacing):
        if isinstance(grid_spacing, tuple):
            grid_spacing_x, grid_spacing_y, grid_spacing_t = grid_spacing
        else:
            grid_spacing_x = grid_spacing
            grid_spacing_y = grid_spacing
            grid_spacing_t = grid_spacing
        test_x_range = np.linspace(self.xmin, self.xmax, num=grid_spacing_x)
        test_y_range = np.linspace(self.ymin, self.ymax, num=grid_spacing_y)
        test_t_range = np.linspace(self.tmin, self.tmax, num=grid_spacing_t)

        test_x, test_y, test_t = np.meshgrid(test_x_range, test_y_range, test_t_range)
        test_x = test_x.flatten()
        test_y = test_y.flatten()
        test_t = test_t.flatten()

        t_X = np.stack((test_x, test_y, test_t), axis=-1)
        return t_X

    def get_meshgrid_with_resolution(self, resolution):
        if isinstance(resolution,tuple):
            x_resolution = resolution[0]
            y_resolution = resolution[1]
            z_resolution = resolution[2]
        else:
            x_resolution = resolution
            y_resolution = resolution
            z_resolution = resolution

        num_x_points = int((self.xmax - self.xmin) / x_resolution)
        num_y_points = int((self.ymax - self.ymin) / y_resolution)
        num_t_points = int((self.tmax - self.tmin) / z_resolution)
        return self.get_meshgrid((num_x_points, num_y_points, num_t_points))

    def is_inside(self,p):
        return self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax and self.tmin <= p[2] <= self.tmax



class RectangularPlaneWorkspace(Workspace):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def dimensions(self) -> typing.SupportsInt:
        return 2

    # def is_inside(self, x, y, z):
    #     return self.xmin < x < self.xmax and self.ymin < y < self.ymax

    def get_point_inside(self, rs=None):
        if rs is None:
            rs = np.random.RandomState()
        return rs.uniform(self.xmin, self.xmax), \
               rs.uniform(self.ymin, self.ymax)

    def get_meshgrid(self, grid_spacing):
        if isinstance(grid_spacing, tuple):
            grid_spacing_x, grid_spacing_y = grid_spacing
        else:
            grid_spacing_x = grid_spacing
            grid_spacing_y = grid_spacing
        test_x_range = np.linspace(self.xmin, self.xmax, num=grid_spacing_x)
        test_y_range = np.linspace(self.ymin, self.ymax, num=grid_spacing_y)

        test_x, test_y = np.meshgrid(test_x_range, test_y_range)
        test_x = test_x.flatten()
        test_y = test_y.flatten()

        t_X = np.stack((test_x, test_y), axis=-1)
        return t_X

    def get_meshgrid_with_resolution(self, resolution):
        num_x_points = int((self.xmax - self.xmin) / resolution)
        num_y_points = int((self.ymax - self.ymin) / resolution)
        return self.get_meshgrid((num_x_points, num_y_points))


class RectangularPrismWorkspace(Workspace):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def dimensions(self) -> typing.SupportsInt:
        return 3

    #
    # def is_inside(self, x, y, z):
    #     return self.xmin < x < self.xmax and self.ymin < y < self.ymax and self.zmin < z < self.zmax

    def get_point_inside(self, rs=None):
        if rs is None:
            rs = np.random.RandomState()
        return rs.uniform(self.xmin, self.xmax), \
               rs.uniform(self.ymin, self.ymax), \
               rs.uniform(self.zmin, self.zmax)

    def get_meshgrid(self, grid_spacing):
        if isinstance(grid_spacing, tuple) or isinstance(grid_spacing,list):
            grid_spacing_x, grid_spacing_y, grid_spacing_z = grid_spacing
        else:
            grid_spacing_x = grid_spacing
            grid_spacing_y = grid_spacing
            grid_spacing_z = grid_spacing
        test_x_range = np.linspace(self.xmin, self.xmax, num=grid_spacing_x)
        test_y_range = np.linspace(self.ymin, self.ymax, num=grid_spacing_y)
        test_z_range = np.linspace(self.zmin, self.zmax, num=grid_spacing_z)

        test_x, test_y, test_z = np.meshgrid(test_x_range, test_y_range, test_z_range)
        test_x = test_x.flatten()
        test_y = test_y.flatten()
        test_z = test_z.flatten()

        t_X = np.stack((test_x, test_y, test_z), axis=-1)
        return t_X
    def get_meshgrid_with_resolution(self, resolution):
        if isinstance(resolution,tuple) or isinstance(resolution,list):
            x_resolution = resolution[0]
            y_resolution = resolution[1]
            z_resolution = resolution[2]
        else:
            x_resolution = resolution
            y_resolution = resolution
            z_resolution = resolution
        num_x_points = int((self.xmax - self.xmin) / x_resolution)
        num_y_points = int((self.ymax - self.ymin) / y_resolution)
        num_z_points = int((self.zmax - self.zmin) / z_resolution)
        return self.get_meshgrid((num_x_points, num_y_points, num_z_points))

    def is_inside(self, p):
        return self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax and self.zmin <= p[2] <= self.zmax


    


