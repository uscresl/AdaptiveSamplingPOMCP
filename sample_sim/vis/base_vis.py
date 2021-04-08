import abc
import matplotlib.pyplot as plt


class Visualizer(abc.ABC):
    def __init__(self,fig):
        self.fig = fig
    def set_title(self,title):
        self.fig.suptitle(title)
    @abc.abstractmethod
    def update(self):
        pass
    def save(self,name):
        self.fig.savefig(name)
