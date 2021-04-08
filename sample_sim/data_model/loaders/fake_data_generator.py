import logging
import math
import pickle

import random
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from tqdm import tqdm

from sample_sim.data_model.data_model import TorchApproximateGPBackedDataModel, TorchExactGPBackedDataModel
from sample_sim.data_model.workspace import RectangularPrismWorkspace
import numpy as np

def cache_name(seed):
    return f".cache/{seed}"

def uncache(seed):
    try:
        fname = cache_name(seed)
        with open(f"{fname}w.pkl","rb") as f:
            workspace = pickle.load(f)
            X = pickle.load(f)
            Y = pickle.load(f)
        model = TorchExactGPBackedDataModel(X=X, Y=Y,logger=logging.getLogger("default"),verbose=True)
        #model.update(X,Y)
        model.load(fname)
        #model.fit(10)
        return model,workspace
    except:
         print(f"Rebuilding cache for fake data seed: {seed}")


def create_fake_data(seed=1,num_blooms=50,xymin=0,xymax=300,zmin=0,zmax=18, chl_max=100):
    r = uncache(seed)
    if r is not None:
        return r
    else:
        rs = np.random.RandomState(seed)
        workspace = RectangularPrismWorkspace(xmin=xymin, xmax=xymax, ymin=xymin, ymax=xymax, zmin=zmin, zmax=zmax)
        blooms = []
        for i in range(num_blooms):
            blooms.append(multivariate_normal(workspace.get_point_inside(rs),make_spd_matrix(3,rs) * 400))
        X = workspace.get_meshgrid((20,20,10))
        y = []
        for point in tqdm(X,desc="Generating Fake Datapoints",file=TqdmToLogger(logging.getLogger("default"))):
            y_cur = 0
            for bloom in blooms:
                y_cur += bloom.pdf(point)
            y_cur /= num_blooms
            assert 0 <= y_cur <= 1
            #y_cur = math.exp(math.log(chl_max) * y_cur)
            y.append(y_cur)

        Y = np.array(y)
        y_max = np.max(y)
        y_min = np.min(Y)
        slope = chl_max / (y_max - y_min)
        Y *= slope


        use_exact = True
        model = TorchExactGPBackedDataModel(logger=logging.getLogger("default"),X=X, Y=Y)
        #model.set_Y_transform(Mean0Var1())
        #model.update(X, Y, input_uncertainties=None)
        model.fit()

        fname = cache_name(seed)
        with open(f"{fname}w.pkl","wb") as f:
            pickle.dump(workspace,f)
            pickle.dump(X,f)
            pickle.dump(Y,f)
        model.save(fname)
    return model,workspace




