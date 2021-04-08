import logging
import pickle

import numpy as np

from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.data_model.workspace import RectangularPrismWorkspace


def curved_function(X_t):
    """
    https://www.sfu.ca/~ssurjano/detpep10curv.html
    :return:
    """
    #assert 0 <= X_t <= 1
    return 4 * (X_t[:,0] - 2 + 8 * X_t[:,1] - 8 * X_t[:,1]**2)**2 + (3 - 4 * X_t[:,1])**2 + 16 * np.sqrt(X_t[:,2] + 1) * (2 * X_t[:,2] - 1) ** 2
def exponential_function(X_t):
    """
    https://www.sfu.ca/~ssurjano/detpep10exp.html
    """
    X_t = np.clip(X_t,0.001,1)
    x1 = np.exp(-2/(X_t[:,0]**1.75))
    x2 = np.exp(-2/(X_t[:,1]**1.5))
    x3 = np.exp(-2/(X_t[:,2]**1.25))
    return (100 * (x1 + x2 + x3))**2

def sbo_function(X_t):
    #The t has a period of 1, make it have a period of 12
    t = X_t[:,2] * 12
    f1 = 1.5 * np.sin(2 * np.pi * t)
    f2 = 1.5 * np.cos(2 * np.pi * t)

    #SBO Test function assumes x_i \in [0,5] but our api  assumes it's \in [0,1]
    x_1 = X_t[:,0] * 5
    x_2 = X_t[:,1] * 5
    top1 = x_1 - 2 - f1
    top2 = x_2 - 2 - f2
    exp1 = top1 / 0.7
    exp2 = top2 / 0.7
    return np.exp(-(exp1)**2) * np.exp(-(exp2)**2)


def cache_name(function_name):
    return f".cache/{function_name}"

def uncache(function_name):
    try:
        fname = cache_name(function_name)
        with open(f"{fname}w.pkl","rb") as f:
            workspace = pickle.load(f)
            X = pickle.load(f)
            Y = pickle.load(f)
        model = TorchExactGPBackedDataModel(X=X, Y=Y,logger=logging.getLogger("default"),workspace=workspace)
        model.load(fname)
        return model,workspace
    except Exception as e:
        print(e)
        print(f"Rebuilding cache for function: {function_name}")
def create_data(fn_name):
    r = uncache(fn_name)
    if r is not None:
        return r
    else:
        if fn_name == "fn:exp":
            fn = exponential_function
        elif fn_name == "fn:curved":
            fn = curved_function
        elif fn_name == "fn:sbo":
            fn = sbo_function
        else:
            raise Exception("Function not understood")
        fn_workspace = RectangularPrismWorkspace(0,1,0,1,0,1)
        X_fn = fn_workspace.get_meshgrid((10,10,10))
        Y = fn(X_fn)
        workspace = RectangularPrismWorkspace(0,30,0,30,0,30)
        X = workspace.get_meshgrid((10,10,10))

        model = TorchExactGPBackedDataModel(logger="default",X=X, Y=Y,workspace=workspace)
        model.update(X,Y,input_uncertainties=None)
        model.fit(1000)

        fname = cache_name(fn_name)
        with open(f"{fname}w.pkl","wb") as f:
            pickle.dump(workspace,f)
            pickle.dump(X,f)
            pickle.dump(Y,f)
        model.save(fname)
        return model,workspace


