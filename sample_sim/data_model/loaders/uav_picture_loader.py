import logging
import pickle
import subprocess

from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.data_model.workspace import RectangularPlaneWorkspace
import rasterio


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

def load_from_egret_data(picture_fname,channels=None):

    r = uncache(picture_fname)
    if r is not None:
        return r
    else:
        p = subprocess.Popen(
            ["gdalinfo", picture_fname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        try:
            out, err = p.communicate(timeout=10)
            out = out.decode()
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            print("gladinfo timed out.")
            print("STDOUT:\n{}".format(out))
            print("STDERR:\n{}".format(err))
            raise RuntimeError

        ur = list(map(float, out[out.find("Upper Right") + 15:
                                 out.find("Upper Right") + 38].split(",")))
        ll = list(map(float, out[out.find("Lower Left") + 15:
                                 out.find("Lower Left") + 38].split(",")))

        workspace = RectangularPlaneWorkspace(min(ll[0],ur[0]),max(ll[0],ur[0]),min(ll[1],ur[1]),max(ll[1],ur[1]))
        with rasterio.open(picture_fname) as src:
            t = src.read()
        X = workspace.get_meshgrid((20,20,20))

        model = TorchExactGPBackedDataModel(logger="default",X=X, Y=Y,workspace=workspace)
        model.update(X,Y,input_uncertainties=None)
        model.fit(1000)



        fname = cache_name(picture_fname)
        with open(f"{fname}w.pkl","wb") as f:
            pickle.dump(workspace,f)
            pickle.dump(X,f)
            pickle.dump(Y,f)
        model.save(fname)
        return model,workspace


