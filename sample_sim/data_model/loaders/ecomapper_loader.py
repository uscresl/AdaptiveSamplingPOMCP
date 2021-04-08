import logging

import pandas as pd
import utm
import numpy as np
from cachier import cachier

from sample_sim.data_model.data_model import TorchApproximateGPBackedDataModel, TorchExactGPBackedDataModel
from sample_sim.data_model.workspace import RectangularPrismWorkspace
import pickle
import matplotlib.pyplot as plt

MESHGRID_SIZE = 10

def cache_name(csv_fname):
    return ".cache/{}".format(csv_fname.split("/")[-1])

def uncache(csv_fname):
    try:
        fname = cache_name(csv_fname)
        with open(fname + "w.pkl","rb") as f:
            workspace = pickle.load(f)
            X = pickle.load(f)
            Y = pickle.load(f)
            model = TorchExactGPBackedDataModel(X=X, Y=Y,logger="default",force_cpu=False,workspace=workspace)
        model.load(fname)
        return model,workspace
    except Exception as e:
         logging.getLogger("default").debug("Rebuilding Cached files for {} due to exception {}".format(csv_fname,e))
         return None



def load_from_ecomapper_data(csv_filename,first_waypoint_id,last_waypoint_id,reading_type="YSI-Chl ug/L"):
    r = uncache(csv_filename)
    if r is not None:
        return r
    else:
        data = pd.read_csv(csv_filename, sep=";")
        lat = data["Latitude"].values
        long = data["Longitude"].values
        waypoints = data["Current Step"].values
        assert len(lat) == len(long)

        first_idx = 0
        if first_waypoint_id is not None:
            first_idx = np.min(np.where(waypoints == first_waypoint_id)[0])

        last_idx = lat.size
        if last_waypoint_id is not None:
            last_idx = np.min(np.where(waypoints == last_waypoint_id)[0])

        waypoints = waypoints[first_idx:last_idx]
        lat = lat[first_idx:last_idx]
        long = long[first_idx:last_idx]

        m_xs = []
        m_ys = []
        for cur_lat,cur_lon in zip(lat, long):
            m_x, m_y, zone_number, zone_letter = utm.from_latlon(cur_lat,cur_lon)
            m_xs.append(m_x)
            m_ys.append(m_y)
        m_x = np.array(m_xs)
        m_y = np.array(m_ys)
        m_x = m_x - np.min(m_x)
        m_y = m_y - np.min(m_y)


        height = data["DTB Height (m)"].values
        height = height[first_idx:last_idx]
        values = data[reading_type].values
        values = values[first_idx:last_idx]

        workspace = RectangularPrismWorkspace(np.min(m_x), np.max(m_x), np.min(m_y), np.max(m_y),
                                              np.min(height), np.max(height))
        assert len(values) == len(height) == len(m_x) == len(m_y)
        X = np.stack((m_x,m_y, height), axis=-1)
        Y = values
        # plt.figure()
        # plt.hist(Y,bins=100)
        # plt.title("Non Log")
        #Y[Y < 1 ] = 1
        #Y = np.log(Y)
        # plt.figure()
        # plt.title("Log")
        # plt.hist(np.log(Y),bins=100)
        #
        # plt.show()
        #Y = (Y - np.mean(Y)) / np.std(Y)
        #assert np.isclose(np.mean(Y),0)
        #assert np.isclose(np.std(Y),1)

        model = TorchExactGPBackedDataModel(X,Y,"default",force_cpu=False,workspace=workspace)
        model.update(X,Y,input_uncertainties=None)
        model.fit(1*10**3)
        #



        fname = cache_name(csv_filename)
        with open(fname + "w.pkl","wb") as f:
            pickle.dump(workspace,f)
            pickle.dump(X,f)
            pickle.dump(Y,f)
        model.save(fname)

        return model, workspace
