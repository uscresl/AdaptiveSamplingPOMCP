import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import utm
from scipy.io import netcdf

# To download more data: https://thredds.cencoos.org/thredds/ncss/ERDMWCHLA1DAY.nc/dataset.html
from sample_sim.data_model.data_model import TorchExactGPBackedDataModel
from sample_sim.data_model.workspace import RectangularPlaneWorkspace

utm_zones_and_data = dict()


def cache_name(time):
    return ".cache/{}".format(f"cencoos_chl{time}")


def uncache(time):
    try:
        fname = cache_name(time)
        with open(fname + "w.pkl", "rb") as f:
            workspace = pickle.load(f)
            X = pickle.load(f)
            Y = pickle.load(f)
            model = TorchExactGPBackedDataModel(X=X, Y=Y, logger="default", verbose=True)

        model.load(fname)
        return model, workspace
    except Exception as e:
        logging.getLogger("default").debug("Rebuilding Cached files for {} due to exception {}".format(time, e))
        return None


def load_sat_chl(time_idx, stride=50):
    '''
    corners is north_start,north_end,easting_start,easting_end
    :param time_idx:
    :param stride:
    :param corners:
    :return:
    '''
    ds = netcdf.NetCDFFile("chl_data.nc")
    vars = ds.variables

    chl = vars["chlorophyll"].data
    times = vars["time"].data
    lats = vars["latitude"].data
    longs = vars["longitude"].data
    # On the chl data the order is time, altitude, lat, long
    assert 0 <= time_idx <= times.shape[0]
    des_time = times[time_idx]

    r = uncache(des_time)
    if r is not None:
        return r

    utm_zones_and_data = dict()

    for i, lat in enumerate(lats):
        for j, long in enumerate(longs):
            cur_chl = chl[time_idx, 0, i, j]
            northing, easting, zone_number, zone_letter = utm.from_latlon(lat, -1 * (360 - long))
            # if (prev_zone_letter is not None and prev_zone_letter != zone_letter) or (prev_zone_number is not None and prev_zone_number != zone_number):
            #     continue
            utm_key = (zone_number, zone_letter)
            if utm_key not in utm_zones_and_data:
                utm_zones_and_data[utm_key] = ([], [])  # xs and ys
            if cur_chl > 0.0:
                utm_zones_and_data[utm_key][0].append([northing/5000, easting/5000])
                utm_zones_and_data[utm_key][1].append(cur_chl)

    utm_zones_and_data = {k: (np.array(v[0]), np.array(v[1])) for k, v in utm_zones_and_data.items()}
    best_key = None
    most_good_values = float("-inf")
    for key, value in utm_zones_and_data.items():
        if value[1].shape[0] > most_good_values:
            best_key = key
            most_good_values = value[1].shape[0]

    data = utm_zones_and_data[best_key]
    plt.ioff()
    plt.figure()
    X = data[0]
    Y = data[1]
    X = X[::stride, :]
    Y = Y[::stride]
    Y = (Y - np.mean(Y)) / np.std(Y)
    xmin = np.min(X[:, 0])
    ymin = np.min(X[:, 1])
    X -= np.array([xmin,ymin])


    #keep_indexes = []
    # for i in range(X.shape[0]):
    #     if xmin + corners[0] <= X[i, 0] <= min(xmin + corners[1], xmax) and ymin + corners[0] <= X[i, 1] <= min(
    #         ymin + corners[1], ymax):
    #         keep_indexes.append(i)
    # keep_indexes = np.array(keep_indexes)
    # X = X[keep_indexes,:]
    # Y = Y[keep_indexes]


    print(X.shape)
    # plt.scatter(data[0][:,0],data[0][:,1],c=data[1])
    # plt.colorbar()
    # plt.show()
    model = TorchExactGPBackedDataModel(X, Y, "default")
    model.fit(1 * 10 ** 3)
    workspace = RectangularPlaneWorkspace(np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1]))
    fname = cache_name(des_time)
    with open(fname + "w.pkl", "wb") as f:
        pickle.dump(workspace, f)
        pickle.dump(X, f)
        pickle.dump(Y, f)
    model.save(fname)
    ds.close()
    return model, workspace
