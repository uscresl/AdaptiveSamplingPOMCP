
#import seaborn# as sns
#sns.set_theme()

import matplotlib
matplotlib.use('Agg')

from sample_sim.data_model.loaders.analytical_function_loader import (create_data, sbo_function, curved_function,
                                                                      exponential_function)
from sample_sim.data_model.loaders.ecomapper_loader import load_from_ecomapper_data
from sample_sim.data_model.workspace import RectangularPrismWorkspace
from sample_sim.general_utils import coordinates_to_unit_cube
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

files = ["fn:sbo"]



def rename_title(s):
    if "fn" in s:
        return "Dynamic Function"
    elif "163636" in s:
        return "Validation Environment 1"
    else:
        return "Validation Environment 2"

def get_vmin_vmax(s):
    if s == "Dynamic Function":
        return 0,1
    elif "1" in s:
        return 6,23
    elif "2" in s:
        return 20,90

for file in files:
    print(file)
    if "fn" in file:
        oracle_model, workspace = create_data(file)
        oracle_model_for_hps = oracle_model
        if file == "fn:exp":
            chosen_function = exponential_function
        elif file == "fn:curved":
            chosen_function = curved_function
        elif file == "fn:sbo":
            chosen_function = sbo_function
    else:
        oracle_model, workspace = load_from_ecomapper_data(file, 15, 50)
        oracle_model_for_hps = oracle_model
        oracle_model = oracle_model
        workspace = workspace
    # print([workspace.xmin, workspace.xmax, workspace.ymin, workspace.ymax, workspace.zmin, workspace.zmax])
    oracle_model.model.eval_model()
    grid = (25,25,50)

    X_t = workspace.get_meshgrid(grid)
    plot_workspace = RectangularPrismWorkspace(0,50,0,50,0,200) 
    if "fn" in file:
        samples = chosen_function(
            coordinates_to_unit_cube(X_t, workspace))
    else:
        samples = oracle_model.query_many(X_t, return_std=False)

    X_t = plot_workspace.get_meshgrid(grid)
    Y = samples
    out_X = []
    out_Y = []
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    std_dev_below = 0
    if "fn" in file:
        std_dev_below = -1
    if np.isfinite(y_std) and np.isfinite(y_mean):
        for x, y in zip(X_t, Y):
            if y >= y_mean - std_dev_below * y_std:
                out_X.append(x)
                out_Y.append(y)
        X_t = np.array(out_X)
        Y = np.array(out_Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    vmin,vmax = get_vmin_vmax(rename_title(file))
    ax.grid(False)
    c = ax.scatter(X_t[:,0], X_t[:,1], X_t[:,2],c=Y.ravel(),vmin=vmin,vmax=vmax)
    plt.title(rename_title(file))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    plt.tight_layout()
    plt.savefig(f"{rename_title(file)}-environment.pdf",dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
