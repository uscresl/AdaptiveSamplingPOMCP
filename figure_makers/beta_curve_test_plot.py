import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
from matplotlib.collections import LineCollection
from smallab.file_locations import get_experiment_save_directory
from tqdm import tqdm
from matplotlib.lines import Line2D

os.chdir("..")
experiment_name = "beta_curve_grid_sbo"

# x_var = ("result","objective_c_vis")
# y_var = ("result","MMSE")
y_var = ("result", "total_used_rollouts")
# y_var = ("result","rollouts_used_this_iteration")
x_var = ("result", "used_budget")
dash_var = ("specification", "rollout_number_goal")
# hue_var = ("specification","rollout_allocation_method")
# hue_var = ("specification","seed")
# hue_var = ("specification", "bezier_control_point")
graph_split_var = ("specification", "file")

x_name = "Budget"
y_name = "Used Rollouts"
# y_name = "RMSE"
# dash_var_name = "UCB $C$"
dash_var_name = "Rollout Number Goal"
# hue_var_name = "Rollout Allocation Method"
# hue_var_name = "Seed"
hue_var_name = "Alpha Beta Point"
graph_split_name = "Environment {}"
# graph_split_name = "Rollout Allocation Method"


df = pandas.DataFrame(columns=[x_name, y_name, dash_var_name, hue_var_name, graph_split_name, 'mmse', "seed"])
df = df.astype({y_name: 'float64', x_name: 'float64', dash_var_name: 'category', hue_var_name: 'category',
                graph_split_name: 'category', 'mmse': "float64"})


def tqdm_experiment_iterator(name):
    for root, _, files in tqdm(list(os.walk(get_experiment_save_directory(name)))):
        for fname in files:
            if ".pkl" in fname:
                with open(os.path.join(root, fname), "rb") as f:
                    yield pickle.load(f)
            if ".json" in fname and fname != "specification.json":
                with open(os.path.join(root, fname), "r") as f:
                    yield json.load(f)


try:
    df = pandas.read_pickle(f"{experiment_name}.pkl")
except:
    for experiment in tqdm_experiment_iterator(experiment_name):
        d = dict()
        if float(experiment["specification"]["alpha_param"]) > 6 or float(experiment["specification"]["beta_param"] > 6):
            continue

        if experiment['result'] != []:
            d[x_name] = experiment[x_var[0]][x_var[1]]
            d[y_name] = experiment[y_var[0]][y_var[1]]
            d[dash_var_name] = str(experiment[dash_var[0]][dash_var[1]])
            d[hue_var_name] = (
            str(experiment["specification"]["alpha_param"]), str(experiment["specification"]["beta_param"]))
            d[graph_split_name] = str(experiment[graph_split_var[0]][graph_split_var[1]])
            d['mmse'] = experiment["result"]["MMSE"]
            d["seed"] = experiment["specification"]["seed"]
            d['Ys'] = experiment["result"]["Ys"][-1]
            d['Xs'] = experiment["result"]["Xs"][-1]
            df = df.append(d, True)
    df.to_pickle(f"{experiment_name}.pkl")
#for alpha_beta in df[hue_var_name].values:
#    if alpha_beta[0] == str(0.0) or alpha_beta[1] == str(0.0):
#        df = df.loc[df[hue_var_name] != alpha_beta]

rmse_to_control_point = []
for control_point in df[hue_var_name].unique():
    df_cur = df.loc[df[hue_var_name] == control_point]
    max_budget = df_cur[x_name].max()
    df_cur = df_cur.loc[df_cur[x_name] == max_budget]
    rmse = np.mean(list(map(np.sum,df_cur["Ys"].values)))
    df.loc[df[hue_var_name] == control_point, "EndRMSE"] = rmse
    rmse_to_control_point.append((rmse, control_point))


def multiline(xs, ys, c, legend_entries, ax=None,**kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    #def make_proxy(zvalue, scalar_mappable, **kwargs):
    #    color = scalar_mappable.cmap(zvalue)
    #    return Line2D([0, 1], [0, 1], color=color, **kwargs)
    #proxies = [make_proxy(item, lc, linewidth=5) for item in c]
    #ax.legend(proxies, legend_entries)

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


rmse_to_control_point = list(reversed(sorted(rmse_to_control_point)))
top_control_points = list(map(lambda x: x[1], rmse_to_control_point))[:3]
bottom_control_points = list(map(lambda x: x[1], filter(lambda x: float(x[1][0]) == 1.0 and float(x[1][1]) == 1.0, rmse_to_control_point)))
top_control_points += bottom_control_points

print(top_control_points)
# #top_control_points = bottom_control_points
# top_control_points = list(map(lambda x: x[1], rmse_to_control_point))

for split_var in df[graph_split_name].unique():
    df_cur = df.loc[df[graph_split_name] == split_var]
    # df_cur = df_cur.loc[df_cur[hue_var_name].isin(bottom_control_points)]
    # for control_point in df_cur[hue_var_name].unique():
    #     df_cur = df_cur.loc[df_cur[hue_var_name] == control_point]
    xs = []
    ys = []
    cs = []
    for control_point in reversed(top_control_points):
        df_cur = df.loc[(df[hue_var_name] == control_point)]
        xys = sorted(zip(df_cur[x_name].values , df_cur[y_name].values))
        xys = np.array(xys)
        #xys = np.concatenate((xys,np.array([[200,50000]])),axis=0)
        xs.append(xys[:, 0])
        ys.append(xys[:, 1])
        cs.append(df_cur["EndRMSE"].values[0])
       
        df_s1 = df_cur.loc[(df_cur["seed"] == 1) & (df_cur["Budget"] == 199)]

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #Xs = df_s1["Xs"].values[0]
        #Ys = df_s1["Ys"].values[0]
        #p = ax.scatter(Xs[:,0],Xs[:,1],Xs[:,2],c=Ys.ravel(),vmin=0,vmax=1)
        #ax.set_title(f'{control_point}')
        #ax.set_xlim(0,30)
        #ax.set_ylim(0,30)
        #ax.set_zlim(0,30)
        #plt.colorbar(p)
        #, Isns.lineplot(xs,ys,data=df_cur)

    fig = plt.figure()
    lc = multiline(xs, ys, cs, top_control_points)
    axcb = fig.colorbar(lc)
    axcb.set_label("Mean Final Accumulated Reward")
    plt.xlabel("Time Steps")
    plt.ylabel("Used Rollouts")
    # sns.lineplot(x=x_name, y=y_name, data=df_cur)
    plt.title("Top Three Rollout Curves and Fixed")
    plt.savefig("grid_search.pdf")
plt.show()
