import os

import pandas
import sys
from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

if "figure_makers" in os.getcwd():
    os.chdir("..")
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    experiment_name = "allocator_test_sbo"
print(f"Experiment: {experiment_name}")

x_var = ("result","used_budget"); x_name = "Time steps"
# x_var = ("result","objective_c_vis")
# x_var = ("result","used_budget"); x_name = "Time steps"

y_var = ("result","Ys"); y_name = "Acc Reward"
# y_var = ("result","total_used_rollouts"); y_name = "Used Rollouts"
# y_var = ("result","rollouts_used_this_iteration")

#dash_var = ("specification","plan_commitment_algorithm"); dash_var_name = "Plan Commitment Algorithm"
# dash_var = ("specification", "alpha_param"); dash_var_name = "Alpha"
dash_var = ("specification","rollout_number_goal"); dash_var_name = "Max Rollouts"
# dash_var = ("specification","control_point"); dash_var_name= "Control Point"
hue_var = ("specification", "rollout_allocation_method"); hue_var_name = "Rollout Allocation Method"
# hue_var = ("specification","bezier_control_point"); hue_var_name = "Control Point"
# hue_var = ("specification","beta_param"); hue_var_name = "beta"
# hue_var = ("specification","rollout_allocation_method"); hue_var_name = "Rollout Allocation Method"

graph_split_var = ("specification","file"); graph_split_name= "Environment {}"
#graph_split_var = ("specification","control_point"); graph_split_name= "Control Point {}"
#graph_split_var = ("specification","file"); graph_split_name= "Environment {}"

hue_skip_list = ["inf"]
dash_skip_list = ["ugapec"]

df = pandas.DataFrame(columns=[x_name, y_name, dash_var_name, hue_var_name,graph_split_name])
df = df.astype({y_name:'float64',x_name:'float64',dash_var_name:'category',hue_var_name:'category'})
for experiment in experiment_iterator(experiment_name):
    d = dict()
    if "ugapec" in experiment["specification"]["rollout_allocation_method"] :
        continue
    if experiment['result'] != []:

        d[x_name] = experiment[x_var[0]][x_var[1]]
        d[y_name] = np.sum(experiment[y_var[0]][y_var[1]][-1])
        d[dash_var_name] = experiment[dash_var[0]][dash_var[1]]
        d[hue_var_name] = str(experiment[hue_var[0]][hue_var[1]])
        d[graph_split_name] = experiment[graph_split_var[0]][graph_split_var[1]]
        if d[x_name] == str(199) or d[x_name] == 199.0 and experiment["specification"]["seed"] == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            Xs = experiment["result"]["Xs"][-1]
            Ys = experiment["result"]["Ys"][-1]
            p = ax.scatter(Xs[:,0],Xs[:,1],Xs[:,2],c=Ys.ravel(),vmin=0,vmax=1)
            ax.set_title(f'{experiment["specification"]["rollout_allocation_method"]} , R: {np.sum(Ys)}')
            ax.set_xlim(0,30)
            ax.set_ylim(0,30)
            ax.set_zlim(0,30)
            plt.colorbar(p)

        df = df.append(d,True)
plt.show()
exit()
print(df)
for split_var in df[graph_split_name].unique():
    fig = plt.figure()
    df_cur = df.loc[df[graph_split_name] == split_var]
    sns.lineplot(x=x_name, y=y_name, data=df_cur, style=dash_var_name,hue=hue_var_name)
    plt.title(graph_split_name.format(split_var))
    file_name = f"{experiment_name}_{y_name}_{graph_split_name.format(split_var)}"
    
    # plt.savefig(f"{file_name}.png")
    plt.savefig(f"{file_name}.pdf", dpi=300)
    print(f"Saved figure {file_name}")
    plt.show()
