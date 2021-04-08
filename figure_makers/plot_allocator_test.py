import os

import pandas
import sys
from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import itertools
if "figure_makers" in os.getcwd():
    os.chdir("..")
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    experiment_name1 = "allocator_test"

print(f"Experiment: {experiment_name1}")

x_var = ("result","used_budget"); x_name = "Time steps" 
y_var = ("result","MMSE"); y_name = "Accumulated Reward"
# y_var = ("result","total_used_rollouts"); y_name = "Used Rollouts"
# y_var = ("result","rollouts_used_this_iteration")

hue_var = ("specification","rollout_allocation_method"); 

graph_split_var = ("specification","file"); graph_split_name= "Environment {}"
#graph_split_var = ("specification","control_point"); graph_split_name= "Control Point {}"
#graph_split_var = ("specification","file"); graph_split_name= "Environment {}"
dash_var_name = "Allocator Curve"
hue_var_name = "Arm Selection"

def rename_arm_selector(s):
    if s=="beta-sr":
        return "Successive Rejects, Beta"
    elif s == "sr":
        return "Successive Rejects, Fixed"
    elif "beta-ugapeb" == s:
        return "UGapEb, Beta"
    elif "ugapeb" == s:
        return "UGapEb, Fixed"
    elif "beta" == s:
        return "UCT, Beta"
    elif "fixed" == s:
        return "UCT, Fixed"
    else:
        print(s)
        raise Exception()

def rename_title(s):
    if "fn" in s:
        return "Dynamic Function"
    elif "163636" in s:
        return "Validation Environment 1"
    else:
        return "Validation Environment 2"

df = pandas.DataFrame(columns=[x_name, y_name, hue_var_name,graph_split_name])
df = df.astype({y_name:'float64',x_name:'float64',hue_var_name:'str'})
for experiment in experiment_iterator(experiment_name1):
    d = dict()
    if experiment["specification"]["rollout_allocation_method"] in ["sr","ugapeb"]:
        continue
    if experiment['result'] != []:
        d[x_name] = experiment[x_var[0]][x_var[1]]
        d[y_name] = np.sum(experiment["result"]["Ys"][-1]) 
        d[hue_var_name] = rename_arm_selector(str(experiment[hue_var[0]][hue_var[1]]))
        d[graph_split_name] = experiment[graph_split_var[0]][graph_split_var[1]]
        df = df.append(d,True)

for split_var in df[graph_split_name].unique():
    fig = plt.figure()
    df_cur = df.loc[df[graph_split_name] == split_var].sort_values(by=[hue_var_name])
    #sns.lineplot(x=x_name, y=y_name, data=df_cur,hue=hue_var_name,style=dash_var_name)
    ax = sns.lineplot(x=x_name, y=y_name, data=df_cur,hue=hue_var_name)
    plt.xlabel("Time Steps")
    plt.ylabel("Accumlated Reward")
    plt.title(rename_title(graph_split_name.format(split_var)))
    file_name = f"allocator_test_{y_name}_{graph_split_name.format(split_var)}"
    #ax.add_legend(label_order = sorted(ax._legend_data.keys(), key = int))

    # plt.savefig(f"{file_name}.png")
    plt.savefig(f"allocator_test_{graph_split_name.format(split_var).split('/')[-1]}.pdf", dpi=300)
    print(f"Saved to allocator_test_{graph_split_name.format(split_var).split('/')[-1]}.pdf")
plt.show()
