import os

import pandas
import sys
from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from smallab.file_locations import get_experiment_save_directory

if "figure_makers" in os.getcwd():
    os.chdir("..")
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    experiment_name = "sample_obs"
print(f"Experiment: {experiment_name}")
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


x_var = ("result","used_budget"); x_name = "Time steps"
# x_var = ("result","objective_c_vis")
# x_var = ("result","used_budget"); x_name = "Time steps"

y_var = ("result","Ys"); y_name = "Acc Reward"
# y_var = ("result","total_used_rollouts"); y_name = "Used Rollouts"
# y_var = ("result","rollouts_used_this_iteration")

dash_var = ("specification","sample_observations"); dash_var_name = "Sample Observations"
hue_var = ("specification", "rollout_allocation_method"); hue_var_name = "Rollout Allocation Method"

graph_split_var = ("specification","file"); graph_split_name= "Environment {}"


def tqdm_experiment_iterator(name):
    for root, _, files in tqdm(list(os.walk(get_experiment_save_directory(name)))):
        for fname in files:
            if ".pkl" in fname:
                with open(os.path.join(root, fname), "rb") as f:
                    yield pickle.load(f)
            if ".json" in fname and fname != "specification.json":
                with open(os.path.join(root, fname), "r") as f:
                    yield json.load(f)


df = pandas.DataFrame(columns=[x_name, y_name, dash_var_name, hue_var_name,graph_split_name])
df = df.astype({y_name:'float64',x_name:'float64',dash_var_name:'category',hue_var_name:'category'})
for experiment in tqdm_experiment_iterator(experiment_name):
    d = dict()
    if experiment['result'] != []:
        d[x_name] = experiment[x_var[0]][x_var[1]]
        d[y_name] = np.sum(experiment[y_var[0]][y_var[1]][-1])
        d[dash_var_name] = experiment[dash_var[0]][dash_var[1]]
        d[hue_var_name] = rename_arm_selector(str(experiment[hue_var[0]][hue_var[1]]))
        d[graph_split_name] = experiment[graph_split_var[0]][graph_split_var[1]]
        df = df.append(d,True)
print(df)
for split_var in df[graph_split_name].unique():
    fig = plt.figure()
    df_cur = df.loc[df[graph_split_name] == split_var].sort_values(by=[hue_var_name,dash_var_name])
    sns.lineplot(x=x_name, y=y_name, data=df_cur, style=dash_var_name,hue=hue_var_name)
    plt.title(rename_title(graph_split_name.format(split_var)))
    file_name = f"{experiment_name}_{y_name}_{graph_split_name.format(split_var)}"
    
    # plt.savefig(f"{file_name}.png")
    #plt.savefig(f"{file_name}.pdf", dpi=300)
    print(f"Saved figure {file_name}")
plt.show()
