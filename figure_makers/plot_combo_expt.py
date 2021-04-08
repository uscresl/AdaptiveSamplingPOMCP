import os
import pandas
import sys
from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
from smallab.specification_generator import SpecificationGenerator
from sample_sim.data_model.loaders.ecomapper_loader import load_from_ecomapper_data
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
np.set_printoptions(suppress=True) #prevent numpy exponential notation on print
from mpl_toolkits.mplot3d import Axes3D 
if "figure_makers" in os.getcwd():
    os.chdir("..")
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    experiment_name = "camerareadytest"
print(f"Experiment: {experiment_name}")

#### DEFINE VARS
# X & Y (ensure correct datatype below)
y1_var = ("result","total_used_rollouts"); y1_name = "Used Rollouts"
# y2_var = ("result","MMSE"); y2_name = "RMSE"
y2_var = ("result","Ys"); y2_name = "Accumulated Reward"
y3_var = ("result","pomcp_traj_length"); y3_name = "PlannerTrajLength"
y4_var = ("result","tTest_pval"); y4_name = "TTestPVal"
x_var = ("result","used_budget"); x_name = "Environment steps"

dash_var = ("specification","plan_commitment_algorithm"); dash_var_name = "Plan Commitment Algorithm"
# dash_var = ("specification","rollout_number_goal"); dash_var_name = "Max Rollouts"
hue_var = ("specification", "plan_threshold"); hue_var_name = "p-value"
# hue_var = ("specification","rollout_allocation_method"); hue_var_name = "Rollout Allocation Method"
graph_split_var = ("specification","file"); graph_split_name= "{}"

### FILTER SPECS
num_specs = 0 # number of specs considered
file_to_write = 'specs_considered.txt'
validation_env = ["data/VEnv1.log", # validation environment 1
        "data/VEnv2.log"] # validation environment 2
plot_traj = False
save_figs = True
omit_ours = { # any experiment runs to omit from the proposed method
    # "file": [validation_env[0], validation_env[1]],
    # "rollout_allocation_method": ["beta"],
    "waste_unused_rollouts": [False], # filter one of the two ONLY
}
omit_baseline_specs = { # any experiment runs to omit from the baseline
    # "file": [validation_env[0], validation_env[1]],
}

def is_baseline(experiment):
    is_pc_baseline = (experiment["specification"]["plan_commitment_algorithm"] == "n_steps") and (experiment["specification"]["plan_threshold"] == 1)
    is_ra_baseline = (experiment["specification"]["rollout_allocation_method"] == "fixed")
    return (is_pc_baseline and is_ra_baseline)

# return True if any spec from omit_list matches
def omit_this_spec(experiment, omit_list):
    for spec_var in omit_list: # assume specs to omit is defined above
        for spec_val in omit_list[spec_var]:
            if experiment["specification"][spec_var] == spec_val:
                return True
    return False

#### PLOTTING
def env_name(full_name):
    if full_name == "fn:sbo":
        name = "Dynamic Function"
    elif full_name == validation_env[0]:
        name = "Validation Environment 1"
    elif full_name == validation_env[1]:
        name = "Validation Environment 2"
    else:
        name = full_name
    return name

def get_vmin_vmax(env_name):
    if env_name == "Dynamic Function":
        return 0,1
    elif "1" in env_name:
        return 6,23
    elif "2" in env_name:
        return 20,90
    else:
        raise NotImplementedError

open(file_to_write, 'w').close()
df = pandas.DataFrame(columns=[x_name, y1_name, y2_name, y3_name, y4_name, dash_var_name, hue_var_name, graph_split_name])
# DATATYPES: y1 float, y2 float, y3_name, y4name float, x float
df = df.astype({y1_name:'float64', y2_name:'float64', y3_name:'float64', y4_name:'float64', x_name:'float64',dash_var_name:'category',hue_var_name:'category'})
for experiment in experiment_iterator(experiment_name):
    d = dict()
    if experiment['result'] != []:

        d[x_name] = experiment[x_var[0]][x_var[1]]
        d[y1_name] = experiment[y1_var[0]][y1_var[1]]
        # d[y2_name] = experiment[y2_var[0]][y2_var[1]]
        d[y2_name] = np.sum(experiment[y2_var[0]][y2_var[1]][-1]) # applies for accumulated reward
        d[y3_name] = experiment[y3_var[0]][y3_var[1]] + 1
        try:
            d[y4_name] = experiment[y4_var[0]][y4_var[1]]
        except:
            d[y4_name] = -1 # default tTest pvalue to impossible value
        d[dash_var_name] = experiment[dash_var[0]][dash_var[1]]
        d[hue_var_name] = str(experiment[hue_var[0]][hue_var[1]])
        d[graph_split_name] = experiment[graph_split_var[0]][graph_split_var[1]]

        if is_baseline(experiment):
            d[hue_var_name] = "baseline"
            d[dash_var_name] = "baseline"
        else:
            d[hue_var_name] = "Proposed method"
            d[dash_var_name] = "Proposed method"

        # omit experiment if any spec from omit_list matches
        if omit_this_spec(experiment, omit_baseline_specs) and is_baseline(experiment):
            continue
        # omit experiment if any spec from omit_list matches
        if omit_this_spec(experiment, omit_ours) and not is_baseline(experiment):
            continue

        if plot_traj:
            Xs = experiment["result"]["Xs"][-1]
            budget = experiment["result"]["used_budget"]
            seed = experiment["specification"]["seed"]
            env = experiment["specification"]["file"]
            env = env.replace("data/", "").replace("_UTC_0_lawnmower_undulate_1dfs3hfb_IVER2-135.log", "")
            title_name = env_name(experiment["specification"]["file"])
            rollout_alloc = experiment["specification"]["rollout_allocation_method"]
            alpha = experiment["specification"]["alpha_param"]
            beta = experiment["specification"]["beta_param"]
            pcAlgo = experiment["specification"]["plan_commitment_algorithm"]
            pcThresh = experiment["specification"]["plan_threshold"]
            waste = str(experiment["specification"]["waste_unused_rollouts"])[0] # take first letter
            vmin, vmax = get_vmin_vmax(title_name)
            if budget == 199:
                file_name = f"plots/traj/{seed}/{d[dash_var_name]} {d[hue_var_name]}_{title_name}.pdf"
                # print(file_name)
                # print(Xs)
                rew = np.round(np.sum(experiment["result"]["Ys"][-1]), 2)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plt.plot(Xs[:,0],Xs[:,1],Xs[:,2])
                p = ax.scatter(Xs[:,0], Xs[:,1], Xs[:,2], c=experiment["result"]["Ys"][-1].ravel(), vmin=vmin, vmax=vmax)
                # plt.colorbar(p)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                ax.grid(False)
                plt.tight_layout()
                if env == "fn:curved" or env == "fn:sbo":
                    ax.set_xlim(0,30)
                    ax.set_ylim(0,30)
                    ax.set_zlim(0,30)
                else:
                    _, workspace = load_from_ecomapper_data(experiment["specification"]["file"], 15, 50)
                    ax.set_xlim(workspace.xmin, workspace.xmax)
                    ax.set_ylim(workspace.ymin, workspace.ymax)
                    ax.set_zlim(workspace.zmin, workspace.zmax)
                # if is_baseline(experiment):
                #     traj_title = f"{title_name} baseline, Reward: {rew}"
                # else:
                #     traj_title = f"{title_name} {rollout_alloc} & {pcAlgo}, Reward: {rew}"
                # plt.title(traj_title)
                # plt.title(f"{title_name} {rollout_alloc} {alpha},{beta} W{waste} {pcAlgo} {pcThresh}, Reward: {rew}")
                plt.savefig(file_name, dpi=300, bbox_inches = 'tight', pad_inches = 0)
                print(f"\tSaved trajectory to {file_name}")
                plt.close()
        # Print specs you are appending (only those that have completed)
        if experiment["result"]["used_budget"] == 199:
            with open(file_to_write, 'a') as f:
                f.write(f"Using spec: {experiment['specification']}\n")
            num_specs = num_specs + 1
            if num_specs%10 == 0:
                print(f"Added {num_specs} specs to {file_to_write}")
        if save_figs:
            df = df.append(d,True)
print(df)
print(f"{num_specs} specs considered. Written to {file_to_write}")
if save_figs:
    for split_var in df[graph_split_name].unique():
        env = env_name(graph_split_name.format(split_var))
        df_cur = df.loc[df[graph_split_name] == split_var].sort_values(by=[hue_var_name,dash_var_name])

        # y1 plot
        fig1 = plt.figure()
        sns.lineplot(x=x_name, y=y1_name, data=df_cur, hue=hue_var_name) #style=dash_var_name
        plt.title(f"{env}")
        plt.legend(loc="upper left")
        file_name = f"plots/{experiment_name}_{env}_{y1_name}_W{not omit_ours['waste_unused_rollouts'][0]}"
        plt.tight_layout()
        plt.savefig(f"{file_name}.pdf", dpi=300)
        print(f"Saved figure {file_name}")
        # plt.show()
        plt.close()

        # y2 plot
        fig2 = plt.figure()
        sns.lineplot(x=x_name, y=y2_name, data=df_cur, hue=hue_var_name)
        plt.title(f"{env}")
        plt.legend(loc="upper left")
        file_name = f"plots/{experiment_name}_{env}_{y2_name}_W{not omit_ours['waste_unused_rollouts'][0]}"
        plt.tight_layout()
        plt.savefig(f"{file_name}.pdf", dpi=300)
        print(f"Saved figure {file_name}")
        # plt.show()
        plt.close()

        # # y3 plot
        # fig3 = plt.figure()
        # sns.lineplot(x=x_name, y=y3_name, data=df_cur, hue=hue_var_name)
        # plt.title(f"{env}")
        # plt.legend(loc="upper left")
        # file_name = f"plots/{experiment_name}_{env}_{y3_name}_W{not omit_ours['waste_unused_rollouts'][0]}"
        # plt.tight_layout()
        # plt.savefig(f"{file_name}.pdf", dpi=300)
        # print(f"Saved figure {file_name}")
        # # plt.show()
        # plt.close()

        # # y4 plot
        # fig4 = plt.figure()
        # sns.lineplot(x=x_name, y=y4_name, data=df_cur, hue=hue_var_name)
        # plt.title(f"{env}")
        # plt.legend(loc="upper left")
        # file_name = f"plots/{experiment_name}_{env}_{y4_name}_W{not omit_ours['waste_unused_rollouts'][0]}"
        # plt.ylim(0,1)
        # plt.tight_layout()
        # plt.savefig(f"{file_name}.pdf", dpi=300)
        # print(f"Saved figure {file_name}")
        # # plt.show()
        # plt.close()
