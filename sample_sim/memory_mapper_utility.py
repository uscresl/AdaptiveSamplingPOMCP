import pickle

import numpy as np
import os

from sample_sim.data_model.loaders.analytical_function_loader import create_data
from sample_sim.data_model.loaders.ecomapper_loader import load_from_ecomapper_data
from sample_sim.data_model.loaders.uav_picture_loader import load_from_egret_data
from sample_sim.planning.pomcp_utilities import adjacency_dict_to_numpy, create_adjacency_dict, ActionModel
import itertools

memory_mapped_dict = dict()


def map_memory(files, state_space_dimensionality):
    if memory_mapped_dict == dict():
        pickle_file_name = os.path.join(".cache",
                                        str(list(map(lambda x: x.replace("/",""), files))) + str(state_space_dimensionality) + "mmdict.pkl")
        try:
            with open(pickle_file_name, 'rb') as f:
                for key,value in pickle.load(f).items():
                    memory_mapped_dict[key] = value
        except Exception as e:
            print(f"Setting up memory mapped due to {e}")
            for file, cur_state_space_dimensionality in itertools.product(files,
                                                            state_space_dimensionality):
                print(f"Setting up memory mapped stuff")
                if "fn" in file:
                    oracle_model, workspace = create_data(file)
                    action_model = ActionModel.XYT
                elif "drone:" in file:
                    oracle_model, workspace = load_from_egret_data(file.split(":")[1])
                    action_model = ActionModel.XYT
                else:
                    oracle_model, workspace = load_from_ecomapper_data(file, 15, 50)
                    action_model = ActionModel.XYZ
                x_step = round((workspace.xmax - workspace.xmin) / cur_state_space_dimensionality[0], 2)
                y_step = round((workspace.ymax - workspace.ymin) / cur_state_space_dimensionality[1], 2)
                z_step = round((workspace.zmax - workspace.zmin) / cur_state_space_dimensionality[2], 2)
                step_sizes = (x_step, y_step, z_step)
                adjacency_dict = create_adjacency_dict(
                    np.array([int((workspace.xmin + workspace.xmax) / 2), int((workspace.ymin + workspace.ymax) / 2), workspace.zmin]),
                    workspace, step_sizes, action_model=action_model)

                # These need to get mapped into memory
                Sreal = np.array(list(adjacency_dict.keys()))
                Sreal_ndarrays = np.array(list(map(lambda x: np.array(x), Sreal)))
                S = np.array(list(range(len(Sreal))))
                S_real_pts_to_idxs = {tuple(Sreal[v]): v for v in S}
                transition_matrix = adjacency_dict_to_numpy(adjacency_dict, S_real_pts_to_idxs, step_sizes, action_model=action_model)

                memory_mapped_dict[str(file) + str(cur_state_space_dimensionality) + "Sreal"] = Sreal
                memory_mapped_dict[str(file) + str(cur_state_space_dimensionality) + "Sreal_ndarrays"] = Sreal_ndarrays
                memory_mapped_dict[str(file) + str(cur_state_space_dimensionality) + "S"] = S
                memory_mapped_dict[
                    str(file) + str(cur_state_space_dimensionality) + "S_real_pts_to_idxs"] = S_real_pts_to_idxs
                memory_mapped_dict[
                    str(file) + str(cur_state_space_dimensionality) + "transition_matrix"] = transition_matrix
                print(f"Done memory mapping {file} {cur_state_space_dimensionality}")
            with open(pickle_file_name, "wb") as f:
                pickle.dump(memory_mapped_dict, f)
