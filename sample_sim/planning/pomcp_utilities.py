import json
import logging
import time
from collections import defaultdict
from json import JSONDecodeError

import enum
import numpy as np

from sample_sim.timeit import timeit

class ActionXYT(enum.Enum):
    STAY_STILL = 0
    POSITIVE_X = 1
    NEGATIVE_X = 2
    POSITIVE_Y = 3
    NEGATIVE_Y = 4


class ActionXYZ(enum.Enum):
    STAY_STILL = 0
    POSITIVE_X = 1
    NEGATIVE_X = 2
    POSITIVE_Y = 3
    NEGATIVE_Y = 4
    POSITIVE_Z = 5
    NEGATIVE_Z = 6


class ActionModel(enum.Enum):
    XYT = 0
    XYZ = 1


def action_enum(model: ActionModel):
    if model == ActionModel.XYT:
        return ActionXYT
    elif model == ActionModel.XYZ:
        return ActionXYZ
    else:
        raise Exception(f"action model not understood {model}")


def apply_action_to_state(state, action, step_sizes):
    if isinstance(action, ActionXYT):
        state = np.array(state)
        if action == ActionXYT.POSITIVE_X:
            sprime = state + np.array([step_sizes[0], 0, step_sizes[2]])
        elif action == ActionXYT.NEGATIVE_X:
            sprime = state + np.array([-1 * step_sizes[0], 0, step_sizes[2]])
        elif action == ActionXYT.POSITIVE_Y:
            sprime = state + np.array([0, step_sizes[1], step_sizes[2]])
        elif action == ActionXYT.NEGATIVE_Y:
            sprime = state + np.array([0, -1 * step_sizes[1], step_sizes[2]])
        elif action == ActionXYT.STAY_STILL:
            sprime = state + np.array([0, 0, step_sizes[2]])
        else:
            raise Exception(f"Action not understood: {action}")

    elif isinstance(action, ActionXYZ):
        state = np.array(state)
        if action == ActionXYZ.POSITIVE_X:
            sprime = state + np.array([step_sizes[0], 0, 0])
        elif action == ActionXYZ.NEGATIVE_X:
            sprime = state + np.array([-1 * step_sizes[0], 0, 0])
        elif action == ActionXYZ.POSITIVE_Y:
            sprime = state + np.array([0, step_sizes[1], 0])
        elif action == ActionXYZ.NEGATIVE_Y:
            sprime = state + np.array([0, -1 * step_sizes[1], 0])
        elif action == ActionXYZ.POSITIVE_Z:
            sprime = state + np.array([0, 0, step_sizes[2]])
        elif action == ActionXYZ.NEGATIVE_Z:
            sprime = state + np.array([0, 0, -1 * step_sizes[2]])
        elif action == ActionXYZ.STAY_STILL:
            sprime = state
        else:
            raise Exception(f"Action not understood: {action}")

    else:
        raise Exception(f"action_model not understood: {action_model}")

    return sprime



@timeit
def create_adjacency_dict(start_state, workspace, step_sizes, action_model: ActionModel):
    queue = [start_state]
    visited = set()
    adjacency_dict = defaultdict(dict)
    while queue != []:
        cur_state = queue.pop()
        for action in action_enum(action_model):
            next_state = tuple(apply_action_to_state(cur_state, action, step_sizes))
            if workspace.is_inside(next_state):
                adjacency_dict[tuple(cur_state)][action] = next_state
                if next_state not in visited:
                    queue.append(next_state)
                    visited.add(next_state)
    return adjacency_dict


def adjacency_dict_to_numpy(adjacency_dict, S_real_pts_to_idxs, step_sizes, action_model: ActionModel):
    action_enums = action_enum(action_model)
    transition_matrix = np.zeros((len(adjacency_dict.keys()), len(action_enums)), dtype="int32")
    for i, state in enumerate(adjacency_dict.keys()):
        assert i == S_real_pts_to_idxs[state]
        try:
            transition_matrix[S_real_pts_to_idxs[state], :] = S_real_pts_to_idxs[
                tuple(apply_action_to_state(state, action_enums.STAY_STILL, step_sizes))]
        except KeyError:
            # This is when you get to the end, just go to the same state?
            transition_matrix[S_real_pts_to_idxs[state], :] = -1
        for action in action_enums:
            if action not in adjacency_dict[state]:
                # 4 is the "do nothing"
                action = action_enums.STAY_STILL
            if adjacency_dict[state][action] in S_real_pts_to_idxs:
                transition_matrix[S_real_pts_to_idxs[state], action.value] = S_real_pts_to_idxs[
                    adjacency_dict[state][action]]

    return transition_matrix
def get_default_low_param():
    return 100
def get_default_hi_param():
    return 500

def get_uct_c(objective_c, filename, logger_name):
    retries = 10
    while retries >= 0:
        try:
            with open(f"params/{objective_c}-{filename.replace('/', '')}.json", "r") as f:
                file = json.load(f)
                return file["low"], file["high"]
        except FileNotFoundError:
            assert False
            logging.getLogger(logger_name).warning("Params File not found, Returning default")
            return get_default_low_param(), get_default_hi_param()  #extreme high and low numbers seen during running fn:curved env
        except JSONDecodeError:
            logging.getLogger(logger_name).warning("Params File not corrupt, retrying")
            time.sleep(0.1)
        retries -= 1
    assert False
    logging.getLogger(logger_name).warning("Params file not recoverable, returning defaults")
    return get_default_low_param(), get_default_hi_param()  # extreme high and low numbers seen during running fn:curved env


