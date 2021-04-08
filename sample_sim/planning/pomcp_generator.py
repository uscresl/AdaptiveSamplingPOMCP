import numba
from copy import deepcopy

from sample_sim.data_model.data_model import DataModel
from sample_sim.data_model.gp_wrapper import TorchExactGp
from sample_sim.memory_mapper_utility import memory_mapped_dict
import numpy as np

from scipy.stats import norm

def expected_improvment(mu,sigma,ymax,xi=0.01):
    if sigma == 0.0:
        return 0.0
    else:
        improvment = mu - ymax - xi
        Z = improvment/sigma
        ei = improvment * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei


#@numba.jit(nopython=True)
def Generator(s, a, s_history, extra_data):
    # Extra_data is (mean,stdv,transition_matrix,c, data_model, filename, state_space_dimensionality,sample_observations, use_expected_improvement)
    sprime_idx = extra_data[2][s][a]
    if sprime_idx == -1:
        s_history.append((None,None))
        return s, s, 0

    # Observations are deterministic with state, we can work on later adding in more observations
    if extra_data[7] == True:
        ps = np.array([.3989,.2419,.2419,0.05,0.05])
        ps /= np.sum(ps)
        sigmas = [0,-1,1,-2,2]
        sample_idx = np.random.choice(sigmas,p=ps)
        o = sprime_idx * 5 + sigmas.index(sample_idx)
    else:
        sample_idx = 0
        o = sprime_idx


    # #assert isinstance(extra_data[4], DataModel)
    Sreal_ndarrays = memory_mapped_dict[str(extra_data[5]) + str(extra_data[6]) + "Sreal_ndarrays"]
    gp = extra_data[4].model
    if s_history != []:
        s_ndarray_history = np.array(list(map(lambda s: s[0], s_history)))
        s_ndarray_history_ys = np.array(list(map(lambda s: s[1], s_history)))
        X = np.vstack((extra_data[4].Xs,s_ndarray_history))
        Y = np.concatenate((extra_data[4].Ys, s_ndarray_history_ys))
        assert isinstance(gp,TorchExactGp)
        gp.update_prior(X,Y)
        #gp.model = gp.model.get_fantasy_model(s_ndarray_history[-1],s_ndarray_history_ys[-1])
    else:
        Y = extra_data[4].Ys
    sprime_loc = Sreal_ndarrays[sprime_idx]
    mean,stdv = gp.predict(np.array([sprime_loc]),return_std=True)
    if extra_data[8] == True:
        rw = expected_improvment(mean,stdv,np.max(Y))
    else:
        rw = ((mean + sample_idx * stdv) + extra_data[3] * stdv)[0]
    #gp.add_to_prior(sprime_loc.reshape((1,3)),mean)
    s_history.append((sprime_loc,mean[0]))
    #rw = extra_data[0][sprime_idx] + extra_data[3] * extra_data[1][sprime_idx]
    return sprime_idx, o, rw
    
