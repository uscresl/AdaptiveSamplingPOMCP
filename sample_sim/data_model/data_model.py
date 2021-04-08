import logging
import pickle
import math

import abc
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sample_sim.data_model.gp_wrapper import TorchSparseUncertainGPModel, TorchExactGp, GPWrapper
from sample_sim.data_model.workspace import Workspace
from sample_sim.general_utils import is_good_matrix, wmae, rwmse
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from tqdm import tqdm
import random

from scipy.stats import norm,multivariate_normal


class DataModel(abc.ABC):
    def __init__(self, logger, use_uncertainty=False, verbose=False, cov_fn=None):
        self.Xs = []
        self.Ys = []
        self.input_uncertanties = []
        self.use_uncertainty = use_uncertainty
        self.verbose = verbose
        self.logger = logger
        self.cov_fn = cov_fn

    def error_against_ground_truth(self, other_datamodel,use_mc):
        assert isinstance(other_datamodel, DataModel)
        self._flatten_data()
        if use_mc:
            predicted_ys_other,_ = other_datamodel.monte_carlo_query(self.Xs)
        else:
            predicted_ys = other_datamodel.query_many(self.Xs, return_std=False)
        return mean_absolute_error(self.Ys, predicted_ys), np.sqrt(mean_squared_error(self.Ys, predicted_ys)),  wmae(self.Ys,predicted_ys),rwmse(self.Ys,predicted_ys),

    def error_against_model(self, other_datamodel, points,use_mc):
        self._flatten_data()
        if use_mc:
            predicted_ys_other,_ = other_datamodel.mcmc_query2(points,weight=True)
        else:
            predicted_ys_other = other_datamodel.query_many(points, return_std=False)
        predicted_ys_self = self.query_many(points, return_std=False)
        return mean_absolute_error(predicted_ys_self, predicted_ys_other), np.sqrt(
            mean_squared_error(predicted_ys_self, predicted_ys_other)), wmae(predicted_ys_self,predicted_ys_other), rwmse(predicted_ys_self,predicted_ys_other)

    def error_against_track(self, points, ys, use_mc,mc_iterations=1000,mc_keep=100,weight=False):
        self._flatten_data()
        if use_mc:
            if mc_keep is None:
                mc_keep = mc_iterations
            predicted_ys,_ = self.mcmc_query2(points,iterations=mc_iterations,keep=mc_keep,weight=weight)
        else:
            predicted_ys = self.query_many(points, return_std=False)

        return mean_absolute_error(predicted_ys, ys), np.sqrt(
            mean_squared_error(predicted_ys, ys)), wmae(predicted_ys,ys), rwmse(predicted_ys,ys)

        



    def log_error_gt(self,use_mc):
        mae, mse,wmae,rwmse = self.error_against_ground_truth(self,use_mc)
        logging.getLogger(self.logger).info(f"Error Against GT = MAE: {mae}, RMSE: {mse}, WMAE: {wmae}, RWMSE: {rwmse}")
        

    def mcmc_query2(self,points,point_noises=None,iterations=500,keep=None,weight=True):
        if keep is not None:
            means = np.zeros((keep, points.shape[0]))
            stds = np.zeros((keep, points.shape[0]))
            weights = np.zeros(keep)
        else:
            means = np.zeros((iterations, points.shape[0]))
            stds = np.zeros((iterations, points.shape[0]))
            weights = np.zeros(iterations)

        calculate_probability = weight or (keep is not None)
        if not calculate_probability:
            logging.getLogger(self.logger).debug("Using fast MC")
        else:
            logging.getLogger(self.logger).debug("Using slow MC")


        i = 0
        rollouts = []
        with tqdm(total=iterations, desc="Fully MCMC sample", file=TqdmToLogger(logging.getLogger(self.logger), logging.INFO)) as pbar:
            while i < iterations:
                log_likliehoods = []
                prior_logs = []

                current_inputs = []
                current_targets = []
                for x, noise,y in zip(self.Xs, self.input_uncertanties,self.Ys):
                    prob_fn = multivariate_normal(mean=x,cov=np.diag(noise))
                    new_input = prob_fn.rvs()
                    if calculate_probability:
                        prior_logs.append(prob_fn.logpdf(new_input))
                    current_inputs.append(new_input)
                    current_targets.append(y)
                self.model.update_prior(np.array(current_inputs), np.array(current_targets))
                if calculate_probability:
                    cur_means,cur_vars = self.query_many(np.array(current_inputs))
                    for mean, var in zip(cur_means,cur_vars):
                        log_likliehoods.append(norm.logpdf(y,loc=mean,scale=var))

                    logging.getLogger(self.logger).info(f"Prior : {min(prior_logs)} - {max(prior_logs)})")
                    logging.getLogger(self.logger).info(f"GP: {min(log_likliehoods)} - {max(log_likliehoods)})")

                    log_likelihood = sum(log_likliehoods) + sum(prior_logs)
                #logging.getLogger(self.logger).info(f"Log Likelihood {log_likelihood}")
                    pbar.set_postfix(likeliehood=log_likelihood )
                else:
                    log_likelihood = 0 
                if point_noises is None:
                    mean, std = self.query_many(points)
                else:
                    inputs = []
                    for x, noise in zip(points,point_noises):
                        prob_fn = multivariate_normal(mean=x,cov=np.diag(noise))
                        inputs.append(prob_fn.rvs())
                    mean, std = self.query_many(np.array(inputs))
                rollouts.append((log_likelihood,(mean,std)))
                pbar.update(1)
                i += 1
        i = 0
        if keep is None:
            iterator = rollouts
        else:
            iterator = sorted(rollouts,reverse=True)[:keep]
        for log_likliehood, output in iterator:
            logging.getLogger(self.logger).info(f"LL:{ log_likliehood}")
            mean,std = output
            if keep == 1:
                return mean,std
            means[i] = mean
            stds[i] = std
            weights[i] = log_likliehood
            i += 1
        if weight:
            weights = weights/sum(weights)
            return np.average(means, axis=0,weights=weights), np.var(means, axis=0) + np.mean(stds, axis=0)

        else:
            return np.mean(means, axis=0), np.var(means, axis=0) + np.mean(stds, axis=0)



    def update(self, X, Y, input_uncertainties=None):
        #assert input_uncertainties is not None
        if self.Xs == []:
            self.Xs = X
        else:
            self.Xs = np.vstack((self.Xs, X))
        if self.Ys == []:
            self.Ys = Y
        else:
            self.Ys = np.append(self.Ys,Y)
        if self.input_uncertanties == []:
            self.input_uncertanties = input_uncertainties
        else:
            self.input_uncertanties = np.vstack((self.input_uncertanties, input_uncertainties))

    def query(self, p, return_std=True):
        return self.query_many(np.array([p]), return_std=return_std)

    def query_many(self, Xs, return_std=True):
        if return_std:
            mean, std = self.__query_many_implementation__(Xs, return_std)
        else:
            std = None
            mean = self.__query_many_implementation__(Xs, return_std)
        if return_std:
            return mean, std
        else:
            return mean

    @abc.abstractmethod
    def __query_many_implementation__(self, Xs, return_std=True):
        pass

    def _flatten_data(self):
        if isinstance(self.Xs, list):
            self.Xs = np.vstack(self.Xs)
            #assert self.Xs.shape[1] == 3
            if self.input_uncertanties is not None:
                self.input_uncertanties = np.vstack(self.input_uncertanties)
            self.Ys = np.concatenate(self.Ys)
            assert is_good_matrix(self.Xs)
            assert is_good_matrix(self.input_uncertanties)
            assert is_good_matrix(self.Ys)

class TorchDataModel(DataModel):
    def __init__(self, logger, model: GPWrapper,use_uncertainty,workspace:Workspace,cov_fn=None):
        super().__init__(logger,use_uncertainty,verbose=True)
        self.logger = logger
        self.model = model
        self.workspace = workspace
        self.cov_fn = cov_fn

    def update(self, X, Y, input_uncertainties=None):
        super().update(X, Y, input_uncertainties)
        self.model.update_prior(self.Xs,self.Ys,self.input_uncertanties)

    def fit(self,steps=200):
        self.model.fit(self.Xs,self.Ys,self.input_uncertanties,optimization_steps=steps)

    def __query_many_implementation__(self, Xs, return_std=True):
        return self.model.predict(Xs, return_std)


    def save(self, fname):
        with open(fname + "dm.pkl", "wb") as f:
            pickle.dump(self.Xs, f)
            pickle.dump(self.Ys, f)
            pickle.dump(self.input_uncertanties, f)
            pickle.dump(self.use_uncertainty, f)
        self.model.save(fname)

    def load(self, fname):
        with open(fname + "dm.pkl", "rb") as f:
            self.Xs = pickle.load(f)
            self.Ys = pickle.load(f)
            self.input_uncertanties = pickle.load(f)
            self.use_uncertainty = pickle.load(f)
        self.model.load(fname)

class TorchApproximateGPBackedDataModel(TorchDataModel):
    def __init__(self, logger, workspace:Workspace,inducing_points=None, verbose=False, use_x_as_inducing=True,cov_fn=None):

        self.refit = True
        self.gp = TorchSparseUncertainGPModel(logger, inducing_points, use_fast_strategy=False)
        self.gp.verbose = verbose
        self.use_x_as_inducing = use_x_as_inducing

        super().__init__(logger, model=self.gp,use_uncertainty=True,workspace=workspace,cov_fn=cov_fn)


class TorchExactGPBackedDataModel(TorchDataModel):
    def __init__(self, X, Y, logger, workspace:Workspace,use_better_mean=False,force_cpu=False,cov_fn=None,device=None):
        self.gp = TorchExactGp(X, Y, logger=logger, use_mlp_mean=use_better_mean,force_cpu=True,gpu_num=device)
        super().__init__(logger,model=self.gp, use_uncertainty=False,workspace=workspace,cov_fn=cov_fn)

