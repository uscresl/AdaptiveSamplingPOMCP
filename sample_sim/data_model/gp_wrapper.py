from itertools import chain


import abc
import logging
import math
import pickle
import subprocess

import gpytorch
import gpytorch.settings as settings
import numpy as np
import sklearn.metrics as metrics
import torch
from copy import deepcopy
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_absolute_error, mean_squared_error
from smallab.utilities.tqdm_to_logger import TqdmToLogger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

settings.fast_computations(True, True, True)


class GPWrapper():
    def __init__(self, logger, verbose=False):
        self.verbose = verbose
        self.logger = logger

    def update_prior(self, X, Y, input_uncertanties=None):
        raise NotImplementedError

    def fit(self, X, Y, input_uncertanties=None):
        raise NotImplementedError

    def predict(self, X_t, return_std=False, full_cov=False):
        raise NotImplementedError

    def compute_rmse(self, Y_t: np.ndarray, X_t: np.ndarray) -> float:
        return math.sqrt(metrics.mean_squared_error(Y_t, self.predict(X_t)))

    def save(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def deepcopy(self, model=None):
        return deepcopy(self)


class MLPMean(gpytorch.means.Mean):
    def __init__(self, dim):
        super(MLPMean, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        count = 0
        for n, p in self.mlp.named_parameters():
            self.register_parameter(name='mlp' + str(count), parameter=p)
            count += 1

    def forward(self, x):
        m = self.mlp(x)
        return m.squeeze()


class AffineMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size()):
        super().__init__()
        self.register_parameter(name='weight', parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size)))
        self.register_parameter(name='bias', parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))

    def forward(self, x):
        return torch.einsum('...j, ...ij->...i', self.weight, x) + self.bias


# https://gpytorch.readthedocs.io/en/latest/examples/04_Variational_and_Approximate_GPs/GP_Regression_with_Uncertain_Inputs.html
class ApproximateUncertainGPModel(ApproximateGP):
    def __init__(self, inducing_points, use_fast_strategy):
        inducing_points = torch.from_numpy(inducing_points).float()
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        dims = inducing_points.shape[1]
        # self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=dims)) + ScaleKernel(LinearKernel(ard_num_dims=dims))
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, use_mlp_mean=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        dims = train_x.shape[1]

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class TorchGP(GPWrapper):
#     @abc.abstractmethod
#     def get_samples(self) -> (np.ndarry,np.ndarry):
#         pass
#     #def fit(self,X,Y,input_uncertanties):
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    result = result.decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def sort_cuda_devices():
    memory_map = get_gpu_memory_map()
    return {'cuda:' + str(k): v for k, v in sorted(memory_map.items(), key=lambda item: item[1])}


class TorchExactGp(GPWrapper):
    def __init__(self, X, Y, logger, verbose=True, use_mlp_mean=False,force_cpu=True,gpu_num=None):
        super().__init__(verbose)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(X, Y, self.likelihood, use_mlp_mean=use_mlp_mean)
        self.using_cuda = False
        if torch.cuda.is_available() and not force_cpu:
            if gpu_num is not None:
                device = f"cuda:{gpu_num}"
                self.dev = device
                self.likelihood = self.likelihood.cuda(device)
                self.model = self.model.cuda(device)
                self.using_cuda = True
            else:
                found_device = False
                for device,_ in sort_cuda_devices().items():
                    try:
                        self.dev = device
                        self.likelihood = self.likelihood.cuda(device)
                        self.model = self.model.cuda(device)
                        self.using_cuda = True
                        found_device = True
                    except RuntimeError as e:
                        logging.getLogger(self.logger).warning(f"Couldn't use best device {device} due to {e}")
                if not found_device:
                    raise Exception("Couldn't fit into any cuda memory")

        self.logger = logger

    def update_prior(self, X, Y, input_uncertanties=None):
        #logging.getLogger(self.logger).info(f"GP Data Size: X - {X.shape} Y - {Y.shape} Noise - {input_uncertanties.shape}, ymean: {np.mean(Y)} ystd: {np.std(Y)}")
        # This is a weird name for this
        # X, Y = remove_close_entries(X, Y)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
   
        self.model.set_train_data(X, Y, strict=False)
        if self.using_cuda:
            self.model = self.model.cuda(self.dev)
            self.likelihood = self.likelihood.cuda(self.dev)

        # X,Y = remove_close_entries(X,Y)
        # self.fit(X,Y,input_uncertanties,optimization_steps=50)
    def add_to_prior(self,X,Y):
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        self.model = self.model.get_fantasy_model(X,Y)
    def copy_hyperparameters_from(self,other):
       logging.getLogger(self.logger).info(other.state_dict())
       self.model.load_state_dict(other.state_dict())



    def fit(self, X, Y, input_uncertanties=None, early_stopping_delta=0.001,
            early_stopping_iterations=3, optimization_steps=500):
        # X, Y = remove_close_entries(X, Y)
        train_x_mean = torch.from_numpy(X).float()
        #assert input_uncertanties is None or np.sum(input_uncertanties) == 0
        train_y = torch.from_numpy(Y).float()
        #self.model = ExactGPModel(train_x_mean, train_y, self.likelihood)
        if self.using_cuda:
            self.model = self.model.cuda(self.dev)
            self.likelihood = self.likelihood.cuda(self.dev)
            train_x_mean = train_x_mean.cuda(self.dev)
            train_y = train_y.cuda(self.dev)

        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        train_x_sample = train_x_mean
        iterator = tqdm(range(optimization_steps), file=TqdmToLogger(logging.getLogger(self.logger), logging.INFO),
                        mininterval=early_stopping_iterations, desc="Training Exact GP")
        scheduler = ReduceLROnPlateau(optimizer)
        best_loss = float("inf")
        best_params = None

        for i in iterator:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x_sample)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=loss.item())
            # if i % early_stopping_iterations == 0:
            #     mae = mean_absolute_error(Y, self.predict(X))
            #     rmse = np.sqrt(mean_squared_error(Y, self.predict(X)))
            scheduler.step(loss)
            if loss < best_loss:
                best_params = deepcopy(self.model.state_dict())
                best_loss = loss
                logging.getLogger(self.logger).debug(f"New Best found Loss: {best_loss}")
        self.model.load_state_dict(best_params)
    def eval_model(self):
        self.model.float().eval()
        self.likelihood.float().eval()

    def predict(self, X_t, return_std=False, full_cov=False):
        with torch.no_grad(),gpytorch.settings.fast_computations(), gpytorch.settings.fast_pred_var(),gpytorch.settings.max_root_decomposition_size(10):
            pytorch_input = torch.from_numpy(X_t).float()
            if self.using_cuda:
                pytorch_input = pytorch_input.cuda(self.dev)
            pred = self.likelihood(self.model(pytorch_input))
            #pred = self.model(pytorch_input)
            mu, var = pred.mean.cpu().detach().numpy(), pred.variance.cpu().detach().numpy()
            if return_std:
                return mu, var
            elif not return_std and not full_cov:
                return mu
            else:
                raise Exception()

    def save(self, fname):
        torch.save(self.model.state_dict(), fname)

    def load(self,fname):
        state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def deepcopy(self, model=None):
        raise Exception()


class TorchSparseUncertainGPModel(GPWrapper):
    def __init__(self, logger, inducing_inputs, use_fast_strategy=False):
        super().__init__(logger)
        self.use_fast_strategy = use_fast_strategy
        self.inducing_inputs = inducing_inputs
        self.model = ApproximateUncertainGPModel(inducing_inputs, use_fast_strategy)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.filter = False
        self.using_cuda = False

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.using_cuda = True
        else:
            raise Exception("No Cuda")
  
    def copy_hyperparameters_from(self, m):
        '''
        Copys the hyper parameters but doesn't affect the variational paramters
        :param gp_model:
        :return:
        '''
        self.model.load_state_dict(self.keep_only_nonvariational_params(m.state_dict()),False)
        #self.model.covar_module.length_scale = 2.0

    def keep_only_nonvariational_params(self,state_dict):
        return deepcopy({k: v for k, v in state_dict.items() if "variational" not in k})

    def update_prior(self, X, Y, input_uncertanties=None):
        current_parms = self.keep_only_nonvariational_params(self.model.state_dict())
        self.model_old = self.model
        self.model = ApproximateUncertainGPModel(X, self.use_fast_strategy)
        self.copy_hyperparameters_from(self.model_old)
        self.model_old = None

        # This is done so that the variational parameters are refit but not the underlying kernel params
        #for param in chain(self.model.covar_module.parameters(), self.model.mean_module.parameters()):
        #     param.requires_grad = False
        #self.fit(X,Y,input_uncertanties,optimization_steps=400)
        if self.using_cuda:
            self.model = self.model.cuda()
        #new_params = self.keep_only_nonvariational_params(self.model

    def fit(self, X, Y, input_uncertanties=None, optimization_steps=200):
        # if self.filter:
        #     X, Y, input_uncertanties = remove_close_entries(X, Y, input_uncertanties)

        self.model.float().train()
        self.likelihood.float().train()
        if self.using_cuda:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        # We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=1)
        scheduler = ReduceLROnPlateau(optimizer)
        train_x_mean = torch.from_numpy(X).float()
        train_x_stdv = torch.from_numpy(input_uncertanties).float()
        train_y = torch.from_numpy(Y).float()
        if self.using_cuda:
            train_x_mean = train_x_mean.cuda()
            train_x_stdv = train_x_stdv.cuda()
            train_y = train_y.cuda()


        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=train_y.size(0))
        training_iter = optimization_steps

        iterator = tqdm(range(training_iter), file=TqdmToLogger(logging.getLogger(self.logger)),
                        desc="Training GP")
        l_prev = float('inf')
        stopping_iterations = 0
        loader = DataLoader(list(zip(train_x_mean, train_x_stdv, train_y)), batch_size=200, shuffle=True)
        best_params = None
        best_loss = float("inf")
        for i in iterator:
            train_x_sample = torch.distributions.Normal(train_x_mean, train_x_stdv).rsample()
            train_y_sample = train_y

            # Now do the rest of the training loop
            optimizer.zero_grad()
            output = self.model(train_x_sample)
            # logging.getLogger(self.logger).info(f"Y Min: {torch.min(train_y)}, Max: {torch.max(train_y)}, AVG: {torch.mean(train_y)}, STD: {torch.std(train_y)}")
            # logging.getLogger(self.logger).info(f"Y_Hat Min: {torch.min(output.mean)}, Max: {torch.max(output.mean)}, AVG: {torch.mean(output.mean)}, STD: {torch.std(output.mean)}")
            loss = -mll(output, train_y_sample)
            epoch_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch_loss)

            l_prev = epoch_loss

            iterator.set_postfix(loss=l_prev)
            if epoch_loss < best_loss:
                logging.getLogger(self.logger).debug(f"New Best found Loss: {best_loss}")
                best_params = deepcopy(self.model.state_dict())
                best_loss = epoch_loss
        self.model.load_state_dict(best_params)
        mae = mean_absolute_error(Y, self.predict(X))
        rmse = np.sqrt(mean_squared_error(Y, self.predict(X)))
        logging.getLogger(self.logger).debug(f"RMSE: {rmse}, MAE: {mae}")

    def predict(self, X_t, return_std=False, full_cov=False):
        # X_t = remove_close_entries(X_t)
        self.model.float().eval()
        self.likelihood.float().eval()

        with torch.no_grad():
            pytorch_input = torch.from_numpy(X_t).float()
            if self.using_cuda:
                pytorch_input = pytorch_input.cuda()
            pred = self.likelihood(self.model(pytorch_input))
            if self.using_cuda:
                mu, var = pred.mean.cpu().detach().numpy(), pred.variance.cpu().detach().numpy()
            else:
                mu, var = pred.mean.detach().numpy(), pred.variance.detach().numpy()

            if return_std:
                return mu, var
            elif not return_std and not full_cov:
                return mu
            else:
                raise Exception()

    def save(self, fname):
        np.save(fname + "inducing", self.inducing_inputs)
        torch.save(self.model.state_dict(), fname)

    @staticmethod
    def load(fname):
        inducing_inputs = np.load(fname + "inducing.npy")
        model = TorchSparseUncertainGPModel(inducing_inputs)
        state_dict = torch.load(fname)
        model.model.load_state_dict(state_dict)
        return model


# def remove_close_entries(X_t, y=None, input_uncertanties=None, tol=10 ** -2):
#     pairwise_distances = squareform(pdist(X_t, "euclidean"))
#     indexes_above_tolerance = np.where(np.any(pairwise_distances > tol, axis=1))[0]
#     X_t_out = X_t[indexes_above_tolerance, :]
#     assert X_t_out.shape[0] <= X_t.shape[0]
#     if y is not None and input_uncertanties is not None:
#         return X_t_out, y[indexes_above_tolerance], input_uncertanties[indexes_above_tolerance, :]
#     elif y is not None:
#         return X_t_out, y[indexes_above_tolerance]
#     else:
#         return X_t_out
