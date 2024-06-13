import models
import sys
import numpy as np
import models
import torch
import math
import math
import torch

from botorch.test_functions import Branin
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import matplotlib.pyplot as plt
from pyro.infer.mcmc import NUTS, MCMC, HMC
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
import pyro
import pickle
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior, MultivariateNormalPrior
import pyro.distributions as dist
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
import time
import scipy
from scipy.optimize import minimize
import botorch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy


class experiment_BOtorch_MF(object):
    def __init__(self, target):
        self.target = target
        self.X = None
        self.index_X = None
        self.Y = None
        self.Y_true = None
        self.fidelity = target.fidelity
        self.bounds = target.bounds
        self.fix_fidelity = None
        self.candidate_set = torch.cat([torch.cat([torch.arange(1000)/500 - 1, torch.arange(1000)/500 - 1]).unsqueeze(1), torch.cat([torch.zeros(1000), torch.ones(1000)]).unsqueeze(1)], dim=1)

    def initialize(self, X, index_X, Y):
        train_X = torch.cat([X, index_X.unsqueeze(1)], dim=1)
        self.model = SingleTaskGP(train_X, Y.unsqueeze(1), covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(self.mll)
        self.model.covar_module.base_kernel.lengthscale = torch.tensor([.3])
        self.model.covar_module.outputscale = 1 * torch.ones_like(self.model.covar_module.outputscale)
        self.model.mean_module.constant.data = 0 * torch.ones_like(self.model.mean_module.constant.data)
        # fit_gpytorch_model(self.mll)
        self.qMES = qMaxValueEntropy(self.model, self.candidate_set)

    def update(self, new_x, new_index_x, last=False):
        # print(indices, index_x)
        if self.X is None:
            self.X = new_x
            self.index_X = new_index_x
            self.Y = self.target.query(new_x, new_index_x)
            self.Y_true = self.target.query_ground_truth(new_x, new_index_x)
        else:
            self.X = torch.cat([self.X, new_x], dim=0)
            self.index_X = torch.cat([self.index_X, new_index_x], dim=0)
            self.Y = torch.cat([self.Y, self.target.query(new_x, new_index_x)], dim=0)
            self.Y_true = torch.cat([self.Y_true, self.target.query_ground_truth(new_x, new_index_x)], dim=0)
        if not last:
            self.initialize(self.X, self.index_X, self.Y)
    def initialize_given(self, init_x, index_x):
        self.update(init_x, index_x)


    def save(self, dir):
        torch.save(self.X, f"{dir}x")
        torch.save(self.index_X, f"{dir}ind_x")
        torch.save(self.Y, f"{dir}Y")
        torch.save(self.Y_true, f"{dir}Y_true")

    def run_iterations(self, num):

        for iter in range(num - 1):
            print(f"Iteration {iter + 1} begin!")
            acq_value_best = -torch.inf
            candidates_best = None
            for f in [0., 1.]:
                candidates, acq_value = optimize_acqf(
                    acq_function=self.qMES,
                    bounds=torch.cat([self.target.bounds, torch.tensor([-1, 2]).unsqueeze(1)], dim=1),
                    q=1,
                    fixed_features={1:f},
                    num_restarts=10,
                    raw_samples=512,
                )
                if acq_value > acq_value_best:
                    acq_value_best = acq_value
                    candidates_best = candidates
            self.update(candidates_best[:, 0:1], candidates_best[:, 1].long())
        print(f"Iteration {num} begin!")
        acq_value_best = -torch.inf
        candidates_best = None
        for f in [0., 1.]:
            candidates, acq_value = optimize_acqf(
                acq_function=self.qMES,
                bounds=torch.cat([self.target.bounds, torch.tensor([-1, 2]).unsqueeze(1)], dim=1),
                q=1,
                fixed_features={1: f},
                num_restarts=10,
                raw_samples=512,
            )
            if acq_value > acq_value_best:
                acq_value_best = acq_value
                candidates_best = candidates
        self.update(candidates_best[:, 0:1], candidates_best[:, 1].long(), last=True)

    def save(self, dir):
        torch.save(self.X, f"{dir}x")
        torch.save(self.index_X, f"{dir}ind_x")
        torch.save(self.Y, f"{dir}Y")
        torch.save(self.Y_true, f"{dir}Y_true")


class experiment_continious(object):
    def __init__(self, target, surrogate, acquisition):
        self.target = target
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.X = None
        self.index_X = None
        self.Y = None
        self.Y_true = None
        self.fidelity = target.fidelity
        self.bounds = target.bounds
        self.fix_fidelity = None

    def update(self, new_x, new_index_x, last=False):
        # print(indices, index_x)
        if self.X is None:
            self.X = new_x
            self.index_X = new_index_x
            self.Y = self.target.query(new_x, new_index_x)
            self.Y_true = self.target.query_ground_truth(new_x, new_index_x)
        else:
            self.X = torch.cat([self.X, new_x], dim=0)
            self.index_X = torch.cat([self.index_X, new_index_x], dim=0)
            self.Y = torch.cat([self.Y, self.target.query(new_x, new_index_x)], dim=0)
            self.Y_true = torch.cat([self.Y_true, self.target.query_ground_truth(new_x, new_index_x)], dim=0)
        if not last:
            self.surrogate.initialize(self.X, self.index_X, self.Y)

    def initialize(self, sample_num):
        init_x = (self.bounds[1:2, :] - self.bounds[0:1, :]) * torch.rand([sample_num, self.target.x_dim]) + self.bounds[0:1, :]
        print(self.target.fidelity)
        if self.target.fidelity != 1:

            index_x = torch.ones([init_x.shape[0]]).type(torch.long)
            index_x[:int(index_x.shape[0]/2)] = torch.zeros([int(index_x.shape[0]/2)]).type(torch.long)
        else:
            index_x = torch.zeros([init_x.shape[0]]).type(torch.long)
        self.update(init_x, index_x)

    def initialize_given(self, init_x, index_x):
        self.update(init_x, index_x)

    def optimize_acquisition_one_fidelity(self, init_x, iterations, f_i):
        init_x.requires_grad = True
        optimizer = optim.LBFGS([init_x], lr=0.1, max_iter=20)
        trajectory_a = torch.zeros(iterations)
        trajectory_x = torch.zeros(iterations, init_x.shape[1])
        def closure():
            optimizer.zero_grad()
            if self.surrogate.num_samples is not None:
                pred, noise_var = self.surrogate.predict(init_x.unsqueeze(0).repeat(self.surrogate.num_samples, 1, 1), f_i * torch.ones([self.surrogate.num_samples, 1]))
            else:
                pred, noise_var = self.surrogate.predict(init_x, f_i.clone().detach())
            objective = -self.acquisition(pred.mean, pred.variance, noise_var)
            objective.backward()
            # print(objective.item())
            return objective

        for i in range(iterations):
            optimizer.step(closure)
            with torch.no_grad():
                if self.surrogate.num_samples is not None:
                    pred, noise_var = self.surrogate.predict(
                        init_x.unsqueeze(0).repeat(self.surrogate.num_samples, 1, 1),
                        f_i * torch.ones([self.surrogate.num_samples, 1]))
                else:
                    pred, noise_var = self.surrogate.predict(init_x, torch.tensor([f_i]))
                ucb_v = self.acquisition(pred.mean, pred.variance, noise_var)
                trajectory_x[i, :] = init_x[0].detach()
                trajectory_a[i] = ucb_v.detach()
        lb_x_satis = torch.sum((trajectory_x >= self.bounds[0:1, ].repeat([trajectory_x.shape[0], 1])), dim=1) == init_x.shape[1]
        ub_x_satis = torch.sum((trajectory_x <= self.bounds[1:2, ].repeat([trajectory_x.shape[0], 1])), dim=1) == init_x.shape[1]
        satis = (lb_x_satis * ub_x_satis)
        if not any(satis):
            fake_x = trajectory_x[-1, :]
            fake_x[trajectory_x[-1, :] < self.bounds[0,]] = self.bounds[0, trajectory_x[-1, :] < self.bounds[0,]]
            fake_x[trajectory_x[-1, :] > self.bounds[1,]] = self.bounds[1, trajectory_x[-1, :] > self.bounds[1,]]
            fake_x = fake_x.unsqueeze(0)
            if self.surrogate.num_samples is not None:
                pred, noise_var = self.surrogate.predict(fake_x.unsqueeze(0).repeat(self.surrogate.num_samples, 1, 1),
                                                         f_i * torch.ones([self.surrogate.num_samples, 1]))
            else:
                pred, noise_var = self.surrogate.predict(fake_x, torch.tensor([f_i]))
            ucb_v = self.acquisition(pred.mean, pred.variance, noise_var)
            return fake_x.detach(), ucb_v.detach()
        else:
            return trajectory_x[satis, :][-1:, :].detach(), trajectory_a[satis][-1].detach()
        # pred, noise_var = self.surrogate.predict(init_x.unsqueeze(0).repeat(self.surrogate.num_samples, 1, 1),
        #                                              f_i * torch.ones([self.surrogate.num_samples, 1]))
        #     ucb_v = self.acquisition(pred.mean, pred.variance, noise_var)
        # return init_x.detach(), ucb_v.detac
    def optimize_acquisition(self, restarts=3, fix_fidelity=None):
        best_a = -torch.inf
        best_x = None
        best_f = None
        if fix_fidelity is not None:
            f = torch.tensor([fix_fidelity], dtype=torch.long)
            for j in range(restarts):
                init_x = (self.bounds[1:2, :] - self.bounds[0:1, :]) * torch.rand(
                    [1, self.target.x_dim]) + self.bounds[0:1, :]
                opt_x, opt_a = self.optimize_acquisition_one_fidelity(init_x, 20, f)
                if opt_a > best_a:
                    best_a = opt_a
                    best_x = opt_x
            best_f = fix_fidelity
        else:
            for f in range(self.fidelity):
                f = torch.tensor([f], dtype=torch.long)
                for j in range(restarts):
                    init_x = (self.bounds[1:2, :] - self.bounds[0:1, :]) * torch.rand(
                        [1, self.target.x_dim]) + self.bounds[0:1, :]
                    opt_x, opt_a = self.optimize_acquisition_one_fidelity(init_x, 20, f)
                    if opt_a > best_a:
                        best_a = opt_a
                        best_x = opt_x
                        best_f = f
                    # print(f, j, opt_a, best_a)
        return best_x, best_f

    def save(self, dir):
        torch.save(self.X, f"{dir}x")
        torch.save(self.index_X, f"{dir}ind_x")
        torch.save(self.Y, f"{dir}Y")
        torch.save(self.Y_true, f"{dir}Y_true")
    def run_iterations(self, num):
        for iter in range(num-1):
            print(f"Iteration {iter + 1} begin!")
            best_x, best_f = self.optimize_acquisition()
            self.update(best_x, best_f)
        print(f"Iteration {num} begin!")
        best_x, best_f = self.optimize_acquisition()
        self.update(best_x, best_f, last=True)
        # self.update()


class experiment_two_step(experiment_continious):
    def __init__(self, target, surrogate, acquisition):
        super().__init__(target, surrogate, acquisition)
    def optimize_acquisition(self, restarts=3, fix_fidelity=None):
        best_a = -torch.inf
        best_x = None
        best_f = None

        for f in range(self.fidelity):
            f = torch.tensor([f], dtype=torch.long)
            for j in range(restarts):
                init_x = (self.bounds[1:2, :] - self.bounds[0:1, :]) * torch.rand(
                    [1, self.target.x_dim]) + self.bounds[0:1, :]
                opt_x, opt_a = self.optimize_acquisition_one_fidelity(init_x, 20, f)
                if opt_a > best_a:
                    best_a = opt_a
                    best_x = opt_x
                    best_f = f
                # print(f, j, opt_a, best_a)
        best_f = 0
        least_noise = torch.inf
        for f_i in range(self.fidelity):
            if self.surrogate.num_samples is not None:
                pred, noise_var = self.surrogate.predict(best_x.unsqueeze(0).repeat(self.surrogate.num_samples, 1, 1),
                                                         f_i * torch.ones([self.surrogate.num_samples, 1]))
            else:
                pred, noise_var = self.surrogate.predict(best_x, torch.tensor([f_i]))
            if least_noise > torch.mean(noise_var):
                best_f = f_i
                least_noise = torch.mean(noise_var)
        return best_x, torch.tensor([best_f])


class experiment_num(object):
    def __init__(self, target, surrogate, acquisition):
        self.target = target
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.X = None
        self.index_X = None
        self.Y = None
        self.Y_true = None
        self.fidelity = target.fidelity
        temp_number = torch.cat([torch.arange(target.size)]*self.fidelity, dim=0).unsqueeze(1)
        temp_index = torch.cat([f*torch.ones(target.size) for f in torch.arange(self.fidelity)], dim=0).unsqueeze(1)
        self.candidate = torch.cat([temp_number, temp_index], dim=1).type(torch.long)
        self.cost_list = []
    def update(self, indices, index_x, last=False):
        # print(indices, index_x)
        if self.X is None:
            self.X = self.target.input_by_num(indices)
            self.index_X = index_x
            self.Y = self.target.query_by_num(indices, index_x)
            self.Y_true = self.target.query_ground_truth_by_num(indices)
        else:
            self.X = torch.cat([self.X, self.target.input_by_num(indices)], dim=0)
            self.index_X = torch.cat([self.index_X, index_x], dim=0)
            self.Y = torch.cat([self.Y, self.target.query_by_num(indices, index_x)], dim=0)
            self.Y_true = torch.cat([self.Y_true, self.target.query_ground_truth_by_num(indices)], dim=0)
            self.cost_list.append(self.target.cost[index_x])
        if not last:
            self.surrogate.initialize(self.X, self.index_X, self.Y)
            new_cand_index_ex = []
            for i in range(indices.shape[0]):
                for j in range(self.candidate.shape[0]):
                    if self.candidate[j][0] == indices[i] and self.candidate[j][1] == index_x[i]:
                        new_cand_index_ex.append(j)
            new_cand_index = [not i in new_cand_index_ex for i in range(self.candidate.shape[0])]
            self.candidate = self.candidate[new_cand_index, :]

    def initialize(self, sample_num):
        indices = torch.randperm(self.target.size)[:sample_num]
        index_x = torch.ones([indices.shape[0]]).type(torch.long)
        index_x[:int(index_x.shape[0]/2)] = torch.zeros([int(index_x.shape[0]/2)]).type(torch.long)
        # print(torch.sum(index_x))
        self.update(indices, index_x)

    def initialize_given(self, indices, index_x):
        # print(torch.sum(index_x))
        self.update(indices, index_x)

    def run_one_iteration(self):
        best_ucb = -torch.inf
        best_num = 0
        for i in range(self.candidate.shape[0]):
            num_x = self.candidate[i, 0]
            index_f = self.candidate[i, 1]
            sample_x = self.target.input_by_num([num_x])
            with gpytorch.settings.cholesky_jitter(1e-4):
                pred, noise = self.surrogate.predict(sample_x, index_f.unsqueeze(0))
                cur_ucb = self.acquisition.query(pred.mean, pred.variance, noise, cost=self.target.cost[index_f])
            if cur_ucb > best_ucb:
                best_num = i
                best_ucb = cur_ucb
        # print(best_num, best_ucb)
        return best_num, best_ucb
    def save(self, dir):
        torch.save(self.X, f"{dir}x")
        torch.save(self.index_X, f"{dir}ind_x")
        torch.save(self.Y, f"{dir}Y")
        torch.save(self.Y_true, f"{dir}Y_true")
        torch.save(torch.tensor(self.cost_list), f"{dir}cost_list")
    def run_iterations(self, num):
        for iter in range(num-1):
            print(f"Iteration {iter + 1} begin!")
            best_num, _ = self.run_one_iteration()
            self.update(torch.tensor([self.candidate[best_num, 0], ]), torch.tensor([self.candidate[best_num, 1], ]))
        print(f"Iteration {num} begin!")
        best_num, _ = self.run_one_iteration()
        self.update(torch.tensor([self.candidate[best_num, 0], ]), torch.tensor([self.candidate[best_num, 1], ]), last=True)
        # self.update()


class experiment_num_two_step(experiment_num):
    def run_one_iteration(self):
        best_ucb = -torch.inf
        best_num = 0
        for i in range(self.candidate.shape[0]):
            num_x = self.candidate[i, 0]
            index_f = self.candidate[i, 1]
            sample_x = self.target.input_by_num([num_x])
            with gpytorch.settings.cholesky_jitter(1e-4):
                pred, noise = self.surrogate.predict(sample_x, index_f)
                cur_ucb = self.acquisition.query(pred.mean, pred.variance, None)
            if cur_ucb > best_ucb:
                best_num = i
                best_ucb = cur_ucb
        # print(best_num, best_ucb)
        min_noise = torch.inf
        best_f_i = 0
        num_x = self.candidate[best_num, 0]
        for f_i in range(self.fidelity):
            sample_x = self.target.input_by_num([num_x])
            pred, noise = self.surrogate.predict(sample_x, f_i)
            if noise.mean*self.target.cost[f_i] < min_noise:
                min_noise = noise.mean
                best_f_i = f_i
        for k in range(self.candidate.shape[0]):
            if self.candidate[k, 0] == num_x and self.candidate[k, 1] == best_f_i:
                best_num = k
                break
        return best_num, best_ucb

    def save(self, dir):
        torch.save(self.X, f"{dir}x")
        torch.save(self.index_X, f"{dir}ind_x")
        torch.save(self.Y, f"{dir}Y")
        torch.save(self.Y_true, f"{dir}Y_true")


class experiment_BOtorch_MF_num(object):
    def __init__(self, target):
        self.target = target
        self.X = None
        self.index_X = None
        self.Y = None
        self.Y_true = None
        self.fidelity = target.fidelity
        self.fix_fidelity = None
        temp_number = torch.cat([torch.arange(target.size)] * self.fidelity, dim=0).unsqueeze(1)
        temp_index = torch.cat([f * torch.ones(target.size) for f in torch.arange(self.fidelity)], dim=0).unsqueeze(1)
        self.candidate = torch.cat([temp_number, temp_index], dim=1).type(torch.long)
        self.candidate_set = torch.cat([target.input_by_num(torch.arange(target.size))]*self.fidelity)
        self.candidate_set = torch.cat([self.candidate_set, temp_index], dim=1)
        # self.candidate_set = target.input_by_num(torch.arange(target.size))
        self.cost_list = []

    def initialize(self, X_ind, index_x, Y):
        X = self.X
        index_X = index_x
        train_X = torch.cat([X, index_X.unsqueeze(1)], dim=1)
        self.model = SingleTaskGP(train_X, Y.unsqueeze(1), covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(self.mll)
        self.model.covar_module.base_kernel.lengthscale = torch.tensor([.5])
        self.model.covar_module.outputscale = 2 * torch.ones_like(self.model.covar_module.outputscale)
        self.model.mean_module.constant.data = 6 * torch.ones_like(self.model.mean_module.constant.data)
        # fit_gpytorch_model(self.mll)
        self.qMES = qMaxValueEntropy(self.model, self.candidate_set)

    def update(self, indices, index_x, last=False):
        # print(indices, index_x)
        if self.X is None:
            self.X = self.target.input_by_num(indices)
            self.index_X = index_x
            self.Y = self.target.query_by_num(indices, index_x)
            self.Y_true = self.target.query_ground_truth_by_num(indices)
        else:
            if len(indices.shape)>1:
                self.X = torch.cat([self.X, indices], dim=0)
            else:
                self.X = torch.cat([self.X, self.target.input_by_num(indices)], dim=0)
            self.index_X = torch.cat([self.index_X, index_x], dim=0)
            self.Y = torch.cat([self.Y, self.target.query_by_value(indices, index_x)], dim=0)
            self.Y_true = torch.cat([self.Y_true, self.target.query_ground_truth_by_value(indices)], dim=0)
            self.cost_list.append(self.target.cost[index_x])
        if not last:
            self.initialize(self.X, self.index_X, self.Y)
    def initialize_given(self, init_x, index_x):
        self.update(init_x, index_x)


    def save(self, dir):
        torch.save(self.X, f"{dir}x")
        torch.save(self.index_X, f"{dir}ind_x")
        torch.save(self.Y, f"{dir}Y")
        torch.save(self.Y_true, f"{dir}Y_true")
        torch.save(torch.tensor(self.cost_list), f"{dir}cost_list")
    def run_iterations(self, num):

        for iter in range(num - 1):
            print(f"Iteration {iter + 1} begin!")
            acq_value_best = -torch.inf
            candidates_best = None
            for f in [0., 1.]:
                candidates, acq_value = optimize_acqf(
                    acq_function=self.qMES,
                    bounds=torch.cat([torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([-1, 2]).unsqueeze(1)], dim=1),
                    q=1,
                    fixed_features={2:f},
                    num_restarts=10,
                    raw_samples=512,
                )
                if acq_value > acq_value_best:
                    acq_value_best = acq_value
                    candidates_best = candidates
            self.update(candidates_best[:, 0:2], candidates_best[:, 2].long())
        print(f"Iteration {num} begin!")
        acq_value_best = -torch.inf
        candidates_best = None
        for f in [0., 1.]:
            candidates, acq_value = optimize_acqf(
                acq_function=self.qMES,
                bounds=torch.cat([torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([-1, 2]).unsqueeze(1)], dim=1),
                q=1,
                fixed_features={2: f},
                num_restarts=10,
                raw_samples=512,
            )
            if acq_value > acq_value_best:
                acq_value_best = acq_value
                candidates_best = candidates
        self.update(candidates_best[:, 0:2], candidates_best[:, 2].long(), last=True)


class experiment_num_random(experiment_num):
    def run_one_iteration(self):
        best_num = torch.randint(self.candidate.shape[0], (1,)).item()
        return best_num, 0

    def save(self, dir):
        torch.save(self.X, f"{dir}x")
        torch.save(self.index_X, f"{dir}ind_x")
        torch.save(self.Y, f"{dir}Y")
        torch.save(self.Y_true, f"{dir}Y_true")

    def update(self, indices, index_x, last=False):
        # print(indices, index_x)
        if self.X is None:
            self.X = self.target.input_by_num(indices)
            self.index_X = index_x
            self.Y = self.target.query_by_num(indices, index_x)
            self.Y_true = self.target.query_ground_truth_by_num(indices)
        else:
            self.X = torch.cat([self.X, self.target.input_by_num(indices)], dim=0)
            self.index_X = torch.cat([self.index_X, index_x], dim=0)
            self.Y = torch.cat([self.Y, self.target.query_by_num(indices, index_x)], dim=0)
            self.Y_true = torch.cat([self.Y_true, self.target.query_ground_truth_by_num(indices)], dim=0)
            self.cost_list.append(self.target.cost[index_x])
        if not last:
            new_cand_index_ex = []
            for i in range(indices.shape[0]):
                for j in range(self.candidate.shape[0]):
                    if self.candidate[j][0] == indices[i] and self.candidate[j][1] == index_x[i]:
                        new_cand_index_ex.append(j)
            new_cand_index = [not i in new_cand_index_ex for i in range(self.candidate.shape[0])]
            self.candidate = self.candidate[new_cand_index, :]
