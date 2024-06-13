import sys
import numpy as np
import models
import torch
import math
import gpytorch
import matplotlib.pyplot as plt
from pyro.infer.mcmc import NUTS, MCMC, HMC
from gpytorch.likelihoods import GaussianLikelihood
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


def Model_initialization_multi(num_fidelity, x_dim, tr_x, index_x, tr_y):
    noise_model = models.LinearNoise(num_fidelity, x_dim)
    likelihood = models.LearntNoiseLikelihood(tr_x, index_x, noise_model)
    model = models.ExactGPModel(tr_x, tr_y, likelihood)
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
    for name, para in likelihood.noise_model.named_parameters():
        if len(para.shape) == 1:
            likelihood.noise_model.register_prior(name + "prior", NormalPrior(0, 1), name)
        else:
            likelihood.noise_model.register_prior(name + "prior", MultivariateNormalPrior(torch.zeros_like(para), torch.eye(para.shape[0])), name)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    return model, likelihood, mll


def Model_initialization_our(num_fidelity, x_dim, tr_x, index_x, tr_y):
    noise_model = models.LinearNoise(num_fidelity, x_dim)
    likelihood = models.LearntNoiseLikelihood(tr_x, index_x, noise_model)
    model = models.ExactGPModel(tr_x, tr_y, likelihood)
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
    for name, para in likelihood.noise_model.named_parameters():
        if len(para.shape) == 1:
            likelihood.noise_model.register_prior(name + "prior", NormalPrior(0, 1), name)
        else:
            likelihood.noise_model.register_prior(name + "prior", MultivariateNormalPrior(torch.zeros_like(para), torch.eye(para.shape[1])), name)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    return model, likelihood, mll


def Model_initialization_our_2(num_fidelity, x_dim, tr_x, index_x, tr_y):
    noise_model = models.LinearNoise(num_fidelity, x_dim)
    likelihood = models.LearntNoiseLikelihood(tr_x, index_x, noise_model)
    model = models.ExactGPModel(tr_x, tr_y, likelihood)
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
    for name, para in likelihood.noise_model.named_parameters():
        if len(para.shape) == 1:
            likelihood.noise_model.register_prior(name + "prior", UniformPrior(0, 1), name)
        else:
            likelihood.noise_model.register_prior(name + "prior", UniformPrior(-torch.ones_like(para), torch.ones_like(para)), name)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    return model, likelihood, mll


def Model_initialization_naive(num_fidelity, x_dim, tr_x, index_x, tr_y):
    likelihood_list = torch.nn.ModuleList()
    model_list = torch.nn.ModuleList()
    mll_list = torch.nn.ModuleList()
    for i in range(num_fidelity):
        likelihood = GaussianLikelihood()
        likelihood_list.append(likelihood)
        model = models.ExactGPModel(tr_x[index_x==i], tr_y[index_x==i], likelihood)
        model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
        model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
        likelihood.register_prior("noise_prior", UniformPrior(0.01, 0.5), "noise")
        model_list.append(model)
        mll_list.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model))
    return model_list, likelihood_list, mll_list


def Train_model_our(num_fidelity, x_dim, tr_x, index_x, tr_y, num_samples):
    model, likelihood, mll = Model_initialization_our(num_fidelity, x_dim, tr_x, index_x, tr_y)

    def pyro_model(train_x, index_x, train_y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(train_x), test_x=train_x, index_x=index_x)
            pyro_output = pyro.sample("obs", output, obs=train_y)
        return pyro_output

    nuts_kernel = NUTS(pyro_model)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, disable_progbar=False)
    mcmc_run.run(tr_x, index_x, tr_y)

    sampled_para = mcmc_run.get_samples()

    model.pyro_load_from_samples(sampled_para)
    model.num_samples = num_samples
    return model, likelihood, mll


def Train_model_our_2(num_fidelity, x_dim, tr_x, index_x, tr_y, num_samples):
    model, likelihood, mll = Model_initialization_our_2(num_fidelity, x_dim, tr_x, index_x, tr_y)

    def pyro_model(train_x, index_x, train_y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(train_x), test_x=train_x, index_x=index_x)
            pyro_output = pyro.sample("obs", output, obs=train_y)
        return pyro_output

    nuts_kernel = NUTS(pyro_model)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, disable_progbar=False)
    mcmc_run.run(tr_x, index_x, tr_y)

    sampled_para = mcmc_run.get_samples()

    model.pyro_load_from_samples(sampled_para)
    model.num_samples = num_samples
    return model, likelihood, mll


def Train_model_naive(num_fidelity, x_dim, tr_x, index_x, tr_y, num_samples):
    model_list, likelihood_list, mll_list = Model_initialization_naive(num_fidelity, x_dim, tr_x, index_x, tr_y)

    for i, model in enumerate(model_list):
        def pyro_model(train_x, index_x, train_y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = model.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(train_x[index_x==i]))
                pyro_output = pyro.sample("obs", output, obs=train_y[index_x==i])
            return pyro_output

        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, disable_progbar=False)
        mcmc_run.run(tr_x, index_x, tr_y)

        sampled_para = mcmc_run.get_samples()
        model.pyro_load_from_samples(sampled_para)
        model.num_samples = num_samples
    return model_list, likelihood_list, mll_list


class UCB_our(Module):
    def __init__(self, model, likelihood, costs, beta):
        super().__init__()
        self.model = model
        self.costs = costs
        self.likelihood = likelihood
        self.beta = beta
    def forward(self, test_x):
        num_samples = self.model.num_samples
        expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
        output = self.model(expanded_test_x)
        mus = output.mean
        vars = output.variance
        noise_var_list = [self.likelihood.noise_model(expanded_test_x, i * torch.ones([num_samples, test_x.shape[0]])) for i in range(self.likelihood.noise_model.num_f)]
        res_list = [mus + self.beta * (1/self.costs[i]) * vars/torch.sqrt(vars + noise_var) for i, noise_var in enumerate(noise_var_list)]
        res = [torch.mean(res_i, dim=0).unsqueeze(0) for res_i in res_list]
        res = torch.cat(res, dim=0)
        return res


class UCB_our_nfc(Module):
    def __init__(self, model, likelihood, costs, beta):
        super().__init__()
        self.model = model
        self.costs = costs
        self.likelihood = likelihood
        self.beta = beta
    def forward(self, test_x):
        num_samples = self.model.num_samples
        expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
        output = self.model(expanded_test_x)
        mus = output.mean
        vars = output.variance
        noise_var_list = [self.likelihood.noise_model(expanded_test_x, i * torch.ones([num_samples, test_x.shape[0]])) for i in range(self.likelihood.noise_model.num_f)]
        res_list = [mus + self.beta * (1/self.costs[i]) * torch.sqrt(vars + noise_var) for i, noise_var in enumerate(noise_var_list)]
        res = [torch.mean(res_i, dim=0).unsqueeze(0) for res_i in res_list]
        res = torch.cat(res, dim=0)
        return res


class UCB_naive(Module):
    def __init__(self, model_list, likelihood, costs, beta):
        super().__init__()
        self.models = model_list
        self.costs = costs
        self.beta = beta
    def forward(self, test_x):
        num_samples = self.models[0].num_samples
        expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
        output_list = [model(expanded_test_x) for model in self.models]
        mus_list = [output.mean for output in output_list]
        vars_list = [output.variance for output in output_list]
        res_list = [mus + self.beta * (1/self.costs[i])* torch.sqrt(vars) for i, (mus, vars) in enumerate(zip(mus_list, vars_list))]
        res = [torch.mean(res_i, dim=0).unsqueeze(0) for res_i in res_list]
        res = torch.cat(res)
        return res


def optimize_acquisition_one_fidelity(UCB, init_x, iterations, f_i, lr=0.01):
    init_x.requires_grad = True
    optimizer = optim.Adam([init_x], lr=lr)
    for i in range(iterations):
        optimizer.zero_grad()
        UCB_value = -UCB(init_x)
        loss = UCB_value[f_i, 0]
        loss.backward()
        optimizer.step()
    return init_x.detach(), UCB(init_x)[f_i, 0].detach()


def optimize_acquisition_one_fidelity_LB(UCB, init_x, iterations, f_i):
    init_x.requires_grad = True
    optimizer = optim.LBFGS([init_x])
    for i in range(iterations):
        optimizer.zero_grad()
        ucb_neg = lambda x: -UCB(x)[f_i, 0]
        UCB_value = ucb_neg(init_x)
        loss = UCB_value
        loss.backward()
        optimizer.step(lambda: ucb_neg(init_x))
    return init_x.detach(), UCB(init_x)[f_i, 0].detach()


def optimize_acquisition_restart(UCB, iterations, restarts, f_num, x_dim, lr=0.01):
    max_our = -torch.inf
    opt_x_our = None
    index_chosen = None
    for f_i in range(f_num):
        for _ in range(restarts):
            init_x_our = torch.rand([1, x_dim])
            x_chosen_our, value_our = optimize_acquisition_one_fidelity(UCB, init_x_our, iterations, f_i, lr=lr)
            if max_our < value_our:
                opt_x_our = x_chosen_our
                max_our = value_our
                index_chosen = f_i
    return opt_x_our, torch.tensor([index_chosen])

def optimize_acquisition_LB(UCB, iterations, f_num, x_dim):
    max_our = -torch.inf
    opt_x_our = None
    index_chosen = None
    for f_i in range(f_num):
        init_x_our = torch.rand([1, x_dim])
        x_chosen_our, value_our = optimize_acquisition_one_fidelity_LB(UCB, init_x_our, iterations, f_i)
        if max_our < value_our:
            opt_x_our = x_chosen_our
            max_our = value_our
            index_chosen = f_i
    return opt_x_our, torch.tensor([index_chosen])

#
# def optimize_acquisition_LBFGSB(UCB, iterations, f_num, x_bound):
#     max_our = -torch.inf
#     opt_x_our = None
#     index_chosen = None
#     for f_i in range(f_num):
#         init_x_our = torch.rand([1, 1])
#         x_chosen_our, value_our = optimize_acquisition_one_fidelity_LB(UCB, init_x_our, iterations, f_i)
#         if max_our < value_our:
#             opt_x_our = x_chosen_our
#             max_our = value_our
#             index_chosen = f_i
#     return opt_x_our, torch.tensor([index_chosen])


def BO_performe(query, num_f, x_dim, costs, train_model_func, ucb_func, budget, num_samples=64, initial_samples=None, x_bound=None):
    opt_iterations = 100  # number of optimization iterations (for acquisition function optimization)
    init_beta = 2  # beta value
    restarts = 5
    if initial_samples is None:
        if x_bound is None:
            x_bound = torch.cat([torch.zeros([x_dim, 1]), torch.ones([x_dim, 1])], dim=1).T
        init_tr_x = (x_bound[1, :] - x_bound[0, :])*torch.rand([4, x_dim]) + x_bound[0, :]
        init_index_x = torch.cat([torch.zeros(int(4 / 2)).long(), torch.ones(int(4 / 2)).long()])
    else:
        init_tr_x, init_index_x = initial_samples
    init_tr_y, _ = query(init_tr_x, init_index_x)

    tr_x = init_tr_x
    index_x = init_index_x
    tr_y = init_tr_y

    for i in range(budget):
        print(i + 1, "-th iteration has just begun!")
        model, likelihood, mll = train_model_func(num_f, x_dim, tr_x, index_x, tr_y, num_samples)
        ucb = ucb_func(model, likelihood, costs, beta=init_beta * torch.exp(- 0.1 * torch.tensor(i)))
        model.eval()
        opt_x, index_chosen = optimize_acquisition_LB(ucb, num_f, x_dim, lr=0.01)
        new_y, _ = query(opt_x, index_chosen)
        tr_x = torch.cat([tr_x, opt_x], dim=0)
        tr_y = torch.cat([tr_y, new_y], dim=0)
        index_x = torch.cat([index_x, index_chosen], dim=0)
    saves = [tr_x, tr_y, index_x]
    return saves


def BO_performe_2(query, num_f, x_dim, costs, train_model_func, ucb_func, budget, num_samples=64, initial_samples=None, x_bound=None):
    opt_iterations = 100  # number of optimization iterations (for acquisition function optimization)
    init_beta = 2  # beta value
    restarts = 5
    if initial_samples is None:
        if x_bound is None:
            x_bound = torch.cat([torch.zeros([x_dim, 1]), torch.ones([x_dim, 1])], dim=1).T
        init_tr_x = torch.rand([4, x_dim])
        init_index_x = torch.cat([torch.zeros(int(4 / 2)).long(), torch.ones(int(4 / 2)).long()])
    else:
        init_tr_x, init_index_x = initial_samples
    init_tr_y, _ = query(init_tr_x, init_index_x, x_bound)

    tr_x = init_tr_x
    index_x = init_index_x
    tr_y = init_tr_y

    for i in range(budget):
        print(i + 1, "-th iteration has just begun!")
        model, likelihood, mll = train_model_func(num_f, x_dim, tr_x, index_x, tr_y, num_samples)
        ucb = ucb_func(model, likelihood, costs, beta=init_beta * torch.exp(- 0.1 * torch.tensor(i)))
        model.eval()
        opt_x, index_chosen = optimize_acquisition_restart(ucb, opt_iterations, 5, num_f, x_dim)
        new_y, _ = query(opt_x, index_chosen, x_bound)
        tr_x = torch.cat([tr_x, opt_x], dim=0)
        tr_y = torch.cat([tr_y, new_y], dim=0)
        index_x = torch.cat([index_x, index_chosen], dim=0)
    saves = [tr_x, tr_y, index_x]
    return saves
