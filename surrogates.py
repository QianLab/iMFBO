import models
import sys
import numpy as np
import models
import torch
import math
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

class NVSurrogate(object):
    def __init__(self, num_fidelity, x_dim, num_samples=64, lengthscale=None, outputscale=None, mean=None, bias = False):
        self.num_fidelity = num_fidelity
        self.x_dim = x_dim
        self.num_samples = num_samples
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.mean = mean
        self.bias=bias

    def initialize(self, tr_x, index_x, tr_y):
        if self.bias:
            for i in range(self.num_fidelity):
                tr_y[index_x==i] -= torch.mean(tr_y[index_x==i])

        self.noise_model = models.LinearNoise(self.num_fidelity, self.x_dim)
        self.likelihood = models.LearntNoiseLikelihood(tr_x, index_x, self.noise_model)
        self.model = models.ExactGPModelSE(tr_x, tr_y, self.likelihood)
        if self.lengthscale is not None:
            self.model.covar_module.base_kernel.lengthscale = self.lengthscale
        if self.outputscale is not None:
            self.model.covar_module.outputscale = self.outputscale
        if self.mean is not None:
            self.model.mean_module.constant.data = self.mean
        #
        # model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
        for name, para in self.likelihood.noise_model.named_parameters():
            if len(para.shape) == 1:
                self.likelihood.noise_model.register_prior(name + "prior", NormalPrior(0, 1), name)
            else:
                self.likelihood.noise_model.register_prior(name + "prior",
                                                      MultivariateNormalPrior(torch.zeros_like(para), torch.eye(para.shape[0])), name)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        def pyro_model(train_x, index_x, train_y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = self.model.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(train_x), test_x=train_x, index_x=index_x)
                pyro_output = pyro.sample("obs", output, obs=train_y)
            return pyro_output

        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(nuts_kernel, num_samples=self.num_samples, disable_progbar=True)
        mcmc_run.run(tr_x, index_x, tr_y)

        sampled_para = mcmc_run.get_samples()

        self.model.pyro_load_from_samples(sampled_para)
        self.model.num_samples = self.num_samples

    def predict(self, X, index_X):
        self.model.eval()
        self.noise_model.eval()
        pred = self.model(X)
        noise = self.noise_model(X, index_X)
        return pred, noise

    def predict_truth(self, X):
        self.model.eval()
        pred = self.model(X)
        return pred

class NPNVSurrogate(object):
    def __init__(self, num_fidelity, x_dim, n_mean=0, n_length_scale=0.5, n_scale=0.1, m_mean=None, m_l=None, m_scale=None):
        self.num_fidelity = num_fidelity
        self.x_dim = x_dim
        self.mean = n_mean
        self.length_scale = n_length_scale
        self.scale = n_scale
        self.mean_m = m_mean
        self.m_l = m_l
        self.m_scale = m_scale

    def initialize(self, tr_x, index_x, tr_y):
        self.noise_model = models.GPNoise(self.num_fidelity, self.x_dim, self.mean, self.length_scale, self.scale)
        self.noise_model.store_train(tr_x, index_x)
        likelihood_pretrain = GaussianLikelihood()
        likelihood_noise = FixedNoiseGaussianLikelihood(self.noise_model.Y_square, learn_additional_noise=False)
        self.model_pretrain = models.ExactGPModelSE(tr_x, tr_y, likelihood_pretrain)
        self.model_noise = models.ExactGPModelSE(tr_x, tr_y, likelihood_noise)
        mll_pretrain = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_pretrain, self.model_pretrain)
        mll_noise = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_noise, self.model_noise)
        optimizer_pretrain = torch.optim.Adam(self.model_pretrain.parameters(), lr=0.01)
        optimizer_noise = torch.optim.Adam(self.noise_model.parameters(), lr=0.03)
        optimizer_noise_2 = torch.optim.Adam(self.noise_model.parameters(), lr=0.003)
        for f in range(self.num_fidelity):
            likelihood_inner = FixedNoiseGaussianLikelihood(noise=torch.zeros_like(tr_y[index_x==f]), learn_additional_noise=False)
            self.noise_model.mlls[f]=gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_inner, self.noise_model.train_models[f])
        # for aaa in noise_model.parameters():
        #     print(aaa)
        if self.m_l is None:
            for i in range(1000):
                optimizer_pretrain.zero_grad()
                output = self.model_pretrain(tr_x)
                # Output from model
                posterior_pretrain = mll_pretrain(output, tr_y)
                # Calc loss and backprop gradients
                loss_pretrain = -posterior_pretrain
                loss_pretrain.backward()
                optimizer_pretrain.step()
            #     print(i, loss_pretrain.item(), mll_pretrain(output, tr_y))
            # print(self.model_pretrain.mean_module.constant)
            # print(self.model_pretrain.covar_module.outputscale)
            # print(self.model_pretrain.covar_module.base_kernel.lengthscale)

            self.model_noise.mean_module.constant.data = self.model_pretrain.mean_module.constant
            self.model_noise.covar_module.outputscale = self.model_pretrain.covar_module.outputscale
            self.model_noise.covar_module.base_kernel.lengthscale = self.model_pretrain.covar_module.base_kernel.lengthscale
        else:
            self.model_noise.mean_module.constant.data = self.mean_m * torch.ones_like(self.model_noise.mean_module.constant.data)
            self.model_noise.covar_module.outputscale = self.m_scale * torch.ones_like(self.model_noise.covar_module.outputscale)
            self.model_noise.covar_module.base_kernel.lengthscale = self.m_l * torch.ones_like(self.model_noise.covar_module.base_kernel.lengthscale)
        with gpytorch.settings.fast_computations.log_prob(False):
            for i in range(1000):
                # Zero gradients from previous iteration
                optimizer_noise.zero_grad()
                output_noise = self.model_noise(tr_x)
                # Output from model
                # posterior_noise = mll_noise(output_noise, tr_y)
                try:
                    posterior_noise = self.noise_model.Y_mll() + mll_noise(output_noise, tr_y)
                except:
                    break
                # Calc loss and backprop gradients
                loss_noise = -posterior_noise
                loss_noise.backward()
                optimizer_noise.step()
                # print(self.noise_model.Y_square[:5])
                # print(i, loss_noise.item(), self.noise_model.Y_mll(), mll_noise(output_noise, tr_y))

            for i in range(1000):
                # Zero gradients from previous iteration
                optimizer_noise_2.zero_grad()
                output_noise = self.model_noise(tr_x)
                # Output from model
                # posterior_noise = mll_noise(output_noise, tr_y)
                # posterior_noise = self.noise_model.Y_mll() + 5 * mll_noise(output_noise, tr_y)#for surrogate illustrate
                try:
                    posterior_noise = 0.01*self.noise_model.Y_mll() + mll_noise(output_noise, tr_y)
                except:
                    # print(self.noise_model.Y_square)
                    break
                if mll_noise(output_noise, tr_y) > 0.05:
                    break
                # Calc loss and backprop gradients
                loss_noise = -posterior_noise
                loss_noise.backward()
                optimizer_noise_2.step()
                # print(self.noise_model.Y_square[:5])
                # try:
                #     # print(i, loss_noise.item(), self.noise_model.Y_mll(), mll_noise(output_noise, tr_y))
                # except:
                #     # print("!!!!!!!!!!", self.noise_model.Y_square)
                #     break
        # # for aaa in noise_model.parameters():
        # #     print(aaa)
        self.noise_model.equip_y()

    def predict(self, X, index_X):
        self.model_noise.eval()
        self.noise_model.eval()
        with gpytorch.settings.cholesky_jitter(1e-3):
            pred = self.model_noise(X)
            noise = self.noise_model(X, index_X)
        return pred, noise

    def predict_truth(self, X):
        self.model_noise.eval()
        pred = self.model_noise(X)
        return pred

class SepSurrogate(object):
    def __init__(self, num_fidelity, x_dim, lengthscale=None, outputscale=None, mean=None):
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.mean = mean
        self.num_fidelity = num_fidelity
        self.x_dim = x_dim
        self.num_samples = None

    def initialize(self, tr_x, index_x, tr_y):
        for i in range(torch.max(index_x).int()):
            tr_y[index_x==i] -= torch.mean(tr_y[index_x==i])

        self.likelihood_list = torch.nn.ModuleList()
        self.model_list = torch.nn.ModuleList()
        self.mll_list = torch.nn.ModuleList()
        for i in range(self.num_fidelity):
            likelihood = GaussianLikelihood()
            self.likelihood_list.append(likelihood)
            model = models.ExactGPModelSE(tr_x[index_x == i], tr_y[index_x == i], likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            if self.lengthscale is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
                for j in range(1000):
                    optimizer.zero_grad()
                    output = model(tr_x[index_x == i])
                    # Output from model
                    posterior = mll(output, tr_y[index_x == i])
                    # Calc loss and backprop gradients
                    loss_pretrain = -posterior
                    loss_pretrain.backward()
                    optimizer.step()
            else:
                model.covar_module.base_kernel.lengthscale = self.lengthscale*torch.ones_like(model.covar_module.base_kernel.lengthscale)
                model.covar_module.outputscale = self.outputscale*torch.ones_like(model.covar_module.outputscale)
                model.mean_module.constant.data = self.mean*torch.ones_like(model.mean_module.constant.data)
            self.model_list.append(model)
            self.mll_list.append(mll)

    def predict(self, X, index_X):
        self.model_list.eval()
        pred = self.model_list[index_X[0]](X)
        noise = None
        return pred, noise


if __name__ == "__main__":
    surrogate = NPNVSurrogate(2, 3)
    train_x = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    index_x = torch.tensor([0, 1])
    train_y = torch.tensor([4., 2.])
    surrogate.initialize(train_x, index_x, train_y)
