import torch
import gpytorch
import math
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.nn import Module
import torch.nn as nn
from math import floor


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_f):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_f
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_f, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class ExactGPModelLN(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelLN, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModelSE(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelSE, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModelM(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelM, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LearntNoiseLikelihood(FixedNoiseGaussianLikelihood):
    def __init__(self, train_x, index_x, noise_model):
        # super(LearntNoiseLikelihood, self).__init__(noise=noise, learn_additional_noise=False)
        super(LearntNoiseLikelihood, self).__init__(noise=10000*torch.ones_like(train_x), learn_additional_noise=False)
        self.noise_model = noise_model
        self.train_x = train_x
        self.index_x = index_x
        self.num_f = noise_model.num_f

    def __call__(self, input, *args, **kwargs):
        if not kwargs:
            if input.batch_shape:
                b_shape = input.batch_shape[0]
                return super(LearntNoiseLikelihood, self).__call__(input, noise=self.noise_model(self.train_x.repeat(b_shape, 1), self.index_x.repeat(b_shape, 1)))
            else:
                return super(LearntNoiseLikelihood, self).__call__(input, noise=self.noise_model(self.train_x, self.index_x))
        else:
            return super(LearntNoiseLikelihood, self).__call__(input, noise=self.noise_model(kwargs['test_x'], kwargs['index_x']))


class ScalarNoise(gpytorch.Module):
    def __init__(self, num_x, x_dim):
        super(ScalarNoise, self).__init__()
        self.num_f = num_x
        self.noise = nn.Parameter(torch.rand(num_x, requires_grad=True))

    def forward(self, train_x, index_x):
        return self.noise[index_x]**2


class LinearNoise(gpytorch.Module):
    def __init__(self, num_x, x_dim):
        super(LinearNoise, self).__init__()
        self.num_f = num_x
        self.x_dim = x_dim
        for i in range(num_x):
            self.register_parameter('F' + str(i) + '_weight', torch.nn.Parameter(0.1*torch.randn([1, x_dim])))
        for i in range(num_x):
            self.register_parameter('F' + str(i) + '_bias', torch.nn.Parameter(0.1*torch.randn(1)))

    def forward(self, train_x, index_x):
        if len(index_x.shape) == 1:
            if len(train_x.shape) == 1:
                train_x = train_x.reshape([-1, self.x_dim])
            list_var = [torch.matmul(self.__getattr__('F' + str(fidelity) + '_weight'), train_x.T) +
                        self.__getattr__('F' + str(fidelity) + '_bias') for fidelity in range(self.num_f)]
            ind = torch.cat([torch.arange(index_x.shape[0]).unsqueeze(1), index_x.unsqueeze(1)], dim=-1)
            res = torch.cat(list_var, dim=0).T
            res = res[tuple(ind.T)]
            return res ** 2
        else:
            if train_x.shape[0] != self.F0_weight.shape[0]:
                train_x = train_x.reshape(self.F0_weight.shape[0], -1, self.x_dim)
            if len(train_x.shape) == 2:
                train_x = train_x.unsqueeze(-1)
            list_var = [torch.matmul(self.__getattr__('F' + str(fidelity) + '_weight'), train_x.transpose(1, 2)) +
                        self.__getattr__('F' + str(fidelity) + '_bias').unsqueeze(-1) for fidelity in range(self.num_f)]
            list_res = []
            res = torch.cat(list_var, dim=1).transpose(1, 2)

            for nb in range(res.shape[0]):
                ind = torch.cat([torch.arange(index_x.shape[1]).unsqueeze(1), index_x[nb].unsqueeze(1)], dim=-1)
                list_res.append(res[nb][tuple(ind.T.type(torch.long))].unsqueeze(0))
            res = torch.cat(list_res, dim=0)
            return res**2


class GPNoise(gpytorch.Module):
    def __init__(self, num_f, x_dim, mean=0, length_scale=0.5, scale=0.1):
        super(GPNoise, self).__init__()
        self.num_f = num_f
        self.x_dim = x_dim
        self.train_models = torch.nn.ModuleList()
        self.mlls = torch.nn.ModuleList()
        self.models = torch.nn.ModuleList()
        self.Y_square = None
        self.mean_prior = mean
        self.lengthscale_prior = length_scale
        self.scale_prior = scale
        for f in range(num_f):

            likelihood = FixedNoiseGaussianLikelihood(noise=torch.tensor([1e10]), learn_additional_noise=False)
            likelihood_inner = FixedNoiseGaussianLikelihood(noise=torch.tensor([0]), learn_additional_noise=False)
            self.train_models.append(ExactGPModelSE(torch.zeros([1, self.x_dim]), torch.zeros([1]), likelihood))
            if isinstance(mean, list):
                self.train_model_settings(self.train_models[f], mean[f], length_scale[f], scale[f])
            else:
                self.train_model_settings(self.train_models[f], mean, length_scale, scale)
            self.mlls.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_inner, self.train_models[f]))

    def train_model_settings(self, GPmodel, mean=0, length_scale=0.5, scale=0.1):
        GPmodel.mean_module.constant = torch.nn.Parameter(mean*torch.ones_like(GPmodel.mean_module.constant))
        GPmodel.covar_module.outputscale = scale*torch.ones_like(GPmodel.covar_module.outputscale)
        GPmodel.covar_module.base_kernel.lengthscale = length_scale*torch.ones_like(GPmodel.covar_module.base_kernel.lengthscale)

    def store_train(self, train_x, index_x):
        self.train_x = train_x
        self.index_x = index_x
        # self.Y_square = torch.nn.Parameter(0.0625*torch.ones(train_x.shape[0], requires_grad=True))
        self.Y_square = torch.nn.Parameter(0.25*torch.rand(train_x.shape[0], requires_grad=True) + 0.125)
        # self.Y = torch.nn.Parameter(torch.zeros(train_x.shape[0], requires_grad=True))

    def equip_y(self):
        for f in range(self.num_f):
            likelihood = FixedNoiseGaussianLikelihood(noise=torch.zeros_like(self.Y_square[self.index_x == f]), learn_additional_noise=False)
            # print((self.Y_square[self.index_x == f] + torch.abs(self.Y_square[self.index_x == f])) / 2)
            self.models.append(ExactGPModelSE(self.train_x[self.index_x == f], torch.sqrt((self.Y_square[self.index_x == f] + torch.abs(self.Y_square[self.index_x == f]))/2), likelihood))
            if isinstance(self.mean_prior, list):
                self.train_model_settings(self.models[f], self.mean_prior[f], self.lengthscale_prior[f], self.scale_prior[f])
            else:
                self.train_model_settings(self.models[f], self.mean_prior, self.lengthscale_prior, self.scale_prior)
    def Y_mll(self):
        self.train_models.eval()
        self.train_models.train()

        mll_f = torch.ones(self.num_f)
        self.train_models.eval()
        # print(self.Y.data)
        for f in range(self.num_f):
            if self.train_x[self.index_x == f].shape[0]:
                output = self.train_models[f](self.train_x[self.index_x == f])
                # print(torch.sqrt(torch.abs(self.Y[self.index_x == f])))
                mll_f[f] = self.mlls[f](output, torch.sqrt(torch.abs(self.Y_square[self.index_x == f])))
            else:
                mll_f[f] = 0
        # print(mll_f)
        self.train_models.train()
        return torch.sum(mll_f)

    def forward(self, train_x, index_x):
        self.models[index_x].eval()
        return self.models[index_x](train_x)





if __name__ == '__main__':
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    tr_x = torch.linspace(0, 1, 4)
    tr_y = torch.sin(tr_x * (2 * math.pi))

    model = ExactGPModelSE(tr_x, tr_y, likelihood)
    index = torch.randint(0, 3, [10])
    noise = torch.rand(3, requires_grad=True)