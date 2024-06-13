import torch
from torch.nn import Module
import gpytorch


class UCB_NVUCB(Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.step_num=1
    def forward(self, mean, variance, noise, cost=1):
        uncertainty_eval = variance / torch.sqrt(variance + noise)
        res = mean + 1/cost * self.beta ** self.step_num * uncertainty_eval
        return torch.mean(res)
    def step(self):
        self.step_num += 1

class UCB_NUCB(Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.step_num = 1
    def forward(self, mean, variance, noise, cost=1):
        uncertainty_eval = torch.sqrt(variance + noise)
        res = mean + 1/cost * self.beta ** self.step_num * uncertainty_eval
        return torch.mean(res)
    def step(self):
        self.step_num += 1

class UCB(Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.step_num = 1
    def forward(self, mean, variance, noise, cost=1):
        uncertainty_eval = torch.sqrt(variance)
        res = mean + 1/cost * self.beta ** self.step_num * uncertainty_eval
        return torch.mean(res)
    def step(self):
        self.step_num += 1




class UCB_by_num(object):
    def __init__(self, beta=1):
        self.step_num = 1
        self.beta = beta
    def query(self, mean, variance, noise, sample_num=128, cost=1):
        uncertainty_eval = 0
        if isinstance(noise, gpytorch.distributions.MultivariateNormal):
            noise_samples = noise.sample(torch.Size([sample_num]))
            uncertainty_eval_sample = variance/torch.sqrt(variance + noise_samples**2)
            uncertainty_eval = torch.mean(uncertainty_eval_sample).unsqueeze(0)
        if noise is None:
            uncertainty_eval = torch.sqrt(variance)
        return mean + 1/cost * self.beta**self.step_num * uncertainty_eval
    def step(self):
        self.step_num += 1


class UCB_nfc_by_num(object):
    def __init__(self, beta=1):
        self.step_num = 1
        self.beta = beta
    def query(self, mean, variance, noise, sample_num=128, cost=1):
        uncertainty_eval = 0
        if isinstance(noise, gpytorch.distributions.MultivariateNormal):
            noise_samples = noise.sample(torch.Size([sample_num]))
            uncertainty_eval_sample = torch.sqrt(variance + noise_samples**2)
            uncertainty_eval = torch.mean(uncertainty_eval_sample).unsqueeze(0)
        if noise is None:
            uncertainty_eval = torch.sqrt(variance)
        return mean + 1/cost*self.beta**self.step_num * uncertainty_eval
    def step(self):
        self.step_num += 1

class Random():
    def __init__(self, beta=1):
        self.step_num = 1
        self.beta = beta

    def query(self):
        return torch.tensor(0.)

    def step(self):
        self.step_num += 1