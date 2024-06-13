import sys
import numpy as np
import models
import torch
import math
import gpytorch
import matplotlib.pyplot as plt
from torch.nn import Module
import torch.nn as nn
from target_functions import Branin, Hartmann, Levy

class quad_target(object):
    def __init__(self):
        self.fidelity = 2
        self.x_dim = 1
        self.a = torch.tensor([0.5, -0.5])
        self.b = torch.tensor([0.5, 0.5])
        self.bounds = torch.tensor([[-1.], [1.]])
        self.s = 1
        self.f = 32
    def noise_level(self, tr_x, index_x):
        if len(tr_x.shape) == 2:
            tr_x = tr_x[:, 0]
        return (self.a[index_x] * tr_x + self.b[index_x]) * torch.sin(self.f*torch.pi*tr_x)
    def query_ground_truth(self, tr_x, index_x):
        if len(tr_x.shape) == 2:
            tr_x = tr_x[:, 0]
        tr_y_gt = - (self.s*tr_x ** 2 - 1)*torch.cos(torch.pi*3*tr_x)
        return tr_y_gt

    def query(self, tr_x, index_x):
        return self.query_ground_truth(tr_x, index_x) + self.noise_level(tr_x, index_x)



class sin_target(object):
    def __init__(self, fidelity_fix=None, bias=False):
        if fidelity_fix is None:
            self.fidelity = 2
        else:
            self.fidelity = 1
            self.fidelity_fix = fidelity_fix
        self.x_dim = 1
        self.a = torch.tensor([0.5, -0.5])
        self.b = torch.tensor([0, 0.5])
        self.bounds = torch.tensor([[0.], [1.]])
        self.bias = bias
    def noise_level(self, tr_x, index_x):
        if self.fidelity == 1:
            index_x = self.fidelity_fix * torch.ones_like(index_x, dtype=torch.long)
        if len(tr_x.shape) == 2:
            tr_x = tr_x[:, 0]
        return self.a[index_x] * tr_x + self.b[index_x]

    def query_ground_truth(self, tr_x, index_x):
        if len(tr_x.shape) == 2:
            tr_x = tr_x[:, 0]
        tr_y_gt = torch.sin(tr_x * (2 * math.pi))
        tr_y_gt[torch.logical_or((tr_x < 0), (tr_x > 1))] = 0
        return tr_y_gt

    def query(self, tr_x, index_x):
        bias_sets = torch.tensor([0.5, -0.5])
        if self.fidelity == 1:
            index_x = self.fidelity_fix * torch.ones(tr_x.shape[0], dtype=torch.long)

        noise_level = self.noise_level(tr_x, index_x)
        bias_value = self.bias*bias_sets[index_x]
        return self.query_ground_truth(tr_x, index_x) + torch.randn(tr_x.shape[0])*noise_level + bias_value

class band_gap_target(object):
    def __init__(self, dir, follow, cost=None):
        if cost is None:
            cost = [1, 1]
        self.fidelity = 2
        self.Z = torch.load(dir+'/Z'+follow+'.ts')
        self.Y = torch.load(dir+'/Y'+follow+'.ts')
        self.Y_0 = torch.load(dir+'/Y_0'+follow+'.ts')
        self.Y_1 = torch.load(dir+'/Y_1'+follow+'.ts')+0.9
        self.size = self.Z.shape[0]
        self.Y_low = [self.Y_0, self.Y_1]
        self.cost = cost
    def input_by_num(self, num_x):
        return self.Z[num_x, :]

    def query_ground_truth_by_num(self, num_x):
        return self.Y[num_x, 0]

    def query_by_num(self, num_x, index_x):
        output = torch.ones([num_x.shape[0]])
        for i in range(num_x.shape[0]):
            output[i] = self.Y_low[index_x[i]][num_x[i], 0]
        return output

    def query_by_value(self, value, index_x):
        closest_index = torch.argmin(torch.sum((self.Z - value)**2, dim=1)).unsqueeze(0)
        return self.query_by_num(closest_index, index_x)

    def query_ground_truth_by_value(self, value):
        closest_index = torch.argmin(torch.sum((self.Z - value)**2, dim=1)).unsqueeze(0)
        return self.query_ground_truth_by_num(closest_index)



class band_gap_target_three(object):
    def __init__(self, dir, follow, cost=None):
        if cost is None:
            cost = [1, 1, 1]
        self.fidelity = 3
        self.Z = torch.load(dir+'/Z'+follow+'.ts')
        self.Y = torch.load(dir+'/Y'+follow+'.ts')
        self.Y_0 = torch.load(dir+'/Y_0'+follow+'.ts')
        self.Y_1 = torch.load(dir+'/Y_1'+follow+'.ts')+0.9
        self.Y_2 = torch.load(dir + '/Y_2' + follow + '.ts')
        self.size = self.Z.shape[0]
        self.Y_low = [self.Y_0, self.Y_1, self.Y_2]
        self.cost = cost
    def input_by_num(self, num_x):
        return self.Z[num_x, :]

    def query_ground_truth_by_num(self, num_x):
        return self.Y[num_x, 0]

    def query_by_num(self, num_x, index_x):
        output = torch.ones([num_x.shape[0]])
        for i in range(num_x.shape[0]):
            output[i] = self.Y_low[index_x[i]][num_x[i], 0]
        return output

    def query_by_value(self, value, index_x):
        closest_index = torch.argmin(torch.sum((self.Z - value)**2, dim=1)).unsqueeze(0)
        return self.query_by_num(closest_index, index_x)

    def query_ground_truth_by_value(self, value):
        closest_index = torch.argmin(torch.sum((self.Z - value)**2, dim=1)).unsqueeze(0)
        return self.query_ground_truth_by_num(closest_index)






class br_target(object):
    def __init__(self, fidelity_fix=None):
        if fidelity_fix is None:
            self.fidelity = 2
        else:
            self.fidelity = 1
            self.fidelity_fix = fidelity_fix
        self.x_dim = 1
        self.a = torch.tensor([[[0.5, 0.5]], [[-0.5, -0.50]]])
        self.b = torch.tensor([[0], [1]])
    #
    # self.bounds = torch.tensor([[0.], [1.]])
    def noise_level(self, tr_x, index_x):
        if self.fidelity == 1:
            index_x = self.fidelity_fix * torch.ones_like(index_x, dtype=torch.long)
        if len(tr_x.shape) == 2:
            tr_x = tr_x[:, 0]
        return self.a[index_x] * tr_x + self.b[index_x]

    def query_ground_truth(self, tr_x, index_x):
        if len(tr_x.shape) == 2:
            tr_x = tr_x[:, 0]
        tr_y_gt = torch.sin(tr_x * (2 * math.pi))
        tr_y_gt[torch.logical_or((tr_x < 0), (tr_x > 1))] = 0
        return tr_y_gt

    def query(self, tr_x, index_x):
        if self.fidelity == 1:
            index_x = self.fidelity_fix * torch.ones(tr_x.shape[0], dtype=torch.long)

        noise_level = self.noise_level(tr_x, index_x)
        return self.query_ground_truth(tr_x, index_x) + torch.randn(tr_x.shape[0])*noise_level




if __name__ == '__main__':
    aaa = band_gap_target("./real_experiement", '')
    print(aaa.Y.max())