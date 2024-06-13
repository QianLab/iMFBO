import torch
import matplotlib.pyplot as plt

a_br = 1
b_br = 5.1 / (4 * torch.pi**2)
c_br = 5 / torch.pi
r_br = 6
s_br = 10
t_br = 1 / (8 * torch.pi)

alpha_hm = torch.tensor([1.0, 1.2, 3.0, 3.2])
A_hm = torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
     [0.05, 10, 17, 0.1, 8, 14],
     [3, 3.5, 1.7, 10, 17, 8],
     [17, 8, 0.05, 10, 0.1, 14]])
P_hm = 1e-4 * torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
               [2329, 4135, 8307, 3736, 1004, 9991],
               [2348, 1451, 3522, 2883, 3047, 6650],
               [4047, 8828, 8732, 5743, 1091, 381]])


def Branin(x):
    f_value = a_br * (x[:, 1] - b_br * x[:, 0] ** 2 + c_br * x[:, 0] - r_br) ** 2 + s_br * (1 - t_br) * torch.cos(x[:, 0]) + s_br
    return -f_value


def Hartmann(x):
    outer = torch.zeros(x.shape[0])
    for ii in range(4):
        inner = torch.zeros(x.shape[0])
        for jj in range(6):
            xj = x[:, jj]
            Aij = A_hm[ii, jj]
            Pij = P_hm[ii, jj]
            inner = inner + Aij*(xj-Pij)**2
        new = alpha_hm[ii] * torch.exp(-inner)
        outer = outer + new
    y = -(2.58 + outer) / 1.94
    return -y


def Levy(x):
    d = x.shape[1]
    w = []
    for ii in range(d):
        w += [1 + (x[:, ii] - 1)/4]
    term1 = (torch.sin(torch.pi*w[0]))**2
    term3 = (w[d-1]-1)**2 * (1+(torch.sin(2*torch.pi*w[d-1]))**2)
    sum_res = torch.zeros(x.shape[0])
    for ii in range(d-1):
        wi = w[ii]
        new = (wi-1)**2 * (1+10*(torch.sin(torch.pi*wi+1))**2)
        sum_res = sum_res + new
    y = term1 + sum_res + term3
    return -y


if __name__ == "__main__":
    test_x_lv = torch.tensor([[1, 1, 1], [2, 1, 3], [-4, 5, 3]])
    print(Levy(test_x_lv))

    test_x_hm = torch.tensor([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], [0.20119, 1, 0.476874, 0.275332, 0.311652, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    print(Hartmann(test_x_hm))

    x_star = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    for i in range(6):
        x_arange = torch.arange(100)/100
        X_cand = x_star.unsqueeze(0).repeat([100, 1])
        X_cand[:, i] = x_arange
        plt.plot(x_arange, Hartmann(X_cand))
        plt.show()

    test_x_br = torch.tensor([[-torch.pi, 12.275], [torch.pi, 2.275], [9.42478, 2.475]])
    print(Branin(test_x_br))
