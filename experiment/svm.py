import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 3)
        self.act = nn.ReLU()
        # self.BN2 = nn.BatchNorm1d(3)
        # self.BN1 = nn.BatchNorm1d(input_dim)

    def forward(self, X):
        # l1_out = self.act(self.l1(self.BN1(X)))
        # l2_out = self.act(self.l2(l1_out))
        # out = self.BN2(l2_out)
        # return out
        l1_out = self.l1(X)
        # out = self.act(l1_out)
        return l1_out


class LinearClassifier(nn.Module):
    def __init__(self, extractor, ex_dim):
        super().__init__()
        self.extractor = extractor
        self.l = nn.Linear(ex_dim, 1)

    def forward(self, X):
        extracted = self.extractor(X)
        return self.l(extracted)


if __name__ == "__main__":
    data_df = pd.read_csv('./queried_data.csv')
    print(data_df.head())
    X = data_df.loc[:,
        ~data_df.columns.isin(['Eg (eV)', 'Computed band gap', 'Unnamed: 0', 'formula', 'composition', 'composition_oxid'])]
    X = np.array(X)

    Y = np.array(data_df['Eg (eV)'])
    Y_0 = np.array(data_df['Computed band gap small'])
    valid_index = np.sum(np.isnan(X), axis=1) == 0

    X = X[valid_index, :]
    Y = Y[valid_index]
    Y_0 = Y_0[valid_index]
    valid_feature = X.var(0) == 0
    X = X[:, ~valid_feature]
    X = (X - X.mean(0)[np.newaxis, :])/np.sqrt(X.var(0)[np.newaxis, :])

    print(X.shape)

    Used_to_train = np.random.random(X.shape[0]) < 1
    print(Used_to_train)
    X_train = X[Used_to_train, :]
    Y_train = Y[Used_to_train]
    Y_0_train = Y_0[Used_to_train]

    X_test = X[~Used_to_train, :]
    Y_test = Y[~Used_to_train]
    Y_0_test = Y_0[~Used_to_train]

    Used_to_validate = np.random.random(X_train.shape[0]) < 0.2
    X_validate = X_train[Used_to_validate, :]
    Y_validate = Y_train[Used_to_validate]

    X_train = X_train[~Used_to_validate, :]
    Y_train = Y_train[~Used_to_validate]

    X_train = torch.tensor(X_train, dtype=torch.float)
    Y_train = torch.tensor(Y_train, dtype=torch.float)
    X_validate = torch.tensor(X_validate, dtype=torch.float)
    Y_validate = torch.tensor(Y_validate, dtype=torch.float)


    hidden_dim = 64
    extracted_dim = 3
    feature_extractor = FeatureExtractor(X_train.shape[-1], extracted_dim)
    model = LinearClassifier(feature_extractor, 3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.5)
    loss_func = nn.MSELoss()
    model.train()
    loss_list = []
    vali_list = []
    for i in range(5000):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_func(pred, Y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())
        pred_validate = model(X_validate)
        loss_validate = loss_func(pred, Y_validate)
        vali_list.append(loss_validate.item())
        if i % 100 == 0:
            print(i, loss.item(), optimizer.param_groups[0]['lr'])
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.plot(np.arange(len(loss_list)), vali_list)
    plt.show()
    with torch.no_grad():
        plt.scatter(Y_train, pred)
        plt.show()
    # regr = svm.SVR(kernel='rbf', gamma=2)
    # regr.fit(X_train, Y_train)
    # print(regr.score(X_train, Y_train))
    # print(regr.score(X_test, Y_test))
    # plt.scatter(Y_train, regr.predict(X_train))
    # plt.show()
    #
    # plt.scatter(Y_test, regr.predict(X_test))
    # plt.show()
    #
    # regr = svm.SVR(kernel='rbf', gamma=2)
    # regr.fit(Y_0_train[:, np.newaxis], Y_train)
    # print(regr.score(Y_0_train[:, np.newaxis], Y_train))
    # print(regr.score(Y_0_test[:, np.newaxis], Y_test))
    # plt.scatter(Y_train, regr.predict(Y_0_train[:, np.newaxis]))
    # plt.show()
    #
    # plt.scatter(Y_test, regr.predict(Y_0_test[:, np.newaxis]))
    # plt.show()