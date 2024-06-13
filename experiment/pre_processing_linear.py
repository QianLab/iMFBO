import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, X):
        # l1_out = self.act(self.l1(X))
        # l2_out = self.act(self.l2(l1_out))
        # out = self.BN(l2_out)
        out = self.l1(X)
        return out
#
# class LinearClassifier(nn.Module):
#     def __init__(self, extractor, ex_dim):
#         super().__init__()
#         self.extractor = extractor
#         self.l = nn.Linear(ex_dim, 1)
#
#     def forward(self, X):
#         extracted = self.extractor(X)
#         return self.l(extracted)

if __name__ == "__main__":
    RETRAIN = True
    # RETRAIN = False
    df = pd.read_csv('queried_data.csv')
    print("Original Data Shape:", df.shape)
    print(df.head(100))
    df = df.drop_duplicates(subset=['formula'])
    print("After Drop Duplicated Data Shape:", df.shape)
    df2 = df.loc[df['Computed band gap small'].isna(),]
    df = df.loc[~df['Computed band gap small'].isna(),]

    Y = np.array(df['Eg (eV)'])
    Y_0 = np.array(df['Computed band gap small'])
    X = df.loc[:,
        ~df.columns.isin(['Eg (eV)', 'Computed band gap', 'Unnamed: 0', 'formula', 'composition', 'composition_oxid'])]
    X_np = np.array(X)[:, :-6]


    Y_train = np.array(df2['Eg (eV)'])
    Y_train_0 = np.array(df2['Computed band gap small'])
    X_train = df2.loc[:,
        ~df2.columns.isin(['Eg (eV)', 'Computed band gap', 'Unnamed: 0', 'formula', 'composition', 'composition_oxid'])]
    X_train_np = np.array(X_train)[:, :-6]

    valid_index = np.sum(np.isnan(X_np), axis=1) == 0
    valid_index_train = np.sum(np.isnan(X_train_np), axis=1) == 0
    X_train_np = X_train_np[valid_index_train]
    X_np = X_np[valid_index]
    Y_train = Y_train[valid_index_train]
    Y = Y[valid_index]
    Y_train_0 = Y_train_0[valid_index_train]
    Y_0 = Y_0[valid_index]
    print(X_train_np.shape, X_np.shape)
    print(np.sum(np.isnan(X_np)), np.sum(np.isnan(X_train_np)))

    if RETRAIN == True:
        torch.manual_seed(42)

        Y_train = torch.tensor(Y_train).float().unsqueeze(-1)
        Y_train_0 = torch.tensor(Y_train_0).float().unsqueeze(-1)
        # print(X_np.shape, X.shape, Y.shape, Y_1.shape)
        X_train = torch.tensor(X_train_np).float()

        Y = torch.tensor(Y).float().unsqueeze(-1)
        Y_0 = torch.tensor(Y_0).float().unsqueeze(-1)
        # print(X_np.shape, X.shape, Y.shape, Y_1.shape)
        X = torch.tensor(X_np).float()

        # feature_extractor = FeatureExtractor(X_train.shape[-1], extracted_dim)
        model = Linear(X_train.shape[-1], 1)
        optimizer = optim.Adam(model.parameters(), lr=.01)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60000, gamma=0.5)
        loss_func = nn.MSELoss()
        model.train()
        loss_list = []
        for i in range(200):
            optimizer.zero_grad()
            pred = model(X_train)
            # print(torch.sum(pred.isnan()))
            loss = loss_func(pred, Y_train)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            loss_list.append(loss.item())
            if i % 100 == 0:
                print(i, loss.item(), optimizer.param_groups[0]['lr'])
                with torch.no_grad():

                    plt.scatter(Y_train, pred, alpha=0.2)
                    plt.scatter(Y, model(X), alpha=0.2)
                    plt.plot([0, 13], [0, 13])
                    plt.show()
                    Y_1 = model(X)
        model.train()
    #
        with torch.no_grad():
            plt.plot(np.arange(len(loss_list)), loss_list)
            # plt.ylim([0.2, .3])
            plt.show()
    #
            plt.scatter(Y_train, pred)
            plt.plot([0, 13], [0, 13])
            plt.show()
    #
            plt.scatter(Y, model(X))
            plt.plot([0, 13], [0, 13])
            plt.show()
            Y_1 = model(X)

    #     torch.save(feature_extractor.state_dict(), './feature_extractor.md')
    #     torch.save(X_train, './X_train.ts')
    #     torch.save(Y_train, './Y_train.ts')
    #     torch.save(Y_train_0, './Y_0_train.ts')
    # #
    #     torch.save(X, './X_test.ts')
    #     torch.save(Y, './Y_test.ts')
    #     torch.save(Y_0, './Y_0_test.ts')
        torch.save(Y_1, './Y_2_test.ts')
    #
    # with torch.no_grad():
    #     extracted_dim = 2
    #     feature_extractor = FeatureExtractor(X_np.shape[-1], extracted_dim)
    #     feature_extractor.load_state_dict(torch.load('./feature_extractor.md'))
    #     X_test = torch.load('./X_test.ts')
    #     Y_test = torch.load('./Y_test.ts')
    #     Y_0_test = torch.load('./Y_0_test.ts')
    #     X_train = torch.load('./X_train.ts')
    #     Y_1_test = torch.load('./Y_1_test.ts')
    #     Z_test = feature_extractor(X_test)
    #     Y_train = torch.load('./Y_train.ts')
    #     Y_0_train = torch.load('./Y_0_train.ts')
    #     Z_train = feature_extractor(X_train)
    #     plt.scatter(Z_test[:, 0], Z_test[:, 1])
    #     plt.show()
    # #     # plt.scatter(Z_test[:, 2], Z_test[:, 1])
    # #     # plt.show()
    # # Z_test = Z_test[:, [0, 2]]
    # Z_min, _ = torch.min(Z_test, dim=0)
    # Z_max, _ = torch.max(Z_test, dim=0)
    # Z_test = (Z_test - Z_min.unsqueeze(0))/(Z_max.unsqueeze(0) - Z_min.unsqueeze(0))
    # # torch.save(Z_test, './Z_test.ts')
    # print(X_test.shape, X_train.shape, Y_test.shape, Y_train.shape, Y_0_test.shape, Y_0_train.shape, Y_1_test.shape)
    # print(Z_test[:, 0].min(), Z_test[:, 0].max())
    # print(Z_test[:, 1].min(), Z_test[:, 1].max())
    # Z_test_np = np.array(Z_test)
    # Y_test_np = np.array(Y_test)
    # Y_0_test_np = np.array(Y_0_test)
    #
    # Z_train_np = np.array(Z_train[:, [0, 2]])
    # Y_train_np = np.array(Y_train)
    # Y_0_train_np = np.array(Y_0_train)
    #
    #
    # linear = LinearRegression()
    # linear.fit(Z_test_np, Y_test_np)
    # print(linear.score(Z_test_np, Y_test_np))
    #
    # linear2 = LinearRegression()
    # linear2.fit(Y_0_train_np, Y_train_np)
    # print(linear2.score(Y_0_test_np, Y_test_np))
    #
    # plt.scatter(Y_test, linear2.predict(Y_0_test_np))
    # plt.plot([0, 13], [0, 13])
    # plt.show()