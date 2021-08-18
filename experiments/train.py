import os
import argparse
import time
import sys
sys.path.append('../')

import numpy as np
from scipy.stats import norm, invwishart, multivariate_normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from collections import Counter

import pdb

class VanillaNN(nn.Module):
    """
    A `vanilla` neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dims, non_linearity=F.relu):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the output.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.non_linearity = non_linearity

        self.layers = nn.ModuleList()

        for dim in range(len(hidden_dims) + 1):
            if dim == 0:
                self.layers.append(nn.Linear(self.in_dim, hidden_dims[dim]))
            elif dim == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[-1], self.out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[dim - 1],
                                             hidden_dims[dim]))

    def forward(self, x):
        """
        :param self:
        :param x: (torch tensor, (batch_size, in_dim)) Input to the network.
        :return: (torch tensor, (batch_size, out_dim)) Output of the network.
        """
        assert len(x.shape) == 2, 'Input must be of shape [batch_size, in_dim].'

        for i in range(len(self.layers) - 1):
            x = self.non_linearity(self.layers[i](x))

        return self.layers[-1](x)

def system_subsample(conc, lb, n_samples, ptd):
    ptf = ptd + 'X_{}_{}'.format(conc, lb).replace('.', '-') + '.npy'
    x = np.load(ptf)
    return np.random.choice(x, size=n_samples, replace=False)

def sample_batch(concs, lbs, y, y_err, mu_x=None, std_x=None, n_systems=5, n_samples=5000, ptd='data/'):
    nt = concs.shape[0]
    pdb.set_trace()
    assert concs.shape[0] == lbs.shape[0] == y.shape[0] == y_err.shape[0]
    pdb.set_trace()
    ids = np.random.choice(nt, size=n_systems, replace=False)

    y_batch = torch.tensor(y[ids]).float()
    y_batch_err = torch.tensor(y_err[ids]).float()
    X_batch = []

    for i in range(n_systems):
        X_batch.append(system_subsample(concs[i], lbs[i], n_samples, ptd+'processed/'))
    pdb.set_trace()
    X_batch = np.vstack(X_batch)
    if mu_x is not None:
        X_batch = (X_batch - mu_x) / std_x
    X_batch = torch.tensor(X_batch).float()
    return X_batch, y_batch, y_batch_err

def train_test_split(y, y_err, concs, lbs, seed=10, fraction_valid=0.1, fraction_test=0.1):
    np.random.seed(seed)
    n = concs.shape[0]
    idx = np.random.permutation(n)
    idx_te = idx[0:int(fraction_test*n)]
    idx_val = idx[int(fraction_test*n):(int(fraction_test*n)+int(fraction_valid*n))]
    idx_tr = idx[(int(fraction_test*n)+int(fraction_valid*n)):]

    data_test = (y[idx_te], y_err[idx_te], concs[idx_te], lbs[idx_te])
    data_val = (y[idx_val], y_err[idx_val], concs[idx_val], lbs[idx_val])
    data_train = (y[idx_tr], y_err[idx_tr], concs[idx_tr], lbs[idx_tr])

    return data_train, data_val, data_test

def x_scaler(concs, lbs, ptd):
    X = []
    for i in range(concs.shape[0]):
        conc = concs[i]
        lb = lbs[i]
        ptf = ptd + 'X_{}_{}'.format(conc, lb).replace('.', '-') + '.npy'
        x = np.load(ptf)
        print('Conc {}\t lB {}:\t N = {}'.format(conc, lb, x.shape[0]))
        X.append(x)
    X = np.vstack(X)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    print(mu)
    print(std)
    return mu, std

def main(args):
    ptd = args.ptd
    ptx = ptd + 'processed/'
    # Model parameters
    hidden_dims = [8, 8]
    n_systems = 500
    n_samples = 2500
    lr = 0.01
    epochs = 5000
    print_freq = 200
    seed = 10
    standardise = False
    # Load ion positions
    y = np.load(ptd + 'molar_conductivities.npy')
    y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    y_err = y_err.reshape(-1, 1)
    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load(ptd + 'bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)
    data_train, data_valid, data_test = train_test_split(y, y_err, concs, lbs, seed=seed)
    (y_train, y_err_train, concs_train, lbs_train) = data_train

    # Get scalers, and scale the y_data
    if standardise:
        mu_x, std_x = x_scaler(concs_train, lbs_train, ptx)
        in_dim = mu_x.shape[0]
    else:
        mu_x = None
        std_x = None
        ptf = ptx + 'X_{}_{}'.format(concs[0], lbs[0]).replace('.', '-') + '.npy'
        x = np.load(ptf)
        in_dim = x.shape[-1]

    pdb.set_trace()


    sc_y = StandardScaler()
    sc_y.fit_transform(y_train)

    pdb.set_trace()

    # Pre-train using the full dataset
    model = VanillaNN(in_dim=in_dim, out_dim=1, hidden_dims=hidden_dims)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    pdb.set_trace()
    running_loss = 0

    for epoch in range(epochs):
        optimiser.zero_grad()

        X_batch, y_batch, y_batch_err = sample_batch(concs, lbs, y, y_err, mu_x, std_x, n_systems, n_samples, ptx)
        local_pred_batch = model(x_batch)

        local_pred_batch = local_pred_batch.reshape(n_systems, n_samples, -1)

        # y is the mean over all samples
        pred_batch = torch.mean(local_pred_batch, dim=1).reshape(n_systems, -1)

        loss = criterion(pred_batch, y_batch)

        running_loss += loss
        if epoch % print_freq == 0:
            print('Epoch {}\tLoss: {}'.format(epoch, running_loss / print_freq))
            running_loss = 0
            """
            pred_train = model(X_train).float().detach().numpy()
            pred_test = model(X_test).float().detach().numpy()
            pred_train = sc_y.inverse_transform(pred_train)
            pred_test = sc_y.inverse_transform(pred_test)
            true_train = sc_y.inverse_transform(y_train.float().detach().numpy())
            true_test = y_test

            print('Train RMSE: {:.2f}\t Test RMSE: {:.2f}\t Train R2: {:.2f}\t Test R2: {:.2f}\n'.format(np.sqrt(mean_squared_error(true_train, pred_train)),
                                                                                         np.sqrt(mean_squared_error(true_test, pred_test)),
                                                                                         r2_score(true_train, pred_train),
                                                                                         r2_score(true_test, pred_test)))
            """
        loss.backward()
        optimiser.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../results/models/',
                        help='Path to directory containing data.')

    args = parser.parse_args()

    main(args)
