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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

    def predict(self, data, mu_x=None, std_x=None, n_systems=5, n_samples=20000, ptd='../data/processed/'):
        X_batch, y_batch, y_batch_err = sample_batch(data, mu_x, std_x, n_systems, n_samples, ptd)
        local_pred_batch = self.forward(X_batch)
        local_pred_batch = local_pred_batch.reshape(n_systems, n_samples, 1)
        # y is the mean over all samples
        pred_batch = torch.mean(local_pred_batch, dim=1)
        return pred_batch, y_batch, y_batch_err

def system_subsample(conc, lb, n_samples, ptd):
    ptf = ptd + 'X_{}_{}'.format(conc, lb).replace('.', '-') + '.npy'
    x = np.load(ptf)
    idx = np.random.choice(x.shape[0], size=n_samples, replace=False)
    return x[idx]

def sample_batch(data, mu_x=None, std_x=None, n_systems=5, n_samples=5000, ptd='../data/processed/'):
    (y, y_err, concs, lbs) = data
    nt = concs.shape[0]
    assert concs.shape[0] == lbs.shape[0] == y.shape[0] == y_err.shape[0]
    ids = np.random.choice(nt, size=n_systems, replace=False)
    y_batch = torch.tensor(y[ids]).float()
    y_batch_err = torch.tensor(y_err[ids]).float()
    X_batch = []
    for i in range(n_systems):
        X_batch.append(system_subsample(concs[ids[i]], lbs[ids[i]], n_samples, ptd))
    X_batch = np.vstack(X_batch)
    if mu_x is not None:
        X_batch = (X_batch - mu_x) / std_x
    X_batch = torch.tensor(X_batch).float()
    return X_batch, y_batch, y_batch_err

def train_test_split(data, seed=10, fraction_test=0.1):
    np.random.seed(seed)
    (y, y_err, concs, lbs) = data
    n = concs.shape[0]
    idx = np.random.permutation(n)
    idx_te = idx[0:int(fraction_test*n)]
    idx_tr = idx[int(fraction_test*n):]

    data_test = (y[idx_te], y_err[idx_te], concs[idx_te], lbs[idx_te])
    data_train = (y[idx_tr], y_err[idx_tr], concs[idx_tr], lbs[idx_tr])

    return data_train, data_test

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
    hidden_dims = [20,20]
    n_systems = args.n_systems
    n_samples = 5000
    lr = 0.001
    epochs = 3000
    print_freq = 200
    standardise = False


    experiment_name = 'A'
    log_name = 'log_{}.txt'.format(experiment_name)
    dir_to_save = 'results/{}/'.format(experiment_name)
    import os
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
        os.makedirs(dir_to_save + 'predictions/')
        os.makedirs(dir_to_save + 'models/')

    save_predictions = True
    save_models = True
    # Load ion positions
    y = np.load(ptd + 'molar_conductivities.npy')
    y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    y_err = y_err.reshape(-1, 1)
    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load(ptd + 'bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)

    #y = y[0:99]
    #y_err = y_err[0:99]
    #concs = concs[0:99]
    #lbs = lbs[0:99]

    data = (y, y_err, concs, lbs)

    data_train_valid, data_test = train_test_split(data, seed=10)

    r2s_tr = []
    r2s_val = []
    rmses_tr = []
    rmses_val = []

    for seed in range(5):
        print('\nTraining... seed {}'.format(seed))
        data_train, data_valid = train_test_split(data_train_valid, seed=seed)
        n_train = data_train[0].shape[0]
        n_valid = data_valid[0].shape[0]
        n_test = data_test[0].shape[0]
        (y_train, y_err_train, concs_train, lbs_train) = data_train
        (y_val, y_err_val, concs_val, lbs_val) = data_valid

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

        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)
        y_val = sc_y.transform(y_val)

        data_train = (y_train, y_err_train, concs_train, lbs_train)
        data_valid = (y_val, y_err_val, concs_val, lbs_val)

        # Pre-train using the full dataset
        model = VanillaNN(in_dim=in_dim, out_dim=1, hidden_dims=hidden_dims)
        optimiser = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        running_loss = 0

        for epoch in range(epochs):
            optimiser.zero_grad()

            X_batch, y_batch, y_batch_err = sample_batch(data_train, mu_x, std_x, n_systems, n_samples, ptx)

            local_pred_batch = model(X_batch)

            local_pred_batch = local_pred_batch.reshape(n_systems, n_samples, 1)

            # y is the mean over all samples
            pred_batch = torch.mean(local_pred_batch, dim=1)

            loss = criterion(pred_batch, y_batch)
            running_loss += loss
            if epoch % print_freq == 0:
                print('Epoch {}\tLoss: {}'.format(epoch, running_loss / print_freq))
                running_loss = 0
                pred_tr, y_tr, y_err_tr = model.predict(data_train, n_systems=n_train)
                pred_val, y_val, y_err_val = model.predict(data_valid, n_systems=n_valid)
                pred_tr = sc_y.inverse_transform(pred_tr.detach().numpy())
                y_tr = sc_y.inverse_transform(y_tr.detach().numpy())
                pred_val = sc_y.inverse_transform(pred_val.detach().numpy())
                y_val = sc_y.inverse_transform(y_val.detach().numpy())
                print('Train RMSE: {:.6f}\t Valid RMSE: {:.6f}\t Train R2: {:.2f}\t Valid R2: {:.2f}'.format(np.sqrt(mean_squared_error(y_tr, pred_tr)),
                                                                                             np.sqrt(mean_squared_error(y_val, pred_val)),
                                                                                             r2_score(y_tr, pred_tr),
                                                                                             r2_score(y_val, pred_val)))

            loss.backward()
            optimiser.step()

        if save_predictions:
            np.save(dir_to_save + 'predictions/' + '{}{}_pred_tr.npy'.format(experiment_name, seed), pred_tr)
            np.save(dir_to_save + 'predictions/' + '{}{}_y_tr.npy'.format(experiment_name, seed), y_tr)
            np.save(dir_to_save + 'predictions/' + '{}{}_y_err_tr.npy'.format(experiment_name, seed), y_err_tr)
            np.save(dir_to_save + 'predictions/' + '{}{}_pred_val.npy'.format(experiment_name, seed), pred_val)
            np.save(dir_to_save + 'predictions/' + '{}{}_y_val.npy'.format(experiment_name, seed), y_val)
            np.save(dir_to_save + 'predictions/' + '{}{}_y_err_val.npy'.format(experiment_name, seed), y_err_val)

        if save_models:
            torch.save(model.state_dict(), dir_to_save + 'models/' + 'model{}{}.pkl'.format(experiment_name, seed))

        r2s_tr.append(r2_score(y_tr, pred_tr))
        rmses_tr.append(np.sqrt(mean_squared_error(y_tr, pred_tr)))
        r2s_val.append(r2_score(y_val, pred_val))
        rmses_val.append(np.sqrt(mean_squared_error(y_val, pred_val)))

    r2s_tr = np.array(r2s_tr)
    rmses_tr = np.array(rmses_tr)
    r2s_val = np.array(r2s_val)
    rmses_val = np.array(rmses_val)
    print('\nRMSE (train):{:.6f}\tR2 (train):{:.2f}'.format(np.median(rmses_tr), np.median(r2s_tr)))
    print('RMSE (val):{:.6f}\tR2 (val):{:.2f}'.format(np.median(rmses_val), np.median(r2s_val)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../results/models/',
                        help='Path to directory containing data.')
    parser.add_argument('--n_systems', type=int, default=10,
                        help='Number of systems to use in training.')

    args = parser.parse_args()

    main(args)
