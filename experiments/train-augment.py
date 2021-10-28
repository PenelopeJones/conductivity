# -*- coding: utf-8 -*-
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

    def predict(self, x, ns):

        local_pred = self.forward(x)
        local_pred = torch.split(local_pred, ns)
        pred = torch.stack([torch.mean(chunk) for chunk in local_pred]).reshape(-1, self.out_dim)

        return pred.float()

def maximum_r2(y, y_err, n_samples=10, file=None):
    r2s = []
    for seed in range(n_samples):
        np.random.seed(seed)
        y_sampled = np.random.normal(y, y_err)
        r2_sampled = r2_score(y_sampled, y)
        r2s.append(r2_sampled)
    r2s = np.array(r2s)
    print('Max R2 = {:.2f} +- {:.2f}'.format(np.mean(r2s), np.std(r2s)))
    if file is not None:
        file.write('Max R2 = {:.2f} +- {:.2f}'.format(np.mean(r2s), np.std(r2s)))
    return

def aggregate_metrics(suffix, f, pts, experiment_name, n_splits, n_ensembles):
    trues = []
    preds = []
    preds_err = []
    for ns in range(n_splits):
        preds_split = []
        for ne in range(n_ensembles):
            try:
                true = np.load(pts + 'predictions/' + '{}{}_{}_y_{}.npy'.format(experiment_name, ns, ne, suffix))
                pred = np.load(pts + 'predictions/' + '{}{}_{}_pred_{}.npy'.format(experiment_name, ns, ne, suffix))
                preds_split.append(pred)
            except:
                continue
        if len(preds_split) > 0:
            trues.append(true)
            preds_split = np.vstack(preds_split)
            preds_split = preds_split.reshape((-1, pred.shape[0]))
            preds.append(np.mean(preds_split, axis=0))
            preds_err.append(np.std(preds_split, axis=0))
        else:
            continue
    trues = np.array(trues).reshape(-1)
    preds = np.array(preds).reshape(-1)

    f.write('\n Aggregate RMSE ({}): {:.6f}\t R2 ({}): {:.2f}'.format(suffix, np.sqrt(mean_squared_error(trues, preds)),
                                                            suffix, r2_score(trues, preds)))
    f.flush()
    return

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

def train_valid_split(data, n_split, seed=5, fraction_valid=0.2):
    np.random.seed(seed)
    (y, y_err, concs, lbs) = data

    n = concs.shape[0]
    n_valid = int(fraction_valid*n)
    idx = np.random.permutation(n)
    idx_te = idx[n_split*n_valid:(n_split+1)*n_valid]
    idx_tr = np.delete(idx, slice(n_split*n_valid, (n_split+1)*n_valid))
    print(idx_te)
    print(idx_tr)

    data_valid = (y[idx_te], y_err[idx_te], concs[idx_te], lbs[idx_te])
    data_train = (y[idx_tr], y_err[idx_tr], concs[idx_tr], lbs[idx_tr])

    return data_valid, data_train


def data_loader_augmented(concs_train, lbs_train, concs_valid, lbs_valid, ptd, n_augments=10):
    mus = []
    ns_train = []
    ns_train_na = []
    vars = []

    for i in range(concs_train.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        n_batch = x.shape[0]//n_augments
        x = x[0:n_augments*n_batch]
        ns_train.extend([n_batch,]*n_augments)
        ns_train_na.append(n_batch*n_augments)
        mus.append(np.mean(x, axis=0))
        vars.append(np.var(x, axis=0))

        if i == 0:
            X_train = x

        else:
            X_train = np.concatenate((X_train, x), axis=0)

    ns_na = np.hstack(ns_train_na)
    mus = np.vstack(mus)
    vars = np.vstack(vars)

    mu = np.sum(ns_na*mus.T, axis=1) / np.sum(ns_na)
    std = np.sqrt(np.sum((ns_na*(vars.T + (mus.T - mu.reshape(-1, 1))**2)), axis=1) / np.sum(ns_na))

    ns_valid = []

    for i in range(concs_valid.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_valid[i], lbs_valid[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        n_batch = x.shape[0]//n_augments
        x = x[0:n_augments*n_batch]
        ns_valid.extend([n_batch,]*n_augments)
        if i == 0:
            X_valid = x
        else:
            X_valid = np.concatenate((X_valid, x), axis=0)

    X_train = torch.tensor(((X_train - mu) / std), dtype=torch.float32)
    X_valid = torch.tensor(((X_valid - mu) / std), dtype=torch.float32)

    print('Size of training data = {}'.format(sys.getsizeof(X_train)))
    print('Size of validation data = {}'.format(sys.getsizeof(X_valid)))
    return X_train, ns_train, X_valid, ns_valid, mu, std

def data_loader(concs_train, lbs_train, concs_valid, lbs_valid, ptd):
    mus = []
    ns_train = []
    vars = []
    idxs_train = [] # idxs[n] gives start row of the nth system in X i.e. x_n = X[idx[n]:idx[n+1], :]

    idx = 0

    for i in range(concs_train.shape[0]):
        idxs_train.append(idx)
        ptf = ptd + 'X_{}_{}_soap'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        mus.append(np.mean(x, axis=0))
        vars.append(np.var(x, axis=0))
        ns_train.append(x.shape[0])
        idx += x.shape[0]

        if i == 0:
            X_train = x

        else:
            X_train = np.concatenate((X_train, x), axis=0)

    mus = np.vstack(mus)
    ns = np.hstack(ns_train)
    vars = np.vstack(vars)
    idxs_train = np.hstack(idxs_train)

    mu = np.sum(ns*mus.T, axis=1) / np.sum(ns)
    std = np.sqrt(np.sum((ns*(vars.T + (mus.T - mu.reshape(-1, 1))**2)), axis=1) / np.sum(ns))

    idxs_valid = [] # idxs[n] gives start row of the nth system in X i.e. x_n = X[idx[n]:idx[n+1], :]
    ns_valid = []
    idx = 0

    for i in range(concs_valid.shape[0]):
        idxs_valid.append(idx)
        ptf = ptd + 'X_{}_{}_soap'.format(concs_valid[i], lbs_valid[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        ns_valid.append(x.shape[0])
        idx += x.shape[0]

        if i == 0:
            X_valid = x

        else:
            X_valid = np.concatenate((X_valid, x), axis=0)

    idxs_valid = np.hstack(idxs_valid)
    X_train = torch.tensor(((X_train - mu) / std), dtype=torch.float32)
    X_valid = torch.tensor(((X_valid - mu) / std), dtype=torch.float32)

    print('Size of training data = {}'.format(sys.getsizeof(X_train)))
    print('Size of validation data = {}'.format(sys.getsizeof(X_valid)))
    return X_train, ns_train, X_valid, ns_valid, mu, std

def x_scaler(concs, lbs, ptd):
    mus = []
    ns = []
    vars = []
    for i in range(concs.shape[0]):
        conc = concs[i]
        lb = lbs[i]
        ptf = ptd + 'X_{}_{}_soap'.format(conc, lb).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        mu_i = np.mean(x, axis=0)
        var_i = np.var(x, axis=0)
        n_i = x.shape[0]
        mus.append(mu_i)
        vars.append(var_i)
        ns.append(n_i)
    mus = np.vstack(mus)
    ns = np.hstack(ns)
    vars = np.vstack(vars)
    mu = np.sum(ns*mus.T, axis=1) / np.sum(ns)
    var = np.sum((ns*(vars.T + (mus.T - mu.reshape(-1, 1))**2)), axis=1) / np.sum(ns)
    std = np.sqrt(var)
    return mu, std

def main(args):

    t0 = time.time()

    ptd = args.ptd
    ptx = ptd + 'processed/'

    n_splits = args.n_splits
    n_ensembles = args.n_ensembles
    n_augments = args.n_augments
    hidden_dims = args.hidden_dims
    run_id = args.run_id
    n_split = args.n_split
    experiment_name = args.experiment_name
    lr = args.lr
    epochs = args.epochs
    print_freq = args.print_freq
    standardise = True
    save_predictions = True
    save_models = True
    log_name = '{}_log_{}_{}.txt'.format(experiment_name, n_split, run_id)
    args_name = '{}_args.txt'.format(experiment_name)
    pts = '../results/{}/'.format(experiment_name)

    # Load ion positions
    if (n_split == 0) and (run_id == 0):
        with open(pts + args_name, 'w') as f:
            f.write(str(args))

    y = np.load(ptd + 'molar_conductivities.npy')
    y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    y_err = y_err.reshape(-1, 1)
    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load(ptd + 'bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)

    data = (y, y_err, concs, lbs)
    data_train_valid, data_test = train_test_split(data, seed=10)
    data_valid, data_train = train_valid_split(data_train_valid, n_split=n_split)
    (true_train, true_err_train, concs_train, lbs_train) = data_train
    (true_valid, true_err_valid, concs_valid, lbs_valid) = data_valid
    print('n_split {}: Loaded data train / valid sets'.format(n_split))

    f = open(pts + log_name, 'w')
    f.write('\nLoading data for split {}...'.format(n_split))
    f.flush()

    # Load training and validation data
    X_train, ns_train, X_valid, ns_valid, mu_x, std_x = data_loader_augmented(concs_train, lbs_train, concs_valid, lbs_valid, ptx, n_augments=n_augments)

    # Scale y data
    sc_y = StandardScaler()
    y_train = torch.tensor(sc_y.fit_transform(true_train), dtype=torch.float32)
    y_err_train = torch.tensor(true_err_train / sc_y.scale_, dtype=torch.float32)
    y_valid = torch.tensor(sc_y.transform(true_valid), dtype=torch.float32)
    y_err_valid = torch.tensor(true_err_valid / sc_y.scale_, dtype=torch.float32)

    # Augment y data if needed
    y_train = torch.repeat_interleave(y_train, n_augments).reshape(-1, 1)
    y_err_train = torch.repeat_interleave(y_err_train, n_augments).reshape(-1, 1)
    y_valid = torch.repeat_interleave(y_valid, n_augments).reshape(-1, 1)
    y_err_valid = torch.repeat_interleave(y_err_valid, n_augments).reshape(-1, 1)

    f.write('\nLoaded data for split {}... Load Time: {:.1f}'.format(n_split, time.time() - t0))
    f.flush()
    t0 = time.time()
    # Pre-train using the full dataset
    model = VanillaNN(in_dim=mu_x.shape[0], out_dim=1, hidden_dims=hidden_dims)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    running_loss = 0

    f.write('\nTraining... Set Up Time: {:.1f}'.format(time.time() - t0))
    f.flush()

    t0 = time.time()

    rmse_best = 100000

    for epoch in range(epochs):
        optimiser.zero_grad()
        pred = model.predict(X_train, ns_train)
        loss = criterion(pred, y_train)
        running_loss += loss

        if epoch % print_freq == 0:
            t1 = time.time()
            # Make prediction
            pred_train = model.predict(X_train, ns_train)
            pred_valid = model.predict(X_valid, ns_valid)
            pred_train = sc_y.inverse_transform(pred_train.detach().numpy())
            pred_valid = sc_y.inverse_transform(pred_valid.detach().numpy())

            f.write('\nEpoch {}\tLoss: {:.4f}\t Train Time: {:.1f}\t Predict Time: {:.1f}'.format(epoch, running_loss / print_freq, t1-t0, time.time()-t1))
            f.flush()
            running_loss = 0
            t0 = time.time()

            rmse_valid = np.sqrt(mean_squared_error(np.repeat(true_valid, n_augments), pred_valid))

            f.write('\nSaving RMSE (valid): {}'.format(rmse_valid))
            f.flush()
            np.save(pts + 'predictions/' + '{}{}_{}_pred_train.npy'.format(experiment_name, n_split, run_id), pred_train)
            np.save(pts + 'predictions/' + '{}{}_{}_y_train.npy'.format(experiment_name, n_split, run_id), np.repeat(true_train, n_augments))
            np.save(pts + 'predictions/' + '{}{}_{}_y_err_train.npy'.format(experiment_name, n_split, run_id), np.repeat(true_err_train, n_augments))
            np.save(pts + 'predictions/' + '{}{}_{}_concs_train.npy'.format(experiment_name, n_split, run_id), np.repeat(concs_train, n_augments))
            np.save(pts + 'predictions/' + '{}{}_{}_lbs_tr.npy'.format(experiment_name, n_split, run_id), np.repeat(lbs_train, n_augments))
            np.save(pts + 'predictions/' + '{}{}_{}_pred_valid.npy'.format(experiment_name, n_split, run_id), pred_valid)
            np.save(pts + 'predictions/' + '{}{}_{}_y_valid.npy'.format(experiment_name, n_split, run_id), np.repeat(true_valid, n_augments))
            np.save(pts + 'predictions/' + '{}{}_{}_y_err_valid.npy'.format(experiment_name, n_split, run_id), np.repeat(true_err_valid, n_augments))
            np.save(pts + 'predictions/' + '{}{}_{}_concs_valid.npy'.format(experiment_name, n_split, run_id), np.repeat(concs_valid, n_augments))
            np.save(pts + 'predictions/' + '{}{}_{}_lbs_valid.npy'.format(experiment_name, n_split, run_id), np.repeat(lbs_valid, n_augments))

            if save_models:
                torch.save(model.state_dict(), pts + 'models/' + 'model{}{}_{}.pkl'.format(experiment_name, n_split, run_id))

            if (n_split == 0) and (run_id == 0):
                aggregate_metrics('train', f, pts, experiment_name, n_splits, n_ensembles)
                aggregate_metrics('valid', f, pts, experiment_name, n_splits, n_ensembles)

        loss.backward()
        optimiser.step()

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--experiment_name', type=str, default='ENSEMBLE_A',
                        help='Name of experiment.')
    parser.add_argument('--hidden_dims', nargs='+', type=int,
                        default=[50, ],
                        help='Dimensionality of network hidden layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Number of systems to use in training.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs.')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency.')
    parser.add_argument('--run_id', type=int, default=0,
                        help='Run ID.')
    parser.add_argument('--n_ensembles', type=int, default=5,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_augments', type=int, default=10, help='Factor by which to augment the training data.')
    parser.add_argument('--n_split', type=int, default=0,
                        help='Split number.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')

    args = parser.parse_args()

    main(args)
