# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


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

def data_loader_full(concs_train, lbs_train, concs_valid, lbs_valid, concs_test, lbs_test, ptd):
    mus = []
    ns_train = []
    vars = []

    for i in range(concs_train.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        mus.append(np.mean(x, axis=0))
        vars.append(np.var(x, axis=0))
        ns_train.append(x.shape[0])

        if i == 0:
            X_train = x

        else:
            X_train = np.concatenate((X_train, x), axis=0)

    mus = np.vstack(mus)
    ns = np.hstack(ns_train)
    vars = np.vstack(vars)

    mu = np.sum(ns*mus.T, axis=1) / np.sum(ns)
    std = np.sqrt(np.sum((ns*(vars.T + (mus.T - mu.reshape(-1, 1))**2)), axis=1) / np.sum(ns))

    ns_valid = []

    for i in range(concs_valid.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_valid[i], lbs_valid[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        ns_valid.append(x.shape[0])

        if i == 0:
            X_valid = x

        else:
            X_valid = np.concatenate((X_valid, x), axis=0)

    ns_test = []

    for i in range(concs_test.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_test[i], lbs_test[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        ns_test.append(x.shape[0])

        if i == 0:
            X_test = x

        else:
            X_test = np.concatenate((X_test, x), axis=0)

    X_train = torch.tensor(((X_train - mu) / std), dtype=torch.float32)
    X_valid = torch.tensor(((X_valid - mu) / std), dtype=torch.float32)
    X_test = torch.tensor(((X_test - mu) / std), dtype=torch.float32)

    print('Size of training data = {}'.format(sys.getsizeof(X_train)))
    print('Size of validation data = {}'.format(sys.getsizeof(X_valid)))
    print('Size of test data = {}'.format(sys.getsizeof(X_test)))
    return X_train, ns_train, X_valid, ns_valid, X_test, ns_test, mu, std

def data_loader(concs_train, lbs_train, concs_valid, lbs_valid, ptd):
    mus = []
    ns_train = []
    vars = []

    for i in range(concs_train.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        mus.append(np.mean(x, axis=0))
        vars.append(np.var(x, axis=0))
        ns_train.append(x.shape[0])

        if i == 0:
            X_train = x

        else:
            X_train = np.concatenate((X_train, x), axis=0)

    mus = np.vstack(mus)
    ns = np.hstack(ns_train)
    vars = np.vstack(vars)

    mu = np.sum(ns*mus.T, axis=1) / np.sum(ns)
    std = np.sqrt(np.sum((ns*(vars.T + (mus.T - mu.reshape(-1, 1))**2)), axis=1) / np.sum(ns))

    ns_valid = []

    for i in range(concs_valid.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_valid[i], lbs_valid[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        ns_valid.append(x.shape[0])

        if i == 0:
            X_valid = x

        else:
            X_valid = np.concatenate((X_valid, x), axis=0)

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