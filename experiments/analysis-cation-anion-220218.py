# -*- coding: utf-8 -*-
"""
Script to predict local conductivity given SOAP descriptors of both cations and anions.
"""
import argparse
import time
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import pdb

from utils.util import train_test_split, train_valid_split, data_loader_full, VanillaNN, aggregate_metrics

def data_loader_all(concs_train, lbs_train, concs_valid, lbs_valid, concs_test, lbs_test, ptd):

    # Get original scaling parameters mu and std from anion training data.
    mus = []
    ns_train = []
    vars = []

    for i in range(concs_train.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        mus.append(np.mean(x, axis=0))
        vars.append(np.var(x, axis=0))
        ns_train.append(x.shape[0])

    mus = np.vstack(mus)
    ns = np.hstack(ns_train)
    vars = np.vstack(vars)

    mu = np.sum(ns*mus.T, axis=1) / np.sum(ns)
    std = np.sqrt(np.sum((ns*(vars.T + (mus.T - mu.reshape(-1, 1))**2)), axis=1) / np.sum(ns))

    # We are now testing on the data generated only from cation local environments. Need to check what these plots look like.
    ns_train = []
    for i in range(concs_train.shape[0]):
        ptf = ptd + 'X_{}_{}_soap_cat_an'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        ns_train.append(x.shape[0])

        if i == 0:
            X_train = x

        else:
            X_train = np.concatenate((X_train, x), axis=0)

    ns_valid = []

    for i in range(concs_valid.shape[0]):
        ptf = ptd + 'X_{}_{}_soap_cat_an'.format(concs_valid[i], lbs_valid[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        ns_valid.append(x.shape[0])

        if i == 0:
            X_valid = x

        else:
            X_valid = np.concatenate((X_valid, x), axis=0)

    ns_test = []

    for i in range(concs_test.shape[0]):
        ptf = ptd + 'X_{}_{}_soap_cat_an'.format(concs_test[i], lbs_test[i]).replace('.', '-') + '.npy'
        x = np.load(ptf, allow_pickle=True)
        ns_test.append(x.shape[0])

        if i == 0:
            X_test = x

        else:
            X_test = np.concatenate((X_test, x), axis=0)

    X_train = torch.tensor(((X_train - mu) / std), dtype=torch.float32)
    X_valid = torch.tensor(((X_valid - mu) / std), dtype=torch.float32)
    X_test = torch.tensor(((X_test - mu) / std), dtype=torch.float32)

    return X_train, ns_train, X_valid, ns_valid, X_test, ns_test, mu, std

def main(args):
    ptd = args.ptd
    ptx = ptd + 'processed/'
    n_splits = args.n_splits
    n_ensembles = args.n_ensembles
    hidden_dims = args.hidden_dims
    experiment_name = args.experiment_name

    log_name = '{}_log_analysis.txt'.format(experiment_name)
    pts = '../results/{}/'.format(experiment_name)

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
    (true_test, true_err_test, concs_test, lbs_test) = data_test

    f = open(pts + log_name, 'w')

    for n_split in range(n_splits):

        f.write('\nLoading data for split {}...'.format(n_split))
        f.flush()

        data_valid, data_train = train_valid_split(data_train_valid, n_split=n_split)
        (true_train, true_err_train, concs_train, lbs_train) = data_train
        (true_valid, true_err_valid, concs_valid, lbs_valid) = data_valid

        # Load training and validation data
        X_train, ns_train, X_valid, ns_valid, X_test, ns_test, mu_x, std_x = data_loader_all(concs_train, lbs_train, concs_valid, lbs_valid, concs_test, lbs_test, ptx)

        # Scale y data
        sc_y = StandardScaler()
        y_train = torch.tensor(sc_y.fit_transform(true_train), dtype=torch.float32).reshape(-1, 1)

        for run_id in range(n_ensembles):
            # Load saved models
            model = VanillaNN(in_dim=mu_x.shape[0], out_dim=1, hidden_dims=hidden_dims)
            model.load_state_dict(torch.load(pts + 'models/' + 'model{}{}_{}.pkl'.format(experiment_name, n_split,
                                                                                         run_id)))
            model.eval()

            # Make predictions
            idx_train = []
            idx = 0
            for i in range(len(ns_train)):
                idx += ns_train[i]
                idx_train.append(idx)
            idx_valid = []
            idx = 0
            for i in range(len(ns_valid)):
                idx += ns_valid[i]
                idx_valid.append(idx)
            idx_test = []
            idx = 0
            for i in range(len(ns_test)):
                idx += ns_test[i]
                idx_test.append(idx)
            local_preds_train = sc_y.inverse_transform(model.forward(X_train).detach().numpy())
            local_preds_train = np.split(local_preds_train, idx_train)
            local_preds_valid = sc_y.inverse_transform(model.forward(X_valid).detach().numpy())
            local_preds_valid = np.split(local_preds_valid, idx_valid)
            local_preds_test = sc_y.inverse_transform(model.forward(X_test).detach().numpy())
            local_preds_test = np.split(local_preds_test, idx_test)

            for i in range(concs_train.shape[0]):
                pts_local = pts + 'predictions/all/local_pred_{}_{}_{}_{}_cat_an'.format(concs_train[i], lbs_train[i], n_split, run_id).replace('.', '-') + '.npy'
                np.save(pts_local, local_preds_train[i].reshape(-1))

            for i in range(concs_valid.shape[0]):
                pts_local = pts + 'predictions/all/local_pred_{}_{}_{}_{}_cat_an'.format(concs_valid[i], lbs_valid[i], n_split, run_id).replace('.', '-') + '.npy'
                np.save(pts_local, local_preds_valid[i].reshape(-1))

            for i in range(concs_test.shape[0]):
                pts_local = pts + 'predictions/all/local_pred_{}_{}_{}_{}_cat_an'.format(concs_test[i], lbs_test[i], n_split, run_id).replace('.', '-') + '.npy'
                np.save(pts_local, local_preds_test[i].reshape(-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--experiment_name', type=str, default='220104_WL_ENSEMBLE',
                        help='Name of experiment.')
    parser.add_argument('--hidden_dims', nargs='+', type=int,
                        default=[50, ],
                        help='Dimensionality of network hidden layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Number of systems to use in training.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency.')
    parser.add_argument('--n_ensembles', type=int, default=1,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')

    args = parser.parse_args()

    main(args)
