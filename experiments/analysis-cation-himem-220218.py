# -*- coding: utf-8 -*-
"""
Script to predict local conductivity given SOAP descriptors of just cations.
"""
import argparse
import time
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import pdb

from utils.util import train_test_split, train_valid_split, data_loader_full, VanillaNN, aggregate_metrics

def load_scalers(concs_train, lbs_train, ptd):

    # Get original scaling parameters mu and std from anion training data.
    mus = []
    ns_train = []
    vars = []

    for i in range(concs_train.shape[0]):
        ptf = ptd + 'X_{}_{}_soap'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
        print(ptf)
        x = np.load(ptf, allow_pickle=True)
        mus.append(np.mean(x, axis=0))
        vars.append(np.var(x, axis=0))
        ns_train.append(x.shape[0])

    mus = np.vstack(mus)
    ns = np.hstack(ns_train)
    vars = np.vstack(vars)

    mu = np.sum(ns*mus.T, axis=1) / np.sum(ns)
    std = np.sqrt(np.sum((ns*(vars.T + (mus.T - mu.reshape(-1, 1))**2)), axis=1) / np.sum(ns))

    return mu, std

def main(args):
    ptd = args.ptd
    ptx = ptd + 'processed/'
    n_splits = args.n_splits
    n_ensembles = args.n_ensembles
    hidden_dims = args.hidden_dims
    experiment_name = args.experiment_name

    log_name = '{}_log_every_cation.txt'.format(experiment_name)
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

        mu_x, std_x = load_scalers(concs_train, lbs_train, ptx)

        # Scale y data
        sc_y = StandardScaler()
        y_train = torch.tensor(sc_y.fit_transform(true_train), dtype=torch.float32).reshape(-1, 1)

        for run_id in range(n_ensembles):
            # Load saved models
            model = VanillaNN(in_dim=mu_x.shape[0], out_dim=1, hidden_dims=hidden_dims)
            model.load_state_dict(torch.load(pts + 'models/' + 'model{}{}_{}.pkl'.format(experiment_name, n_split,
                                                                                         run_id)))
            model.eval()

            for i in range(concs.shape[0]):
                ptf = ptd + 'X_{}_{}_soap_every_cation'.format(concs[i], lbs[i]).replace('.', '-') + '.npy'
                print(ptf)
                try:
                    x = np.load(ptf, allow_pickle=True)
                except:
                    print('Cannot load full file. Instead use smaller file')
                    ptf = ptd + 'X_{}_{}_soap_cation_reverse'.format(concs[i], lbs[i]).replace('.', '-') + '.npy'
                    x = np.load(ptf, allow_pickle=True)
                x = torch.tensor((x - mu_x) / std_x, dtype=torch.float32)
                print(x.shape)
                local_pred = sc_y.inverse_transform(model.forward(x).detach().numpy())
                pts_local = pts + 'predictions/every/local_pred_{}_{}_{}_{}_every_cation'.format(concs[i], lbs[i], n_split, run_id).replace('.', '-') + '.npy'
                np.save(pts_local, local_pred.reshape(-1))


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
