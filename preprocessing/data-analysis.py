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
from torch import optim
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from collections import Counter

import pdb

fontsize = 12

def train_test_split(X, y, test_fraction=0.2, seed=10):
    n = X.shape[0]
    np.random.seed(seed)
    idx = np.random.permutation(n)
    idx_te = idx[0:int(test_fraction*n)]
    idx_tr = idx[int(test_fraction * n):]
    return X[idx_tr, :], y[idx_tr], X[idx_te, :], y[idx_te], idx_tr, idx_te

def sample_batch(X, y, batch_size=1000):
    idx = np.random.permutation(X.shape[0])[:batch_size]
    return X[idx, :], y[idx, :]


def main(args):
    ptl = args.path

    # Load ion positions
    X = np.load(ptl + 'X_6_new.npy')
    y = np.load(ptl + 'y_6_new.npy')

    print(X.shape)
    print(y.shape)

    recounted = Counter(y)
    plt.hist(y, bins=np.arange(0, 100, 1))
    plt.show()

    n_ensembles = 5

    seeds = [10, ]
    start_seed = 1

    for seed in seeds:
        ensemble_states = np.arange(1, n_ensembles + 1, 1)

        n = X.shape[0]
        np.random.seed(seed)
        idx = np.random.permutation(n)

        # Approximately 25 `test' cells per split.
        n_splits = 5
        n_te = n // n_splits + 1

        means = []
        vars = []
        benchmark = []

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))

        # 10-fold cross validation
        for k in range(n_splits):
            if k == (n_splits-1):
                idx_te = idx[k * n_te:]
                idx_tr = idx[0:k * n_te]
            else:
                idx_te = idx[k * n_te: (k + 1) * n_te]
                idx_tr = np.delete(idx, np.arange(k * n_te, (k + 1) * n_te))

            X_test = X[idx_te, :]
            X_train = X[idx_tr, :]
            y_test = y[idx_te]
            y_train = y[idx_tr]

            y_pred_trs = []
            y_pred_tes = []

            # Random forest model. Different set of random seeds for each train/test split.
            states = n_ensembles*ensemble_states + k + start_seed
            for ensemble_state in states:
                regr = RandomForestRegressor(max_depth=10, random_state=ensemble_state)
                regr.fit(X_train, y_train)

                # Make predictions
                y_pred_tr = regr.predict(X_train)
                y_pred_trs.append(y_pred_tr)
                y_pred_te = regr.predict(X_test)
                y_pred_tes.append(y_pred_te)

            y_pred_trs = np.vstack(y_pred_trs)
            y_pred_tes = np.vstack(y_pred_tes)
            y_pred_tr = np.mean(y_pred_trs, axis=0)
            y_pred_tr_err = np.var(y_pred_trs, axis=0)
            y_pred_te = np.mean(y_pred_tes, axis=0)
            y_pred_te_err = np.var(y_pred_tes, axis=0)

            ax.errorbar(y_test, y_pred_te, yerr=np.sqrt(y_pred_te_err), marker='o', linestyle='', capsize=2.0,
                        label='Split {}'.format(k + 1))
            ax1.errorbar(y_train, y_pred_tr, yerr=np.sqrt(y_pred_tr_err), marker='o', linestyle='', capsize=2.0,
                        label='Split {}'.format(k + 1))
            means.append(y_pred_te)
            vars.append(y_pred_te_err)
            benchmark.append(np.mean(y_train) * np.ones(y_pred_te.shape))

        ax.plot(np.linspace(0, 200, 4), np.linspace(0, 200, 4), '-.', c='grey')
        ax.set_xlabel('True', fontsize=fontsize)
        ax.set_ylabel('Predicted', fontsize=fontsize)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.legend(fontsize=fontsize, ncol=2)
        fig.savefig('figures/predictions/rf{}1.png'.format(seed), dpi=400)
        ax1.plot(np.linspace(0, 200, 4), np.linspace(0, 200, 4), '-.', c='grey')
        ax1.set_xlabel('True', fontsize=fontsize)
        ax1.set_ylabel('Predicted', fontsize=fontsize)
        ax1.set_xlim(0, 200)
        ax1.set_ylim(0, 200)
        ax1.legend(fontsize=fontsize, ncol=2)
        fig1.savefig('rf{}_train2.png'.format(seed), dpi=400)

        means = np.hstack(means)
        vars = np.hstack(vars)

        benchmark = np.hstack(benchmark)
        pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/processed/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../data/processed/',
                        help='Path to directory containing data.')
    parser.add_argument('--subdir', type=str, default='6',
                        help='Sub directory of interest.')

    args = parser.parse_args()

    main(args)
