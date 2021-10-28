import os
import argparse
import time
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
mpl.rc('font', family='Times New Roman')
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from matplotlib.cm import ScalarMappable

import pdb

# Graph parameters
alpha = 0.7
capsize = 2.0
ms = 6
fontsize = 28
linewidth = 3.0

def classify(array, fraction=0.1, top=True):
    if top:
        sorted = np.sort(array)[::-1]
        n_top = int(fraction*array.shape[0])
        boundary = sorted[n_top]
        classified = (array > boundary)
    else:
        sorted = np.sort(array)
        n_bottom = int(fraction * array.shape[0])
        boundary = sorted[n_bottom]
        classified = (array < boundary)

    return classified




def train_test_split(X, y, test_fraction=0.2, seed=10):
    n = X.shape[0]
    np.random.seed(seed)
    idx = np.random.permutation(n)
    idx_tr = idx[0:int(test_fraction*n)]
    idx_te = idx[int(test_fraction * n):]
    return X[idx_tr, :], y[idx_tr], X[idx_te, :], y[idx_te], idx_tr, idx_te



def main():

    experiment_name = 'FULL_C'
    log_name = 'log_{}.txt'.format(experiment_name)
    dir_to_save = 'results/{}/'.format(experiment_name)
    n_seeds = 4

    ptd = '../data/'
    y = np.load(ptd + 'molar_conductivities.npy')
    y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    y_err = y_err.reshape(-1, 1)


    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.0)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    for axis in ['bottom', 'left']:
        ax1.spines[axis].set_linewidth(2.0)
    for axis in ['top', 'right']:
        ax1.spines[axis].set_visible(False)

    # 10-fold cross validation
    for seed in range(n_seeds):
        pred_tr = np.load(dir_to_save + 'predictions/' + '{}{}_pred_train.npy'.format(experiment_name, seed))
        y_tr = np.load(dir_to_save + 'predictions/' + '{}{}_y_train.npy'.format(experiment_name, seed))
        y_err_tr = np.load(dir_to_save + 'predictions/' + '{}{}_y_err_train.npy'.format(experiment_name, seed))
        pred_val = np.load(dir_to_save + 'predictions/' + '{}{}_pred_valid.npy'.format(experiment_name, seed))
        y_val = np.load(dir_to_save + 'predictions/' + '{}{}_y_valid.npy'.format(experiment_name, seed))
        y_err_val = np.load(dir_to_save + 'predictions/' + '{}{}_y_err_valid.npy'.format(experiment_name, seed))

        ax1.scatter(y_tr, pred_tr, marker='o', label='Seed {}'.format(seed))
        ax.scatter(y_val, pred_val, marker='o', label='Seed {}'.format(seed))
        #ax1.errorbar(y_tr, pred_tr, xerr=y_err_tr.reshape(-1), marker='o', linestyle='', capsize=2.0, label='Seed {}'.format(seed))
        #ax.errorbar(y_val, pred_val, xerr=y_err_val.reshape(-1), marker='o', linestyle='', capsize=2.0, label='Seed {}'.format(seed))

    #ax.plot(np.linspace(0, 0.18, 4), np.linspace(0, 0.18, 4), '-.', c='grey')
    ax.set_xlabel('Actual conductivity', fontsize=fontsize+4)
    ax.set_ylabel('Predicted conductivity', fontsize=fontsize+4)
    ax.set_xlim(0.0, 0.17)
    ax.set_ylim(0.0, 0.17)
    ax.set_xticks([0, 0.16])
    ax.set_yticks([0, 0.16])
    ax.set_yticklabels([0, 0.16], fontsize=fontsize)
    ax.set_xticklabels([0, 0.16], fontsize=fontsize)
    ax1.set_xlabel('Actual conductivity', fontsize=fontsize+4)
    ax1.set_ylabel('Predicted conductivity', fontsize=fontsize+4)
    ax1.set_xlim(0.0, 0.17)
    ax1.set_ylim(0.0, 0.17)
    ax1.set_xticks([0, 0.16])
    ax1.set_yticks([0, 0.16])
    ax1.set_yticklabels([0, 0.16], fontsize=fontsize)
    ax1.set_xticklabels([0, 0.16], fontsize=fontsize)
    #ax1.set_xlim(0.07, 0.095)
    #ax1.set_ylim(0.07, 0.095)

    #sm =  ScalarMappable(norm=norm, cmap=cmap)
    #cbar = fig.colorbar(sm, ax=ax)
    #cbar.set_ticks([])
    #ax.legend(fontsize=fontsize, ncol=2)
    fig.tight_layout()
    fig1.tight_layout()
    if not os.path.exists(dir_to_save + 'figures/'):
        os.makedirs(dir_to_save + 'figures/')
    fig.savefig(dir_to_save + 'figures/validation_{}.png'.format(experiment_name), dpi=400)
    fig1.savefig(dir_to_save + 'figures/train_{}.png'.format(experiment_name), dpi=400)


if __name__ == '__main__':
    main()
