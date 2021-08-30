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
fontsize = 16
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

    experiment_name = 'A'
    log_name = 'log_{}.txt'.format(experiment_name)
    dir_to_save = 'results/{}/'.format(experiment_name)
    n_seeds = 5

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.0)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))

    # 10-fold cross validation
    for seed in range(n_seeds):
        pred_tr = np.load(dir_to_save + 'predictions/' + '{}{}_pred_tr.npy'.format(experiment_name, seed))
        y_tr = np.load(dir_to_save + 'predictions/' + '{}{}_y_tr.npy'.format(experiment_name, seed))
        y_err_tr = np.load(dir_to_save + 'predictions/' + '{}{}_y_err_tr.npy'.format(experiment_name, seed))
        pred_val = np.load(dir_to_save + 'predictions/' + '{}{}_pred_val.npy'.format(experiment_name, seed))
        y_val = np.load(dir_to_save + 'predictions/' + '{}{}_y_val.npy'.format(experiment_name, seed))
        y_err_val = np.load(dir_to_save + 'predictions/' + '{}{}_y_err_val.npy'.format(experiment_name, seed))

        ax1.errorbar(y_tr, pred_tr, xerr=y_err_tr.reshape(-1), marker='o', linestyle='', capsize=2.0, label='Seed {}'.format(seed))
        ax.errorbar(y_val, pred_val, xerr=y_err_val.reshape(-1), marker='o', linestyle='', capsize=2.0, label='Seed {}'.format(seed))

    #ax.plot(np.linspace(0, 200, 4), np.linspace(0, 200, 4), '-.', c='grey')
    #ax.set_xlabel('Actual RUL', fontsize=fontsize+4)
    #ax.set_ylabel('Predicted RUL', fontsize=fontsize+4)
    #ax.set_xlim(0, 205)
    #ax.set_ylim(0, 205)
    #ax.set_xticks([0, 200])
    #ax.set_yticks([0, 200])
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    #sm =  ScalarMappable(norm=norm, cmap=cmap)
    #cbar = fig.colorbar(sm, ax=ax)
    #cbar.set_ticks([])
    #ax.legend(fontsize=fontsize, ncol=2)
    plt.tight_layout()
    plt.show()
    pdb.set_trace()

    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
        os.makedirs(dir_to_save + 'figures/')
    fig.savefig(dir_to_save + 'figures/validation_{}.png'.format(experiment_name), dpi=400)
    fig.savefig(dir_to_save + 'figures/train_{}.png'.format(experiment_name), dpi=400)


if __name__ == '__main__':
    main()
