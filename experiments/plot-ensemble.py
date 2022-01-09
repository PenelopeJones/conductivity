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

    experiment_name = '220104_WL_ENSEMBLE'
    n_splits = 5
    n_ensembles = 5

    ptd = '../data/'

    log_name = 'log_{}.txt'.format(experiment_name)
    dir_to_save = '../results/{}/'.format(experiment_name)
    pts = dir_to_save
    n_seeds = 5

    y = np.load(ptd + 'molar_conductivities.npy')
    #y = y.reshape(-1, 1)
    y_err = np.load(ptd + 'molar_conductivities_error.npy')
    #y_err = y_err.reshape(-1, 1)

    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load('../data/bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)
    y = np.load('../data/molar_conductivities.npy')
    y = y.reshape(-1)
    y_err = np.load('../data/molar_conductivities_error.npy')
    y_err = y_err.reshape(-1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.0)
        ax1.spines[axis].set_linewidth(2.0)
        ax2.spines[axis].set_linewidth(2.0)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
        ax1.spines[axis].set_visible(False)
        ax2.spines[axis].set_visible(False)

    concs_ts = []
    lbs_ts = []
    pred_mns = []
    pred_stds = []
    trues = []
    trues_errs = []

    for n_split in range(n_splits):

        pred_tr = []
        pred_val = []

        for run_id in range(n_ensembles):
            pred_train = np.load(pts + 'predictions/' + '{}{}_{}_pred_train.npy'.format(experiment_name, n_split, run_id))
            pred_valid = np.load(pts + 'predictions/' + '{}{}_{}_pred_valid.npy'.format(experiment_name, n_split, run_id))
            pred_tr.append(pred_train)
            pred_val.append(pred_valid)
            if run_id == 0:
                y_tr = np.load(pts + 'predictions/' + '{}{}_{}_y_train.npy'.format(experiment_name, n_split, run_id))
                y_err_tr = np.load(pts + 'predictions/' + '{}{}_{}_y_err_train.npy'.format(experiment_name, n_split, run_id))
                concs_train = np.load(pts + 'predictions/' + '{}{}_{}_concs_train.npy'.format(experiment_name, n_split, run_id))
                lbs_train = np.load(pts + 'predictions/' + '{}{}_{}_lbs_tr.npy'.format(experiment_name, n_split, run_id))

                y_val = np.load(pts + 'predictions/' + '{}{}_{}_y_valid.npy'.format(experiment_name, n_split, run_id))
                y_err_val = np.load(pts + 'predictions/' + '{}{}_{}_y_err_valid.npy'.format(experiment_name, n_split, run_id))
                concs_valid = np.load(pts + 'predictions/' + '{}{}_{}_concs_valid.npy'.format(experiment_name, n_split, run_id))
                lbs_valid = np.load(pts + 'predictions/' + '{}{}_{}_lbs_valid.npy'.format(experiment_name, n_split, run_id))

                trues.append(y_val.reshape(-1))
                trues_errs.append(y_err_val.reshape(-1))
                lbs_ts.append(lbs_valid.reshape(-1))
                concs_ts.append(concs_valid.reshape(-1))


        pred_tr = np.hstack(pred_tr)
        pred_val = np.hstack(pred_val)

        pred_err_tr = np.std(pred_tr, axis=1).reshape(-1)
        pred_tr = np.mean(pred_tr, axis=1).reshape(-1)
        pred_err_val = np.std(pred_val, axis=1).reshape(-1)
        pred_val = np.mean(pred_val, axis=1).reshape(-1)

        pred_mns.append(pred_val)
        pred_stds.append(pred_err_val)


        #ax1.scatter(y_tr, pred_tr, marker='o', label='Split {}'.format(n_split))
        #ax.scatter(y_val, pred_val, marker='o', label='Split {}'.format(n_split))
        ax1.errorbar(y_tr.reshape(-1), pred_tr.reshape(-1), xerr=y_err_tr.reshape(-1), yerr=pred_err_tr.reshape(-1), marker='o', linestyle='', capsize=2.0, label='Split {}'.format(n_split))
        ax.errorbar(y_val.reshape(-1), pred_val.reshape(-1), xerr=y_err_val.reshape(-1), yerr=pred_err_val.reshape(-1), marker='o', linestyle='', capsize=2.0, label='Split {}'.format(n_split))
        """
                if n_split == 0:
                    ax2.scatter(true_err_tr, total_err_tr, marker='o', color='red', label='Train')
                    ax2.scatter(true_err_val, total_err_val, marker='o', color='blue', label='Validation')
                else:
                    ax2.scatter(true_err_tr, total_err_tr, marker='o', color='red')
                    ax2.scatter(true_err_val, total_err_val, marker='o', color='blue')
        if n_split == 0:
            ax2.scatter(true_err_tr, y_err_tr, marker='o', color='red', label='Train')
            ax2.scatter(true_err_val, y_err_val, marker='o', color='blue', label='Validation')
        else:
            ax2.scatter(true_err_tr, y_err_tr, marker='o', color='red')
            ax2.scatter(true_err_val, y_err_val, marker='o', color='blue')
        if n_split == 0:
            ax2.scatter(true_err_tr, pred_err_tr, marker='o', color='red', label='Train')
            ax2.scatter(true_err_val, pred_err_val, marker='o', color='blue', label='Validation')
        else:
            ax2.scatter(true_err_tr, pred_err_tr, marker='o', color='red')
            ax2.scatter(true_err_val, pred_err_val, marker='o', color='blue')

        if n_split == 0:
            ax2.scatter(y_err_tr, pred_err_tr.reshape(-1), marker='o', color='red', label='Train')
            ax2.scatter(y_err_val, pred_err_val.reshape(-1), marker='o', color='blue', label='Validation')
        else:
            ax2.scatter(y_err_tr, pred_err_tr.reshape(-1), marker='o', color='red')
            ax2.scatter(y_err_val, pred_err_val.reshape(-1), marker='o', color='blue')
        """
    ax1.plot(np.linspace(0, 0.18, 4), np.linspace(0, 0.18, 4), '-.', c='grey')
    ax.plot(np.linspace(0, 0.18, 4), np.linspace(0, 0.18, 4), '-.', c='grey')
    ax.set_xlabel('Actual molar conductivity', fontsize=fontsize+4)
    ax.set_ylabel('Predicted molar conductivity', fontsize=fontsize+4)
    ax.set_xlim(0.0, 0.17)
    ax.set_ylim(0.0, 0.17)
    ax.set_xticks([0, 0.16])
    ax.set_yticks([0, 0.16])
    ax.set_yticklabels([0, 0.16], fontsize=fontsize)
    ax.set_xticklabels([0, 0.16], fontsize=fontsize)
    ax1.set_xlabel('Actual molar conductivity', fontsize=fontsize+4)
    ax1.set_ylabel('Predicted molar conductivity', fontsize=fontsize+4)

    """
    ax2.set_xlabel('|Predicted - True|', fontsize=fontsize+4)
    ax2.set_ylabel('(E.U.^2 + A.U.^2)^0.5', fontsize=fontsize+4)
    ax2.set_ylabel('Aleatoric uncertainty', fontsize=fontsize+4)
    ax2.set_ylabel('Predicted error', fontsize=fontsize+4)
    ax2.set_xlabel('Actual error', fontsize=fontsize+4)
    ax2.set_ylabel('Predicted error', fontsize=fontsize+4)
    ax2.legend()
    ax2.set_xlim(0.0, 0.04)
    ax2.set_ylim(0.0, 0.04)
    ax2.set_xticks([0, 0.04])
    ax2.set_yticks([0, 0.04])
    ax2.set_yticklabels([0, 0.04], fontsize=fontsize)
    ax2.set_xticklabels([0, 0.04], fontsize=fontsize)
    fig2.tight_layout()
    fig2.savefig(dir_to_save + 'figures/total_err_{}.png'.format(experiment_name), dpi=400)
    """
    ax1.set_xlim(0.0, 0.17)
    ax1.set_ylim(0.0, 0.17)
    ax1.set_xticks([0, 0.16])
    ax1.set_yticks([0, 0.16])
    ax1.set_yticklabels([0, 0.16], fontsize=fontsize)
    ax1.set_xticklabels([0, 0.16], fontsize=fontsize)
    fig.tight_layout()
    fig1.tight_layout()

    if not os.path.exists(dir_to_save + 'figures/'):
        os.makedirs(dir_to_save + 'figures/')
    fig.savefig(dir_to_save + 'figures/validation_{}.png'.format(experiment_name), dpi=400)
    fig1.savefig(dir_to_save + 'figures/train_{}.png'.format(experiment_name), dpi=400)

    pdb.set_trace()
    concs_ts = np.hstack(concs_ts).reshape(-1)
    lbs_ts = np.hstack(lbs_ts).reshape(-1)
    trues = np.hstack(trues).reshape(-1)
    trues_errs = np.hstack(trues_errs).reshape(-1)
    pred_mns = np.hstack(pred_mns).reshape(-1)
    pred_stds = np.hstack(pred_stds).reshape(-1)
    pdb.set_trace()

    np.save(dir_to_save + 'predictions/concs.npy', concs_ts)
    np.save(dir_to_save + 'predictions/lbs.npy', lbs_ts)
    np.save(dir_to_save + 'predictions/trues.npy', trues)
    np.save(dir_to_save + 'predictions/trues_errs.npy', trues_errs)
    np.save(dir_to_save + 'predictions/pred_mns.npy', pred_mns)
    np.save(dir_to_save + 'predictions/pred_stds.npy', pred_stds)

    pdb.set_trace()


if __name__ == '__main__':
    main()
