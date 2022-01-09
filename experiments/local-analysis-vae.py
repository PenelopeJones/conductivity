# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from utils.util import train_test_split, train_valid_split, data_loader_full, VanillaNN, aggregate_metrics

import pdb

def main(args):

    n_splits = args.n_splits
    n_ensembles = args.n_ensembles

    experiment_name = args.experiment_name
    ptd = args.ptd
    pts = '../results/{}/'.format(experiment_name)
    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load(ptd + 'bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)

    figsize = (10, 10)
    cmap = cm.get_cmap('YlOrRd_r')
    norm = Normalize(vmin=2.5, vmax=13.0)

    xmin = -2.5
    xmax = 2.5
    ymin = -0.8
    ymax = 1.2
    fontsize = 20
    bins = np.linspace(xmin, xmax, 100)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.0)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    sys_hists = []
    for i in range(concs.shape[0]):
        preds = []
        for n_split in range(n_splits):
            for run_id in range(n_ensembles):
                pts_local = pts + 'predictions/local_pred_{}_{}_{}_{}'.format(concs[i], lbs[i], n_split, run_id).replace('.', '-') + '.npy'
                pred = np.load(pts_local)
                preds.append(pred)
        preds = np.vstack(preds)
        preds_mn = np.mean(preds, axis=0)
        preds_std = np.std(preds, axis=0)
        sys_hist, _ = np.histogram(preds_mn, bins, density=True)
        sys_hists.append(sys_hist.reshape(-1))
    sys_hist_mn = np.mean(np.vstack(sys_hists), axis=0)
    bincentres = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]

    for i in range(concs.shape[0]):



        ax.step(bincentres, sys_hists[i] - sys_hist_mn, where='mid', color=cmap(norm(lbs[i])), alpha=0.7, label='Conc {} lB {}'.format(concs[i], lbs[i]))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('symlog')

        if i % 9 == 8:
            ax.legend(fontsize=14, frameon=False)
            ax.set_xticks([xmin, xmax])
            ax.set_xticklabels([str(xmin), str(xmax)], fontsize=fontsize)
            ax.set_yticks([ymin, ymax])
            ax.set_yticklabels([str(ymin), str(ymax)], fontsize=fontsize)
            ax.set_ylabel('Residual probability density', fontsize=fontsize)
            ax.set_xlabel('Local conductivity', fontsize=fontsize)
            fig.savefig(pts + 'figures/histogram_{}'.format(i//9).replace('.', '-') + '.png', dpi=400)
            plt.close(fig)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            for axis in ['bottom', 'left']:
                ax.spines[axis].set_linewidth(2.0)
            for axis in ['top', 'right']:
                ax.spines[axis].set_visible(False)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2.0)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    ax.step(bincentres, sys_hist_mn, where='mid', alpha=0.7, label='Mean')
    ax.set_yscale('symlog')
    ax.set_ylabel('Residual probability density', fontsize=fontsize)
    ax.set_xlabel('Local conductivity', fontsize=fontsize)
    fig.savefig(pts + 'figures/histogram_mean.png', dpi=400)
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--experiment_name', type=str, default='220104_WL_ENSEMBLE',
                        help='Name of experiment.')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency.')
    parser.add_argument('--n_ensembles', type=int, default=5,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')

    args = parser.parse_args()

    main(args)
