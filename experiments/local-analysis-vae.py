# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

    bins = np.hstack([np.linspace(-12, -5, 8), np.linspace(-4, 4, 41), np.linspace(5, 12, 8)])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
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
        ax.step(bincentres, sys_hists[i] - sys_hist_mn, where='mid', alpha=0.3, label='Conc {} lB {}'.format(concs[i], lbs[i]))
        ax.set_xlim(-12, 12)
        ax.set_ylim(0.00001, 10)
        if i % 9 == 8:
            ax.legend(fontsize=14, frameon=False)
            fig.savefig(pts + 'figures/histogram_{}'.format(i//9).replace('.', '-') + '.png', dpi=400)
            plt.close(fig)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--experiment_name', type=str, default='NEW_VAE_ENSEMBLE',
                        help='Name of experiment.')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency.')
    parser.add_argument('--n_ensembles', type=int, default=5,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')

    args = parser.parse_args()

    main(args)
