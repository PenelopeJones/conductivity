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
    pts = '../results/{}/'.format(experiment_name)
    concs = np.load(ptd + 'concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load(ptd + 'bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)

    figsize = (7, 7)

    for i in range(concs.shape[0]):
        preds = []
        for n_split in range(n_splits):
            for run_id in range(n_ensembles):
                pts_local = pts + 'predictions/local_pred_{}_{}_{}_{}'.format(concs[i], lbs[i], n_split, run_id).replace('.', '-') + '.npy'
                pred = np.load(pts_local)
                preds.append(pred)
        preds = np.vstack(preds)
        pdb.set_trace()
        preds_mn = np.mean(preds, axis=0)
        preds_std = np.std(preds, axis=0)
        pdb.set_trace()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.hist(preds_mn, bins=50, density=True)
        fig.savefig(pts + 'figures/histogram_{}_{}'.format(concs[i], lbs[i]).replace('.', '-') + '.png', dpi=400)
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
