import os
import sys
sys.path.append('../../')
import argparse

from preprocessing.utils.mda_util import mda_to_numpy

import pdb

import numpy as np

def weighted_stats(means, stds):
    var = 1.0 / np.sum(np.reciprocal(stds**2))
    mean = var * np.sum(np.multiply(means, np.reciprocal(stds**2)))
    return mean, np.sqrt(var)

def correlation_function(anions, conductivities, min_r_value=0, max_r_value=4.0, bin_size=0.1, box_length=12.0):
    x = np.arange(min_r_value+0.5*bin_size, max_r_value+0.5*bin_size, bin_size)
    y = np.zeros(x.shape[0])
    n = np.zeros(x.shape[0])
    k_avg = np.mean(conductivities)
    k_std = np.std(conductivities)
    product = np.matmul((conductivities - k_avg).reshape(-1, 1), (conductivities - k_avg).reshape(1, -1))
    distances = np.zeros(product.shape)
    for i in range(anions.shape[0]):
        anion = anions[i, :].reshape(1, 3)
        distances[i, :] = np.linalg.norm(np.minimum(((anions - anion) % box_length), ((anion - anions) % box_length)),
                                         axis=1)
    product = product.reshape(-1)
    distances = distances.reshape(-1)

    for j in range(x.shape[0]):
        selected = product[np.where(np.abs(distances - x[j]) < 0.5*bin_size)].reshape(-1)
        y[j] += selected.sum()
        n[j] += selected.shape[0]

    unweighted_cf = np.zeros_like(y)
    non_zero = n != 0
    unweighted_cf[non_zero] = np.divide(y[non_zero], n[non_zero])
    weighted_cf = (1.0/(k_std*k_std))*unweighted_cf

    return x, unweighted_cf, weighted_cf, y, n


def main(args):

    conc = args.conc
    lb = args.lb
    experiment_name = args.experiment_name
    if conc == 0.001:
        ptd = args.ptd + '0001/'
    else:
        ptd = args.ptd + '{}/'.format(conc)
    nt = 25000
    n_splits = 5
    n_ensembles = 5
    min_r_value = args.min_r_value
    max_r_value = args.max_r_value
    bin_size = args.bin_size

    # Load parameter lists
    concs = np.load('../../data/concentrations.npy')
    concs = concs.repeat(9).reshape(-1)
    lbs = np.load('../../data/bjerrum_lengths.npy')
    lbs = np.tile(lbs, 12).reshape(-1)
    y = np.load('../../data/molar_conductivities.npy')
    y = y.reshape(-1)
    y_err = np.load('../../data/molar_conductivities_error.npy')
    y_err = y_err.reshape(-1)

    k_avg = y[np.where((concs == conc) & (lbs == lb))][0]
    k_avg_err = y_err[np.where((concs == conc) & (lbs == lb))][0]
    print('Concentration {}\t lB {} True k = {:.4f}+-{:.4f}'.format(conc, lb, k_avg, k_avg_err))

    # Load ion positions
    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptd)

    # Load local conductivity predictions
    ptp = '../results/{}/'.format(experiment_name)
    preds = []
    for n_split in range(n_splits):
        for run_id in range(n_ensembles):
            pts_local = ptp + 'predictions/local_pred_{}_{}_{}_{}'.format(conc, lb, n_split, run_id).replace('.', '-') + '.npy'
            pred = np.load(pts_local)
            preds.append(pred)
    preds = np.vstack(preds)
    preds_mn = np.mean(preds, axis=0) # Gives the mean conductivity predicted for each ion (across all models)
    preds_std = np.std(preds, axis=0)

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_snaps = int(nt / n_anions) # number of snapshots needed to get dataset size > nt
    skip_snaps = n_snapshots // n_snaps
    print(n_snaps)
    print(skip_snaps)

    if not os.path.exists(ptp):
        os.makedirs(ptp)

    if not os.path.exists(ptp + 'snapshots'):
        os.makedirs(ptp + 'snapshots')
    idx = 0
    for snapshot_id in range(0, min(n_snapshots, 25*skip_snaps), max(1, skip_snaps)):
        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        conductivities_mn = preds_mn[idx:(idx+anions.shape[0])]
        conductivities_std = preds_std[idx:(idx + anions.shape[0])]
        idx += anions.shape[0]
        np.save(ptp + 'snapshots/anions_{}_{}_{}.npy'.format(conc, lb, snapshot_id), anions)
        np.save(ptp + 'snapshots/cations_{}_{}_{}.npy'.format(conc, lb, snapshot_id), cations)
        np.save(ptp + 'snapshots/conductivity_mn_{}_{}_{}.npy'.format(conc, lb, snapshot_id), conductivities_mn)


    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--conc', type=float, default=0.045,
                        help='Concentration.')
    parser.add_argument('--lb', type=float, default=10.0,
                        help='Bjerrum length.')
    parser.add_argument('--min_r_value', type=float, default=0.0,
                        help='Minimum r value.')
    parser.add_argument('--max_r_value', type=float, default=12.0,
                        help='Maximum r value.')
    parser.add_argument('--bin_size', type=float, default=0.25,
                        help='Bin size.')
    parser.add_argument('--experiment_name', type=str, default='220104_WL_ENSEMBLE',
                        help='Name of experiment.')
    args = parser.parse_args()

    main(args)
