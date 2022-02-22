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

def spatial_correlation_function(anions, cations, conductivities_a, conductivities_c, min_r_value=0, max_r_value=4.0, bin_size=0.1, box_length=12.0):
    x = np.arange(min_r_value+0.5*bin_size, max_r_value+0.5*bin_size, bin_size)
    y_aa = np.zeros(x.shape[0])
    y_ac = np.zeros(x.shape[0])
    y_cc = np.zeros(x.shape[0])

    n_aa = np.zeros(x.shape[0])
    n_ac = np.zeros(x.shape[0])
    n_cc = np.zeros(x.shape[0])

    k_avg_a = np.mean(conductivities_a)
    k_std_a = np.std(conductivities_a)
    k_avg_c = np.mean(conductivities_c)
    k_std_c = np.std(conductivities_c)

    # --
    product = np.matmul((conductivities_a - k_avg_a).reshape(-1, 1), (conductivities_a - k_avg_a).reshape(1, -1))
    distances = np.zeros(product.shape)
    for i in range(anions.shape[0]):
        anion = anions[i, :].reshape(1, 3)
        distances[i, :] = np.linalg.norm(np.minimum(((anions - anion) % box_length), ((anion - anions) % box_length)),
                                         axis=1)
    product = product.reshape(-1)
    distances = distances.reshape(-1)

    for j in range(x.shape[0]):
        selected = product[np.where(np.abs(distances - x[j]) < 0.5*bin_size)].reshape(-1)
        y_aa[j] += selected.sum()
        n_aa[j] += selected.shape[0]

    # -+
    product = np.matmul((conductivities_a - k_avg_a).reshape(-1, 1), (conductivities_c - k_avg_c).reshape(1, -1))
    distances = np.zeros(product.shape)
    for i in range(anions.shape[0]):
        anion = anions[i, :].reshape(1, 3)
        distances[i, :] = np.linalg.norm(np.minimum(((cations - anion) % box_length), ((anion - cations) % box_length)),
                                         axis=1)
    product = product.reshape(-1)
    distances = distances.reshape(-1)

    for j in range(x.shape[0]):
        selected = product[np.where(np.abs(distances - x[j]) < 0.5*bin_size)].reshape(-1)
        y_ac[j] += selected.sum()
        n_ac[j] += selected.shape[0]

    # ++
    product = np.matmul((conductivities_c - k_avg_c).reshape(-1, 1), (conductivities_c - k_avg_c).reshape(1, -1))
    distances = np.zeros(product.shape)
    for i in range(cations.shape[0]):
        cation = cations[i, :].reshape(1, 3)
        distances[i, :] = np.linalg.norm(np.minimum(((cations - cation) % box_length), ((cation - cations) % box_length)),
                                         axis=1)
    product = product.reshape(-1)
    distances = distances.reshape(-1)

    for j in range(x.shape[0]):
        selected = product[np.where(np.abs(distances - x[j]) < 0.5*bin_size)].reshape(-1)
        y_aa[j] += selected.sum()
        n_aa[j] += selected.shape[0]

    return x, y_aa, n_aa, y_ac, n_ac


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

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_snaps = int(nt / n_anions) # number of snapshots needed to get dataset size > nt

    print(n_snaps)

    # Check if can load the higher memory file
    ptfa = '../../data/processed/X_{}_{}_soap_every_anion'.format(conc, lb).replace('.', '-') + '.npy'
    ptfc = '../../data/processed/X_{}_{}_soap_every_cation'.format(conc, lb).replace('.', '-') + '.npy'
    try:
        xa = np.load(ptfa, allow_pickle=True)
        #xc = np.load(ptfc, allow_pickle=True)
        skip_snaps = 1
    except:
        print('Cannot load full file. Instead using smaller file.')
        skip_snaps = n_snapshots // n_snaps

    print(skip_snaps)

    # Load local conductivity predictions
    ptp = '../results/{}/'.format(experiment_name)
    preds_a = []
    preds_c = []
    for n_split in range(n_splits):
        for run_id in range(n_ensembles):
            ptsa_local = ptp + 'predictions/every/local_pred_{}_{}_{}_{}_every_anion'.format(conc, lb, n_split, run_id).replace('.', '-') + '.npy'
            ptsc_local = ptp + 'predictions/every/local_pred_{}_{}_{}_{}_every_cation'.format(conc, lb, n_split, run_id).replace('.', '-') + '.npy'
            pred_a = np.load(ptsa_local)
            preds_a.append(pred_a)
            pred_c = np.load(ptsc_local)
            preds_c.append(pred_c)

    preds_a = np.vstack(preds_a)
    preds_c = np.vstack(preds_c)
    preds_a_mn = np.mean(preds_a, axis=0) # Gives the mean conductivity predicted for each anion (across all models)
    preds_c_mn = np.mean(preds_c, axis=0) # Gives the mean conductivity predicted for each cation (across all models)

    sys_mn_a = np.mean(preds_a)
    sys_mn_c = np.mean(preds_c)

    if not os.path.exists(ptp):
        os.makedirs(ptp)

    if not os.path.exists(ptp + 'correlation_functions'):
        os.makedirs(ptp + 'correlation_functions')

    if not os.path.exists(ptp + 'correlation_functions/220222/'):
        os.makedirs(ptp + 'correlation_functions/220222/')

    idx = 0

    nums = []
    means = []
    stds = []

    n_snaps = skip_snaps * preds_a_mn.shape[0] // n_anions
    print(n_snaps)
    print(n_snapshots)

    for snapshot_id in range(0, n_snaps, max(1, skip_snaps)):
        # Select ion positions at a given snapshot
        anions = anion_positions[snapshot_id, :, :]
        cations = cation_positions[snapshot_id, :, :]
        conductivities_a_mn = preds_a_mn[idx:(idx+anions.shape[0])]
        conductivities_c_mn = preds_c_mn[idx:(idx+cations.shape[0])]
        snapshot_a_mn = np.mean(conductivities_a_mn)
        snapshot_c_mn = np.mean(conductivities_c_mn)

        if ((snapshot_a_mn < 0) or (snapshot_c_mn < 0)):
            print('Snapshot {} Conductivity {:.3f} {:.3f}'.format(snapshot_id, snapshot_a_mn, snapshot_c_mn))
        idx += anions.shape[0]

        if snapshot_id == 0:
            x, y_aa, n_aa, y_ac, n_ac = spatial_correlation_function(anions, cations, conductivities_a_mn,
                                                                         conductivities_c_mn, min_r_value=min_r_value,
                                                                         max_r_value=max_r_value, bin_size=bin_size,
                                                                         box_length=box_length)
            ncf_nom_aa = y_aa
            ncf_denom_aa = n_aa
            ncf_nom_ac = y_ac
            ncf_denom_ac = n_ac


        else:
            _, y_aa, n_aa, y_ac, n_ac = spatial_correlation_function(anions, cations, conductivities_a_mn,
                                                                         conductivities_c_mn, min_r_value=min_r_value,
                                                                         max_r_value=max_r_value, bin_size=bin_size,
                                                                         box_length=box_length)
            ncf_nom_aa += y_aa
            ncf_denom_aa += n_aa
            ncf_nom_ac += y_ac
            ncf_denom_ac += n_ac

    ncf_aa = np.zeros_like(y_aa)
    ncf_ac = np.zeros_like(y_ac)

    non_zero = ncf_denom_aa != 0
    ncf_aa[non_zero] = np.divide(ncf_nom_aa[non_zero], ncf_denom_aa[non_zero])

    non_zero = ncf_denom_ac != 0
    ncf_ac[non_zero] = np.divide(ncf_nom_ac[non_zero], ncf_denom_ac[non_zero])

    np.save(ptp + 'correlation_functions/220222/220222_bin_positions_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', x)
    np.save(ptp + 'correlation_functions/220222/220222_scf_aa_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', ncf_aa)
    np.save(ptp + 'correlation_functions/220222/220222_scf_ac_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', ncf_ac)

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
