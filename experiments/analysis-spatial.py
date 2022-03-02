import os
import sys
sys.path.append('../../')
import argparse

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import MDAnalysis as mda

from utils.util import train_test_split, train_valid_split, VanillaNN
from preprocessing.utils.mda_util import mda_to_numpy


def spatial_correlation_function(anion_positions, cation_positions, kas, kcs, min_r_value=0, max_r_value=4.0, bin_size=0.1, box_length=12.0):
    # anions = [n_snapshots, n_anions, 3]
    x = np.arange(min_r_value+0.5*bin_size, max_r_value+0.5*bin_size, bin_size)
    assert kas.shape == kcs.shape # [n_snapshots, n_anions]
    print(kas.shape)
    T = kas.shape[0] # number of snapshots
    n = kas.shape[1] # number of anions

    y_aa = np.zeros(x.shape[0])
    y_ac = np.zeros(x.shape[0])
    y_cc = np.zeros(x.shape[0])

    n_aa = np.zeros(x.shape[0])
    n_ac = np.zeros(x.shape[0])
    n_cc = np.zeros(x.shape[0])

    ss_mn_a = np.mean(kas, axis=1) # Average conductivity per snapshot
    ss_mn_c = np.mean(kcs, axis=1)

    for tau in range(0, T):

        anions = anion_positions[tau, :, :]
        cations = cation_positions[tau, :, :]

        product_aa = np.matmul((kas[tau, :] - ss_mn_a[tau]).reshape(-1, 1), (kas[tau, :] - ss_mn_a[tau]).reshape(1, -1))
        distances_aa = np.zeros(product_aa.shape)


        product_ac = np.matmul((kas[tau, :] - ss_mn_a[tau]).reshape(-1, 1), (kcs[tau, :] - ss_mn_c[tau]).reshape(1, -1))
        distances_ac = np.zeros(product_ac.shape)

        product_cc = np.matmul((kcs[tau, :] - ss_mn_c[tau]).reshape(-1, 1), (kcs[tau, :] - ss_mn_c[tau]).reshape(1, -1))
        distances_cc = np.zeros(product_cc.shape)

        for i in range(anions.shape[0]):
            anion = anions[i, :].reshape(1, 3)
            distances_aa[i, :] = np.linalg.norm(np.minimum(((anions - anion) % box_length), ((anion - anions) % box_length)),
                                             axis=1)
            distances_ac[i, :] = np.linalg.norm(np.minimum(((cations - anion) % box_length), ((anion - cations) % box_length)),
                                             axis=1)
        product_aa = product_aa.reshape(-1)
        distances_aa = distances_aa.reshape(-1)
        product_ac = product_ac.reshape(-1)
        distances_ac = distances_ac.reshape(-1)

        for i in range(cations.shape[0]):
            cation = cations[i, :].reshape(1, 3)
            distances_cc[i, :] = np.linalg.norm(np.minimum(((cations - cation) % box_length), ((cation - cations) % box_length)),
                                                axis=1)
        product_cc = product_cc.reshape(-1)
        distances_cc = distances_cc.reshape(-1)

        for j in range(x.shape[0]):
            selected_aa = product_aa[np.where(np.abs(distances_aa - x[j]) < 0.5*bin_size)].reshape(-1)
            selected_ac = product_ac[np.where(np.abs(distances_ac - x[j]) < 0.5*bin_size)].reshape(-1)
            selected_cc = product_cc[np.where(np.abs(distances_cc - x[j]) < 0.5*bin_size)].reshape(-1)

            y_aa[j] += selected_aa.sum()
            n_aa[j] += selected_aa.sum()
            y_ac[j] += selected_ac.sum()
            n_ac[j] += selected_ac.sum()
            y_aa[j] += selected_cc.sum()
            n_aa[j] += selected_cc.sum()

    scf_aa = np.zeros_like(y_aa)
    scf_ac = np.zeros_like(y_ac)

    non_zero = n_aa != 0
    scf_aa[non_zero] = np.divide(y_aa[non_zero], n_aa[non_zero])

    non_zero = n_ac != 0
    scf_ac[non_zero] = np.divide(y_ac[non_zero], n_ac[non_zero])

    return x, scf_aa, scf_ac


def save_scalers(n_split, ptd):
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

    data_valid, data_train = train_valid_split(data_train_valid, n_split=n_split)
    (true_train, true_err_train, concs_train, lbs_train) = data_train
    (true_valid, true_err_valid, concs_valid, lbs_valid) = data_valid

    # Get original scaling parameters mu and std from anion training data.
    mus = []
    ns_train = []
    vars = []

    for i in range(concs_train.shape[0]):
        ptf = ptd + 'processed/X_{}_{}_soap'.format(concs_train[i], lbs_train[i]).replace('.', '-') + '.npy'
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

    np.save(ptd + 'processed/mu_x_{}.npy'.format(n_split), mu)
    np.save(ptd + 'processed/std_x_{}.npy'.format(n_split), std)

    sc_y = StandardScaler()
    sc_y.fit(true_train)
    np.save(ptd + 'processed/mu_y_{}.npy'.format(n_split), sc_y.mean_)
    np.save(ptd + 'processed/std_y_{}.npy'.format(n_split), sc_y.scale_)

    return

def weighted_stats(means, stds):
    var = 1.0 / np.sum(np.reciprocal(stds**2))
    mean = var * np.sum(np.multiply(means, np.reciprocal(stds**2)))
    return mean, np.sqrt(var)


def main(args):

    conc = args.conc
    lb = args.lb
    experiment_name = args.experiment_name
    ptd = args.ptd
    if conc == 0.001:
        ptt = args.ptd + 'md-trajectories/' + '0001/'
    else:
        ptt = args.ptd + 'md-trajectories/' + '{}/'.format(conc)

    pts = '../results/{}/'.format(experiment_name)

    n_splits = args.n_splits
    n_ensembles = args.n_ensembles
    hidden_dims = args.hidden_dims
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
    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptt)

    assert anion_positions.shape == cation_positions.shape
    (n_snapshots, n_anions, _) = anion_positions.shape
    n_cations = cation_positions.shape[1]

    nt = n_snapshots * n_anions
    file_ids = nt // 25000 + 1

    ka = []
    kc = []

    mus_x = []
    stds_x = []
    mus_y = []
    stds_y = []

    models = []
    # Load trained models in eval mode and also the scalers (both x and y)
    for n_split in range(n_splits):
        #save_scalers(n_split, ptd)
        mu_x = np.load(args.ptd + 'processed/mu_x_{}.npy'.format(n_split))
        std_x = np.load(args.ptd + 'processed/std_x_{}.npy'.format(n_split))
        mu_y = np.load(args.ptd + 'processed/mu_y_{}.npy'.format(n_split))
        std_y = np.load(args.ptd + 'processed/std_y_{}.npy'.format(n_split))
        mus_x.append(mu_x)
        stds_x.append(std_x)
        mus_y.append(mu_y)
        stds_y.append(std_y)

        for run_id in range(n_ensembles):
            # Load saved models
            model = VanillaNN(in_dim=mu_x.shape[0], out_dim=1, hidden_dims=hidden_dims)
            model.load_state_dict(torch.load(pts + 'models/' + 'model{}{}_{}.pkl'.format(experiment_name, n_split,
                                                                                         run_id)))
            model.eval()
            models.append(model)

    kas = []
    kcs = []
    for file_id in range(file_ids):
        try:
            xa = np.load(args.ptd + 'processed/X_{}_{}_soap_anion_spatial_{}'.format(conc, lb, file_id).replace('.', '-') + '.npy')
            xc = np.load(args.ptd + 'processed/X_{}_{}_soap_cation_spatial_{}'.format(conc, lb, file_id).replace('.', '-') + '.npy')
            preds_a = []
            preds_c = []
            for n_split in range(n_splits):
                xas = torch.tensor((xa - mus_x[n_split]) / stds_x[n_split], dtype=torch.float32) # apply appropriate scaling
                xcs = torch.tensor((xc - mus_x[n_split]) / stds_x[n_split], dtype=torch.float32) # apply appropriate scaling
                for run_id in range(n_ensembles):
                    pred_a = models[n_ensembles*n_split+run_id].forward(xas).detach().numpy()*stds_y[n_split] + mus_y[n_split]
                    preds_a.append(pred_a.reshape(-1))
                    pred_c = models[n_ensembles*n_split+run_id].forward(xcs).detach().numpy()*stds_y[n_split] + mus_y[n_split]
                    preds_c.append(pred_c.reshape(-1))

            preds_a = np.vstack(preds_a)
            preds_c = np.vstack(preds_c)
            preds_a = np.mean(preds_a, axis=0) # Gives the mean conductivity predicted for each anion (across all models)
            preds_c = np.mean(preds_c, axis=0) # Gives the mean conductivity predicted for each cation (across all models)
            sys_mn_a = np.mean(preds_a)
            sys_mn_c = np.mean(preds_c)
            print('File ID {} Mean (Anion) {:.3f} Mean (Cation) {:.3f}'.format(file_id, sys_mn_a, sys_mn_c))
            kas.append(preds_a)
            kcs.append(preds_c)

        except:
            print('Did not find File {}'.format(file_id))

    kas = np.hstack(kas)
    kcs = np.hstack(kcs)

    kas = kas.reshape((-1, n_anions))
    kcs = kcs.reshape((-1, n_anions))

    print('Predicted total mean (Anion) {:.4f} (Cation) {:.4f}'.format(np.mean(kas), np.mean(kcs)))
    print(kas.shape)
    print(kcs.shape)

    if not os.path.exists(pts + 'predictions/spatial/'):
        os.makedirs(pts + 'predictions/spatial/')

    np.save(pts + 'predictions/spatial/local_pred_{}_{}_anions'.format(conc, lb).replace('.', '-') + '.npy', kas)
    np.save(pts + 'predictions/spatial/local_pred_{}_{}_cations'.format(conc, lb).replace('.', '-') + '.npy', kcs)

    if not os.path.exists(pts + 'predictions/correlation_functions/spatial/'):
        os.makedirs(pts + 'predictions/correlation_functions/spatial/')

    x, scf_aa, scf_ac = spatial_correlation_function(anion_positions, cation_positions, kas, kcs,
                                                     min_r_value=min_r_value, max_r_value=max_r_value,
                                                     bin_size=bin_size,box_length=box_length)
    print(x)
    print(np.round(100*scf_aa, decimals=1))
    print(np.round(100*scf_ac, decimals=1))

    np.save(ptp + 'correlation_functions/spatial/bin_positions_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', x)
    np.save(ptp + 'correlation_functions/spatial/scf_aa_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', scf_aa)
    np.save(ptp + 'correlation_functions/spatial/scf_ac_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', scf_ac)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
                        help='Path to directory containing data.')
    parser.add_argument('--conc', type=float, default=0.005,
                        help='Concentration.')
    parser.add_argument('--lb', type=float, default=10.0,
                        help='Bjerrum length.')
    parser.add_argument('--min_r_value', type=float, default=0.0,
                        help='Minimum r value.')
    parser.add_argument('--max_r_value', type=float, default=6.0,
                        help='Maximum r value.')
    parser.add_argument('--bin_size', type=float, default=0.2,
                        help='Bin size.')
    parser.add_argument('--experiment_name', type=str, default='220104_WL_ENSEMBLE',
                        help='Name of experiment.')
    parser.add_argument('--hidden_dims', nargs='+', type=int,
                        default=[100, 100],
                        help='Dimensionality of network hidden layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Number of systems to use in training.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency.')
    parser.add_argument('--n_ensembles', type=int, default=5,
                        help='Number of ensembles to train per split.')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of train/valid splits.')
    args = parser.parse_args()

    main(args)
