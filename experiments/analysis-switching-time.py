import os
import sys
sys.path.append('../../')
import argparse

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import MDAnalysis as mda

from utils.util import train_test_split, train_valid_split, VanillaNN

import pdb

def measure_switching_times(ks):

    diff_matrix = np.diff(np.sign(ks), axis=0) / 2 # [T-1, n].

    # 1 if sign changes from negative to positive,
    # 0 if sign doesn't change
    # -1 if sign changes from positive to negative

    time_matrix = np.zeros(ks.shape) # initialise matrix to store switching times

    idx = np.where(diff_matrix != 0) # particles which switch sign in next timeframe last 1 snapshot

    count = 1
    k = count + 1

    while len(idx[0]) != 0:
        time_matrix[idx] = count # particles which switch sign in next timeframe last "count" snapshots
        diff_matrix = diff_matrix[1:, :]
        idx = np.where((diff_matrix != 0) & (time_matrix[:-k] == 0))
        count += 1
        k +=1

    return time_matrix

def compute_histogram(ks, time_matrix, lower_k, upper_k, time_bins):

    mid_k = 0.5*(lower_k + upper_k)
    bin_size = upper_k - lower_k
    times = time_matrix[np.where(np.abs(ks - mid_k) <= 0.5*bin_size)]

    counts, _ = np.histogram(times, bins=time_bins)

    return counts


def compute_histograms(ks, k_bins=np.arange(-1.5, 1.75, 0.25), time_bins=np.arange(0.5, 10.5, 1.0)):

    T = ks.shape[0] # number of snapshots
    n = ks.shape[1] # number of particles

    time_matrix = measure_switching_times(ks) # matrix of time taken to switch sign
    hist_matrix = np.zeros((len(k_bins) - 1, len(time_bins) - 1))
    for i in range(len(k_bins) - 1):
        lower_k = k_bins[i]
        upper_k = k_bins[i+1]
        hist_matrix[i, :] = compute_histogram(ks, time_matrix, lower_k, upper_k, time_bins)

    print(k_bins)
    print(hist_matrix)

    return hist_matrix

def create_mda(dcd_file, data_file): # loads trajectory with unwrapped coordinates
    u = mda.Universe(data_file, dcd_file)
    return u

def check_files_exist(dcd_file, data_file):
    print('Trying to find files at')
    print(dcd_file)
    print(data_file)
    if os.path.isfile(dcd_file) and os.path.isfile(data_file):
        print('Located dcd and data files.')
    else:
        print('Did not locate dcd and data files.')
        sys.exit()
    return

def define_atom_types(u):
    # sort atoms into type of molecule
    anions = u.select_atoms("type Anion")
    cations = u.select_atoms("type Cation")
    solvent = u.select_atoms("type Solvent")
    return cations, anions, solvent

def create_position_arrays(u, anions, cations, solvent):
    # generate numpy arrays with all atom positions
    # position arrays: [time, ion index, spatial dimension (x/y/z)]
    time = 0
    n_times = u.trajectory.n_frames
    anion_positions = np.zeros((n_times, len(anions), 3))
    cation_positions = np.zeros((n_times, len(cations), 3))
    solvent_positions = np.zeros((n_times, len(solvent), 3))
    for ts in u.trajectory:
        anion_positions[time, :, :] = anions.positions - u.atoms.center_of_mass(pbc=True)
        cation_positions[time, :, :] = cations.positions - u.atoms.center_of_mass(pbc=True)
        solvent_positions[time, :, :] = solvent.positions - u.atoms.center_of_mass(pbc=True)
        time += 1
    return anion_positions, cation_positions, solvent_positions

def mda_to_numpy(conc, lb, ptd='../../data/md-trajectories/'):
    if conc == 0.001:
        ptdf = ptd + '0001/'
    else:
        ptdf = ptd + '{}/'.format(conc)
    dcd_file = '{}/conc{}_lb{}_0o5tau.dcd'.format(ptdf, conc, lb)
    data_file = '{}initial_config_conc{}.gsd'.format(ptdf, conc)
    check_files_exist(dcd_file, data_file)
    u = create_mda(dcd_file, data_file)
    box_length = u.dimensions[0] # box length (use to wrap coordinates with periodic boundary conditions)
    cations, anions, solvent = define_atom_types(u)
    anion_positions, cation_positions, solvent_positions = (create_position_arrays(u, anions, cations, solvent))

    return anion_positions, cation_positions, solvent_positions, box_length


def weighted_stats(means, stds):
    var = 1.0 / np.sum(np.reciprocal(stds**2))
    mean = var * np.sum(np.multiply(means, np.reciprocal(stds**2)))
    return mean, np.sqrt(var)


def main(args):

    conc = args.conc
    lb = args.lb
    experiment_name = args.experiment_name
    ptd = args.ptd

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
    anion_positions, cation_positions, solvent_positions, box_length = mda_to_numpy(conc, lb, ptd+'md-trajectories/')

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

        mu_x = np.load(ptd + 'processed/mu_x_{}.npy'.format(n_split))
        std_x = np.load(ptd + 'processed/std_x_{}.npy'.format(n_split))
        mu_y = np.load(ptd + 'processed/mu_y_{}.npy'.format(n_split))
        std_y = np.load(ptd + 'processed/std_y_{}.npy'.format(n_split))
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
            xa = np.load(ptd + 'processed/X_{}_{}_soap_anion_0o5tau_{}'.format(conc, lb, file_id).replace('.', '-') + '.npy')
            xc = np.load(ptd + 'processed/X_{}_{}_soap_cation_0o5tau_{}'.format(conc, lb, file_id).replace('.', '-') + '.npy')
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

    kas = kas.reshape((-1, n_anions)) # Gives the local conductivity of each anion at each snapshot
    kcs = kcs.reshape((-1, n_anions)) # Gives the local conductivity of each cation at each snapshot

    print('Predicted total mean (Anion) {:.4f} (Cation) {:.4f}'.format(np.mean(kas), np.mean(kcs)))
    print(kas.shape)
    print(kcs.shape)

    if not os.path.exists(pts + 'predictions/correlation_functions/temporal/switching-220513/'):
        os.makedirs(pts + 'predictions/correlation_functions/temporal/switching-220513/')

    ks = np.hstack([kas, kcs])
    k_bins = np.arange(-1.75, 1.75, 0.25)
    time_bins = np.arange(0.5, 10.5, 1.0)

    hist_matrix = compute_histograms(ks, k_bins, time_bins)
    np.save(pts + 'predictions/correlation_functions/temporal/switching-220513/k_bins_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', k_bins)
    np.save(pts + 'predictions/correlation_functions/temporal/switching-220513/time_bins_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', time_bins)
    np.save(pts + 'predictions/correlation_functions/temporal/switching-220513/switching_time_counter_{}_{}'.format(conc, lb).replace('.', '-') + '.npy', hist_matrix)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptd', type=str, default='../../data/',
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
